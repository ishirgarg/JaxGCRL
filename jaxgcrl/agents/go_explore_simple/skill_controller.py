"""Hierarchical discrete-skill controller training routine.

Implements ``GoExploreSimple._train_skill_controller`` (``agent_type="sac_discrete"``):
freeze a pretrained OGBench skill-conditioned policy ``π(a | s, z)`` and train an
online SAC-discrete *high-level controller* that selects discrete skills ``z`` to
maximize task reward, with fixed temporal commitment ``k``.

See ``SKILL_CONTROLLER_DESIGN.md`` for the full spec. Key pieces:
  - ``load_frozen_skill_policy`` / ``make_skill_action_fn``: a generic adapter over
    any OGBench skill-conditioned agent whose policy submodule has signature
    ``policy(obs, skills_onehot)`` (true for ``empowerment_skill``).
  - ``rollout_macro_step``: a ``lax.scan`` over ``k`` env steps under one fixed
    per-env skill, producing one SMDP macro-transition ``(s_t, z, R, s_{t+k}, done)``.
  - ``train_skill_controller``: prefill -> epochs of (collect macro-steps + SAC
    updates) -> dedicated deterministic eval rollout.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxgcrl.envs.wrappers import (
    EpisodeWrapper,
    EvalAutoResetWrapper,
    TrainAutoResetWrapper,
    TrajectoryIdWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import EvalWrapper
from jaxgcrl.agents.go_explore.utils import save_params
from jaxgcrl.agents.go_explore.goal_proposers import create_random_env_goals_proposer
from jaxgcrl.agents.go_explore.empowerment import (
    infer_empowerment_override_indices_from_env,
    load_offline_empowerment_agent,
    make_empowerment_full_obs_builder,
    make_empowerment_obs_builder,
)
from jaxgcrl.agents.go_explore_simple.sac_discrete import (
    DiscreteActorNet,
    DiscreteQNet,
    create_controller_state,
    init_controller_replay,
    insert_controller_replay,
    sample_controller_replay,
    sac_discrete_update,
)


# ── Frozen skill-policy adapter (generic over OGBench skill agents) ─────────────


def load_frozen_skill_policy(self, unwrapped_env, template_key):
    """Load + freeze an OGBench skill-conditioned policy and resolve num_skills.

    Returns ``(emp_agent, num_skills, skill_obs_builder)``. The skill policy is
    never differentiated; we only call its frozen ``policy`` submodule.
    """
    state_size = int(unwrapped_env.state_dim)
    emp_agent, ex_obs_dim, base_obs_template = load_offline_empowerment_agent(
        run_dir=self.skill_policy_run_dir,
        jax_env=unwrapped_env,
        template_rng=template_key,
        epoch=self.skill_policy_epoch,
        use_full_obs=self.use_full_skill_obs,
    )

    if self.use_full_skill_obs:
        if state_size != int(ex_obs_dim):
            raise ValueError(
                "use_full_skill_obs=True requires state_size == ex_obs_dim; "
                f"got state_size={state_size}, ex_obs_dim={ex_obs_dim}."
            )
        skill_obs_builder = make_empowerment_full_obs_builder()
    else:
        ogbench_obs_indices, jaxgcrl_state_indices = (
            infer_empowerment_override_indices_from_env(unwrapped_env)
        )
        skill_obs_builder = make_empowerment_obs_builder(
            jnp.asarray(base_obs_template),
            ogbench_obs_indices,
            jaxgcrl_state_indices,
            state_size=state_size,
        )

    # num_skills: explicit override, else inferred from the checkpoint config.
    ckpt_num_skills = int(emp_agent.config["num_skills"])
    if self.num_skills is not None:
        num_skills = int(self.num_skills)
        if num_skills != ckpt_num_skills:
            raise ValueError(
                f"num_skills override ({num_skills}) disagrees with the loaded skill "
                f"checkpoint's num_skills ({ckpt_num_skills}); they must match."
            )
    else:
        num_skills = ckpt_num_skills

    return emp_agent, num_skills, skill_obs_builder


def make_skill_action_fn(emp_agent, skill_obs_builder, num_skills, *, deterministic):
    """Adapter: (jaxgcrl states, discrete skill indices) -> low-level actions.

    Generic over any OGBench skill-conditioned agent whose policy submodule has
    signature ``policy(obs, skills_onehot)`` returning a ``distrax`` dist. The
    frozen ``emp_agent`` params are captured by closure (no gradient ever flows
    into them) — identical pattern to ``make_offline_empowerment_scorer``.
    """

    def skill_action_fn(states: jnp.ndarray, skill_indices: jnp.ndarray, key: jax.Array):
        emp_obs = skill_obs_builder(states)                  # OGBench-mapped state
        z_onehot = jax.nn.one_hot(skill_indices, num_skills)
        dist = emp_agent.network.select("policy")(emp_obs, z_onehot)
        a = dist.mode() if deterministic else dist.sample(seed=key)
        return jnp.clip(a, -1.0, 1.0)

    return skill_action_fn


# ── Training routine ────────────────────────────────────────────────────────────


def train_skill_controller(
    self,
    config,
    train_env,
    eval_env,
    progress_fn: Callable = lambda *a, **k: None,
):
    """SAC-discrete high-level controller over a frozen skill set.

    ``self`` is the ``GoExploreSimple`` dataclass instance (config holder).
    """
    unwrapped_env = train_env
    eval_env_base = eval_env if eval_env is not None else train_env

    action_size = int(unwrapped_env.action_size)
    state_size = int(unwrapped_env.state_dim)
    goal_size = len(unwrapped_env.goal_indices)
    obs_size = state_size + goal_size
    num_envs = config.num_envs
    k = self.skill_commitment_k
    gamma_low = self.gamma_low
    gamma_high = self.discounting

    rng = jax.random.PRNGKey(config.seed)
    rng, skill_key, actor_key, critic_key, buf_key, env_key, eval_key = jax.random.split(rng, 7)
    np.random.seed(config.seed)

    # ── Frozen skill policy + adapters ───────────────────────────────────────
    emp_agent, num_skills, skill_obs_builder = load_frozen_skill_policy(
        self, unwrapped_env, skill_key
    )
    logging.info("skill controller: num_skills=%d, k=%d", num_skills, k)
    skill_action_train = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills,
        deterministic=self.deterministic_skill_actions,
    )
    skill_action_eval = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills, deterministic=True,
    )

    target_entropy = float(self.controller_target_entropy_scale) * float(np.log(num_skills))

    # ── Controller networks + state ──────────────────────────────────────────
    actor_net = DiscreteActorNet(
        num_skills=num_skills, h_dim=self.h_dim, n_hidden=self.n_hidden,
        skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
    )
    critic_net = DiscreteQNet(
        num_skills=num_skills, n_critics=2, h_dim=self.h_dim, n_hidden=self.n_hidden,
        skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
    )
    controller_state = create_controller_state(
        actor_net, critic_net, obs_size,
        policy_lr=self.policy_lr, critic_lr=self.critic_lr, alpha_lr=self.alpha_lr,
        actor_key=actor_key, critic_key=critic_key,
    )

    replay_state = init_controller_replay(self.controller_replay_size, obs_size)

    # ── Env stacks ───────────────────────────────────────────────────────────
    # Training: TrajectoryIdWrapper -> VmapWrapper -> EpisodeWrapper -> TrainAutoResetWrapper.
    # NOTE: no GoExploreWrapper and no goal proposers — task goals come from the
    # env reset. TrainAutoResetWrapper auto-resets done envs to the goal stored
    # in info['proposed_goals']; we refresh that field with a fresh task goal
    # every macro-step so each auto-reset draws a new random env goal.
    t_env = TrajectoryIdWrapper(unwrapped_env)
    t_env = VmapWrapper(t_env)
    t_env = EpisodeWrapper(t_env, config.episode_length, config.action_repeat)
    t_env = TrainAutoResetWrapper(t_env)

    # Eval: same stack as the rest of the repo (EvalAutoResetWrapper), wrapped in
    # EvalWrapper for episode-metric accumulation.
    e_env = TrajectoryIdWrapper(eval_env_base)
    e_env = VmapWrapper(e_env)
    e_env = EpisodeWrapper(e_env, config.episode_length, config.action_repeat)
    e_env = EvalAutoResetWrapper(e_env)
    e_env = EvalWrapper(e_env)

    random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, num_envs)

    # ── Initial train reset ──────────────────────────────────────────────────
    env_keys = jax.random.split(env_key, num_envs)
    initial_goals = jax.vmap(random_goals_proposer)(env_keys)
    env_state = t_env.reset(env_keys, goal=initial_goals)

    # ── Macro-step rollout (one SMDP transition per env) ─────────────────────
    def rollout_macro_step(env_state, z, key, skill_action_fn):
        """Execute skill ``z`` for ``k`` env steps; return SMDP transition fields.

        R = Σ_{i<k} alive_i · γ_low^i · r_i  (accumulated only up to the first
        in-window termination); done = any termination in the window; next_obs is
        the obs at window end (frozen at the first termination via ``alive``).
        Bootstrap is zeroed by (1−done) so the exact next_obs on done is moot.

        NOTE: this assumes termination is *truncation-only* (episodes end at the
        length limit, which — with the asserted ``episode_length % k == 0`` — lands
        exactly on a macro boundary). If the env can terminate early (e.g.
        ``terminate_when_unhealthy`` or success-terminating tasks), the auto-reset
        happens mid-window and the same committed skill drives the first steps of
        the next episode; those steps are masked out of this transition but are not
        recorded as a fresh macro-step, mildly biasing the initial-state
        distribution. Use truncation-only envs for the controller.
        """
        R0 = jnp.zeros((num_envs,), dtype=jnp.float32)
        disc0 = jnp.ones((num_envs,), dtype=jnp.float32)
        alive0 = jnp.ones((num_envs,), dtype=jnp.float32)
        done0 = jnp.zeros((num_envs,), dtype=jnp.float32)
        final_obs0 = env_state.obs

        def body(carry, _):
            es, R, disc, alive, done_any, final_obs, key = carry
            key, akey, skey = jax.random.split(key, 3)
            state = es.obs[:, :state_size]
            a = skill_action_fn(state, z, akey)
            nstate = t_env.step(es, a, skey)
            r = nstate.reward
            alive_b = alive > 0.5
            R = R + alive * disc * r
            final_obs = jnp.where(alive_b[:, None], nstate.obs, final_obs)
            step_done = nstate.done.astype(jnp.float32)
            done_any = jnp.where(alive_b, jnp.maximum(done_any, step_done), done_any)
            alive = alive * (1.0 - step_done)
            disc = disc * gamma_low
            return (nstate, R, disc, alive, done_any, final_obs, key), None

        (env_state, R, _, _, done_any, final_obs, _), _ = jax.lax.scan(
            body, (env_state, R0, disc0, alive0, done0, final_obs0, key), (), length=k
        )
        return env_state, R, final_obs, done_any

    # ── Collect one macro-step of experience for all envs ────────────────────
    def collect_macro(controller_state, env_state, replay_state, key):
        goal_key, z_key, roll_key = jax.random.split(key, 3)

        # Refresh task goals so any auto-reset this macro-step draws a fresh goal.
        goal_keys = jax.random.split(goal_key, num_envs)
        fresh_goals = jax.vmap(random_goals_proposer)(goal_keys)
        info = dict(env_state.info)
        info["proposed_goals"] = fresh_goals
        env_state = env_state.replace(info=info)

        controller_obs = env_state.obs  # [state, goal]
        logits = actor_net.apply(controller_state.actor_state.params, controller_obs)
        z = jax.random.categorical(z_key, logits)  # stochastic during training

        env_state, R, next_obs, done = rollout_macro_step(
            env_state, z, roll_key, skill_action_train
        )
        batch_in = {
            "obs": controller_obs,
            "skill": z,
            "reward": R,
            "next_obs": next_obs,
            "done": done,
        }
        replay_state = insert_controller_replay(replay_state, batch_in)
        return env_state, replay_state, R, done

    # ── One training step: collect + N SAC-discrete updates ──────────────────
    n_grad_steps = max(1, self.train_step_multiplier)

    def training_step(controller_state, env_state, replay_state, key):
        collect_key, train_key = jax.random.split(key)
        env_state, replay_state, R, done = collect_macro(
            controller_state, env_state, replay_state, collect_key
        )

        def upd(carry, _):
            cs, k_ = carry
            k_, sk, uk = jax.random.split(k_, 3)
            batch = sample_controller_replay(replay_state, sk, self.batch_size)
            cs, m = sac_discrete_update(
                cs, batch, uk,
                actor_net=actor_net, critic_net=critic_net,
                gamma=gamma_high, tau=self.tau, target_entropy=target_entropy,
            )
            return (cs, k_), m

        (controller_state, _), metrics = jax.lax.scan(
            upd, (controller_state, train_key), (), length=n_grad_steps
        )
        controller_state = controller_state.replace(
            env_steps=controller_state.env_steps + num_envs * k
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics["macro_reward_mean"] = jnp.mean(R)
        metrics["macro_done_mean"] = jnp.mean(done)
        return controller_state, env_state, replay_state, metrics

    # ── Step bookkeeping ─────────────────────────────────────────────────────
    env_steps_per_macro = num_envs * k
    num_prefill_macro_steps = int(np.ceil(self.min_replay_size / num_envs))
    num_prefill_env_steps = num_prefill_macro_steps * env_steps_per_macro
    available_env_steps = max(env_steps_per_macro, config.total_env_steps - num_prefill_env_steps)
    env_steps_per_epoch = available_env_steps // config.num_evals
    macro_steps_per_epoch = max(1, env_steps_per_epoch // env_steps_per_macro)

    logging.info("controller num_prefill_macro_steps: %d", num_prefill_macro_steps)
    logging.info("controller macro_steps_per_epoch:   %d", macro_steps_per_epoch)

    @jax.jit
    def prefill(controller_state, env_state, replay_state, key):
        def f(carry, _):
            cs, es, rs, k_ = carry
            k_, ck = jax.random.split(k_)
            es, rs, _, _ = collect_macro(cs, es, rs, ck)
            return (cs, es, rs, k_), None
        (controller_state, env_state, replay_state, _), _ = jax.lax.scan(
            f, (controller_state, env_state, replay_state, key), (), length=num_prefill_macro_steps
        )
        return controller_state, env_state, replay_state

    @jax.jit
    def training_epoch(controller_state, env_state, replay_state, key):
        def f(carry, _):
            cs, es, rs, k_ = carry
            k_, sk = jax.random.split(k_)
            cs, es, rs, m = training_step(cs, es, rs, sk)
            return (cs, es, rs, k_), m
        (controller_state, env_state, replay_state, _), metrics = jax.lax.scan(
            f, (controller_state, env_state, replay_state, key), (), length=macro_steps_per_epoch
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return controller_state, env_state, replay_state, metrics

    # ── Dedicated eval: argmax controller + deterministic skill policy ───────
    num_eval_envs = config.num_eval_envs
    episode_length = config.episode_length

    @jax.jit
    def evaluate(controller_state, key):
        reset_keys = jax.random.split(key, num_eval_envs)
        state = e_env.reset(reset_keys)  # goal=None -> env's eval task goals

        skill0 = jnp.zeros((num_eval_envs,), dtype=jnp.int32)

        def body(carry, i):
            state, skill = carry
            controller_obs = state.obs
            logits = actor_net.apply(controller_state.actor_state.params, controller_obs)
            new_skill = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            reselect = (i % k) == 0
            skill = jnp.where(reselect, new_skill, skill)
            # `key` is intentionally reused across steps: skill_action_eval is
            # deterministic (dist.mode()), so the key is inert here. If eval is
            # ever switched to stochastic skill actions, split a per-step key.
            a = skill_action_eval(state.obs[:, :state_size], skill, key)
            nstate = e_env.step(state, a)
            return (nstate, skill), None

        (state, _), _ = jax.lax.scan(
            body, (state, skill0), jnp.arange(episode_length)
        )
        eval_metrics = state.info["eval_metrics"]
        return eval_metrics

    def run_eval(controller_state, base_metrics, key):
        eval_metrics = evaluate(controller_state, key)
        em = eval_metrics.episode_metrics
        out = dict(base_metrics)
        for name in ("reward", "success", "success_easy", "dist"):
            if name in em:
                out[f"eval/episode_{name}"] = float(np.mean(np.asarray(em[name])))
        if "success" in em:
            out["eval/episode_success_any"] = float(np.mean(np.asarray(em["success"]) > 0.0))
        out["eval/avg_episode_length"] = float(np.mean(np.asarray(eval_metrics.episode_steps)))
        return out

    # ── Prefill ──────────────────────────────────────────────────────────────
    rng, prefill_key = jax.random.split(rng)
    controller_state, env_state, replay_state = prefill(
        controller_state, env_state, replay_state, prefill_key
    )

    # ── Main loop ────────────────────────────────────────────────────────────
    def make_policy(params, deterministic: bool = True):
        # Thin per-step hierarchical policy (re-selects each call; used only for
        # optional rendering). params = (actor_params,).
        actor_params = params[0]

        def policy(obs, rng_key):
            obs_b = obs[None] if obs.ndim == 1 else obs
            logits = actor_net.apply(actor_params, obs_b)
            z = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            a = skill_action_eval(obs_b[:, :state_size], z, rng_key)
            a = a[0] if obs.ndim == 1 else a
            return a, {}
        return policy

    training_walltime = 0.0
    metrics = {}
    current_step = 0
    logging.info("starting skill-controller training....")

    for ne in range(config.num_evals):
        t = time.time()
        rng, epoch_key, eval_rng = jax.random.split(rng, 3)
        controller_state, env_state, replay_state, raw_metrics = training_epoch(
            controller_state, env_state, replay_state, epoch_key
        )
        raw_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), raw_metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time

        current_step = int(controller_state.env_steps) + num_prefill_env_steps
        sps = (env_steps_per_macro * macro_steps_per_epoch) / max(epoch_time, 1e-6)

        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": current_step,
        }
        for name, value in raw_metrics.items():
            metrics[f"training/{name}"] = float(value)

        metrics = run_eval(controller_state, metrics, eval_rng)
        logging.info("step: %d", current_step)

        params = (
            controller_state.actor_state.params,
            controller_state.critic_state.params,
            controller_state.alpha_state.params,
        )
        if config.checkpoint_logdir:
            save_params(f"{config.checkpoint_logdir}/step_{current_step}.pkl", params)

        # Rendering disabled for the hierarchical controller (do_render=False).
        progress_fn(current_step, metrics, make_policy, (controller_state.actor_state.params,),
                    unwrapped_env, do_render=False)

    logging.info("total steps: %s", current_step)
    final_params = (
        controller_state.actor_state.params,
        controller_state.critic_state.params,
        controller_state.alpha_state.params,
    )
    return make_policy, final_params, metrics
