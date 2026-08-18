"""Hierarchical discrete-skill controller training routine.

Implements ``GoExploreSimple._train_skill_controller`` (``agent_type="sac_discrete"``):
freeze a pretrained OGBench skill-conditioned policy ``π(a | s, z)`` and train an
online SAC-discrete *high-level controller* that selects discrete skills ``z`` to
maximize task reward, with fixed temporal commitment ``k``.

See ``SKILL_CONTROLLER_DESIGN.md`` for the full spec. Key pieces:
  - ``load_frozen_skill_policy`` / ``make_skill_action_fn``: a generic adapter over
    any OGBench skill-conditioned agent. Three skill parameterizations are
    supported: (1) agents with a policy submodule of signature
    ``policy(obs, skills_onehot)`` (e.g. ``empowerment_skill``), (2) DDS
    (``dds``, "Discrete Diffusion Skills"), whose discrete skill index selects a
    VQ codebook embedding that a diffusion/categorical decoder turns into an
    action, and (3) ``dads`` — OGBench's ``opal`` agent trained with
    ``config["latent_type"]="discrete"``, whose discrete skill index is
    one-hot encoded and concatenated onto the obs before a single ``decoder``
    call. In all cases the controller selects a discrete skill in
    ``[0, num_skills)`` and gets back a low-level action.
  - ``rollout_macro_step``: a ``lax.scan`` over ``k`` env steps under one fixed
    per-env skill, producing one SMDP macro-transition ``(s_t, z, R, s_{t+k}, done)``.
  - ``train_skill_controller``: prefill -> epochs of (collect macro-steps + SAC
    updates) -> dedicated deterministic eval rollout.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb

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
    her_relabel_sequence,
    init_controller_replay,
    insert_controller_replay,
    sample_controller_replay,
    sac_discrete_update,
)


# ── Frozen skill-policy adapter (generic over OGBench skill agents) ─────────────


def load_frozen_skill_policy(self, unwrapped_env, template_key):
    """Load + freeze an OGBench skill-conditioned policy and resolve the skill space.

    Returns ``(emp_agent, num_skills, skill_obs_builder)``, where ``num_skills``
    is the discrete skill count for a discrete controller, or the continuous
    skill vector dimensionality when ``self.continuous_skill`` is True. The
    skill policy is never differentiated; we only call its frozen ``policy``
    submodule.
    """
    state_size = int(unwrapped_env.state_dim)
    emp_agent, ex_obs_dim, base_obs_template = load_offline_empowerment_agent(
        run_dir=self.skill_policy_run_dir,
        jax_env=unwrapped_env,
        template_rng=template_key,
        epoch=self.skill_policy_epoch,
        use_full_obs=self.use_full_skill_obs,
        ogbench_root=self.ogbench_root,
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

    if getattr(self, "continuous_skill", False):
        # skill_dim: explicit override, else inferred from the checkpoint config
        # (falls back to 8 if the checkpoint doesn't record it).
        ckpt_skill_dim = int(emp_agent.config.get("skill_dim", 8))
        if self.skill_dim is not None:
            num_skills = int(self.skill_dim)
            if num_skills != ckpt_skill_dim:
                raise ValueError(
                    f"skill_dim override ({num_skills}) disagrees with the loaded skill "
                    f"checkpoint's skill_dim ({ckpt_skill_dim}); they must match."
                )
        else:
            num_skills = ckpt_skill_dim
    else:
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

    # skill_policy_type: assert (never derive) the checkpoint's agent family so
    # the wandb-logged config value is guaranteed to describe the real ckpt.
    skill_policy_type = getattr(self, "skill_policy_type", None)
    if skill_policy_type is not None:
        expected_agent_name = _SKILL_POLICY_TYPE_TO_AGENT_NAME.get(skill_policy_type)
        if expected_agent_name is None:
            raise ValueError(
                f"skill_policy_type={skill_policy_type!r} is not one of "
                f"{sorted(_SKILL_POLICY_TYPE_TO_AGENT_NAME)}."
            )
        ckpt_agent_name = str(emp_agent.config.get("agent_name", ""))
        if ckpt_agent_name != expected_agent_name:
            raise ValueError(
                f"skill_policy_type={skill_policy_type!r} expects checkpoint "
                f"agent_name={expected_agent_name!r}, but the loaded skill "
                f"checkpoint has agent_name={ckpt_agent_name!r}; they must match."
            )
        # "dads" and OPAL's continuous VAE path share agent_name="opal"; the
        # only thing distinguishing them is config["latent_type"]. Assert it
        # explicitly so skill_policy_type="dads" can never silently load a
        # continuous-latent OPAL checkpoint (whose "decoder" submodule expects
        # a Gaussian skill_dim vector, not a one-hot over num_skills).
        if skill_policy_type == "dads":
            ckpt_latent_type = str(emp_agent.config.get("latent_type", ""))
            if ckpt_latent_type != "discrete":
                raise ValueError(
                    "skill_policy_type='dads' expects checkpoint "
                    f"config['latent_type']='discrete', but the loaded checkpoint has "
                    f"latent_type={ckpt_latent_type!r}; they must match."
                )

    return emp_agent, num_skills, skill_obs_builder


# Maps the human-facing ``skill_policy_type`` config value to the OGBench
# ``agent_name`` recorded in the checkpoint's flags.json. Used to ASSERT the
# configured type against the loaded checkpoint (never to derive it). "dads"
# and OPAL's own continuous VAE variant share agent_name="opal" — see the
# config["latent_type"] assert in ``load_frozen_skill_policy`` and
# ``_is_dads_agent`` below for how they're actually told apart.
_SKILL_POLICY_TYPE_TO_AGENT_NAME = {
    "dds": "dds",
    "empowerment": "empowerment_skill",
    "dads": "opal",
}


def _is_dds_agent(emp_agent) -> bool:
    """Whether ``emp_agent`` is a DDS ("Discrete Diffusion Skills") checkpoint.

    DDS has no ``policy(obs, skills_onehot)`` submodule; it uses a VQ codebook +
    decoder, so it needs the ``_make_dds_skill_action_fn`` adapter instead of the
    default ``policy``-based one.
    """
    try:
        if str(emp_agent.config.get("agent_name", "")) == "dds":
            return True
    except Exception:
        pass
    return "modules_codebook" in emp_agent.network.params


def _is_dads_agent(emp_agent) -> bool:
    """Whether ``emp_agent`` is a "dads" checkpoint: OGBench's ``opal`` agent
    (``impls/agents/opal.py``) trained with ``config["latent_type"]="discrete"``
    (the paper's Appendix F offline-DADS path — EM-clustered discrete skills +
    BC), as opposed to OPAL's own default continuous VAE latent (also
    ``agent_name="opal"``, ``latent_type="continuous"``).

    The discrete path has no ``policy(obs, skills_onehot)`` or ``actor(obs,
    skills_onehot)`` submodule; its skill-conditioned policy is a ``decoder``
    submodule called with a SINGLE concatenated ``[obs, skill_onehot]`` array
    (not two separate args), so it needs the
    ``_make_dads_skill_action_fn`` adapter instead of the default one.
    """
    try:
        if (
            str(emp_agent.config.get("agent_name", "")) == "opal"
            and str(emp_agent.config.get("latent_type", "")) == "discrete"
        ):
            return True
    except Exception:
        pass
    params = emp_agent.network.params
    return "modules_skill_prior" in params and "modules_traj_model" in params


def _make_dds_skill_action_fn(emp_agent, skill_obs_builder, *, deterministic):
    """Adapter for a frozen DDS checkpoint (arXiv:2503.20176).

    DDS has no ``policy(obs, skills_onehot)`` submodule. Instead each discrete
    skill index ``z`` selects a VQ codebook embedding ``codebook[z]`` which the
    frozen low-level decoder turns into an action:
      - continuous-action envs: ancestral DDPM sampling from the diffusion
        decoder. This is inherently stochastic — there is no closed-form mode —
        so ``deterministic`` is best-effort here: we still sample, seeded by the
        per-call key. (Feeding a fixed key would freeze the noise but not yield a
        true mode; the controller reuses its eval key across steps anyway.)
      - discrete-action envs: the categorical BC decoder, from which
        ``deterministic`` takes the mode (argmax) and otherwise a sample.
    The frozen ``emp_agent`` params are captured by closure (no gradient ever
    flows into them).
    """
    discrete = bool(emp_agent.config["discrete"])

    def skill_action_fn(states: jnp.ndarray, skill_indices: jnp.ndarray, key: jax.Array):
        emp_obs = skill_obs_builder(states)                  # OGBench-mapped state
        z = emp_agent._codebook_table()[skill_indices]        # [B, D_z] codebook lookup
        if discrete:
            dist = emp_agent.network.select("decoder")(emp_obs, z)
            return dist.mode() if deterministic else dist.sample(seed=key)
        a = emp_agent._ddpm_sample(emp_obs, z, key)
        return jnp.clip(a, -1.0, 1.0)

    return skill_action_fn


def _make_dads_skill_action_fn(emp_agent, skill_obs_builder, num_skills, *, deterministic):
    """Adapter for a frozen "dads" checkpoint (OPAL, ``latent_type="discrete"``).

    The discrete-latent ``decoder`` submodule takes a SINGLE concatenated
    ``[obs, skill_onehot]`` array (not two separate args like
    ``empowerment_skill``'s ``policy``), so the discrete skill index is
    one-hot encoded over ``num_skills`` and concatenated onto the mapped obs
    before the call. Inherently discrete (the skill prior is a categorical
    over ``num_skills`` and the decoder is only ever trained on one-hot
    skills) — there is no continuous-skill variant. The frozen ``emp_agent``
    params are captured by closure (no gradient ever flows into them).
    """

    def skill_action_fn(states: jnp.ndarray, skill_indices: jnp.ndarray, key: jax.Array):
        emp_obs = skill_obs_builder(states)                  # OGBench-mapped state
        z = jax.nn.one_hot(skill_indices, num_skills)
        sz = jnp.concatenate([emp_obs, z], axis=-1)
        dist = emp_agent.network.select("decoder")(sz)
        a = dist.mode() if deterministic else dist.sample(seed=key)
        return jnp.clip(a, -1.0, 1.0)

    return skill_action_fn


def make_skill_action_fn(
    emp_agent, skill_obs_builder, num_skills, *, deterministic, continuous=False
):
    """Adapter: (jaxgcrl states, skill) -> low-level actions.

    Generic over any OGBench skill-conditioned agent. For agents with a policy
    submodule of signature ``policy(obs, z)`` returning a ``distrax`` dist (e.g.
    ``empowerment_skill``), the skill is fed directly to that policy: one-hot
    encoded when ``continuous=False`` (discrete skill index in
    ``[0, num_skills)``), or passed through as a raw continuous vector of
    dimension ``num_skills`` when ``continuous=True``. For DDS checkpoints the
    discrete index instead selects a VQ codebook embedding decoded into an
    action (see ``_make_dds_skill_action_fn``); DDS is inherently discrete and
    is not supported when ``continuous=True``. "dads" checkpoints (OPAL,
    discrete latent) instead concatenate a one-hot skill onto the obs and call
    a single ``decoder`` submodule (see ``_make_dads_skill_action_fn``); also
    inherently discrete. The frozen ``emp_agent`` params are captured by
    closure (no gradient ever flows into them) — identical pattern to
    ``make_offline_empowerment_scorer``.
    """
    if _is_dds_agent(emp_agent):
        if continuous:
            raise ValueError(
                "continuous_skill=True is incompatible with a DDS skill checkpoint "
                "(DDS uses a discrete VQ codebook)."
            )
        return _make_dds_skill_action_fn(
            emp_agent, skill_obs_builder, deterministic=deterministic
        )

    if _is_dads_agent(emp_agent):
        if continuous:
            raise ValueError(
                "continuous_skill=True is incompatible with a 'dads' skill checkpoint "
                "(its skill prior/decoder are trained only on one-hot discrete skills)."
            )
        return _make_dads_skill_action_fn(
            emp_agent, skill_obs_builder, num_skills, deterministic=deterministic
        )

    def skill_action_fn(states: jnp.ndarray, skill: jnp.ndarray, key: jax.Array):
        emp_obs = skill_obs_builder(states)                  # OGBench-mapped state
        z = skill if continuous else jax.nn.one_hot(skill, num_skills)
        dist = emp_agent.network.select("policy")(emp_obs, z)
        a = dist.mode() if deterministic else dist.sample(seed=key)
        return jnp.clip(a, -1.0, 1.0)

    return skill_action_fn


# ── Skill-colored trajectory plot ───────────────────────────────────────────────


def _plot_skill_colored_trajectory(
    xy, skills, num_skills, x_bounds, y_bounds, goal_xy=None, title=""
):
    """Plot a single 2D trajectory with each segment colored by the active skill.

    ``xy``: (T, 2) positions of the tracked entity (goal_indices[:2]); ``skills``:
    (T,) int skill index in effect at each step. Returns a matplotlib Figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    xy = np.asarray(xy, dtype=np.float32)
    skills = np.asarray(skills).astype(int)
    cmap = plt.get_cmap("tab20" if num_skills <= 20 else "gist_ncar", num_skills)
    norm = BoundaryNorm(np.arange(-0.5, num_skills + 0.5, 1.0), num_skills)

    fig, ax = plt.subplots(figsize=(6, 6))
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (T-1, 2, 2)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(skills[:-1])
    lc.set_linewidth(2.0)
    ax.add_collection(lc)

    ax.scatter(xy[0, 0], xy[0, 1], c="black", marker="o", s=60, zorder=5, label="start")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="black", marker="s", s=60, zorder=5, label="end")
    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        ax.scatter(goal_xy[0], goal_xy[1], c="red", marker="*", s=220, zorder=6, label="goal")

    ax.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
    ax.set_ylim(float(y_bounds[0]), float(y_bounds[1]))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        ticks=np.arange(num_skills), fraction=0.046, pad=0.04,
    )
    cbar.set_label("skill")
    fig.tight_layout()
    return fig


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

    state_size = int(unwrapped_env.state_dim)
    goal_size = len(unwrapped_env.goal_indices)
    obs_size = state_size + goal_size
    num_envs = config.num_envs
    k = self.skill_commitment_k
    gamma_low = self.gamma_low
    gamma_high = self.discounting

    rng = jax.random.PRNGKey(config.seed)
    rng, skill_key, actor_key, critic_key, _, env_key, _ = jax.random.split(rng, 7)
    np.random.seed(config.seed)

    # ── Frozen skill policy + adapters ───────────────────────────────────────
    emp_agent, num_skills, skill_obs_builder = load_frozen_skill_policy(
        self, unwrapped_env, skill_key
    )
    logging.info(
        "skill controller: num_skills=%d, k=%d, use_her=%s, p_future_her_goal=%.3f",
        num_skills, k, bool(self.use_her), float(self.p_future_her_goal),
    )
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
    def step_macro(controller_state, env_state, key):
        """One macro-step for all envs; returns the (un-inserted) SMDP transition.

        Captures the ``traj_id`` of the *starting* state so hindsight relabeling
        can restrict future-goal sampling to the same episode.
        """
        goal_key, z_key, roll_key = jax.random.split(key, 3)

        # Refresh task goals so any auto-reset this macro-step draws a fresh goal.
        goal_keys = jax.random.split(goal_key, num_envs)
        fresh_goals = jax.vmap(random_goals_proposer)(goal_keys)
        info = dict(env_state.info)
        info["proposed_goals"] = fresh_goals
        env_state = env_state.replace(info=info)

        controller_obs = env_state.obs  # [state, goal]
        traj_id = env_state.info["traj_id"]
        logits = actor_net.apply(controller_state.actor_state.params, controller_obs)
        z = jax.random.categorical(z_key, logits)  # stochastic during training

        env_state, R, next_obs, done = rollout_macro_step(
            env_state, z, roll_key, skill_action_train
        )
        step = {
            "obs": controller_obs,
            "skill": z,
            "reward": R,
            "next_obs": next_obs,
            "done": done,
            "traj_id": traj_id,
        }
        return env_state, step, R, done

    def collect_macro(controller_state, env_state, replay_state, key):
        # insert_controller_replay ignores the extra "traj_id" key.
        env_state, step, R, done = step_macro(controller_state, env_state, key)
        replay_state = insert_controller_replay(replay_state, step)
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

    # ── HER (hindsight relabel) variant ──────────────────────────────────────
    # Collect a block of macro-steps, relabel each transition's goal with a
    # uniformly-sampled *future* achieved state from the SAME episode (matched by
    # traj_id), then feed the relabeled transitions through the SAME flat i.i.d.
    # replay + SAC update path. The block holds one episode (episode_length / k
    # macro-steps) but is capped at HER_MAX_WINDOW to bound the O(T^2) relabel
    # matrices; traj_id masking keeps relabeling within-episode even when a capped
    # block straddles an episode boundary.
    HER_MAX_WINDOW = 512
    use_her = bool(self.use_her)
    her_window = min(config.episode_length // k, HER_MAX_WINDOW)  # macro-steps / block
    # Match the non-HER total step budget: blocks/epoch * her_window ≈ macro_steps/epoch.
    blocks_per_epoch = max(1, int(round(macro_steps_per_epoch / her_window)))
    _goal_indices = jnp.asarray(
        [int(i) for i in np.asarray(unwrapped_env.goal_indices)]
    )
    _goal_reach_thresh = float(unwrapped_env.goal_reach_thresh)
    _relabel_seq = functools.partial(
        her_relabel_sequence,
        state_size=state_size,
        goal_indices=_goal_indices,
        goal_reach_thresh=_goal_reach_thresh,
        p_future_her_goal=float(self.p_future_her_goal),
    )

    def collect_block(controller_state, env_state, key):
        def f(carry, _):
            es_, k_ = carry
            k_, ck = jax.random.split(k_)
            es_, step, _, _ = step_macro(controller_state, es_, ck)
            return (es_, k_), step
        (env_state, _), block = jax.lax.scan(
            f, (env_state, key), (), length=her_window
        )
        return env_state, block  # block leaves: (her_window, num_envs, ...)

    @jax.jit
    def training_epoch_her(controller_state, env_state, replay_state, key):
        def block_step(carry, _):
            cs, es, rs, k_ = carry
            k_, collect_key, relabel_key, train_key = jax.random.split(k_, 4)

            es, block = collect_block(cs, es, collect_key)

            # Relabel per env over its macro-step sequence, then flatten to i.i.d.
            block_env_major = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), block
            )  # (num_envs, her_window, ...)
            rkeys = jax.random.split(relabel_key, num_envs)
            relabeled = jax.vmap(_relabel_seq)(block_env_major, rkeys)
            flat = jax.tree_util.tree_map(
                lambda x: x.reshape((num_envs * her_window,) + x.shape[2:]), relabeled
            )
            rs = insert_controller_replay(rs, flat)

            def upd(carry2, _):
                cs2, k2 = carry2
                k2, sk, uk = jax.random.split(k2, 3)
                batch = sample_controller_replay(rs, sk, self.batch_size)
                cs2, m = sac_discrete_update(
                    cs2, batch, uk,
                    actor_net=actor_net, critic_net=critic_net,
                    gamma=gamma_high, tau=self.tau, target_entropy=target_entropy,
                )
                return (cs2, k2), m

            # Keep updates-per-collected-macro-step equal to the non-HER path.
            (cs, _), m = jax.lax.scan(
                upd, (cs, train_key), (), length=her_window * n_grad_steps
            )
            cs = cs.replace(env_steps=cs.env_steps + num_envs * k * her_window)
            m = jax.tree_util.tree_map(jnp.mean, m)
            m["macro_reward_mean"] = jnp.mean(block["reward"])
            m["macro_done_mean"] = jnp.mean(block["done"])
            return (cs, es, rs, k_), m

        (controller_state, env_state, replay_state, _), metrics = jax.lax.scan(
            block_step,
            (controller_state, env_state, replay_state, key),
            (), length=blocks_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return controller_state, env_state, replay_state, metrics

    training_epoch_fn = training_epoch_her if use_her else training_epoch

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
            return (nstate, skill), skill

        (state, _), skills = jax.lax.scan(
            body, (state, skill0), jnp.arange(episode_length)
        )
        eval_metrics = state.info["eval_metrics"]
        return eval_metrics, skills  # skills: (episode_length, num_eval_envs)

    def run_eval(controller_state, base_metrics, key, num_steps):
        eval_metrics, skills = evaluate(controller_state, key)
        em = eval_metrics.episode_metrics
        out = dict(base_metrics)
        for name in ("reward", "success", "success_easy", "dist"):
            if name in em:
                out[f"eval/episode_{name}"] = float(np.mean(np.asarray(em[name])))
        if "success" in em:
            out["eval/episode_success_any"] = float(np.mean(np.asarray(em["success"]) > 0.0))
        out["eval/avg_episode_length"] = float(np.mean(np.asarray(eval_metrics.episode_steps)))

        # ── Skill-usage distribution (choices made at the reselection steps) ──
        skills_np = np.asarray(skills)                 # (episode_length, num_eval_envs)
        chosen = skills_np[0::k].reshape(-1).astype(int)  # i % k == 0 -> a skill choice
        counts = np.bincount(chosen, minlength=num_skills).astype(np.float64)
        fracs = counts / max(counts.sum(), 1.0)
        nz = fracs > 0
        out["eval/skill_entropy"] = float(-(fracs[nz] * np.log(fracs[nz])).sum())
        out["eval/skill_max_frac"] = float(fracs.max())
        out["eval/skill_active_count"] = float((counts > 0).sum())
        # Stash the histogram under the reserved media key so progress_fn logs it
        # in the SAME wandb.log call as the scalar metrics. A separate wandb.log
        # at this step would establish the step first and (online) drop the later
        # scalar row — which is why every controller_* metric read back as zero.
        media = out.setdefault("_wandb_media", {})
        media["eval/skill_usage_hist"] = wandb.Histogram(
            np_histogram=(counts, np.arange(num_skills + 1) - 0.5)
        )
        return out

    # ── Render: faithful hierarchical rollout (controller + frozen skill) ────
    # Single un-vmapped env, Python loop honoring the k-step commitment, logging
    # (1) a brax 3D HTML render (proves the two-policy rollout works) and
    # (2) a 2D trajectory whose segments are colored by the active skill.
    prim_idx = np.asarray(unwrapped_env.goal_indices)[:2]  # tracked entity xy

    def render_skill_controller(controller_state, key, num_steps):
        from brax.io import html

        # Returns a dict of wandb media objects to be logged via the shared
        # log_wandb call (same single-wandb.log-per-step rationale as run_eval).
        media = {}
        actor_params = controller_state.actor_state.params

        @jax.jit
        def pick(obs_b):
            logits = actor_net.apply(actor_params, obs_b)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        @jax.jit
        def act(state_b, skill_b, akey):
            return skill_action_eval(state_b, skill_b, akey)

        jit_reset = jax.jit(eval_env_base.reset)
        jit_step = jax.jit(eval_env_base.step)

        key, rk = jax.random.split(key)
        state = jit_reset(rk)
        rollout, xy_list, skill_list = [], [], []
        skill = jnp.zeros((1,), dtype=jnp.int32)
        for i in range(episode_length):
            rollout.append(state.pipeline_state)
            obs_b = state.obs[None]
            if i % k == 0:
                skill = pick(obs_b)
            key, ak = jax.random.split(key)
            a = act(obs_b[:, :state_size], skill, ak)
            obs_np = np.asarray(state.obs)
            xy_list.append(obs_np[prim_idx])
            skill_list.append(int(skill[0]))
            state = jit_step(state, a[0])

        xy = np.stack(xy_list)
        skills = np.asarray(skill_list)
        goal_xy = np.asarray(state.obs)[state_size:][:2]

        # (1) brax 3D HTML
        try:
            sys = eval_env_base.sys.tree_replace({"opt.timestep": eval_env_base.dt})
            url = html.render(sys, rollout, height=1024)
            media["render/skill_html"] = wandb.Html(url)
        except Exception as e:  # rendering must never crash training
            logging.warning("skill HTML render failed: %s", e)

        # (2) skill-colored 2D trajectory
        try:
            fig = _plot_skill_colored_trajectory(
                xy, skills, num_skills,
                unwrapped_env.x_bounds, unwrapped_env.y_bounds,
                goal_xy=goal_xy, title=f"skills over trajectory @ step {num_steps}",
            )
            media["render/skill_trajectory"] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            logging.warning("skill trajectory plot failed: %s", e)

        return media

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
        controller_state, env_state, replay_state, raw_metrics = training_epoch_fn(
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

        metrics = run_eval(controller_state, metrics, eval_rng, current_step)
        logging.info("step: %d", current_step)

        params = (
            controller_state.actor_state.params,
            controller_state.critic_state.params,
            controller_state.alpha_state.params,
        )
        if config.checkpoint_logdir:
            save_params(f"{config.checkpoint_logdir}/step_{current_step}.pkl", params)

        # Skill-colored trajectory + 3D HTML render, every visualization_interval
        # evals. We drive the hierarchical (controller + frozen skill) rollout
        # ourselves, so the generic progress_fn render stays disabled.
        vis_interval = getattr(config, "visualization_interval", 0)
        if vis_interval and (ne % vis_interval == 0):
            rng, render_key = jax.random.split(rng)
            try:
                render_media = render_skill_controller(controller_state, render_key, current_step)
                metrics.setdefault("_wandb_media", {}).update(render_media)
            except Exception as e:  # never let rendering crash training
                logging.warning("render_skill_controller failed: %s", e)
        progress_fn(current_step, metrics, make_policy, (controller_state.actor_state.params,),
                    unwrapped_env, do_render=False)

    logging.info("total steps: %s", current_step)
    final_params = (
        controller_state.actor_state.params,
        controller_state.critic_state.params,
        controller_state.alpha_state.params,
    )
    return make_policy, final_params, metrics
