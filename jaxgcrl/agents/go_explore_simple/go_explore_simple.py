"""Go Explore agent.

Two-phase training loop:
  - Go phase   (phase == 0): GCP navigates to a proposed frontier goal.
  - Explore phase (phase == 1): continuation of go phase with eps-random actions
    and stochastic policy sampling.

Phase management is handled by ``GoExploreWrapper`` (see ``jaxgcrl/envs/wrappers.py``).
"""

import logging
import random
import time
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    GoExploreWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization

from jaxgcrl.agents.go_explore.types import TrainingState, Transition, GoalProposerState
from jaxgcrl.agents.go_explore.algorithms import get_algorithm
from jaxgcrl.agents.go_explore.utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
)
from jaxgcrl.agents.go_explore.losses import update_alpha_sac
from jaxgcrl.agents.go_explore.visualization import all_visualizations, visualize_go_explore_phases
from jaxgcrl.agents.go_explore.goal_proposers import (
    create_goal_proposer,
    create_random_env_goals_proposer,
)
from jaxgcrl.agents.go_explore.empowerment import (
    infer_empowerment_override_indices_from_env,
    load_offline_empowerment_agent,
    make_empowerment_obs_builder,
    make_offline_empowerment_scorer,
)
import io
import numpy as np
import os

try:
    import matplotlib.pyplot as _plt
    import wandb as _wandb
except Exception:  # pragma: no cover
    _plt = None
    _wandb = None


def _log_reward_heatmap(
    reward_viz,
    x_bounds,
    y_bounds,
    current_step: int,
):
    """Log scatters of raw (ant x, ant y) transition samples colored by their
    post-bonus reward and (if present) the scaled exploration bonus alone.
    ``reward_viz`` is the ``(xs, ys, rewards, scaled_bonus)`` snapshot taken
    inside ``training_step`` right after the exploration bonus is added."""
    if _plt is None or _wandb is None:
        return

    xs = np.asarray(reward_viz[0]).reshape(-1)
    ys = np.asarray(reward_viz[1]).reshape(-1)
    rs = np.asarray(reward_viz[2]).reshape(-1)
    bonus = np.asarray(reward_viz[3]).reshape(-1) if len(reward_viz) > 3 else None

    x_low, x_high = float(x_bounds[0]), float(x_bounds[1])
    y_low, y_high = float(y_bounds[0]), float(y_bounds[1])

    has_bonus = bonus is not None and np.any(bonus != 0)
    ncols = 2 if has_bonus else 1
    fig, axes = _plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)

    def _draw(ax, values, label):
        sc = ax.scatter(xs, ys, c=values, cmap="viridis", s=6, linewidths=0)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.set_xlabel("Ant x")
        ax.set_ylabel("Ant y")
        ax.set_title(label)

    _draw(axes[0, 0], rs, "reward (post-bonus)")
    if has_bonus:
        _draw(axes[0, 1], bonus, "scaled exploration bonus")

    fig.suptitle(f"step={current_step}")
    fig.tight_layout()
    _wandb.log({"viz/reward_heatmap": _wandb.Image(fig)}, step=current_step)
    _plt.close(fig)


Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class GoExploreSimple:
    """Go Explore agent with a single goal-conditioned policy.

    The go phase navigates to a proposed frontier goal. The explore phase
    continues with the same policy but samples stochastically and injects
    uniform random actions with probability ``eps_random_action``.
    """

    # Algorithm type for the goal-conditioned policy
    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 30000
    min_replay_size: int = 1000
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    repr_dim: int = 64
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    tau: float = 0.005
    n_critics: int = 2
    use_her: bool = True
    use_sac_critic_mean: bool = False  # SAC: average critic ensemble instead of min

    goal_proposer_name: Literal["random_env_goals", "rb", "q_epistemic", "ucgr", "max_critic_to_env", "mega", "omega", "empowerment", "empowerment_density_ratio"] = "random_env_goals"
    num_candidates: int = 512
    goal_proposer_temperature: float = 0.0
    empowerment_run_dir: Optional[str] = None
    empowerment_epoch: Optional[int] = None
    empowerment_num_splus_samples: int = 128
    empowerment_score_chunk_size: int = 32

    # ── RLPD (offline data mixing) ─────────────────────────────────────────
    use_rlpd: bool = False  # Mix 50% offline OGBench data into each training batch

    # ── Exploration bonus (added to reward after HER) ──────────────────────
    exploration_bonus_type: Optional[str] = None  # "empowerment" or None
    exploration_bonus_weight: float = 0.1

    # ── Go Explore specific parameters ──────────────────────────────────────
    num_gcp_steps: int = 250      # max steps in go phase before forcing explore
    num_ep_steps: int = 250        # steps in explore phase before reset to go
    deterministic_go_phase: bool = False  # if True, go phase uses policy mode
    eps_random_action: float = 0.1        # probability of uniform random action in explore phase

    def check_config(self, config):
        assert config.episode_length - 1 == self.num_gcp_steps + self.num_ep_steps, (
            "episode_length - 1 must be equal to num_gcp_steps + num_ep_steps"
        )
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env

        action_size = train_env.action_size
        state_size  = train_env.state_dim
        goal_size   = len(train_env.goal_indices)
        obs_size    = state_size + goal_size

        # ── Eval env ──────────────────────────────────────────────────────────
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # ── Train env: GoExploreWrapper manages phase transitions ─────────────
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        train_env = GoExploreWrapper(
            train_env,
            num_gcp_steps=self.num_gcp_steps,
            num_ep_steps=self.num_ep_steps,
            state_size=state_size,
            goal_size=goal_size,
            goal_indices=unwrapped_env.goal_indices,
        )

        # ── Step count bookkeeping ────────────────────────────────────────────
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps    = self.min_replay_size * config.num_envs
        num_prefill_actor_steps  = int(np.ceil(self.min_replay_size / self.unroll_length))

        available_env_steps          = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch          = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps:          %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps:         %d", num_prefill_actor_steps)
        logging.info("env_steps_per_epoch:             %d", env_steps_per_epoch)
        logging.info("num_training_steps_per_epoch:    %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        # ── GCP (goal-conditioned policy) — the only policy ───────────────────
        gcp_actor, gcp_critic = get_algorithm(
            agent_type=self.agent_type,
            action_size=action_size,
            obs_size=obs_size,
            state_size=state_size,
            goal_indices=train_env.goal_indices,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            repr_dim=self.repr_dim,
            discounting=self.discounting,
            energy_fn=self.energy_fn,
            n_critics=self.n_critics,
        )

        gcp_actor_params  = gcp_actor.init(actor_key,  np.ones([1, obs_size]))
        gcp_critic_params = gcp_critic.init(critic_key, np.ones([1, obs_size]))

        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        gcp_critic_states = gcp_critic.create_critic_states(gcp_critic_params, self.critic_lr)

        target_entropy = -1 * action_size
        log_alpha      = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state    = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        target_critic_params = None
        if self.agent_type == "sac":
            target_critic_params = gcp_critic_params

        # ── TrainingState ─────────────────────────────────────────────────────
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=gcp_actor_state,
            critic_states=gcp_critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )

        # ── Optional offline-empowerment scorer for goal proposer ───────────
        offline_empowerment_scorer = None
        if self.goal_proposer_name in ("empowerment", "empowerment_density_ratio"):
            if self.empowerment_run_dir is None:
                raise ValueError(f"empowerment_run_dir must be set when goal_proposer_name='{self.goal_proposer_name}'.")
            key, empowerment_template_key = jax.random.split(key)
            ogbench_obs_indices, jaxgcrl_state_indices = infer_empowerment_override_indices_from_env(
                unwrapped_env
            )
            emp_agent, _ex_obs_dim, base_obs_template = load_offline_empowerment_agent(
                run_dir=self.empowerment_run_dir,
                jax_env=unwrapped_env,
                template_rng=empowerment_template_key,
                epoch=self.empowerment_epoch,
                num_splus_samples=self.empowerment_num_splus_samples,
            )
            obs_builder = make_empowerment_obs_builder(
                jnp.asarray(base_obs_template),
                ogbench_obs_indices,
                jaxgcrl_state_indices,
                state_size=state_size,
            )
            offline_empowerment_scorer = make_offline_empowerment_scorer(
                emp_agent,
                obs_builder,
                chunk_size=self.empowerment_score_chunk_size,
            )

        # ── Goal proposer ────────────────────────────────────────────────────
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
            self.num_candidates,
            state_size=unwrapped_env.state_dim,
            goal_indices=unwrapped_env.goal_indices,
            actor=gcp_actor,
            critic=gcp_critic,
            discounting=self.discounting,
            offline_empowerment_scorer=offline_empowerment_scorer,
            goal_proposer_temperature=self.goal_proposer_temperature,
        )

        # ── Env reset ────────────────────────────────────────────────────────
        random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys      = jax.random.split(env_key, config.num_envs)
        initial_goals = jax.vmap(random_goals_proposer)(env_keys)

        env_state = train_env.reset(env_keys, goal=initial_goals)
        info = dict(env_state.info)
        info['proposed_goals'] = initial_goals
        env_state = env_state.replace(info=info)

        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # ── Replay buffer ────────────────────────────────────────────────────
        dummy_transition = create_single_dummy_transition(
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        dummy_batch_transition = create_dummy_transition_for_buffer(
            unroll_length=self.unroll_length,
            num_envs=config.num_envs,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)

        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=gcp_actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(gcp_critic_states)},
        )

        # ── RLPD: offline buffer ─────────────────────────────────────────────
        offline_buffer = None
        if self.use_rlpd:
            from jaxgcrl.utils.offline_buffer import load_and_prepare_offline_buffer
            offline_buffer = load_and_prepare_offline_buffer(
                env_name=config.env,
                episode_length=config.episode_length,
                num_slots=config.num_envs,
                obs_size=obs_size,
                action_size=action_size,
                state_size=state_size,
                agent_type=self.agent_type,
                include_phase=True,
            )

        # ── Exploration bonus ─────────────────────────────────────────────────
        exploration_bonus_fn = None
        if self.exploration_bonus_type is not None:
            from jaxgcrl.agents.go_explore.exploration import create_exploration_bonus
            key, bonus_init_key = jax.random.split(key)
            exploration_bonus_fn = create_exploration_bonus(
                self.exploration_bonus_type,
                env=unwrapped_env,
                state_size=state_size,
                key=bonus_init_key,
                empowerment_run_dir=self.empowerment_run_dir,
                empowerment_epoch=self.empowerment_epoch,
                empowerment_num_splus_samples=self.empowerment_num_splus_samples,
                empowerment_score_chunk_size=self.empowerment_score_chunk_size,
            )

        # ── actor_step ────────────────────────────────────────────────────────
        deterministic_go = self.deterministic_go_phase
        eps_random = self.eps_random_action

        def actor_step(training_state, env, env_state, key, extra_fields):
            """One env step using a single GCP policy for both phases."""
            key, action_key, random_key, eps_key, env_rng = jax.random.split(key, 5)

            phase     = env_state.info['phase']           # (num_envs,)
            go_goal   = env_state.info['go_goal']         # (num_envs, goal_size)
            raw_state = env_state.obs[:, :state_size]     # (num_envs, state_size)

            # GCP always sees [state, go_goal] in both phases
            gcp_obs = jnp.concatenate([raw_state, go_goal], axis=-1)

            # Go phase: deterministic if flag set, else stochastic
            # Explore phase: always stochastic (sample from policy)
            in_go = (phase == 0)  # (num_envs,)
            is_deterministic = jnp.where(in_go, deterministic_go, False)

            # Sample policy actions (per-env deterministic flag handled inside)
            # We compute both deterministic and stochastic, then select per-env
            det_actions = gcp_actor.sample_actions(
                training_state.actor_state.params, gcp_obs, action_key, is_deterministic=True
            )
            stoch_actions = gcp_actor.sample_actions(
                training_state.actor_state.params, gcp_obs, action_key, is_deterministic=False
            )
            policy_actions = jnp.where(is_deterministic[:, None], det_actions, stoch_actions)

            # Explore phase: with probability eps_random_action, use uniform random action
            in_explore = (phase == 1)  # (num_envs,)
            random_actions = jax.random.uniform(
                random_key, shape=policy_actions.shape, minval=-1.0, maxval=1.0
            )
            use_random = jax.random.uniform(eps_key, shape=(policy_actions.shape[0],)) < eps_random
            use_random = jnp.logical_and(in_explore, use_random)
            actions = jnp.where(use_random[:, None], random_actions, policy_actions)

            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras['phase'] = phase

            # Build next_observation for SAC if needed
            next_obs = jnp.concatenate(
                [nstate.obs[:, :state_size], go_goal], axis=-1
            ) if self.agent_type == "sac" else None

            return nstate, Transition(
                observation=gcp_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=next_obs,
                extras={"state_extras": state_extras},
            )

        # ── get_experience ────────────────────────────────────────────────────
        def get_experience(training_state, env_state, buffer_state, key,
                           experience_count, goal_proposer_state):
            buffer_state, transitions_sample = replay_buffer.sample(buffer_state)

            goal_proposer_state = goal_proposer_state.replace(
                transitions_sample=transitions_sample,
                actor_params=training_state.actor_state.params,
                critic_params={i: cs.params for i, cs in enumerate(training_state.critic_states)},
            )

            num_envs_     = config.num_envs
            episode_length = config.episode_length
            info           = dict(env_state.info)

            reset_threshold    = jnp.array(episode_length // (self.unroll_length * 2), dtype=jnp.int32)
            new_experience_count = experience_count + 1

            def propose_new_goals(env_state, key, info, experience_count, goal_proposer_state):
                viz_key, goal_key = jax.random.split(key)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs_)
                goal_keys   = jax.random.split(goal_key, num_envs_)
                first_obs   = info['first_obs']

                def propose_single(rng_key, obs, state):
                    goal, updated_state, log_data = goal_proposer(rng_key, obs, state)
                    return goal, log_data

                new_goals, log_data_tree = jax.vmap(
                    propose_single, in_axes=(0, 0, None)
                )(goal_keys, first_obs, goal_proposer_state)
                info['proposed_goals'] = new_goals

                env_steps = training_state.env_steps

                def log_viz(log_data_tree_np, viz_idx, steps):
                    selected = {k: v[viz_idx] for k, v in log_data_tree_np.items()}
                    handle_goal_proposer_visualization(
                        selected, self.goal_proposer_name,
                        unwrapped_env.x_bounds, unwrapped_env.y_bounds, steps
                    )
                    return jnp.array(0, dtype=jnp.int32)

                jax.experimental.io_callback(
                    log_viz, jnp.array(0, dtype=jnp.int32),
                    log_data_tree, viz_env_idx, env_steps
                )
                return env_state, info, jnp.array(0, dtype=jnp.int32), goal_proposer_state

            def keep_existing_goals(env_state, key, info, new_experience_count, goal_proposer_state):
                return env_state, info, new_experience_count, goal_proposer_state

            # Split key so proposal and rollout use independent randomness.
            _, propose_key, rollout_key = jax.random.split(key, 3)
            env_state, info, updated_experience_count, updated_goal_proposer_state = jax.lax.cond(
                new_experience_count >= reset_threshold,
                propose_new_goals,
                keep_existing_goals,
                env_state, propose_key, info, new_experience_count, goal_proposer_state,
            )
            env_state = env_state.replace(info=info)

            @jax.jit
            def f(carry, _):
                env_state, buffer_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    k,
                    extra_fields=("truncation", "traj_id", "phase"),
                )
                return (env_state, buffer_state, next_k), transition

            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, rollout_key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)

            return env_state, buffer_state, updated_experience_count, updated_goal_proposer_state

        # ── prefill_replay_buffer ─────────────────────────────────────────────
        def prefill_replay_buffer(training_state, env_state, buffer_state, key, goal_proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, new_k = jax.random.split(k)
                es, bs, ec, gps = get_experience(ts, es, bs, k, ts.experience_count, gps)
                ts = ts.replace(
                    env_steps=ts.env_steps + config.num_envs * self.unroll_length,
                    experience_count=ec,
                )
                return (ts, es, bs, new_k, gps), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ── update_networks (GCP only) ────────────────────────────────────────
        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            if self.agent_type == "sac":
                key, alpha_key, critic_key, actor_key = jax.random.split(key, 4)
            else:
                key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(actor=gcp_actor, critic=gcp_critic)

            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
            else:  # sac
                training_state, alpha_metrics  = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)

            # Update SAC target network
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                full_cp = {}
                for i, cs in enumerate(training_state.critic_states):
                    for lname, lparams in cs.params.items():
                        full_cp[f"critic_{i}_{lname}"] = lparams
                new_target = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params, full_cp,
                )
                training_state = training_state.replace(target_critic_params=new_target)

            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
            return (training_state, key), metrics

        # ── training_step ─────────────────────────────────────────────────────
        @jax.jit
        def training_step(training_state, env_state, buffer_state, key, goal_proposer_state):
            exp_key, process_key, train_key = jax.random.split(key, 3)

            env_state, buffer_state, updated_ec, updated_gps = get_experience(
                training_state, env_state, buffer_state, exp_key,
                training_state.experience_count, goal_proposer_state,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
                experience_count=updated_ec,
            )

            # GCP update on all transitions
            buffer_state, online_transitions = replay_buffer.sample(buffer_state)

            if self.use_rlpd:
                # Mix 50% offline data: concatenate along num_envs axis
                offline_key, process_key = jax.random.split(process_key)
                offline_transitions = offline_buffer.sample(offline_key, config.num_envs)
                transitions = jax.tree_util.tree_map(
                    lambda a, b: jnp.concatenate([a, b], axis=0),
                    online_transitions, offline_transitions,
                )
            else:
                transitions = online_transitions

            transitions, _ = gcp_actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                state_size, tuple(train_env.goal_indices),
                train_env.goal_reach_thresh, self.use_her,
            )

            if self.use_rlpd:
                # Slice to original num_batches so gradient step count stays the same.
                # The random permutation in process_transitions already mixed
                # online+offline within each batch (~50/50).
                num_batches = config.num_envs * (config.episode_length - 1) // self.batch_size
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:num_batches], transitions
                )

            # Add exploration bonus to reward (after HER, so it always persists)
            if exploration_bonus_fn is not None:
                bonus_key, train_key = jax.random.split(train_key)
                bonus = exploration_bonus_fn(transitions, bonus_key)
                scaled_bonus = self.exploration_bonus_weight * bonus
                transitions = transitions._replace(
                    reward=transitions.reward + scaled_bonus
                )
            else:
                scaled_bonus = jnp.zeros_like(transitions.reward)

            reward_viz = (
                transitions.observation[..., 0],
                transitions.observation[..., 1],
                transitions.reward,
                scaled_bonus,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
            )

            return (training_state, env_state, buffer_state, updated_gps), (metrics, reward_viz)

        # ── training_epoch ────────────────────────────────────────────────────
        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key, goal_proposer_state):
            # Snapshot cumulative counters *before* the epoch so we can compute
            # epoch-level deltas (rather than a lifetime average).
            pre_completions   = jnp.sum(env_state.info['go_completions_total'])
            pre_successes     = jnp.sum(env_state.info['go_successes_total'])
            pre_success_steps = jnp.sum(env_state.info['go_success_steps_total'])

            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, gps), (metrics, reward_viz) = training_step(ts, es, bs, train_key, gps)
                return (ts, es, bs, k, gps), (metrics, reward_viz)

            (training_state, env_state, buffer_state, key, goal_proposer_state), (metrics, reward_viz) = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_training_steps_per_epoch,
            )

            # Keep only the final step's snapshot for visualization.
            last_reward_viz = jax.tree_util.tree_map(lambda x: x[-1], reward_viz)

            # Go Explore phase metrics — epoch-level (current policy performance)
            epoch_completions   = jnp.sum(env_state.info['go_completions_total']) - pre_completions
            epoch_successes     = jnp.sum(env_state.info['go_successes_total']) - pre_successes
            epoch_success_steps = jnp.sum(env_state.info['go_success_steps_total']) - pre_success_steps

            go_success_rate = jnp.where(epoch_completions > 0,
                                        epoch_successes / epoch_completions,
                                        0.0)
            avg_go_steps    = jnp.where(epoch_successes > 0,
                                        epoch_success_steps / epoch_successes,
                                        0.0)

            scan_shape = jax.tree_util.tree_leaves(metrics)[0].shape if metrics else (1,)
            metrics["go_phase_success_rate"] = jnp.broadcast_to(go_success_rate, scan_shape)
            metrics["avg_go_phase_steps"]    = jnp.broadcast_to(avg_go_steps, scan_shape)
            metrics["buffer_current_size"]   = jnp.broadcast_to(replay_buffer.size(buffer_state), scan_shape)

            return training_state, env_state, buffer_state, goal_proposer_state, metrics, last_reward_viz

        # ── prefill ───────────────────────────────────────────────────────────
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, goal_proposer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state
        )

        # ── Evaluator ─────────────────────────────────────────────────────────
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            actions = gcp_actor.sample_actions(
                training_state.actor_state.params,
                env_state.obs,
                jax.random.PRNGKey(0),
                is_deterministic=True,
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=None,
                extras={"state_extras": state_extras},
            )

        evaluator = ActorEvaluator(
            eval_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ── Main training loop ────────────────────────────────────────────────
        training_walltime      = 0
        last_visualization_step = -1
        logging.info("starting training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, goal_proposer_state, metrics, last_reward_viz = training_epoch(
                training_state, env_state, buffer_state, epoch_key, goal_proposer_state
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time  = time.time() - t
            training_walltime   += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time

            metrics_dict = {}
            for name, value in metrics.items():
                if hasattr(value, 'item'):
                    metrics_dict[f"training/{name}"] = float(value.item())
                elif hasattr(value, '__float__'):
                    metrics_dict[f"training/{name}"] = float(value)
                else:
                    metrics_dict[f"training/{name}"] = value

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **metrics_dict,
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            # Visualize trajectories every 1M steps
            if current_step // 1_000_000 > last_visualization_step // 1_000_000:
                key, viz_key = jax.random.split(key)
                buffer_state = all_visualizations(
                    replay_buffer=replay_buffer,
                    buffer_state=buffer_state,
                    env=unwrapped_env,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                    rng_key=viz_key,
                )
                _, phase_transitions = replay_buffer.sample(buffer_state)
                visualize_go_explore_phases(
                    phase_transitions,
                    unwrapped_env.x_bounds,
                    unwrapped_env.y_bounds,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                )
                if exploration_bonus_fn is not None:
                    _log_reward_heatmap(
                        reward_viz=last_reward_viz,
                        x_bounds=unwrapped_env.x_bounds,
                        y_bounds=unwrapped_env.y_bounds,
                        current_step=current_step,
                    )
                last_visualization_step = current_step

            do_render = ne % config.visualization_interval == 0
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: gcp_actor.apply(param, obs)
            else:
                make_policy = lambda param: lambda obs, rng: (
                    gcp_actor.sample_actions(param, obs, rng, is_deterministic=True), {}
                )

            # Build full GCP critic params for checkpointing
            full_critic_params = {}
            for i, cs in enumerate(training_state.critic_states):
                for lname, lparams in cs.params.items():
                    if lname in ["sa_encoder", "g_encoder"]:
                        full_critic_params[f"{lname}_{i}"] = lparams
                    else:
                        full_critic_params[f"critic_{i}_{lname}"] = lparams

            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                full_critic_params,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)
        return make_policy, params, metrics
