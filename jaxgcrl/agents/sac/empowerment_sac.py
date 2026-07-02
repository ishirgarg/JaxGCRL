import functools
import logging
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    EpisodeWrapper,
    TrajectoryIdWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.algorithms import get_explore_policy
from jaxgcrl.agents.go_explore.empowerment import (
    build_ogbench_empowerment_obs_template,
    infer_ogbench_root_from_run_dir,
    _setup_external_imports,
    _latest_epoch,
)
from jaxgcrl.agents.go_explore.losses import (
    update_alpha_sac,
    soft_update_target_params,
)
from jaxgcrl.agents.go_explore.types import TrainingState as GETrainingState, Transition

Metrics = Any


# ---------------------------------------------------------------------------
# Inline visualization helpers (no external visualization module required)
# ---------------------------------------------------------------------------

def _log_dist_vs_reward_scatter(
    replay_buffer,
    buffer_state,
    ball_indices: tuple[int, int],
    empowerment_reward_scaling: float,
    current_step: int,
) -> Any:
    """Sample the replay buffer and log a scatter of ‖ant – ball‖ vs reward."""
    buf_state, transitions = replay_buffer.sample(buffer_state)
    obs = np.asarray(transitions.observation).reshape(-1, transitions.observation.shape[-1])
    reward = np.asarray(transitions.reward).reshape(-1)

    ant_xy = obs[:, 0:2]
    ball_xy = obs[:, list(ball_indices)]
    dist = np.linalg.norm(ant_xy - ball_xy, axis=-1)
    unscaled_reward = reward / max(empowerment_reward_scaling, 1e-8)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(dist, unscaled_reward, s=4, alpha=0.3)
    ax.set_xlabel("‖ant - ball‖")
    ax.set_ylabel("Empowerment reward (unscaled)")
    ax.set_title(f"Dist vs Reward  step={current_step}")
    wandb.log({"viz/dist_vs_reward": wandb.Image(fig)}, step=current_step)
    plt.close(fig)
    return buf_state


def _log_empowerment_map(
    emp_agent,
    base_og_const: jnp.ndarray,     # fixed head obs (built once at startup)
    ex_obs_dim: int,
    ball_x_idx: int,
    ball_y_idx: int,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    grid_res: int,
    current_step: int,
    rng: jnp.ndarray,
    fixed_ball_xy: tuple[float, float],
) -> None:
    """Build a grid-of-ant-positions empowerment map using the fixed head and log to W&B.

    The fixed head (base_og_const) was constructed once at startup from the canonical
    placement Ant=(0,0), Ball=(5,0).  For every grid point we copy that template and
    overwrite only obs[0], obs[1] (ant x,y) and obs[ball_x_idx], obs[ball_y_idx].
    """
    xs = np.linspace(x_low, x_high, grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    N = flat_x.shape[0]

    base_np = np.asarray(base_og_const)
    obs_batch = np.broadcast_to(base_np[None, :], (N, ex_obs_dim)).copy()
    obs_batch[:, 0] = flat_x
    obs_batch[:, 1] = flat_y
    obs_batch[:, ball_x_idx] = float(fixed_ball_xy[0])
    obs_batch[:, ball_y_idx] = float(fixed_ball_xy[1])

    # Chunked evaluation to avoid OOM
    chunk = min(256, N)
    emps = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        chunk_rng = jax.random.fold_in(rng, start)
        emps.append(
            np.asarray(
                emp_agent.empowerment(
                    jnp.asarray(obs_batch[start:end]),
                    rng=chunk_rng,
                )
            )
        )
    emp_map = np.concatenate(emps, axis=0).reshape(grid_res, grid_res)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        emp_map,
        origin="lower",
        extent=[x_low, x_high, y_low, y_high],
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter(
        [fixed_ball_xy[0]], [fixed_ball_xy[1]],
        c="red", s=60, marker="o", edgecolors="white", linewidths=0.8,
        label="ball",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Empowerment map  step={current_step}")
    ax.set_xlabel("Ant x")
    ax.set_ylabel("Ant y")
    ax.legend(fontsize=7)
    wandb.log({"viz/empowerment_map": wandb.Image(fig)}, step=current_step)
    plt.close(fig)


def _log_replay_xy_reward_scatter(
    replay_buffer,
    buffer_state,
    current_step: int,
) -> Any:
    """Scatter of (ant x, ant y) colored by reward from a replay buffer sample."""
    buf_state, transitions = replay_buffer.sample(buffer_state)
    obs = np.asarray(transitions.observation).reshape(-1, transitions.observation.shape[-1])
    reward = np.asarray(transitions.reward).reshape(-1)
    xs = obs[:, 0]
    ys = obs[:, 1]
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xs, ys, c=reward, s=8, cmap="viridis")
    ax.set_xlabel("Ant x")
    ax.set_ylabel("Ant y")
    ax.set_title(f"Replay (x,y) colored by reward  step={current_step}")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="reward")
    wandb.log({"viz/replay_xy_reward": wandb.Image(fig)}, step=current_step)
    plt.close(fig)
    return buf_state


def _log_replay_empowerment_scatter(
    replay_buffer,
    buffer_state,
    ball_indices: tuple[int, int],
    current_step: int,
) -> Any:
    """Scatter of (ball x, ball y) coords from the replay buffer colored by empowerment.

    Uses the unscaled empowerment value stored in transition extras (``emp_abs``).
    Plots every sampled point — no binning or aggregation.
    """
    buf_state, transitions = replay_buffer.sample(buffer_state)
    obs = np.asarray(transitions.observation).reshape(-1, transitions.observation.shape[-1])
    emp_abs = np.asarray(
        transitions.extras["state_extras"]["emp_abs"]
    ).reshape(-1)

    bx = obs[:, ball_indices[0]]
    by = obs[:, ball_indices[1]]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(bx, by, c=emp_abs, s=8, cmap="viridis")
    ax.set_xlabel("ball x")
    ax.set_ylabel("ball y")
    ax.set_title(f"Replay (ball x,y) colored by empowerment  step={current_step}")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="empowerment")
    wandb.log({"viz/replay_empowerment": wandb.Image(fig)}, step=current_step)
    plt.close(fig)
    return buf_state

# ---------------------------------------------------------------------------
# Main agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmpowermentSAC:
    # SAC hyperparameters
    batch_size: int = 256
    max_replay_size: int = 20000
    min_replay_size: int = 1000
    unroll_length: int = 50
    train_step_multiplier: int = 1
    h_dim: int = 256
    n_hidden: int = 4
    use_ln: bool = True
    n_critics: int = 2
    discounting: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    empowerment_reward_scaling: float = 1.0
    use_empowerment_diff: bool = False
    normalize_rewards: bool = False
    deterministic_eval: bool = True

    # RLPD: mix offline OGBench data into each training batch (50/50)
    use_rlpd: bool = False
    # Chunk size for empowerment scoring of offline transitions (memory control)
    offline_emp_chunk_size: int = 1024

    # Empowerment checkpoint args (OGBench import path is inferred from ``run_dir``).
    run_dir: Optional[str] = None
    epoch: Optional[int] = None
    num_splus_samples: int = 128
    ogbench_xy_offset: Optional[float] = None

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size "
            "(process_transitions drops the last timestep of each trajectory)"
        )

    def train_fn(
        self,
        config,
        train_env,
        eval_env=None,
        randomization_fn: Optional[Callable] = None,
        progress_fn: Callable[[int, Metrics, Callable, Any, Any, bool], None] = lambda *args, **kwargs: None,
    ):
        self.check_config(config)
        # ------------------------------------------------------------------
        # Load empowerment checkpoint
        # ------------------------------------------------------------------
        if self.run_dir is None:
            raise ValueError("run_dir must be provided for empowerment checkpoint.")
        ogbench_root = infer_ogbench_root_from_run_dir(self.run_dir)
        agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(
            ogbench_root
        )
        import os, json
        flags_path = os.path.join(self.run_dir, "flags.json")
        with open(flags_path, "r") as f:
            flags = json.load(f)
        agent_cfg = flags["agent"]
        agent_cfg["num_splus_samples"] = int(self.num_splus_samples)
        env_name_og = flags["env_name"]
        ogbench_env, train_dataset, _ = make_env_and_datasets(
            env_name_og, frame_stack=agent_cfg.get("frame_stack")
        )
        example_batch = train_dataset.sample(1)
        if agent_cfg.get("discrete"):
            example_batch["actions"] = np.full_like(
                example_batch["actions"], ogbench_env.action_space.n - 1
            )
        agent_class = agent_registry[agent_cfg["agent_name"]]
        emp_agent = agent_class.create(
            seed=flags.get("seed", 0),
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=agent_cfg,
        )
        # Choose latest epoch if not provided
        resolved_epoch = _latest_epoch(self.run_dir) if self.epoch is None else self.epoch
        emp_agent = restore_agent(emp_agent, self.run_dir, resolved_epoch)
        ex_obs_dim = int(example_batch["observations"].shape[-1])

        # ------------------------------------------------------------------
        # Base observation template from OGBench env (same stacking rules as offline plotting).
        # Only ant x/y and ball x/y indices are overwritten at inference time.
        # ------------------------------------------------------------------
        base_env = ogbench_env.unwrapped if hasattr(ogbench_env, "unwrapped") else ogbench_env
        base_og_np = build_ogbench_empowerment_obs_template(
            ogbench_env,
            base_env,
            ex_obs_dim,
            head_ant_xy=(0.0, 0.0),
            head_ball_xy=(5.0, 0.0),
        )
        base_og_const = jnp.asarray(base_og_np)

        # Hardcoded OGBench indices for ball x,y in qpos layout
        _ball_x_idx = 15
        _ball_y_idx = 16

        # ------------------------------------------------------------------
        # Wrap envs
        # ------------------------------------------------------------------
        unwrapped_env = train_env
        train_env = VmapWrapper(unwrapped_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        train_env = TrajectoryIdWrapper(train_env)
        train_env = EvalAutoResetWrapper(train_env)

        if eval_env is None:
            eval_env = unwrapped_env
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # Dimensions
        # ant_ball_ogbench obs layout = [qpos(22), qvel(20), target_xy(2)] = 44
        # Ant xy at [0,1]; ball xy at [15,16]; goal target at [-2:].
        # We strip the trailing 2 (target) so the policy never sees the goal.
        action_size = unwrapped_env.action_size
        full_obs_size = int(unwrapped_env.observation_size)
        goal_dim = len(unwrapped_env.goal_indices)
        obs_size_net = full_obs_size - goal_dim   # policy / network obs dim (42)
        ball_indices = (_ball_x_idx, _ball_y_idx)  # (15, 16) — ball in OGBench obs
        goal_indices = ball_indices  # used by process_transitions API

        # ------------------------------------------------------------------
        # Networks
        # ------------------------------------------------------------------
        key = jax.random.PRNGKey(config.seed)
        key, actor_key, critic_key, _, env_key, eval_env_key, buffer_key = jax.random.split(key, 7)
        actor, critic = get_explore_policy(
            explore_policy_type="sac",
            action_size=action_size,
            state_size=obs_size_net,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            use_relu=False,
            use_ln=self.use_ln,
            n_critics=self.n_critics,
        )
        actor_params = actor.init(actor_key, np.ones([1, obs_size_net]))
        critic_params = critic.init(critic_key, np.ones([1, obs_size_net]))
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        critic_states = critic.create_critic_states(critic_params, self.critic_lr)
        target_entropy = -1.0 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )
        target_critic_params = critic_params
        training_state = GETrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=actor_state,
            critic_states=critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )

        # ------------------------------------------------------------------
        # Replay buffer
        # ------------------------------------------------------------------
        dummy_obs = jnp.zeros((obs_size_net,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=jnp.zeros(()),
            discount=jnp.zeros(()),
            next_observation=dummy_obs,
            extras={
                "state_extras": {
                    "truncation": jnp.zeros(()),
                    "traj_id": jnp.zeros(()),
                    "emp_abs": jnp.zeros(()),
                    "emp_delta": jnp.zeros(()),
                }
            },
        )

        def jit_wrap(buf):
            buf.insert_internal = jax.jit(buf.insert_internal)
            buf.sample_internal = jax.jit(buf.sample_internal)
            return buf

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

        # Env init
        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Step bookkeeping
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        logging.info("Num_prefill_actor_steps: %d", num_prefill_actor_steps)

        # ------------------------------------------------------------------
        # RLPD: optional offline OGBench buffer
        # ------------------------------------------------------------------
        offline_buffer = None
        if self.use_rlpd:
            from jaxgcrl.utils.offline_buffer import load_and_prepare_offline_buffer
            # offline_buffer pads the OGBench obs (state_size dim) with `goal_dim`
            # zeros at the tail, producing full_obs_size-dim observations that
            # match the env's raw obs layout.  We then slice off the trailing
            # goal pad in the labelling step before mixing with online data.
            offline_buffer = load_and_prepare_offline_buffer(
                env_name=config.env,
                episode_length=config.episode_length,
                num_slots=config.num_envs,
                obs_size=full_obs_size,
                action_size=action_size,
                state_size=full_obs_size - goal_dim,
                include_phase=False,
            )

        # ------------------------------------------------------------------
        # Empowerment scoring.  Single chunked entry point used by both the
        # online actor step (small batch, fits in one chunk) and offline RLPD
        # labelling (large batch, scanned across many chunks).  Uses the fixed
        # OGBench head template; only ant x/y and ball x/y are overwritten.
        # ant_ball_ogbench obs match the OGBench layout for the first 42 dims,
        # so the brax-side and template-side ball indices are identical.
        # ------------------------------------------------------------------
        offline_chunk = int(self.offline_emp_chunk_size)

        def _empowerment_one_chunk(
            full_obs_batch: jnp.ndarray, rng_key: jnp.ndarray
        ) -> jnp.ndarray:
            batch_size = full_obs_batch.shape[0]
            base_og = jnp.broadcast_to(base_og_const, (batch_size, ex_obs_dim))
            base_og = base_og.at[:, 0].set(full_obs_batch[:, 0])
            base_og = base_og.at[:, 1].set(full_obs_batch[:, 1])
            base_og = base_og.at[:, _ball_x_idx].set(full_obs_batch[:, _ball_x_idx])
            base_og = base_og.at[:, _ball_y_idx].set(full_obs_batch[:, _ball_y_idx])
            return emp_agent.empowerment(base_og, rng=rng_key)

        @jax.jit
        def empowerment_reward(
            full_obs_batch: jnp.ndarray, rng_key: jnp.ndarray
        ) -> jnp.ndarray:
            """Chunked empowerment scoring for any batch size."""
            n = full_obs_batch.shape[0]
            num_chunks = (n + offline_chunk - 1) // offline_chunk
            pad = num_chunks * offline_chunk - n
            padded = jnp.pad(full_obs_batch, ((0, pad), (0, 0)))
            chunks = padded.reshape(num_chunks, offline_chunk, full_obs_batch.shape[-1])
            chunk_keys = jax.random.split(rng_key, num_chunks)

            def step(_, args):
                c, kk = args
                return None, _empowerment_one_chunk(c, kk)

            _, emps = jax.lax.scan(step, None, (chunks, chunk_keys))
            return emps.reshape(-1)[:n]

        # ------------------------------------------------------------------
        # Actor step with empowerment reward
        # ------------------------------------------------------------------
        def actor_step(training_state, env, env_state, action_key, emp_key, extra_fields=()):
            full_obs = env_state.obs[:, :full_obs_size]
            policy_obs = full_obs[:, :obs_size_net]   # drop trailing goal target
            actions = actor.sample_actions(
                training_state.actor_state.params, policy_obs, action_key, is_deterministic=False
            )
            nstate = env.step(env_state, actions)
            next_full_obs = nstate.obs[:, :full_obs_size]
            next_policy_obs = next_full_obs[:, :obs_size_net]
            if self.use_empowerment_diff:
                emp_key_cur, emp_key_next = jax.random.split(emp_key)
                cur_emp = empowerment_reward(full_obs, emp_key_cur)
                next_emp = empowerment_reward(next_full_obs, emp_key_next)
                emp_delta = next_emp - cur_emp
                shaped_reward = emp_delta
            else:
                next_emp = empowerment_reward(next_full_obs, emp_key)
                emp_delta = jnp.zeros_like(next_emp)
                shaped_reward = next_emp
            combined_reward = self.empowerment_reward_scaling * shaped_reward
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras["emp_abs"] = next_emp
            state_extras["emp_delta"] = emp_delta
            return nstate, Transition(
                observation=policy_obs,
                action=actions,
                reward=combined_reward,
                discount=1 - nstate.done,
                next_observation=next_policy_obs,
                extras={"state_extras": state_extras},
            )

        def get_experience(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                es, bs, k = carry
                action_key, emp_key, nk = jax.random.split(k, 3)
                es, transition = actor_step(
                    training_state, train_env, es, action_key, emp_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (es, bs, nk), transition

            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, alpha_key, critic_key, actor_key = jax.random.split(key, 4)
            context = dict(
                discounting=self.discounting,
                target_entropy=target_entropy,
                action_size=action_size,
                state_size=obs_size_net,
            )
            networks = dict(actor=actor, critic=critic)
            training_state, alpha_metrics = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
            training_state, critic_metrics = critic.update(context, networks, transitions, training_state, critic_key)
            training_state, actor_metrics = actor.update(context, networks, transitions, training_state, actor_key)
            new_target = soft_update_target_params(
                training_state.target_critic_params, training_state.critic_states, self.tau
            )
            training_state = training_state.replace(
                target_critic_params=new_target,
                gradient_steps=training_state.gradient_steps + 1,
            )
            metrics = {}
            metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)
            return (training_state, key), metrics

        # ------------------------------------------------------------------
        # Offline empowerment labelling.  Reuses the chunked `empowerment_reward`
        # entry point on the flattened (num_envs, episode_length) sample, then
        # replaces observation/next_observation with the goal-stripped policy
        # obs so shapes match the online buffer.
        # ------------------------------------------------------------------
        def label_offline_with_empowerment(transitions, key):
            full_obs = transitions.observation
            next_full_obs = transitions.next_observation
            N, L = full_obs.shape[0], full_obs.shape[1]
            flat_obs = full_obs.reshape(N * L, -1)
            flat_next = next_full_obs.reshape(N * L, -1)

            k1, k2 = jax.random.split(key)
            next_emp = empowerment_reward(flat_next, k2).reshape(N, L)

            if self.use_empowerment_diff:
                cur_emp = empowerment_reward(flat_obs, k1).reshape(N, L)
                emp_delta = next_emp - cur_emp
                shaped_reward = emp_delta
            else:
                emp_delta = jnp.zeros_like(next_emp)
                shaped_reward = next_emp
            new_reward = self.empowerment_reward_scaling * shaped_reward

            new_extras = dict(transitions.extras["state_extras"])
            new_extras["emp_abs"] = next_emp
            new_extras["emp_delta"] = emp_delta

            sliced_obs = full_obs[..., :obs_size_net]
            sliced_next = next_full_obs[..., :obs_size_net]

            return transitions._replace(
                observation=sliced_obs,
                next_observation=sliced_next,
                reward=new_reward,
                extras={"state_extras": new_extras},
            )

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key, process_key, train_key = jax.random.split(key, 3)
            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, exp_key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )
            buffer_state, online_transitions = replay_buffer.sample(buffer_state)

            if self.use_rlpd:
                offline_key, emp_key, process_key = jax.random.split(process_key, 3)
                raw_offline = offline_buffer.sample(offline_key, config.num_envs)
                offline_transitions = label_offline_with_empowerment(raw_offline, emp_key)
                transitions = jax.tree_util.tree_map(
                    lambda a, b: jnp.concatenate([a, b], axis=0),
                    online_transitions, offline_transitions,
                )
            else:
                transitions = online_transitions

            r = transitions.reward / self.empowerment_reward_scaling
            transitions, _ = actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                obs_size_net, goal_indices, unwrapped_env.goal_reach_thresh, use_her=False,
            )

            if self.use_rlpd:
                # After concat, process_transitions produces 2x the online batch
                # count.  Slicing in half restores the original gradient-step
                # count; the random permutation inside process_transitions has
                # already mixed online and offline samples within each batch.
                num_batches = transitions.observation.shape[0] // 2
                transitions = jax.tree_util.tree_map(lambda x: x[:num_batches], transitions)
            if self.normalize_rewards:
                r_all = transitions.reward
                r_mean = jnp.mean(r_all)
                r_std = jnp.std(r_all)
                normalized = (r_all - r_mean) / (r_std + 1e-8)
                transitions = transitions._replace(reward=normalized)
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics["reward_mean"] = jnp.mean(r)
            metrics["reward_min"] = jnp.min(r)
            metrics["reward_max"] = jnp.max(r)
            emp_abs = transitions.extras["state_extras"]["emp_abs"]
            metrics["emp_abs_mean"] = jnp.mean(emp_abs)
            metrics["emp_abs_min"] = jnp.min(emp_abs)
            metrics["emp_abs_max"] = jnp.max(emp_abs)
            metrics["emp_abs_scaled_mean"] = self.empowerment_reward_scaling * jnp.mean(emp_abs)
            emp_delta = transitions.extras["state_extras"]["emp_delta"]
            metrics["emp_delta_mean"] = jnp.mean(emp_delta)
            metrics["emp_delta_min"] = jnp.min(emp_delta)
            metrics["emp_delta_max"] = jnp.max(emp_delta)
            metrics["emp_delta_scaled_mean"] = self.empowerment_reward_scaling * jnp.mean(emp_delta)
            return (training_state, env_state, buffer_state), metrics

        # Steps-per-epoch scheduling (same as SAC)
        available_env_steps = config.total_env_steps - (self.min_replay_size * config.num_envs)
        env_steps_per_epoch = max(1, available_env_steps // max(config.num_evals, 1))
        num_training_steps_per_epoch = max(1, env_steps_per_epoch // env_steps_per_actor_step)

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (),
                length=num_training_steps_per_epoch,
            )
            return training_state, env_state, buffer_state, metrics

        # Prefill
        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k = carry
                k, nk = jax.random.split(k)
                es, bs = get_experience(ts, es, bs, k)
                ts = ts.replace(env_steps=ts.env_steps + env_steps_per_actor_step)
                return (ts, es, bs, nk), ()

            (training_state, env_state, buffer_state, _), _ = jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (),
                length=num_prefill_actor_steps,
            )
            return training_state, env_state, buffer_state

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # ------------------------------------------------------------------
        # Evaluator
        # ------------------------------------------------------------------
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            full_obs = env_state.obs[:, :full_obs_size]
            policy_obs = full_obs[:, :obs_size_net]
            actions = actor.sample_actions(
                training_state.actor_state.params, policy_obs,
                jax.random.PRNGKey(0), is_deterministic=self.deterministic_eval,
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

        def make_policy(params, deterministic: bool = False):
            def policy(obs, key):
                # Strip trailing goal target so render-time obs (full env obs)
                # matches the policy's expected input dim.
                policy_obs = obs[..., :obs_size_net]
                actions = actor.sample_actions(params, policy_obs, key, is_deterministic=deterministic)
                return actions, {}
            return policy

        # Precompute maze bounds for visualization
        x_low = float(unwrapped_env.x_bounds[0])
        x_high = float(unwrapped_env.x_bounds[1])
        y_low = float(unwrapped_env.y_bounds[0])
        y_high = float(unwrapped_env.y_bounds[1])
        # Fixed ball position for the empowerment map: centre of the maze bounds
        _vis_ball_xy = (0.5 * (x_low + x_high), 0.5 * (y_low + y_high))

        # ------------------------------------------------------------------
        # Main training loop
        # ------------------------------------------------------------------
        training_walltime = 0.0
        current_step = 0
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            epoch_time = time.time() - t
            training_walltime += epoch_time
            current_step = int(training_state.env_steps.item())
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_time
            log = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": current_step,
            }
            for name, value in metrics.items():
                v = float(value.item()) if hasattr(value, "item") else float(value)
                log[f"training/{name}"] = v
            log = evaluator.run_evaluation(training_state, log)

            do_render = (ne % config.visualization_interval) == 0
            if do_render:
                buffer_state = _log_dist_vs_reward_scatter(
                    replay_buffer,
                    buffer_state,
                    ball_indices,
                    self.empowerment_reward_scaling,
                    current_step,
                )
                buffer_state = _log_replay_xy_reward_scatter(
                    replay_buffer,
                    buffer_state,
                    current_step,
                )
                buffer_state = _log_replay_empowerment_scatter(
                    replay_buffer,
                    buffer_state,
                    ball_indices,
                    current_step,
                )
                key, vis_key = jax.random.split(key)
                _log_empowerment_map(
                    emp_agent=emp_agent,
                    base_og_const=base_og_const,
                    ex_obs_dim=ex_obs_dim,
                    ball_x_idx=_ball_x_idx,
                    ball_y_idx=_ball_y_idx,
                    x_low=x_low,
                    x_high=x_high,
                    y_low=y_low,
                    y_high=y_high,
                    grid_res=80,
                    current_step=current_step,
                    rng=vis_key,
                    fixed_ball_xy=_vis_ball_xy,
                )

            progress_fn(
                current_step,
                log,
                functools.partial(make_policy, deterministic=self.deterministic_eval),
                training_state.actor_state.params,
                unwrapped_env,
                do_render,
            )

        params = training_state.actor_state.params
        return functools.partial(make_policy, deterministic=self.deterministic_eval), params, log