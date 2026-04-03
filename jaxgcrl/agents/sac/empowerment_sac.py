import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
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
from jaxgcrl.agents.go_explore.losses import (
    update_alpha_sac,
    update_actor_sac,
    update_critic_sac,
)
from jaxgcrl.agents.sac.visualization import (
                    log_dist_vs_reward_scatter,
                    log_empowerment_map,
                )
from jaxgcrl.agents.go_explore.types import TrainingState as GETrainingState, Transition

def _setup_external_imports(ogbench_root: str):
    """Prepare OGBench imports and return registry/helpers."""
    import sys, os
    impls_root = os.path.join(ogbench_root, "impls")
    for p in (impls_root, ogbench_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agents import agents as agent_registry
    from utils.env_utils import make_env_and_datasets
    from utils.flax_utils import restore_agent
    return agent_registry, make_env_and_datasets, restore_agent


Metrics = Any


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
    deterministic_eval: bool = True

    # Empowerment checkpoint args
    ogbench_root: str = "/home/ishir/ogbench"
    ckpt_root: str = "/home/ishir/ogbench/impls/ckpts"
    run_dir: Optional[str] = "/home/ishir/ogbench/impls/ckpts/sd000_s_32837516.0.20260319_183520"
    epoch: Optional[int] = None
    num_splus_samples: int = 128
    ogbench_xy_offset: Optional[float] = None

    def train_fn(
        self,
        config,
        train_env,
        eval_env=None,
        randomization_fn: Optional[Callable] = None,
        progress_fn: Callable[[int, Metrics, Callable, Any, Any, bool], None] = lambda *args, **kwargs: None,
    ):
        # Load empowerment checkpoint
        agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(
            self.ogbench_root
        )
        if self.run_dir is None:
            raise ValueError("run_dir must be provided for empowerment checkpoint.")
        import os, re, json, glob
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
        # choose latest epoch if not provided
        if self.epoch is None:
            ckpts = glob.glob(os.path.join(self.run_dir, "params_*.pkl"))
            epochs = []
            for p in ckpts:
                m = re.search(r"params_(\d+)\.pkl$", os.path.basename(p))
                if m:
                    epochs.append(int(m.group(1)))
            if not epochs:
                raise FileNotFoundError(f"No params_*.pkl found in {self.run_dir}")
            self_epoch = max(epochs)
        else:
            self_epoch = self.epoch
        emp_agent = restore_agent(emp_agent, self.run_dir, self_epoch)
        ex_obs_dim = int(example_batch["observations"].shape[-1])

        # Wrap envs like go_explore style (not brax training wrap to keep goal intact)
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
        action_size = unwrapped_env.action_size
        state_size = int(unwrapped_env.observation_size)
        obs_size_net = state_size
        # With new layout: tail is [ball_rel(2), goal_rel(2)]
        goal_indices = (obs_size_net - 4, obs_size_net - 3)  # start of ball_rel (used only for APIs that require it)
        goal_target_indices = (obs_size_net - 2, obs_size_net - 1)  # goal_rel

        # Networks
        key = jax.random.PRNGKey(config.seed)
        key, actor_key, critic_key, alpha_key, env_key, eval_env_key, buffer_key = jax.random.split(key, 7)
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
        target_entropy = -0.5 * action_size
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

        # Replay buffer
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

        # Empowerment reward adapter: pure-JAX mapping to OGBench obs without setting goal (goal zeroed).
        ogbench_base_obs_dim = 42
        if ex_obs_dim % ogbench_base_obs_dim != 0:
            raise ValueError(f"Checkpoint obs dim {ex_obs_dim} must be multiple of {ogbench_base_obs_dim}.")
        frame_stack = ex_obs_dim // ogbench_base_obs_dim
        # Capture a constant OGBench head (first 38 dims) from an actual env reset observation.
        # This mirrors the plotting script preference to use env-generated observation structure.
        obs_reset, _ = ogbench_env.reset()
        obs_reset_np = np.asarray(obs_reset, dtype=np.float32)
        base_frame_np = obs_reset_np[:ogbench_base_obs_dim]  # take first frame if stacked
        og_head_np = base_frame_np[: ogbench_base_obs_dim - 4]  # first 38 dims
        og_head_const = jnp.asarray(og_head_np)  # constant used inside jit

        @jax.jit
        def empowerment_reward_online_with_key(full_obs_batch: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
            # State obs layout: [ant_xy(2), ..., ball_rel(2), goal_rel(2)].
            # Build 42-d OGBench obs by taking a fixed head from example obs and inserting current rel terms.
            batch_size = full_obs_batch.shape[0]
            ball_rel = full_obs_batch[:, -4:-2]
            goal_rel = full_obs_batch[:, -2:]
            head = jnp.broadcast_to(og_head_const, (batch_size, ogbench_base_obs_dim - 4))
            base_og = jnp.concatenate([head, ball_rel, goal_rel], axis=1)  # [N, 42]
            if frame_stack > 1:
                base_og = jnp.tile(base_og, (1, frame_stack))  # [N, 42*stack]
            return emp_agent.empowerment(base_og, rng=rng_key)


        # Actor step with empowerment reward
        def actor_step(training_state, env, env_state, action_key, emp_key, extra_fields=()):
            state_obs = env_state.obs[:, :state_size]
            actions = actor.sample_actions(
                training_state.actor_state.params, state_obs, action_key, is_deterministic=False
            )
            nstate = env.step(env_state, actions)
            next_state_obs = nstate.obs[:, :state_size]

            # Absolute-empowerment shaping: reward proportional to next state's empowerment.
            next_emp = empowerment_reward_online_with_key(next_state_obs, emp_key)
            combined_reward = self.empowerment_reward_scaling * next_emp # + nstate.reward
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras["emp_abs"] = next_emp
            return nstate, Transition(
                observation=state_obs,
                action=actions,
                reward=combined_reward,
                discount=1 - nstate.done,
                next_observation=next_state_obs,
                extras={"state_extras": state_extras},
            )

        def get_experience(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                es, bs, k = carry
                action_key, emp_key, nk = jax.random.split(k, 3)
                es, transition = actor_step(training_state, train_env, es, action_key, emp_key, extra_fields=("truncation", "traj_id"))
                return (es, bs, nk), transition

            (env_state, buffer_state, _), data = jax.lax.scan(f, (env_state, buffer_state, key), (), length=self.unroll_length)
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
            full_cp = {}
            for i, cs in enumerate(training_state.critic_states):
                for lname, lparams in cs.params.items():
                    full_cp[f"critic_{i}_{lname}"] = lparams
            new_target = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_critic_params,
                full_cp,
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

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key, process_key, train_key = jax.random.split(key, 3)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, exp_key)
            training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            # Empowerment reward stats over sampled batch
            r = transitions.reward / self.empowerment_reward_scaling
            transitions, _ = actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                obs_size_net, goal_indices, unwrapped_env.goal_reach_thresh, use_her=False,
            )
            (training_state, _), metrics = jax.lax.scan(update_networks, (training_state, train_key), transitions)
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics["reward_mean"] = jnp.mean(r)
            metrics["reward_min"] = jnp.min(r)
            metrics["reward_max"] = jnp.max(r)
            # Log absolute empowerment term separately (unscaled and scaled)
            emp_abs = transitions.extras["state_extras"]["emp_abs"]
            metrics["emp_abs_mean"] = jnp.mean(emp_abs)
            metrics["emp_abs_min"] = jnp.min(emp_abs)
            metrics["emp_abs_max"] = jnp.max(emp_abs)
            metrics["emp_abs_scaled_mean"] = self.empowerment_reward_scaling * jnp.mean(emp_abs)
            return (training_state, env_state, buffer_state), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch,
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
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )
            return training_state, env_state, buffer_state

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_key)

        # Evaluator
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            state_obs = env_state.obs[:, :state_size]
            actions = actor.sample_actions(
                training_state.actor_state.params, state_obs, jax.random.PRNGKey(0), is_deterministic=self.deterministic_eval,
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

        # Main loop
        training_walltime = 0.0
        # Steps per epoch to spread evenly over num_evals (like SAC)
        available_env_steps = config.total_env_steps - (self.min_replay_size * config.num_envs)
        env_steps_per_epoch = max(1, available_env_steps // max(config.num_evals, 1))
        num_training_steps_per_epoch = max(1, env_steps_per_epoch // env_steps_per_actor_step)
        # Precompute layout metadata for script-style empowerment-map plotting.
        from jaxgcrl.envs.ant_ball_maze import BIG_MAZE, SQUARE_MAZE, U_MAZE
        if "square" in config.env:
            layout = np.array(SQUARE_MAZE, dtype=object)
            layout_name = "square_maze"
        elif "u_maze" in config.env:
            layout = np.array(U_MAZE, dtype=object)
            layout_name = "u_maze"
        elif "big_maze" in config.env:
            layout = np.array(BIG_MAZE, dtype=object)
            layout_name = "big_maze"
        else:
            layout = np.array(SQUARE_MAZE, dtype=object)
            layout_name = "square_maze"
        wall_mask = np.equal(layout, 1)
        x_low = float(unwrapped_env.x_bounds[0])
        x_high = float(unwrapped_env.x_bounds[1])
        y_low = float(unwrapped_env.y_bounds[0])
        y_high = float(unwrapped_env.y_bounds[1])
        grid_res = 80
        xs = np.linspace(x_low, x_high, grid_res, dtype=np.float32)
        ys = np.linspace(y_low, y_high, grid_res, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        maze_size_scaling = (x_high - x_low) / float(layout.shape[0])
        half = 0.5 * maze_size_scaling

        # Match SAC's make_policy signature: make_policy(params, deterministic=False) -> (obs, key) -> (actions, extras)
        def make_policy(params, deterministic: bool = False):
            def policy(obs, key):
                actions = actor.sample_actions(params, obs, key, is_deterministic=deterministic)
                return actions, {}
            return policy

        current_step = 0
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            epoch_time = time.time() - t
            training_walltime += epoch_time
            current_step = int(training_state.env_steps.item())
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_time
            log = {"training/sps": sps, "training/walltime": training_walltime, "training/envsteps": current_step}
            for name, value in metrics.items():
                v = float(value.item()) if hasattr(value, "item") else float(value)
                log[f"training/{name}"] = v
            log = evaluator.run_evaluation(training_state, log)
            do_render = (ne % config.visualization_interval) == 0
            if do_render:
                buffer_state = log_dist_vs_reward_scatter(
                    replay_buffer,
                    buffer_state,
                    goal_indices,
                    self.empowerment_reward_scaling,
                    current_step,
                )
                key, vis_emp_key = jax.random.split(key)
                buffer_state = log_empowerment_map(
                    replay_buffer,
                    buffer_state,
                    unwrapped_env,
                    goal_indices,
                    goal_target_indices,
                    obs_size_net,
                    empowerment_reward_online_with_key,
                    current_step,
                    config.env,
                    vis_emp_key,
                )
            progress_fn(
                current_step,
                log,
                functools.partial(make_policy, deterministic=self.deterministic_eval),
                training_state.actor_state.params,  # actor params
                unwrapped_env,
                do_render,
            )

        params = training_state.actor_state.params
        return functools.partial(make_policy, deterministic=self.deterministic_eval), params, log

