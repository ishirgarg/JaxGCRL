import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .types import Actor, Critic, TrainingState, Transition
from .algorithms import get_algorithm
from .utils import save_params
from .losses import update_alpha_sac, update_critic_sac, update_actor_sac
from .visualization import all_visualizations
from .goal_proposers import create_goal_proposer

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

@dataclass
class Baseline:
    """Unified baseline agent supporting both CRL and SAC algorithms.
    
    This agent can be configured to run either CRL (Contrastive Reinforcement Learning)
    or SAC (Soft Actor-Critic) by setting the agent_type parameter.
    """

    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 100000
    min_replay_size: int = 1000
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # SAC specific
    tau: float = 0.005
    n_critics: int = 2
    use_her: bool = True  # Hindsight Experience Replay

    # goal proposer for training
    goal_proposer_name: Literal["random_env_goals"] = "random_env_goals"

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
            NOTE: `episode_length` must be divisible by `unroll_length`.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        assert (config.episode_length - 1) % self.unroll_length == 0, (
            f"episode_length ({config.episode_length}) must be divisible by unroll_length ({self.unroll_length})"
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
        
        # Create goal proposer and modify reset to use it
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
        )
        original_reset = unwrapped_env.reset
        
        def reset_with_goal_proposer(rng: jax.Array) -> State:
            """Reset with goal proposer.
            
            Args:
                rng: Single random key (will be vmap'd by training wrapper)
            """
            goal = goal_proposer(rng)  # Shape: (goal_dim,)
            return original_reset(rng, goal=goal)
        
        unwrapped_env.reset = reset_with_goal_proposer
        
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        # NOTE: an actor_step here is a whole config.episode_length long episode
        # episode_length includes initial state, so we have (episode_length - 1) transitions
        num_unrolls_per_episode = (config.episode_length - 1) // self.unroll_length
        env_steps_per_actor_step = config.num_envs * (config.episode_length - 1)
        
        # Prefill uses unroll_length (original behavior)
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        
        # Calculate training steps per epoch
        # Available env steps for training = total - prefill
        available_env_steps = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_unrolls_per_episode: %d",
            num_unrolls_per_episode,
        )
        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "env_steps_per_epoch: %d",
            env_steps_per_epoch,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        actor, critic = get_algorithm(
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
            n_critics=self.n_critics,  # Number of critics in ensemble (for both CRL and SAC)
        )

        actor_params = actor.init(actor_key, np.ones([1, obs_size]))
        critic_params = critic.init(critic_key, np.ones([1, obs_size]))

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # SAC-specific initialization
        target_critic_params = None
        if self.agent_type == "sac":
            # Initialize target Q-network params (copy of critic params)
            target_critic_params = critic_params

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            next_observation=dummy_obs if self.agent_type == "sac" else None,  # SAC needs next_observation
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
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

        def actor_step(actor_state, env, env_state, key, extra_fields, is_deterministic: bool):
            actions = actor.sample_actions(
                actor_state.params,
                env_state.obs,
                key,
                is_deterministic=is_deterministic
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs, 
                action=actions, 
                reward=nstate.reward, 
                discount=1 - nstate.done,
                next_observation=nstate.obs if self.agent_type == "sac" else None,  # SAC needs next_observation
                extras={"state_extras": state_extras}
            )

        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key, is_deterministic: bool):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                    is_deterministic=is_deterministic,
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_state,
                    key,
                    is_deterministic=False,
                )
                # Prefill uses unroll_length steps per call
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + config.num_envs * self.unroll_length,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

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

            networks = dict(
                actor=actor,
                critic=critic,
            )

            # Update order: match original implementations exactly
            # CRL: actor (with alpha) then critic (original CRL order)
            # SAC: alpha, critic, then actor (original SAC order - line 362, 371, 381)
            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics = actor.update(context, networks, transitions, training_state, actor_key)
                training_state, critic_metrics = critic.update(context, networks, transitions, training_state, critic_key)
            elif self.agent_type == "sac":  # SAC
                # SAC updates: alpha first, then critic, then actor (matching original)
                training_state, alpha_metrics = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
                training_state, critic_metrics = critic.update(context, networks, transitions, training_state, critic_key)
                training_state, actor_metrics = actor.update(context, networks, transitions, training_state, actor_key)
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)
            
            # Update target networks for SAC
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                new_target_critic_params = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params,
                    training_state.critic_state.params,
                )
                training_state = training_state.replace(target_critic_params=new_target_critic_params)
            
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, process_key, training_key = jax.random.split(key, 3)

            # Collect full episode by calling get_experience num_unrolls_per_episode times
            def collect_unroll(carry, unused):
                env_state, buffer_state, key = carry
                key, next_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_state,
                    key,
                    is_deterministic=False,
                )
                return (env_state, buffer_state, next_key), ()
            
            (env_state, buffer_state, _), _ = jax.lax.scan(
                collect_unroll,
                (env_state, buffer_state, experience_key1),
                (),
                length=num_unrolls_per_episode,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            
            # Process transitions (algorithm-specific: flatten_batch, reshape, permute)
            transitions, _ = actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting, state_size, 
                tuple(train_env.goal_indices), train_env.goal_reach_thresh, self.use_her
            )
            
            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    metrics,
                ) = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            lambda ts, env, es, extra_fields=(): actor_step(ts.actor_state, env, es, jax.random.PRNGKey(0), extra_fields, is_deterministic=True),
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            # For CRL: return (mean, log_std) from actor.apply (matches CRL's pattern)
            # For SAC: return (action, {}) from actor.sample_actions (matches SAC's pattern)
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            elif self.agent_type == "sac":  # SAC
                make_policy = lambda param: lambda obs, rng: (actor.sample_actions(param, obs, rng, is_deterministic=True), {})

            # Visualize trajectories
            key, viz_key = jax.random.split(key)
            buffer_state = all_visualizations(
                replay_buffer=replay_buffer,
                buffer_state=buffer_state,
                env=unwrapped_env,
                state_size=state_size,
                goal_indices=tuple(train_env.goal_indices),
                rng_key=viz_key,
            )

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            # Prepare params for return (and optionally save if checkpointing)
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )
            
            if config.checkpoint_logdir:
                # Save current policy and critic params.
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        # assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
