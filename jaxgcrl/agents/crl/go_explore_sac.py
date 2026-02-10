import functools
import logging
from math import e
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types, gradients
from brax.training.agents.sac import losses as sac_losses
from brax.training.acme import running_statistics, specs
from jaxgcrl.agents.sac.sac import Transition as SACTransition
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.visualize import visualize_goals_2d, visualize_kde_heatmap, visualize_q_function_2d, visualize_dual_crl_trajectories_2d, visualize_scatter_sample

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder
from .proposers import (
    FinalReplayBufferProposer, 
    RandomEnvironmentGoalProposer,
    UCGRProposer,
    MaxWaypointRatioOneEnvProposer,
    QEpistemicProposer,
)
from .goals_utils import compute_min_critic_mean_reward, compute_max_critic_reward_per_transition
from jaxgcrl.agents.sac import networks as sac_networks

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    gcp_actor_state: TrainState
    gcp_critic_state: TrainState
    ep_actor_state: TrainState
    ep_q_params: Any  # SAC Q-network params
    ep_target_q_params: Any  # SAC target Q-network params
    ep_q_optimizer_state: Any  # SAC Q-network optimizer state
    ep_actor_optimizer_state: Any  # SAC actor optimizer state
    gcp_alpha_state: TrainState
    ep_alpha_params: jnp.ndarray  # SAC alpha (log_alpha)
    ep_alpha_optimizer_state: Any  # SAC alpha optimizer state
    ep_normalizer_params: Any  # SAC normalizer params


class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    """Process trajectory for CRL training - same as go_explore_crl."""
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    proposed_goals = transition.observation[:, -len(goal_indices):]

    def last_state_for_each_step(obs, traj_ids):
        seq_len = obs.shape[0]
        def last_state_for_t(i):
            mask = traj_ids == traj_ids[i]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs[last_idx]
        return jax.vmap(last_state_for_t)(jnp.arange(seq_len))

    def get_intermediate_trajectory_states(obs, traj_ids):
        """Returns states at evenly spaced points along remaining trajectory."""
        seq_len = obs.shape[0]
        obs_dim = obs.shape[1]
        
        def intermediate_states_for_t(i, num_intermediate):
            same_traj_mask = traj_ids == traj_ids[i]
            future_mask = jnp.arange(seq_len) >= i
            mask = same_traj_mask & future_mask
            indices = jnp.where(mask, jnp.arange(seq_len), seq_len)
            sorted_indices = jnp.sort(indices)
            num_future = jnp.sum(mask)
            fractions = (jnp.arange(1, num_intermediate + 1) / (num_intermediate + 1))
            idxs = jnp.floor(fractions * num_future).astype(jnp.int32)
            idxs = jnp.clip(idxs, 0, jnp.maximum(num_future - 1, 0))
            actual_idxs = sorted_indices[idxs]
            def get_state(idx):
                return jnp.where(num_future > 0, obs[idx], jnp.zeros(obs_dim))
            states = jax.vmap(get_state)(actual_idxs)
            return states
                
        return jax.vmap(functools.partial(intermediate_states_for_t, num_intermediate=6))(jnp.arange(seq_len))

    traj_ids = transition.extras["state_extras"]["traj_id"]
    last_traj_state = last_state_for_each_step(transition.observation, traj_ids)
    intermediate_traj = get_intermediate_trajectory_states(transition.observation, traj_ids)

    def is_last_occurrence(i):
        traj_id = traj_ids[i]
        future_mask = jnp.arange(seq_len) > i
        has_future_same_id = jnp.any((traj_ids == traj_id) & future_mask)
        return ~has_future_same_id
    
    last_traj_state_mask = jax.vmap(is_last_occurrence)(jnp.arange(seq_len))

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)

    original_state_extras = transition.extras["state_extras"]
    state_extras = jax.tree_util.tree_map(
        lambda x: x[:-1] if len(x.shape) > 0 else x,
        original_state_extras
    )
    state_extras["truncation"] = jnp.squeeze(state_extras["truncation"])
    state_extras["traj_id"] = jnp.squeeze(state_extras["traj_id"])
    
    extras = {
        "policy_extras": {},
        "state_extras": state_extras,
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "proposed_goals": proposed_goals[:-1],
        "last_traj_state": last_traj_state[:-1],
        "intermediate_traj": intermediate_traj[:-1],
        "last_traj_state_mask": last_traj_state_mask[:-1],
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class GoExploreSAC:
    """Go-Explore with Soft Actor-Critic for EP.
    
    GCP (Goal-Conditioned Policy): Uses CRL, same as GoExploreCRL
    EP (Exploratory Policy): Uses SAC, NOT goal-conditioned, trained via reward_name
    """
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty (for GCP only)
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension (for GCP only)
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False
    
    # EP observation normalization (same as SAC)
    normalize_observations: bool = False
    
    # EP target update rate (same as SAC)
    tau: float = 0.005

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Goal proposer names for GCP (EP doesn't use goal proposal)
    gcp_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals", "ucgr", "maxwaypointratio_one_env", "q_epistemic"] = "gcp_final_rb"
    goal_sampling_temperature: float = 1.0
    
    # Critic ensemble for Q-epistemic goal proposal (GCP only)
    use_gcp_critic_ensemble: bool = False
    gcp_num_critic_ensemble: int = 5
    
    # EP reward function (EP never uses environment reward, always uses this)
    ep_reward_fn: Literal["max_critic"] = "max_critic"

    # Unroll length for network updates
    unroll_length: int = 50

    def check_config(self, config):
        """Validate configuration parameters."""
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        assert config.num_goal_conditioned_steps % self.unroll_length == 0, (
            f"num_goal_conditioned_steps ({config.num_goal_conditioned_steps}) must be divisible by unroll_length ({self.unroll_length})"
        )
        assert config.num_exploratory_steps % self.unroll_length == 0, (
            f"num_exploratory_steps ({config.num_exploratory_steps}) must be divisible by unroll_length ({self.unroll_length})"
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

        logging.info("Num env: %d", config.num_envs)

        # Step calculations matching go_explore_crl pattern
        unroll_length = self.unroll_length
        total_steps_per_training_step = config.num_goal_conditioned_steps + config.num_exploratory_steps
        env_steps_per_actor_step = config.num_envs * unroll_length  # Steps per chunk
        env_steps_per_training_step = config.num_envs * total_steps_per_training_step  # Steps per full training iteration
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / total_steps_per_training_step))
        # Ceiling division for num_training_steps_per_epoch
        num_training_steps_per_epoch = int(-(-(config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_training_step
        )))

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, gcp_actor_key, gcp_sa_key, gcp_g_key, ep_actor_key, ep_q_key = jax.random.split(key, 9)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)
        
        # Assign unique trajectory IDs per environment
        TRAJ_ID_MULTIPLIER = 100000
        env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
        initial_info = dict(env_state.info)
        initial_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
        # Initialize gc_proposed_goals and ep_proposed_goals for consistent PyTree structure
        initial_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
        initial_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
        env_state = env_state.replace(info=initial_info)

        # Dimensions definitions
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )
        
        # EP observation size (no goals)
        ep_obs_size = state_size

        # ===== Network Setup =====
        # GCP Actor (goal-conditioned)
        gcp_actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
       
        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor.init(gcp_actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # EP Actor (non-goal-conditioned) - uses SAC network
        def normalize_fn(x, y):
            return x
        
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize
        
        ep_sac_network = sac_networks.make_sac_networks(
            observation_size=ep_obs_size,
            action_size=action_size,
            preprocess_observations_fn=normalize_fn,
            layer_norm=self.use_ln,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
        )
        
        make_ep_policy = sac_networks.make_inference_fn(ep_sac_network)
        
        ep_actor_params = ep_sac_network.policy_network.init(ep_actor_key)
        ep_q_params = ep_sac_network.q_network.init(ep_q_key)
        ep_target_q_params = jax.tree_util.tree_map(lambda x: x, ep_q_params)

        # GCP Critic (CRL)
        gcp_sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        gcp_g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        
        if self.use_gcp_critic_ensemble:
            gcp_sa_keys = jax.random.split(gcp_sa_key, self.gcp_num_critic_ensemble)
            gcp_g_keys = jax.random.split(gcp_g_key, self.gcp_num_critic_ensemble)
            gcp_sa_encoder_params = [gcp_sa_encoder.init(k, np.ones([1, state_size + action_size])) for k in gcp_sa_keys]
            gcp_g_encoder_params = [gcp_g_encoder.init(k, np.ones([1, goal_size])) for k in gcp_g_keys]
        else:
            gcp_sa_encoder_params = gcp_sa_encoder.init(gcp_sa_key, np.ones([1, state_size + action_size]))
            gcp_g_encoder_params = gcp_g_encoder.init(gcp_g_key, np.ones([1, goal_size]))
        
        gcp_critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": gcp_sa_encoder_params, "g_encoder": gcp_g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        gcp_alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )
        
        # EP alpha (SAC)
        ep_alpha_params = log_alpha
        
        # EP normalizer (SAC)
        ep_normalizer_params = running_statistics.init_state(
            specs.Array((ep_obs_size,), jnp.float32)
        )

        # ===== SAC Loss Functions and Optimizers for EP =====
        ep_alpha_optimizer = optax.adam(learning_rate=self.alpha_lr)
        ep_policy_optimizer = optax.adam(learning_rate=self.policy_lr)
        ep_q_optimizer = optax.adam(learning_rate=self.critic_lr)
        
        ep_alpha_loss_fn, ep_critic_loss_fn, ep_actor_loss_fn = sac_losses.make_losses(
            sac_network=ep_sac_network,
            reward_scaling=1.0,
            discounting=self.discounting,
            action_size=action_size,
        )
        
        ep_alpha_update = gradients.gradient_update_fn(
            ep_alpha_loss_fn, ep_alpha_optimizer, pmap_axis_name=None
        )
        ep_critic_update = gradients.gradient_update_fn(
            ep_critic_loss_fn, ep_q_optimizer, pmap_axis_name=None
        )
        ep_actor_update = gradients.gradient_update_fn(
            ep_actor_loss_fn, ep_policy_optimizer, pmap_axis_name=None
        )

        # Initialize EP optimizer states
        ep_alpha_optimizer_state = ep_alpha_optimizer.init(ep_alpha_params)
        ep_q_optimizer_state = ep_q_optimizer.init(ep_q_params)
        ep_actor_optimizer_state = ep_policy_optimizer.init(ep_actor_params)
        
        ep_actor_state = TrainState.create(
            apply_fn=None,
            params=ep_actor_params,
            tx=ep_policy_optimizer,
        )
        
        # Training state
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            gcp_actor_state=gcp_actor_state,
            gcp_critic_state=gcp_critic_state,
            ep_actor_state=ep_actor_state,
            ep_q_params=ep_q_params,
            ep_target_q_params=ep_target_q_params,
            ep_q_optimizer_state=ep_q_optimizer_state,
            ep_actor_optimizer_state=ep_actor_optimizer_state,
            gcp_alpha_state=gcp_alpha_state,
            ep_alpha_params=ep_alpha_params,
            ep_alpha_optimizer_state=ep_alpha_optimizer_state,
            ep_normalizer_params=ep_normalizer_params,
        )

        # ===== Replay Buffers =====
        # Unified transition format for all buffers
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_goal = jnp.zeros((goal_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    "in_gc_phase": 0.0,
                    "in_ep_phase": 0.0,
                    "gc_proposed_goals": dummy_goal,
                    "ep_proposed_goals": dummy_goal,
                    "terminated": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        main_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        main_buffer_state = jax.jit(main_replay_buffer.init)(buffer_key)

        gcp_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        gcp_buffer_state = jax.jit(gcp_replay_buffer.init)(buffer_key)

        ep_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        ep_buffer_state = jax.jit(ep_replay_buffer.init)(buffer_key)

        # ===== Goal Proposers (GCP only - EP SAC is non-goal-conditioned) =====
        gcp_final_rb_proposer = FinalReplayBufferProposer()
        ep_final_rb_proposer = FinalReplayBufferProposer()
        env_goals_proposer = RandomEnvironmentGoalProposer()
        ucgr_proposer = UCGRProposer(energy_fn_name=self.energy_fn, num_samples=256)
        maxwaypointratio_one_env_proposer = MaxWaypointRatioOneEnvProposer(
            energy_fn_name=self.energy_fn,
            goal_sampling_temperature=self.goal_sampling_temperature
        )
        q_epistemic_proposer = QEpistemicProposer(
            energy_fn_name=self.energy_fn,
            num_ensemble=self.gcp_num_critic_ensemble,
            use_env_goals=False,
            zero_center=False
        )
        
        if self.gcp_goal_proposer_name == "gcp_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(
                    gcp_replay_buffer, gcp_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, updated_gcp, ep_buffer_state
        elif self.gcp_goal_proposer_name == "ep_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "env_goals":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = env_goals_proposer.propose_goals(
                    None, gcp_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state
        elif self.gcp_goal_proposer_name == "ucgr":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ucgr_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "maxwaypointratio_one_env":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = maxwaypointratio_one_env_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "q_epistemic":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = q_epistemic_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        else:
            raise ValueError(f"Unknown gcp_goal_proposer_name: {self.gcp_goal_proposer_name}")

        # ===== Actor Step Functions =====
        def deterministic_actor_step_with_proposals(actor_state, env, env_state, proposed_goals, extra_fields):
            """Deterministic actor step for GCP with proposed goals."""
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)

            means, _ = actor_state.apply_fn(actor_state.params, env_state.obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )
        
        def ep_actor_step(training_state, env, env_state, key, extra_fields):
            """EP actor step (non-goal-conditioned, uses SAC policy)."""
            state_only = env_state.obs[:, :state_size]
            
            ep_policy_params = (training_state.ep_normalizer_params, training_state.ep_actor_state.params)
            policy = make_ep_policy(ep_policy_params, deterministic=False)
            actions, _ = policy(state_only, key)
            
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            
            # Pad observation with zeros to match obs_size (for replay buffer consistency)
            obs_padded = jnp.concatenate([state_only, jnp.zeros((state_only.shape[0], goal_size))], axis=-1)
            
            return nstate, Transition(
                observation=obs_padded,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # Used for GCP evaluation
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = gcp_actor.apply(training_state.gcp_actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # EP reward setup (EP always uses a specified reward, never environment reward)
        if self.ep_reward_fn == "max_critic":
            assert hasattr(train_env, 'possible_goals') and train_env.possible_goals is not None, (
                "ep_reward_fn='max_critic' requires train_env.possible_goals to be defined"
            )
            env_goals_for_reward = train_env.possible_goals
        else:
            raise ValueError(f"Unknown ep_reward_fn: {self.ep_reward_fn}")

        # ===== Experience Collection (chunk-based, matching go_explore_crl) =====
        @jax.jit
        def get_experience_chunk(training_state, env_state,
                                 main_buffer_state, gcp_buffer_state, ep_buffer_state,
                                 gc_steps_collected, ep_steps_collected, 
                                 gc_goals_proposed, ep_goals_proposed,
                                 gc_proposed_goals_state, ep_proposed_goals_state, key):
            """Collect exactly unroll_length steps of experience.
            
            Tracks which phase we're in (GC or EP) and how many steps collected in each phase.
            """
            num_envs = config.num_envs
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            
            # Determine which phase we're in
            in_gc_phase = gc_steps_collected < num_goal_conditioned_steps
            steps_to_collect = unroll_length
            
            key, gc_key, ep_key = jax.random.split(key, 3)
            
            def collect_gc_phase():
                # Propose GC goals if not already proposed
                def propose_gc_goals_fn():
                    proposed, updated_gcp, updated_ep = gcp_propose_goals(
                        gcp_buffer_state, ep_buffer_state, train_env, env_state, gc_key, training_state
                    )
                    return proposed, updated_gcp, updated_ep
                
                def use_existing_gc_goals():
                    return gc_proposed_goals_state, gcp_buffer_state, ep_buffer_state
                
                gc_proposed_goals_new, gcp_buffer_state_new, ep_buffer_state_new = jax.lax.cond(
                    ~gc_goals_proposed,
                    propose_gc_goals_fn,
                    use_existing_gc_goals
                )
                gc_proposed_goals = jnp.where(~gc_goals_proposed, gc_proposed_goals_new, gc_proposed_goals_state)
                
                # Update env_state with GC goals
                new_info = dict(env_state.info)
                new_info["gc_proposed_goals"] = gc_proposed_goals
                new_info["ep_proposed_goals"] = env_state.info.get("ep_proposed_goals", jnp.zeros_like(gc_proposed_goals))
                
                def update_env_with_goals():
                    return env_state.replace(
                        obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(gc_proposed_goals),
                        info=new_info
                    )
                
                def keep_env_state():
                    return env_state.replace(info=new_info)
                
                env_state_updated = jax.lax.cond(
                    ~gc_goals_proposed,
                    update_env_with_goals,
                    keep_env_state
                )

                def gc_rollout_step(carry, unused_t):
                    env_state_inner, gcp_actor_state_inner, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    
                    gc_proposed_goals_inner = env_state_inner.info.get("gc_proposed_goals", env_state_inner.obs[:, -len(train_env.goal_indices):])
                    
                    nstate, transition = deterministic_actor_step_with_proposals(
                        gcp_actor_state_inner, train_env, env_state_inner, gc_proposed_goals_inner, ("truncation", "traj_id")
                    )
                    
                    truncation = transition.extras["state_extras"]["truncation"]
                    terminated = (transition.discount < 1.0) | (truncation > 0.5)
                    
                    new_info = dict(nstate.info)
                    new_info["gc_proposed_goals"] = gc_proposed_goals_inner
                    new_info["ep_proposed_goals"] = env_state_inner.info.get("ep_proposed_goals", jnp.zeros_like(gc_proposed_goals_inner))
                    env_state_inner = nstate.replace(info=new_info)
                    
                    existing_state_extras = transition.extras["state_extras"]
                    transition_extras = {
                        **existing_state_extras,
                        "in_gc_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                        "in_ep_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                        "gc_proposed_goals": gc_proposed_goals_inner,
                        "ep_proposed_goals": jnp.zeros_like(gc_proposed_goals_inner),
                        "terminated": terminated.astype(jnp.float32),
                    }
                    transition = transition._replace(extras={"state_extras": transition_extras})
                    
                    return (env_state_inner, gcp_actor_state_inner, next_key), transition
                
                (env_state_final, _, _), gc_transitions = jax.lax.scan(
                    gc_rollout_step,
                    (env_state_updated, training_state.gcp_actor_state, key),
                    (),
                    length=steps_to_collect
                )
                
                combined_transitions = gc_transitions
                main_buffer_state_new = main_replay_buffer.insert(main_buffer_state, combined_transitions)
                gcp_buffer_state_final = gcp_replay_buffer.insert(gcp_buffer_state_new, gc_transitions)
                
                new_gc_steps_collected = gc_steps_collected + steps_to_collect
                new_ep_steps_collected = ep_steps_collected
                new_gc_goals_proposed = True
                new_ep_goals_proposed = ep_goals_proposed
                
                return (env_state_final, main_buffer_state_new, gcp_buffer_state_final, ep_buffer_state_new,
                       new_gc_steps_collected, new_ep_steps_collected,
                       new_gc_goals_proposed, new_ep_goals_proposed,
                       gc_proposed_goals, ep_proposed_goals_state, combined_transitions)
            
            def collect_ep_phase():
                # SAC EP: No goal proposals needed
                ep_proposed_goals = jnp.zeros((num_envs, len(train_env.goal_indices)))
                
                new_info = dict(env_state.info)
                new_info["ep_proposed_goals"] = ep_proposed_goals
                new_info["gc_proposed_goals"] = env_state.info.get("gc_proposed_goals", jnp.zeros_like(ep_proposed_goals))
                env_state_updated = env_state.replace(info=new_info)

                def ep_rollout_step(carry, unused_t):
                    env_state_inner, training_state_inner, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    
                    nstate, transition = ep_actor_step(
                        training_state_inner, train_env, env_state_inner, current_key, ("truncation", "traj_id")
                    )
                    
                    truncation = transition.extras["state_extras"]["truncation"]
                    terminated = (transition.discount < 1.0) | (truncation > 0.5)
                    
                    new_info = dict(nstate.info)
                    new_info["ep_proposed_goals"] = env_state_inner.info.get("ep_proposed_goals", jnp.zeros((num_envs, len(train_env.goal_indices))))
                    new_info["gc_proposed_goals"] = env_state_inner.info.get("gc_proposed_goals", jnp.zeros((num_envs, len(train_env.goal_indices))))
                    env_state_inner = nstate.replace(info=new_info)
                    
                    existing_state_extras = transition.extras["state_extras"]
                    gc_proposed_goals_for_transition = env_state_inner.info.get("gc_proposed_goals", jnp.zeros((num_envs, len(train_env.goal_indices))))
                    transition_extras = {
                        **existing_state_extras,
                        "in_gc_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                        "in_ep_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                        "gc_proposed_goals": gc_proposed_goals_for_transition,
                        "ep_proposed_goals": ep_proposed_goals,
                        "terminated": terminated.astype(jnp.float32),
                    }
                    transition = transition._replace(extras={"state_extras": transition_extras})
                    
                    return (env_state_inner, training_state_inner, next_key), transition
                
                (env_state_final, _, _), ep_transitions = jax.lax.scan(
                    ep_rollout_step,
                    (env_state_updated, training_state, key),
                    (),
                    length=steps_to_collect
                )
                
                # Compute EP reward (replaces environment reward entirely)
                # Immediate max-critic rewards: r_t = max_g f(s'_t, a_g, g)
                # ep_transitions.observation shape: (steps_to_collect, num_envs, obs_size)
                # Next state for t = observation at t+1 (for t < steps-1), or env_state_final for last
                next_states = jnp.concatenate([
                    ep_transitions.observation[1:, :, :state_size],
                    env_state_final.obs[:, :state_size][None, :, :]
                ], axis=0)  # (steps_to_collect, num_envs, state_size)
                
                # Flatten and compute immediate rewards
                next_states_flat = next_states.reshape(-1, state_size)
                num_transitions = steps_to_collect * num_envs
                reward_keys = jax.random.split(ep_key, num_transitions)
                
                def compute_reward_for_transition(next_state, rk):
                    return compute_max_critic_reward_per_transition(
                        next_state, env_goals_for_reward,
                        gcp_actor, training_state.gcp_actor_state.params,
                        training_state.gcp_critic_state.params,
                        gcp_sa_encoder, gcp_g_encoder, self.energy_fn, rk
                    )
                
                ep_rewards_flat = jax.vmap(compute_reward_for_transition)(next_states_flat, reward_keys)
                ep_rewards = ep_rewards_flat.reshape(steps_to_collect, num_envs)
                ep_transitions = ep_transitions._replace(reward=ep_rewards)
                
                combined_transitions = ep_transitions
                main_buffer_state_new = main_replay_buffer.insert(main_buffer_state, combined_transitions)
                ep_buffer_state_final = ep_replay_buffer.insert(ep_buffer_state, ep_transitions)
                
                new_gc_steps_collected = gc_steps_collected
                new_ep_steps_collected = ep_steps_collected + steps_to_collect
                new_gc_goals_proposed = gc_goals_proposed
                new_ep_goals_proposed = True  # No goals needed for SAC EP
                
                return (env_state_final, main_buffer_state_new, gcp_buffer_state, ep_buffer_state_final,
                       new_gc_steps_collected, new_ep_steps_collected,
                       new_gc_goals_proposed, new_ep_goals_proposed,
                       gc_proposed_goals_state, ep_proposed_goals, combined_transitions)
            
            return jax.lax.cond(
                in_gc_phase,
                collect_gc_phase,
                collect_ep_phase
            )

        # ===== Prefill Replay Buffer =====
        def prefill_replay_buffer(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            # Initialize env_state.info for consistent PyTree structure
            initial_info = dict(env_state.info)
            initial_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            initial_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            env_state = env_state.replace(info=initial_info)
            
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key = carry
                key, reset_key, new_key = jax.random.split(key, 3)
                
                num_goal_conditioned_steps = config.num_goal_conditioned_steps
                num_exploratory_steps = config.num_exploratory_steps
                num_gc_chunks = num_goal_conditioned_steps // unroll_length
                num_ep_chunks = num_exploratory_steps // unroll_length
                total_chunks = num_gc_chunks + num_ep_chunks
                
                experience_keys = jax.random.split(key, total_chunks)
                
                gc_steps_collected = 0
                ep_steps_collected = 0
                gc_goals_proposed = False
                ep_goals_proposed = False
                gc_proposed_goals_state = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                ep_proposed_goals_state = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                
                def collect_chunk(carry_inner, chunk_idx):
                    (env_state_inner, main_buffer_state_inner, gcp_buffer_state_inner, ep_buffer_state_inner,
                     gc_steps_collected_inner, ep_steps_collected_inner,
                     gc_goals_proposed_inner, ep_goals_proposed_inner,
                     gc_proposed_goals_inner, ep_proposed_goals_inner) = carry_inner
                    
                    (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new,
                     gc_steps_collected_new, ep_steps_collected_new,
                     gc_goals_proposed_new, ep_goals_proposed_new,
                     gc_proposed_goals_new, ep_proposed_goals_new, _) = get_experience_chunk(
                        training_state,
                        env_state_inner,
                        main_buffer_state_inner,
                        gcp_buffer_state_inner,
                        ep_buffer_state_inner,
                        gc_steps_collected_inner,
                        ep_steps_collected_inner,
                        gc_goals_proposed_inner,
                        ep_goals_proposed_inner,
                        gc_proposed_goals_inner,
                        ep_proposed_goals_inner,
                        experience_keys[chunk_idx]
                    )
                    
                    carry_new = (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new,
                                gc_steps_collected_new, ep_steps_collected_new,
                                gc_goals_proposed_new, ep_goals_proposed_new,
                                gc_proposed_goals_new, ep_proposed_goals_new)
                    return carry_new, ()
                
                initial_carry = (env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state,
                                gc_steps_collected, ep_steps_collected,
                                gc_goals_proposed, ep_goals_proposed,
                                gc_proposed_goals_state, ep_proposed_goals_state)
                
                (env_state_final, main_buffer_state_final, gcp_buffer_state_final, ep_buffer_state_final,
                 _, _, _, _, _, _), _ = jax.lax.scan(
                    collect_chunk,
                    initial_carry,
                    jnp.arange(total_chunks)
                )
                
                # Reset environments after collecting experience
                reset_keys = jax.random.split(reset_key, config.num_envs)
                env_state_final = jax.jit(train_env.reset)(reset_keys)
                TRAJ_ID_MULTIPLIER = 10000000
                env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
                final_info = dict(env_state_final.info)
                final_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
                final_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                final_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                env_state_final = env_state_final.replace(info=final_info)
                
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + total_steps_per_training_step * config.num_envs,
                )
                return (training_state, env_state_final, main_buffer_state_final, gcp_buffer_state_final, ep_buffer_state_final, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ===== Network Updates =====
        @jax.jit
        def update_networks(carry, gcp_transitions, ep_sac_transitions):
            """Update GCP (CRL) and EP (SAC) networks.
            
            Args:
                carry: (training_state, key)
                gcp_transitions: Transition from flatten_batch (CRL format)
                ep_sac_transitions: SACTransition with (obs, next_obs, action, reward, discount)
            """
            training_state, key = carry
            
            key, gcp_critic_key, gcp_actor_key, ep_alpha_key, ep_critic_key, ep_actor_key = jax.random.split(key, 6)

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

            gcp_networks = dict(
                actor=gcp_actor,
                sa_encoder=gcp_sa_encoder,
                g_encoder=gcp_g_encoder,
            )

            # Update GCP networks (CRL)
            new_gcp_actor_state, new_gcp_alpha_state, gcp_actor_metrics = update_actor_and_alpha(
                context, gcp_networks, gcp_transitions, 
                training_state.gcp_actor_state, training_state.gcp_critic_state, training_state.gcp_alpha_state,
                gcp_actor_key
            )
            new_gcp_critic_state, gcp_critic_metrics = update_critic(
                context, gcp_networks, gcp_transitions, training_state.gcp_critic_state, gcp_critic_key
            )
            
            # Update EP networks (SAC)
            # ep_sac_transitions already has observation/next_observation as state-only
            ep_alpha = jnp.exp(training_state.ep_alpha_params)
            
            ep_alpha_loss, ep_alpha_params, ep_alpha_optimizer_state = ep_alpha_update(
                training_state.ep_alpha_params,
                training_state.ep_actor_state.params,
                training_state.ep_normalizer_params,
                ep_sac_transitions,
                ep_alpha_key,
                optimizer_state=training_state.ep_alpha_optimizer_state,
            )
            ep_alpha = jnp.exp(ep_alpha_params)
            
            ep_critic_loss, new_ep_q_params, ep_q_optimizer_state = ep_critic_update(
                training_state.ep_q_params,
                training_state.ep_actor_state.params,
                training_state.ep_normalizer_params,
                training_state.ep_target_q_params,
                ep_alpha,
                ep_sac_transitions,
                ep_critic_key,
                optimizer_state=training_state.ep_q_optimizer_state,
            )
            
            ep_actor_loss, ep_actor_params, ep_actor_optimizer_state = ep_actor_update(
                training_state.ep_actor_state.params,
                training_state.ep_normalizer_params,
                training_state.ep_q_params,
                ep_alpha,
                ep_sac_transitions,
                ep_actor_key,
                optimizer_state=training_state.ep_actor_optimizer_state,
            )
            
            # Soft target update for EP Q-network
            new_ep_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.ep_target_q_params,
                new_ep_q_params,
            )
            
            # Update EP normalizer
            new_ep_normalizer_params = running_statistics.update(
                training_state.ep_normalizer_params,
                ep_sac_transitions.observation,
            )
            
            new_ep_actor_state = training_state.ep_actor_state.replace(params=ep_actor_params)
            
            training_state = training_state.replace(
                gcp_actor_state=new_gcp_actor_state,
                gcp_critic_state=new_gcp_critic_state,
                gcp_alpha_state=new_gcp_alpha_state,
                ep_actor_state=new_ep_actor_state,
                ep_q_params=new_ep_q_params,
                ep_target_q_params=new_ep_target_q_params,
                ep_q_optimizer_state=ep_q_optimizer_state,
                ep_actor_optimizer_state=ep_actor_optimizer_state,
                ep_alpha_params=ep_alpha_params,
                ep_alpha_optimizer_state=ep_alpha_optimizer_state,
                ep_normalizer_params=new_ep_normalizer_params,
                gradient_steps=training_state.gradient_steps + 1,
            )

            metrics = {
                "gcp_entropy": gcp_actor_metrics["entropy"],
                "gcp_actor_loss": gcp_actor_metrics["actor_loss"],
                "gcp_alpha_loss": gcp_actor_metrics["alpha_loss"],
                "gcp_log_alpha": gcp_actor_metrics["log_alpha"],
                "gcp_categorical_accuracy": gcp_critic_metrics["categorical_accuracy"],
                "gcp_logits_pos": gcp_critic_metrics["logits_pos"],
                "gcp_logits_neg": gcp_critic_metrics["logits_neg"],
                "gcp_logsumexp": gcp_critic_metrics["logsumexp"],
                "gcp_critic_loss": gcp_critic_metrics["critic_loss"],
                "ep_actor_loss": ep_actor_loss,
                "ep_critic_loss": ep_critic_loss,
                "ep_alpha_loss": ep_alpha_loss,
                "ep_alpha": ep_alpha,
            }

            return (
                training_state,
                key,
            ), metrics

        # ===== Training Step (scan over chunks, matching go_explore_crl) =====
        @jax.jit
        def training_step(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            reset_key, experience_keys, sampling_keys, permute_keys, training_keys = jax.random.split(key, 5)
            
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps
            num_gc_chunks = num_goal_conditioned_steps // unroll_length
            num_ep_chunks = num_exploratory_steps // unroll_length
            total_chunks = num_gc_chunks + num_ep_chunks
            
            experience_keys = jax.random.split(experience_keys, total_chunks)
            sampling_keys = jax.random.split(sampling_keys, total_chunks)
            permute_keys = jax.random.split(permute_keys, total_chunks)
            training_keys = jax.random.split(training_keys, total_chunks)
            
            gc_steps_collected = 0
            ep_steps_collected = 0
            gc_goals_proposed = False
            ep_goals_proposed = False
            gc_proposed_goals_state = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            ep_proposed_goals_state = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            
            def collect_and_update_chunk(carry, chunk_idx):
                (training_state_inner, env_state_inner, main_buffer_state_inner, 
                 gcp_buffer_state_inner, ep_buffer_state_inner,
                 gc_steps_collected_inner, ep_steps_collected_inner,
                 gc_goals_proposed_inner, ep_goals_proposed_inner,
                 gc_proposed_goals_inner, ep_proposed_goals_inner) = carry
                
                # Collect experience chunk
                (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new,
                 gc_steps_collected_new, ep_steps_collected_new,
                 gc_goals_proposed_new, ep_goals_proposed_new,
                 gc_proposed_goals_new, ep_proposed_goals_new, collected_transitions) = get_experience_chunk(
                    training_state_inner,
                    env_state_inner,
                    main_buffer_state_inner,
                    gcp_buffer_state_inner,
                    ep_buffer_state_inner,
                    gc_steps_collected_inner,
                    ep_steps_collected_inner,
                    gc_goals_proposed_inner,
                    ep_goals_proposed_inner,
                    gc_proposed_goals_inner,
                    ep_proposed_goals_inner,
                    experience_keys[chunk_idx]
                )
                
                # Update env_steps
                training_state_inner = training_state_inner.replace(
                    env_steps=training_state_inner.env_steps + env_steps_per_actor_step,
                )
                
                # Sample transitions for training
                # GCP: sample from GCP buffer (only GC transitions with real goals)
                gcp_buffer_state_sampled, gcp_transitions = gcp_replay_buffer.sample(gcp_buffer_state_new)
                gcp_buffer_state_new = gcp_buffer_state_sampled
                
                # EP: sample from EP buffer
                ep_buffer_state_sampled, ep_transitions_raw = ep_replay_buffer.sample(ep_buffer_state_new)
                ep_buffer_state_new = ep_buffer_state_sampled
                
                # Process GCP transitions (CRL format via flatten_batch)
                gcp_batch_keys = jax.random.split(sampling_keys[chunk_idx], gcp_transitions.observation.shape[0])
                
                def process_gcp_transitions(transitions, batch_keys_inner, permute_key_inner):
                    transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                        (self.discounting, state_size, tuple(train_env.goal_indices)),
                        transitions,
                        batch_keys_inner,
                    )
                    transitions = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                    )
                    permutation = jax.random.permutation(permute_key_inner, len(transitions.observation))
                    transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                    transitions = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                        transitions,
                    )
                    return transitions
                
                gcp_last_batch = process_gcp_transitions(gcp_transitions, gcp_batch_keys, permute_keys[chunk_idx])
                
                # Process EP transitions for SAC (extract obs, next_obs, action, reward, discount)
                def process_ep_sac_transitions(transitions, permute_key_inner):
                    """Convert trajectory-based transitions to SAC format with next_observation."""
                    # transitions shape: (sample_batch_size, episode_length, ...)
                    obs = transitions.observation[:, :-1, :state_size]       # (N, L-1, state_size)
                    next_obs = transitions.observation[:, 1:, :state_size]   # (N, L-1, state_size)
                    actions = transitions.action[:, :-1]                      # (N, L-1, action_size)
                    rewards = transitions.reward[:, :-1]                      # (N, L-1)
                    discounts = transitions.discount[:, :-1]                  # (N, L-1)
                    truncations = transitions.extras['state_extras']['truncation'][:, :-1]  # (N, L-1)
                    
                    # Flatten first two dims
                    flat_obs = obs.reshape(-1, state_size)
                    flat_next_obs = next_obs.reshape(-1, state_size)
                    flat_actions = actions.reshape(-1, action_size)
                    flat_rewards = rewards.reshape(-1)
                    flat_discounts = discounts.reshape(-1)
                    flat_truncations = truncations.reshape(-1)
                    
                    # Permute
                    n = flat_obs.shape[0]
                    perm = jax.random.permutation(permute_key_inner, n)
                    flat_obs = flat_obs[perm]
                    flat_next_obs = flat_next_obs[perm]
                    flat_actions = flat_actions[perm]
                    flat_rewards = flat_rewards[perm]
                    flat_discounts = flat_discounts[perm]
                    flat_truncations = flat_truncations[perm]
                    
                    # Reshape into batches (same num_batches as GCP)
                    num_complete = (n // self.batch_size) * self.batch_size
                    num_batches = num_complete // self.batch_size
                    
                    sac_transitions = SACTransition(
                        observation=flat_obs[:num_complete].reshape(num_batches, self.batch_size, state_size),
                        next_observation=flat_next_obs[:num_complete].reshape(num_batches, self.batch_size, state_size),
                        action=flat_actions[:num_complete].reshape(num_batches, self.batch_size, action_size),
                        reward=flat_rewards[:num_complete].reshape(num_batches, self.batch_size),
                        discount=flat_discounts[:num_complete].reshape(num_batches, self.batch_size),
                        extras={
                            'state_extras': {
                                'truncation': flat_truncations[:num_complete].reshape(num_batches, self.batch_size),
                            },
                            'policy_extras': {},
                        },
                    )
                    return sac_transitions
                
                ep_last_batch = process_ep_sac_transitions(ep_transitions_raw, permute_keys[chunk_idx])
                
                # Update networks (scan over batches)
                ((training_state_updated, _), metrics) = jax.lax.scan(
                    lambda carry, xs: update_networks(carry, xs[0], xs[1]),
                    (training_state_inner, training_keys[chunk_idx]),
                    (gcp_last_batch, ep_last_batch)
                )
                
                carry_new = (training_state_updated, env_state_new, main_buffer_state_new,
                            gcp_buffer_state_new, ep_buffer_state_new,
                            gc_steps_collected_new, ep_steps_collected_new,
                            gc_goals_proposed_new, ep_goals_proposed_new,
                            gc_proposed_goals_new, ep_proposed_goals_new)
                
                return carry_new, (metrics, collected_transitions)
            
            # Run collection and updates for all chunks
            initial_carry = (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state,
                            gc_steps_collected, ep_steps_collected,
                            gc_goals_proposed, ep_goals_proposed,
                            gc_proposed_goals_state, ep_proposed_goals_state)
            
            (training_state_final, env_state_final, main_buffer_state_final,
             gcp_buffer_state_final, ep_buffer_state_final, _, _, _, _, _, _), (all_metrics, all_collected) = jax.lax.scan(
                collect_and_update_chunk,
                initial_carry,
                jnp.arange(total_chunks)
            )
            
            # Reset environments after collecting all experience
            reset_keys = jax.random.split(reset_key, config.num_envs)
            env_state_final = jax.jit(train_env.reset)(reset_keys)
            TRAJ_ID_MULTIPLIER = 100000
            env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
            final_info = dict(env_state_final.info)
            final_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
            final_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            final_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            env_state_final = env_state_final.replace(info=final_info)
            
            # Combine collected transitions for visualization
            # all_collected shape: (total_chunks, unroll_length, num_envs, ...)
            # Reshape to (total_chunks * unroll_length, num_envs, ...)
            collected_transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                all_collected
            )
            
            # Average metrics across all chunks
            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), all_metrics)
            
            return (
                training_state_final,
                env_state_final,
                main_buffer_state_final,
                gcp_buffer_state_final,
                ep_buffer_state_final,
                collected_transitions
            ), metrics

        # ===== Training Epoch =====
        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            main_buffer_state,
            gcp_buffer_state,
            ep_buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, mbs, gcbs, ebs, k, _ = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        mbs,
                        gcbs,
                        ebs,
                        collected_transitions
                    ),
                    metrics,
                ) = training_step(ts, es, mbs, gcbs, ebs, train_key)
                return (ts, es, mbs, gcbs, ebs, k, collected_transitions), metrics

            # Run one step to get initial structures for carry
            key, first_key = jax.random.split(key)
            ((training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, init_collected), first_metrics) = training_step(
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, first_key
            )

            (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _, collected_transitions), rest_metrics = jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key, init_collected),
                (),
                length=num_training_steps_per_epoch - 1,
            )

            # Combine metrics from first step with rest
            metrics = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a[None], b]),
                first_metrics,
                rest_metrics,
            )
            return training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, collected_transitions
        
        # ===== Visualization =====
        def visualize_goals(train_env, transitions, wandb_key):
            """Visualize trajectories and goals. EP has no goals, so only GC goals are shown."""
            obs = np.array(transitions.observation)
            last_traj_state = np.array(transitions.extras["last_traj_state"][:, :, :state_size])
            intermediate_traj = np.array(transitions.extras["intermediate_traj"])
            
            obs_flat = obs.reshape(-1, obs.shape[-1])
            
            in_gc_phase = np.array(transitions.extras["state_extras"]["in_gc_phase"])
            in_ep_phase = np.array(transitions.extras["state_extras"]["in_ep_phase"])
            gc_proposed_goals = np.array(transitions.extras["state_extras"]["gc_proposed_goals"])
            traj_ids = np.array(transitions.extras["state_extras"]["traj_id"])
            
            in_gc_phase_flat = in_gc_phase.reshape(-1)
            in_ep_phase_flat = in_ep_phase.reshape(-1)
            gc_proposed_goals_flat = gc_proposed_goals.reshape(-1, len(train_env.goal_indices))
            traj_ids_flat = traj_ids.reshape(-1)
            
            intermediate_traj_flat = intermediate_traj.reshape(-1, intermediate_traj.shape[-2], intermediate_traj.shape[-1])
            
            gc_final_states = []
            ep_final_states = []
            start_states = []
            gc_goals = []
            gc_intermediate_states_list = []
            ep_intermediate_states_list = []
            
            unique_traj_ids = np.unique(traj_ids_flat)
            logging.info(f"Visualization: Found {len(unique_traj_ids)} unique trajectory IDs out of {len(traj_ids_flat)} total transitions")
            
            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_flat == traj_id
                traj_indices = np.sort(np.where(traj_mask)[0])
                
                start_state = obs_flat[traj_indices[0]][train_env.goal_indices]
                start_states.append(start_state)
                
                gc_mask = in_gc_phase_flat[traj_indices] > 0.5
                gc_indices = traj_indices[gc_mask]
                if len(gc_indices) > 0:
                    gc_final_idx = gc_indices[-1]
                    gc_final_state = obs_flat[gc_final_idx][train_env.goal_indices]
                    gc_goal = gc_proposed_goals_flat[gc_final_idx]
                    gc_final_states.append(gc_final_state)
                    gc_goals.append(gc_goal)
                
                ep_mask = in_ep_phase_flat[traj_indices] > 0.5
                ep_indices = traj_indices[ep_mask]
                if len(ep_indices) > 0:
                    ep_final_idx = ep_indices[-1]
                    ep_final_state = obs_flat[ep_final_idx][train_env.goal_indices]
                    ep_final_states.append(ep_final_state)
                
                # GC intermediate states
                gc_intermediate = np.array([]).reshape(0, len(train_env.goal_indices))
                if len(gc_indices) > 0:
                    gc_states = obs_flat[gc_indices][:, train_env.goal_indices]
                    num_gc_steps = len(gc_states)
                    if num_gc_steps > 0:
                        indices = np.linspace(0, num_gc_steps - 1, 6).astype(int)
                        gc_intermediate = gc_states[indices]
                
                # EP intermediate states
                ep_intermediate = np.array([]).reshape(0, len(train_env.goal_indices))
                if len(ep_indices) > 0:
                    ep_states = obs_flat[ep_indices][:, train_env.goal_indices]
                    num_ep_steps = len(ep_states)
                    if num_ep_steps > 0:
                        indices = np.linspace(0, num_ep_steps - 1, 6).astype(int)
                        ep_intermediate = ep_states[indices]
                
                gc_intermediate_states_list.append(gc_intermediate)
                ep_intermediate_states_list.append(ep_intermediate)
            
            start_states = np.array(start_states)
            gc_final_states = np.array(gc_final_states)
            ep_final_states = np.array(ep_final_states)
            gc_goals = np.array(gc_goals)
            
            goal_dim = len(train_env.goal_indices)
            if start_states.ndim == 1:
                start_states = start_states.reshape(-1, goal_dim)
            if gc_final_states.ndim == 1:
                gc_final_states = gc_final_states.reshape(-1, goal_dim)
            if ep_final_states.ndim == 1:
                ep_final_states = ep_final_states.reshape(-1, goal_dim)
            if gc_goals.ndim == 1:
                gc_goals = gc_goals.reshape(-1, goal_dim)
            
            num_trajs = start_states.shape[0]
            num_viz_trajs = min(4, num_trajs)
            if num_viz_trajs == 0:
                return
            sample_indices = np.random.choice(num_trajs, num_viz_trajs, replace=False)
            
            start_xy = start_states[sample_indices]
            gc_final_xy = gc_final_states[sample_indices] if len(gc_final_states) > 0 else start_xy
            ep_final_xy = ep_final_states[sample_indices] if len(ep_final_states) > 0 else start_xy
            gc_proposed_goals_xy = gc_goals[sample_indices] if len(gc_goals) > 0 else start_xy
            
            gc_intermediate_xy_list = [gc_intermediate_states_list[i] for i in sample_indices]
            ep_intermediate_xy_list = [ep_intermediate_states_list[i] for i in sample_indices]
            
            # EP has no goals - use zeros as placeholder
            ep_proposed_goals_xy_placeholder = np.zeros_like(gc_proposed_goals_xy)
            visualize_dual_crl_trajectories_2d(
                start_xy, gc_final_xy, ep_final_xy, gc_proposed_goals_xy, ep_proposed_goals_xy_placeholder,
                gc_intermediate_xy_list, ep_intermediate_xy_list, f"{wandb_key}/dual_crl_trajectories",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # Scatter plots
            if len(gc_final_states) > 0:
                visualize_scatter_sample(
                    gc_final_states, "GC Final States", f"{wandb_key}/gc_final_states_scatter",
                    train_env.x_bounds, train_env.y_bounds
                )
            
            if len(ep_final_states) > 0:
                visualize_scatter_sample(
                    ep_final_states, "EP Final States", f"{wandb_key}/ep_final_states_scatter",
                    train_env.x_bounds, train_env.y_bounds
                )
            
            if len(gc_goals) > 0:
                visualize_scatter_sample(
                    gc_goals, "GC Goal Proposals", f"{wandb_key}/gc_goal_proposals_scatter",
                    train_env.x_bounds, train_env.y_bounds
                )
            
            logging.info(f"Plotted visualizations at env step {training_state.env_steps.item()}")

        # ===== Run Prefill =====
        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, prefill_key
        )

        # ===== Evaluator (GCP only) =====
        key, eval_gcp_key = jax.random.split(key, 2)
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_gcp_key,
        )

        # ===== Main Training Loop =====
        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, collected_transitions = training_epoch(
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, epoch_key
            )

            # Process collected_transitions for visualization
            @jax.jit
            def process_for_viz(transitions, batch_keys):
                processed = jax.vmap(flatten_batch, in_axes=(None, 1, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                processed = jax.tree_util.tree_map(
                    lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, len(x.shape)))),
                    processed
                )
                return processed
            
            viz_key = jax.random.PRNGKey(0)
            num_envs = collected_transitions.observation.shape[1]
            viz_batch_keys = jax.random.split(viz_key, num_envs)
            processed_transitions = process_for_viz(collected_transitions, viz_batch_keys)
            visualize_goals(train_env, processed_transitions, wandb_key="training")

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_training_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            # Run GCP evaluation only
            metrics = evaluator.run_evaluation(training_state, metrics)

            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: gcp_actor_state.apply_fn(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.gcp_actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                params = (
                    training_state.gcp_alpha_state.params,
                    training_state.gcp_actor_state.params,
                    training_state.gcp_critic_state.params,
                    training_state.ep_alpha_params,
                    training_state.ep_actor_state.params,
                    training_state.ep_q_params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
            else:
                params = None

        total_steps = current_step
        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
