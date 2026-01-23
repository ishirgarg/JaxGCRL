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
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.visualize import visualize_goals_2d, visualize_kde_heatmap, visualize_q_function_2d, visualize_dual_crl_trajectories_2d

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder
from .proposers import (
    FinalReplayBufferProposer, 
    RandomEnvironmentGoalProposer,
    UCGRProposer,
    MaxWaypointRatioOneEnvProposer,
    QEpistemicProposer,
)
from .goals_utils import compute_min_critic_mean_reward
from brax.training.agents.sac import networks as sac_networks

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
    # Same as go_explore_crl - for GCP transitions
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

    future_states = last_state_for_each_step(transition.observation, transition.extras["state_extras"]["traj_id"])

    # Preserve all state_extras fields
    state_extras = transition.extras["state_extras"]
    sliced_state_extras = jax.tree_util.tree_map(
        lambda x: x[:-1] if x.ndim > 0 and x.shape[0] == seq_len else x,
        state_extras
    )
    # Explicitly squeeze traj_id and truncation
    sliced_state_extras["traj_id"] = jnp.squeeze(sliced_state_extras["traj_id"], axis=-1) if sliced_state_extras["traj_id"].ndim > 1 else sliced_state_extras["traj_id"]
    sliced_state_extras["truncation"] = jnp.squeeze(sliced_state_extras["truncation"], axis=-1) if sliced_state_extras["truncation"].ndim > 1 else sliced_state_extras["truncation"]

    return Transition(
        observation=transition.observation[:-1],
        action=transition.action[:-1],
        reward=transition.reward[:-1],
        discount=discount[:-1, 1:].sum(axis=1),
        extras={
            "future_state": future_states[:-1],
            "state_extras": sliced_state_extras,
        },
    )


def compute_ep_reward(reward_name: str, transitions: Transition, env) -> jnp.ndarray:
    """Compute rewards for EP transitions based on reward_name.
    
    Args:
        reward_name: Name of the reward function to use
        transitions: EP transitions (non-goal-conditioned)
        env: Environment (for accessing state_dim, goal_indices, etc.)
        
    Returns:
        rewards: (batch_size,) array of computed rewards
    """
    batch_size = transitions.observation.shape[0]
    
    if reward_name == "mean_max_critic":
        # mean_max_critic reward needs the full trajectory to compute,
        # so return zeros here. It will be computed after the rollout is complete.
        return jnp.zeros((batch_size,))
    elif reward_name == "placeholder":
        # Placeholder: return zero rewards
        return jnp.zeros((batch_size,))
    else:
        raise ValueError(f"Unknown reward_name: {reward_name}")


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

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Goal proposer names for GCP (EP doesn't use goal proposal)
    gcp_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals", "ucgr", "maxwaypointratio_one_env", "q_epistemic"] = "gcp_final_rb"
    goal_sampling_temperature: float = 1.0
    
    # Critic ensemble for Q-epistemic goal proposal (GCP only)
    use_gcp_critic_ensemble: bool = False
    gcp_num_critic_ensemble: int = 5
    
    # EP reward computation
    ep_reward_name: str = "placeholder"  # Will be used to compute rewards from trajectories

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
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

        unroll_length = config.num_goal_conditioned_steps + config.num_exploratory_steps
        env_steps_per_actor_step = config.num_envs * unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
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
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

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

        # Network setup
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
        ep_sac_network = sac_networks.make_sac_networks(
            observation_size=ep_obs_size,
            action_size=action_size,
            preprocess_observations_fn=lambda x: x,
            layer_norm=self.use_ln,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
        )
        
        # Initialize EP networks
        dummy_ep_obs = jnp.zeros((ep_obs_size,))
        dummy_ep_action = jnp.zeros((action_size,))
        ep_q_params = ep_sac_network.q_network.init(ep_q_key, dummy_ep_obs, dummy_ep_action)
        ep_target_q_params = jax.tree_util.tree_map(lambda x: x, ep_q_params)
        
        # EP actor params (from SAC network)
        ep_actor_params = ep_sac_network.policy_network.init(ep_actor_key, dummy_ep_obs)
        ep_actor_state = TrainState.create(
            apply_fn=ep_sac_network.policy_network.apply,
            params=ep_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )

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
        
        # Initialize GCP critic params
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
        ep_alpha_optimizer_state = ep_alpha_optimizer.init(ep_alpha_params)
        
        # EP optimizer states
        ep_q_optimizer_state = ep_q_optimizer.init(ep_q_params)
        ep_actor_optimizer_state = ep_policy_optimizer.init(ep_actor_params)
        
        # EP normalizer (SAC)
        ep_normalizer_params = running_statistics.init_state(
            specs.Array((ep_obs_size,), jnp.float32)
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

        # Replay Buffer
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
        
        # EP transition (no goals, but pad with zeros to match obs_size)
        dummy_ep_obs = jnp.concatenate([jnp.zeros((ep_obs_size,)), jnp.zeros((goal_size,))], axis=0)  # (obs_size,)
        dummy_ep_transition = Transition(
            observation=dummy_ep_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    "in_ep_phase": 1.0,
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
                dummy_data_sample=dummy_ep_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        ep_buffer_state = jax.jit(ep_replay_buffer.init)(buffer_key)

        # Initialize GCP goal proposers
        from jaxgcrl.agents.crl.proposers import (
            FinalReplayBufferProposer, 
            RandomEnvironmentGoalProposer,
            UCGRProposer,
            MaxWaypointRatioOneEnvProposer,
            QEpistemicProposer,
        )
        
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
        
        # GCP goal proposer function
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

        # Actor step functions
        def deterministic_actor_step_with_proposals(actor_state, env, env_state, proposed_goals, extra_fields):
            """Deterministic actor step for GCP with proposed goals."""
            # Overwrite goals in observation
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)

            means, _ = gcp_actor.apply(actor_state.params, env_state.obs)
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
        
        def ep_actor_step(actor_state, env, env_state, key, extra_fields):
            """EP actor step (non-goal-conditioned, uses SAC policy)."""
            # Extract state only (no goals) for policy input
            state_only = env_state.obs[:, :state_size]
            
            # Sample action from SAC policy
            policy_dist = ep_sac_network.policy_network.apply(actor_state.params, state_only)
            actions = policy_dist.sample(seed=key)
            
            # Step environment (uses full obs with goals, but we'll store state-only)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            
            # Get next state (state-only)
            next_state_only = nstate.obs[:, :state_size]
            
            # Compute reward based on reward_name
            # Create a transition-like structure for reward computation
            # Note: reward computation will use the full trajectory later
            # For now, use environment reward as placeholder
            computed_reward = compute_ep_reward(
                self.ep_reward_name,
                Transition(
                    observation=state_only,
                    action=actions,
                    reward=nstate.reward,  # Placeholder
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras, "next_observation": next_state_only}
                ),
                unwrapped_env
            )
            
            # Pad observation with zeros to match obs_size (for replay buffer consistency)
            obs_padded = jnp.concatenate([state_only, jnp.zeros((state_only.shape[0], goal_size))], axis=-1)  # (batch_size, obs_size)
            next_obs_padded = jnp.concatenate([next_state_only, jnp.zeros((next_state_only.shape[0], goal_size))], axis=-1)  # (batch_size, obs_size)
            
            return nstate, Transition(
                observation=obs_padded,
                action=actions,
                reward=computed_reward,
                discount=1 - nstate.done,
                extras={
                    "state_extras": state_extras,
                    "next_observation": next_obs_padded,
                },
            )

        # Used for evaluation
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = gcp_actor.apply(training_state.gcp_actor_state.params, env_state.obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, state_extras

        evaluator = ActorEvaluator(
            eval_env,
            deterministic_actor_step,
            extra_fields=("truncation", "traj_id"),
        )

        # Initialize SAC loss functions for EP
        ep_alpha_optimizer = optax.adam(learning_rate=self.alpha_lr)
        ep_policy_optimizer = optax.adam(learning_rate=self.policy_lr)
        ep_q_optimizer = optax.adam(learning_rate=self.critic_lr)
        
        ep_alpha_loss_fn, ep_critic_loss_fn, ep_actor_loss_fn = sac_losses.make_losses(
            sac_network=ep_sac_network,
            reward_scaling=1.0,  # Can be made configurable
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

        @functools.partial(jax.jit, static_argnames=("ep_reward_name",))
        def get_experience(gcp_actor_state, ep_actor_state, env_state, training_state, 
                          main_buffer_state, gcp_buffer_state, ep_buffer_state, key, ep_reward_name):
            """Collect experience with two sequential rollouts.
            
            First: GC policy (deterministic) for num_goal_conditioned_steps
            Second: EP policy (non-deterministic, non-goal-conditioned) for num_exploratory_steps
            """
            num_envs = config.num_envs
            
            # Track initial traj_ids to detect mid-rollout resets
            initial_traj_ids = env_state.info["traj_id"]
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps
            
            # ===== FIRST ROLLOUT: Goal-Conditioned Policy (Deterministic) =====
            gc_key, ep_key = jax.random.split(key, 2)
            gc_proposed_goals, gcp_buffer_state, ep_buffer_state = gcp_propose_goals(
                gcp_buffer_state, ep_buffer_state, train_env, env_state, gc_key, training_state
            )
            
            new_info = dict(env_state.info)
            new_info["gc_proposed_goals"] = gc_proposed_goals
            env_state = env_state.replace(
                obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(gc_proposed_goals),
                info=new_info
            )

            def gc_rollout_step(carry, unused_t):
                env_state, gcp_actor_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                
                gc_proposed_goals = env_state.info.get("gc_proposed_goals", env_state.obs[:, -len(train_env.goal_indices):])
                
                nstate, transition = deterministic_actor_step_with_proposals(
                    gcp_actor_state, train_env, env_state, gc_proposed_goals, ("truncation", "traj_id")
                )
                
                truncation = transition.extras["state_extras"]["truncation"]
                terminated = (transition.discount < 1.0) | (truncation > 0.5)
                
                new_info = dict(nstate.info)
                new_info["gc_proposed_goals"] = gc_proposed_goals
                env_state = nstate.replace(info=new_info)
                
                existing_state_extras = transition.extras["state_extras"]
                transition_extras = {
                    **existing_state_extras,
                    "in_gc_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                    "in_ep_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                    "gc_proposed_goals": gc_proposed_goals,
                    "ep_proposed_goals": gc_proposed_goals,  # Placeholder
                    "terminated": terminated.astype(jnp.float32),
                }
                transition = transition._replace(extras={"state_extras": transition_extras})
                
                obs_goals = transition.observation[:, -len(train_env.goal_indices):]
                goals_match = jnp.allclose(obs_goals, gc_proposed_goals, atol=1e-5, rtol=1e-5)
                _ = 1.0 / jnp.where(goals_match, 1.0, 0.0)
                
                return (env_state, gcp_actor_state, next_key), transition
            
            (env_state, _, _), gc_transitions = jax.lax.scan(
                gc_rollout_step,
                (env_state, gcp_actor_state, gc_key),
                (),
                length=num_goal_conditioned_steps
            )
            
            # ===== SECOND ROLLOUT: Exploratory Policy (Non-Goal-Conditioned, SAC) =====
            # EP does NOT use goal proposal - it's non-goal-conditioned
            
            def ep_rollout_step(carry, unused_t):
                env_state, ep_actor_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                
                # Use ep_actor_step helper function
                nstate, transition = ep_actor_step(
                    ep_actor_state, train_env, env_state, current_key, ("truncation", "traj_id")
                )
                
                # Check for early termination
                truncation = transition.extras["state_extras"]["truncation"]
                terminated = (transition.discount < 1.0) | (truncation > 0.5)
                
                # Add phase markers
                existing_state_extras = transition.extras["state_extras"]
                transition_extras = {
                    **existing_state_extras,
                    "in_gc_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                    "in_ep_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                    "terminated": terminated.astype(jnp.float32),
                }
                transition = transition._replace(extras={"state_extras": transition_extras})
                
                return (nstate, ep_actor_state, next_key), transition
            
            (env_state, _, _), ep_transitions = jax.lax.scan(
                ep_rollout_step,
                (env_state, ep_actor_state, ep_key),
                (),
                length=num_exploratory_steps
            )
            
            # ===== COMPUTE MEAN_MAX_CRITIC REWARD FOR EP TRANSITIONS =====
            # Compute reward: mean over (s,a) of max_g f(s, a, g) for each environment
            if ep_reward_name == "mean_max_critic" and hasattr(train_env, 'possible_goals') and train_env.possible_goals is not None:
                # Extract states and actions from EP transitions
                # ep_transitions shape: (num_exploratory_steps, num_envs, ...)
                ep_states = ep_transitions.observation[:, :, :state_size]  # (num_exploratory_steps, num_envs, state_dim)
                ep_actions = ep_transitions.action  # (num_exploratory_steps, num_envs, action_dim)
                env_goals = train_env.possible_goals  # (num_env_goals, goal_dim)
                
                # Compute reward for each environment's trajectory
                def compute_reward_for_env(env_idx):
                    """Compute min_critic_mean reward for a single environment's EP trajectory."""
                    env_states = ep_states[:, env_idx, :]  # (num_exploratory_steps, state_dim)
                    env_actions = ep_actions[:, env_idx, :]  # (num_exploratory_steps, action_dim)
                    
                    reward = compute_min_critic_mean_reward(
                        env_states, env_actions, env_goals,
                        gcp_actor, training_state.gcp_actor_state.params,
                        training_state.gcp_critic_state.params,
                        gcp_sa_encoder, gcp_g_encoder, self.energy_fn
                    )
                    return reward
                
                # Compute reward for all environments
                min_critic_mean_rewards = jax.vmap(compute_reward_for_env)(jnp.arange(num_envs))  # (num_envs,)
                
                # Add reward to each transition in the EP trajectory
                # Broadcast reward to all time steps for each environment
                reward_broadcast = min_critic_mean_rewards[None, :]  # (1, num_envs)
                reward_broadcast = jnp.broadcast_to(reward_broadcast, (num_exploratory_steps, num_envs))  # (num_exploratory_steps, num_envs)
                
                # Add to existing reward
                ep_transitions = ep_transitions._replace(
                    reward=ep_transitions.reward + reward_broadcast
            )
            
            # Combine transitions
            combined_transitions = jax.tree_util.tree_map(
                lambda gc, ep: jnp.concatenate([gc, ep], axis=0),
                gc_transitions, ep_transitions
            )
            
            # Track mid-rollout resets
            final_traj_ids = env_state.info["traj_id"]
            reset_during_rollout = initial_traj_ids != final_traj_ids
            num_reset_during_rollout = jnp.sum(reset_during_rollout)
            
            # Insert into buffers
            main_buffer_state = main_replay_buffer.insert(main_buffer_state, combined_transitions)
            gcp_buffer_state = gcp_replay_buffer.insert(gcp_buffer_state, gc_transitions)
            ep_buffer_state = ep_replay_buffer.insert(ep_buffer_state, ep_transitions)
         
            return env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, combined_transitions

        def prefill_replay_buffer(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key = carry
                key, reset_key, new_key = jax.random.split(key, 3)
                env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = get_experience(
                    training_state.gcp_actor_state,
                    training_state.ep_actor_state,
                    env_state,
                    training_state,
                    main_buffer_state,
                    gcp_buffer_state,
                    ep_buffer_state,
                    key,
                    self.ep_reward_name
                )
                
                reset_keys = jax.random.split(reset_key, config.num_envs)
                env_state = jax.jit(train_env.reset)(reset_keys)
                TRAJ_ID_MULTIPLIER = 100000
                env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
                initial_info = dict(env_state.info)
                initial_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
                env_state = env_state.replace(info=initial_info)
                
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, gcp_transitions, ep_transitions):
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
            # EP transitions have shape (batch_size, ...) with state-only observations
            # Convert to SAC Transition format (with next_observation)
            ep_obs = ep_transitions.observation  # (batch_size, state_size)
            ep_actions = ep_transitions.action  # (batch_size, action_size)
            ep_rewards = ep_transitions.reward  # (batch_size,)
            ep_discounts = ep_transitions.discount  # (batch_size,)
            ep_next_obs = ep_transitions.extras.get("next_observation", ep_obs)  # (batch_size, state_size)
            
            # Create SAC Transition format
            from brax.training.agents.sac.sac import Transition as SACTransition
            ep_sac_transitions = SACTransition(
                observation=ep_obs,
                next_observation=ep_next_obs,
                action=ep_actions,
                reward=ep_rewards,
                discount=ep_discounts,
                extras=ep_transitions.extras,
            )
            
            # Normalize observations for EP
            ep_normalized_obs = running_statistics.normalize(
                ep_obs, training_state.ep_normalizer_params
            )
            ep_normalized_next_obs = running_statistics.normalize(
                ep_next_obs, training_state.ep_normalizer_params
            )
            ep_sac_transitions = ep_sac_transitions._replace(
                observation=ep_normalized_obs,
                next_observation=ep_normalized_next_obs
            )
            
            # Update EP alpha
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
            
            # Update EP critic
            ep_critic_loss, ep_q_params, ep_q_optimizer_state = ep_critic_update(
                training_state.ep_q_params,
                training_state.ep_actor_state.params,
                training_state.ep_normalizer_params,
                training_state.ep_target_q_params,
                ep_alpha,
                ep_sac_transitions,
                ep_critic_key,
                optimizer_state=training_state.ep_q_optimizer_state,
            )
            
            # Update EP actor
            ep_actor_loss, ep_actor_params, ep_actor_optimizer_state = ep_actor_update(
                training_state.ep_actor_state.params,
                training_state.ep_normalizer_params,
                training_state.ep_q_params,
                ep_alpha,
                ep_sac_transitions,
                ep_actor_key,
                optimizer_state=training_state.ep_actor_optimizer_state,
            )
            
            # Update EP target Q-network (soft update)
            tau = 0.005  # Can be made configurable
            new_ep_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - tau) + y * tau,
                training_state.ep_target_q_params,
                ep_q_params,
            )
            
            # Update EP normalizer
            new_ep_normalizer_params = running_statistics.update(
                training_state.ep_normalizer_params,
                ep_obs,
            )
            
            # Update EP actor state
            new_ep_actor_state = training_state.ep_actor_state.replace(params=ep_actor_params)
            
            # Update training state
            training_state = training_state.replace(
                gcp_actor_state=new_gcp_actor_state,
                gcp_critic_state=new_gcp_critic_state,
                gcp_alpha_state=new_gcp_alpha_state,
                ep_actor_state=new_ep_actor_state,
                ep_q_params=ep_q_params,
                ep_target_q_params=new_ep_target_q_params,
                ep_q_optimizer_state=ep_q_optimizer_state,
                ep_actor_optimizer_state=ep_actor_optimizer_state,
                ep_alpha_params=ep_alpha_params,
                ep_alpha_optimizer_state=ep_alpha_optimizer_state,
                ep_normalizer_params=new_ep_normalizer_params,
                gradient_steps=training_state.gradient_steps + 1,
            )

            # Construct metrics
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

        @jax.jit
        def training_step(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            experience_key1, reset_key, sampling_key, permute_key, training_key = jax.random.split(key, 5)

            env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, collected_transitions = get_experience(
                training_state.gcp_actor_state,
                training_state.ep_actor_state,
                env_state,
                training_state,
                main_buffer_state,
                gcp_buffer_state,
                ep_buffer_state,
                experience_key1,
                self.ep_reward_name
            )
            
            reset_keys = jax.random.split(reset_key, config.num_envs)
            env_state = jax.jit(train_env.reset)(reset_keys)
            TRAJ_ID_MULTIPLIER = 100000
            env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
            initial_info = dict(env_state.info)
            initial_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
            env_state = env_state.replace(info=initial_info)

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # Sample transitions
            main_buffer_state, gcp_transitions = main_replay_buffer.sample(main_buffer_state)
            ep_buffer_state, ep_transitions = ep_replay_buffer.sample(ep_buffer_state)

            # Process transitions for training
            batch_keys = jax.random.split(sampling_key, gcp_transitions.observation.shape[0] + ep_transitions.observation.shape[0])
            gcp_batch_keys = batch_keys[:gcp_transitions.observation.shape[0]]
            ep_batch_keys = batch_keys[gcp_transitions.observation.shape[0]:]

            def process_gcp_transitions(transitions, batch_keys):
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )
                permutation = jax.random.permutation(permute_key, len(transitions.observation))
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )
                return transitions
            
            def process_ep_transitions(transitions, batch_keys):
                # EP transitions don't need flatten_batch (no goal-conditioning)
                # Just reshape and permute
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )
                permutation = jax.random.permutation(permute_key, len(transitions.observation))
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )
                return transitions
            
            gcp_last_batch = process_gcp_transitions(gcp_transitions, gcp_batch_keys)
            ep_last_batch = process_ep_transitions(ep_transitions, ep_batch_keys)

            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(
                lambda carry, xs: update_networks(carry, xs[0], xs[1]),
                (training_state, training_key),
                (gcp_last_batch, ep_last_batch)
            )

            return (
                training_state,
                env_state,
                main_buffer_state,
                gcp_buffer_state,
                ep_buffer_state,
                collected_transitions
            ), metrics

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

            metrics = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a[None], b]),
                first_metrics,
                rest_metrics,
            )
            return training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, collected_transitions
        
        def visualize_goals(train_env, transitions, wandb_key):
            """Visualize trajectories and goals for go_explore_sac.
            Note: EP does not have goals, so we only visualize GC goals."""
            # Shape is (episode_len-1, batch_size, ...) since we only keep the last training step's batch
            obs = np.array(transitions.observation)  # (episode_len-1, batch_size, obs_dim)
            last_traj_state = np.array(transitions.extras["last_traj_state"][:, :, :state_size])  # (episode_len-1, batch_size, state_size)
            last_traj_state_flat = last_traj_state.reshape(-1, state_size)
            intermediate_traj = np.array(transitions.extras["intermediate_traj"])  # (episode_len-1, batch_size, num_intermediate_states, obs_dim)
            
            obs_flat = obs.reshape(-1, obs.shape[-1])  # (total_samples, obs_dim)
            
            # Extract GC and EP specific data
            in_gc_phase = np.array(transitions.extras["state_extras"]["in_gc_phase"])  # (episode_len-1, batch_size)
            in_ep_phase = np.array(transitions.extras["state_extras"]["in_ep_phase"])  # (episode_len-1, batch_size)
            gc_proposed_goals = np.array(transitions.extras["state_extras"]["gc_proposed_goals"])  # (episode_len-1, batch_size, goal_dim)
            traj_ids = np.array(transitions.extras["state_extras"]["traj_id"])  # (episode_len-1, batch_size)
            
            in_gc_phase_flat = in_gc_phase.reshape(-1)
            in_ep_phase_flat = in_ep_phase.reshape(-1)
            gc_proposed_goals_flat = gc_proposed_goals.reshape(-1, len(train_env.goal_indices))
            traj_ids_flat = traj_ids.reshape(-1)
            
            intermediate_traj_flat = intermediate_traj.reshape(-1, intermediate_traj.shape[-2], intermediate_traj.shape[-1])
            
            # Extract GC and EP final states for each trajectory
            gc_final_states = []
            ep_final_states = []
            start_states = []
            gc_goals = []
            gc_intermediate_states_list = []
            ep_intermediate_states_list = []
            
            unique_traj_ids = np.unique(traj_ids_flat)
            
            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_flat == traj_id
                traj_indices = np.sort(np.where(traj_mask)[0])
                
                start_state = obs_flat[traj_indices[0]][train_env.goal_indices]
                start_states.append(start_state)
                
                # Find last GC phase transition
                gc_mask = in_gc_phase_flat[traj_indices] > 0.5
                gc_indices = traj_indices[gc_mask]
                if len(gc_indices) > 0:
                    gc_final_idx = gc_indices[-1]
                    gc_final_state = obs_flat[gc_final_idx][train_env.goal_indices]
                    gc_goal = gc_proposed_goals_flat[gc_final_idx]
                    gc_final_states.append(gc_final_state)
                    gc_goals.append(gc_goal)
                
                # Find last EP phase transition
                ep_mask = in_ep_phase_flat[traj_indices] > 0.5
                ep_indices = traj_indices[ep_mask]
                if len(ep_indices) > 0:
                    ep_final_idx = ep_indices[-1]
                    ep_final_state = obs_flat[ep_final_idx][train_env.goal_indices]
                    ep_final_states.append(ep_final_state)
                
                # Get intermediate states from transitions in their respective phases
                # Intermediate states are computed per transition as fractions of the remaining trajectory
                # We use intermediate states from the first transition of each phase and filter to correct phase
                
                # GC intermediate states: use intermediate states from first GC transition
                gc_intermediate = np.array([]).reshape(0, len(train_env.goal_indices))
                if len(gc_indices) > 0:
                    gc_first_idx = gc_indices[0]  # First GC transition in this trajectory
                    gc_intermediate_full = intermediate_traj_flat[gc_first_idx, :, train_env.goal_indices]  # (num_intermediate, goal_dim)
                    # Map each intermediate state back to its actual trajectory position and filter by phase
                    gc_local_idx = gc_first_idx - traj_indices[0]  # Local index within this trajectory
                    remaining_length = len(traj_indices) - gc_local_idx
                    gc_intermediate_list = []
                    for inter_idx in range(gc_intermediate_full.shape[0]):
                        if remaining_length > 0:
                            # Intermediate states are at fractions: [1/(n+1), 2/(n+1), ..., n/(n+1)] of remaining trajectory
                            fraction = (inter_idx + 1) / (gc_intermediate_full.shape[0] + 1)
                            target_local_idx = gc_local_idx + int(fraction * remaining_length)
                            target_local_idx = min(target_local_idx, len(traj_indices) - 1)
                            target_global_idx = traj_indices[target_local_idx]
                            # Only include if the target state is actually in GC phase
                            if target_global_idx < len(in_gc_phase_flat) and in_gc_phase_flat[target_global_idx] > 0.5:
                                gc_intermediate_list.append(gc_intermediate_full[inter_idx])
                    if len(gc_intermediate_list) > 0:
                        gc_intermediate = np.array(gc_intermediate_list)
                
                # EP intermediate states: use intermediate states from first EP transition
                ep_intermediate = np.array([]).reshape(0, len(train_env.goal_indices))
                if len(ep_indices) > 0:
                    ep_first_idx = ep_indices[0]  # First EP transition in this trajectory
                    ep_intermediate_full = intermediate_traj_flat[ep_first_idx, :, train_env.goal_indices]  # (num_intermediate, goal_dim)
                    # Map each intermediate state back to its actual trajectory position and filter by phase
                    ep_local_idx = ep_first_idx - traj_indices[0]  # Local index within this trajectory
                    remaining_length = len(traj_indices) - ep_local_idx
                    ep_intermediate_list = []
                    for inter_idx in range(ep_intermediate_full.shape[0]):
                        if remaining_length > 0:
                            # Intermediate states are at fractions: [1/(n+1), 2/(n+1), ..., n/(n+1)] of remaining trajectory
                            fraction = (inter_idx + 1) / (ep_intermediate_full.shape[0] + 1)
                            target_local_idx = ep_local_idx + int(fraction * remaining_length)
                            target_local_idx = min(target_local_idx, len(traj_indices) - 1)
                            target_global_idx = traj_indices[target_local_idx]
                            # Only include if the target state is actually in EP phase
                            if target_global_idx < len(in_ep_phase_flat) and in_ep_phase_flat[target_global_idx] > 0.5:
                                ep_intermediate_list.append(ep_intermediate_full[inter_idx])
                    if len(ep_intermediate_list) > 0:
                        ep_intermediate = np.array(ep_intermediate_list)
                
                # Store separate GC and EP intermediate states
                gc_intermediate_states_list.append(gc_intermediate)
                ep_intermediate_states_list.append(ep_intermediate)
            
            # Convert to numpy arrays
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
            
            # Sample exactly 4 trajectories for 2x2 grid visualization
            num_trajs = start_states.shape[0]
            num_viz_trajs = min(4, num_trajs)
            sample_indices = np.random.choice(num_trajs, num_viz_trajs, replace=False)
            
            start_xy = start_states[sample_indices]
            gc_final_xy = gc_final_states[sample_indices]
            ep_final_xy = ep_final_states[sample_indices]
            gc_proposed_goals_xy = gc_goals[sample_indices]
            
            # Extract GC and EP intermediate states for sampled trajectories
            gc_intermediate_xy_list = [gc_intermediate_states_list[i] for i in sample_indices]
            ep_intermediate_xy_list = [ep_intermediate_states_list[i] for i in sample_indices]
            
            # Visualize trajectories (without EP goals - pass zeros as placeholder)
            ep_proposed_goals_xy_placeholder = np.zeros_like(gc_proposed_goals_xy)  # Placeholder, won't be plotted
            visualize_dual_crl_trajectories_2d(
                start_xy, gc_final_xy, ep_final_xy, gc_proposed_goals_xy, ep_proposed_goals_xy_placeholder,
                gc_intermediate_xy_list, ep_intermediate_xy_list, f"{wandb_key}/dual_crl_trajectories",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # Heatmaps (no EP goal proposals heatmap)
            visualize_kde_heatmap(
                gc_final_states, "GC Final States", f"{wandb_key}/gc_final_states_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            visualize_kde_heatmap(
                ep_final_states, "EP Final States", f"{wandb_key}/ep_final_states_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            visualize_kde_heatmap(
                gc_goals, "GC Goal Proposals", f"{wandb_key}/gc_goal_proposals_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            logging.info(f"Plotted visualizations at env step {training_state.env_steps.item()}")
        
        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, prefill_key
        )

        evaluator = ActorEvaluator(
            deterministic_actor_step,
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

            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, collected_transitions = training_epoch(
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, epoch_key
            )

            # Process collected_transitions for visualization
            @jax.jit
            def process_for_viz(transitions, batch_keys):
                # Process transitions through flatten_batch (vmap over environments)
                processed = jax.vmap(flatten_batch, in_axes=(None, 1, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                # processed now has shape (num_envs, episode_length-1, obs_dim)
                # Reshape to (episode_length-1, num_envs, ...) for visualization
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

            training_walltime += time.time() - t

            # Aggregate metrics
            eval_metrics = {}
            eval_metrics.update(
                {
                    f"training/{k}": np.array(v).mean() for k, v in metrics.items()
                }
            )
            eval_metrics["training/walltime"] = training_walltime
            eval_metrics["training/env_steps"] = training_state.env_steps

            # Evaluation
            eval_state = evaluator.run_evaluation(
                (training_state.gcp_actor_state.params,),
                training_state.env_steps,
                eval_metrics,
            )

            progress_fn(int(training_state.env_steps), eval_metrics)

            # Save checkpoint
            if config.save_dir:
                save_params(
                    str(epath.Path(config.save_dir) / f"params_{ne}.pkl"),
                    training_state.gcp_actor_state.params,
                )

        return {"params": training_state.gcp_actor_state.params}
