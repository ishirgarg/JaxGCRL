import functools
import logging
from math import e
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union
import wandb

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
from jaxgcrl.utils.visualize import visualize_goals_2d, visualize_kde_heatmap, visualize_q_function_2d, visualize_dual_crl_trajectories_2d, visualize_scatter_sample

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder
from jaxgcrl.agents.crl.proposers import (
    FinalReplayBufferProposer, 
    RandomEnvironmentGoalProposer,
    UCGRProposer,
    MaxWaypointRatioOneEnvProposer,
    QEpistemicProposer,
    MEGAProposer,
    OMEGAProposer,
    NearestEnvGoalProposer,
    EmpowermentDifferenceGoalProposer,
)
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
    ep_critic_state: TrainState
    gcp_alpha_state: TrainState
    ep_alpha_state: TrainState
    max_traj_id: jnp.ndarray  # Track maximum trajectory ID for globally unique IDs


class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    # transition.observations has size (episode_length, obs_dim)
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
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
        """Returns states at 1/3 and 2/3 of remaining trajectory for each timestep"""
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


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class GoExploreCRL:
    """Go-Explore Contrastive Reinforcement Learning (CRL) agent."""
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

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Goal proposer names for go-explore style algorithms
    gcp_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals", "ucgr", "maxwaypointratio_one_env", "q_epistemic", "mega", "omega", "empowerment_diff"] = "gcp_final_rb"
    ep_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals", "nearest_env_goal"] = "ep_final_rb"
    goal_sampling_temperature: float = 0.0
    
    # Replay buffer goal sampling parameters
    num_rb_goals: int = 256
    candidate_goals_type: Literal["final", "any"] = "final"
    filter_successful_waypoints: bool = False

    # Empowerment difference goal proposer parameters
    empowerment_num_outer_samples: int = 10   # N – outer MC samples
    empowerment_num_inner_actions: int = 10   # M – contrastive inner actions
    gcp_empowerment_penalty: float = 1.0      # beta coefficient

    train_ep_on_main_buffer: bool = False

    use_same_policy: bool = True
    
    # Critic ensemble for Q-epistemic goal proposal
    use_gcp_critic_ensemble: bool = False
    gcp_num_critic_ensemble: int = 5

    # Unroll length for network updates
    unroll_length: int = 50

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
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

        unroll_length = self.unroll_length
        total_steps_per_training_step = config.num_goal_conditioned_steps + config.num_exploratory_steps
        env_steps_per_actor_step = config.num_envs * unroll_length
        env_steps_per_training_step = config.num_envs * total_steps_per_training_step
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / (config.num_goal_conditioned_steps + config.num_exploratory_steps))
        num_training_steps_per_epoch = -(-(config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_training_step
        ))

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, gcp_actor_key, gcp_sa_key, gcp_g_key, ep_actor_key, ep_sa_key, ep_g_key = jax.random.split(key, 10)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)
        
        # Initialize environment info with tracking fields
        TRAJ_ID_MULTIPLIER = 100000
        env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
        initial_info = dict(env_state.info)
        initial_info["traj_id"] = env_indices * TRAJ_ID_MULTIPLIER
        initial_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
        initial_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
        initial_info["policy_phase"] = jnp.ones((config.num_envs,), dtype=bool)  # All start in GCP
        initial_info["needs_reset"] = jnp.zeros((config.num_envs,), dtype=bool)
        initial_info["gc_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
        initial_info["ep_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
        env_state = env_state.replace(info=initial_info)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        # Actor
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

        ep_actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
       
        ep_actor_state = TrainState.create(
            apply_fn=ep_actor.apply,
            params=ep_actor.init(ep_actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
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
        ep_sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        ep_g_encoder = Encoder(
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
        
        ep_sa_encoder_params = ep_sa_encoder.init(ep_sa_key, np.ones([1, state_size + action_size]))
        ep_g_encoder_params = ep_g_encoder.init(ep_g_key, np.ones([1, goal_size]))
        
        gcp_critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": gcp_sa_encoder_params, "g_encoder": gcp_g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )
        ep_critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": ep_sa_encoder_params, "g_encoder": ep_g_encoder_params},
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
        ep_alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            gcp_actor_state=gcp_actor_state,
            ep_actor_state=ep_actor_state,
            gcp_critic_state=gcp_critic_state,
            ep_critic_state=ep_critic_state,
            gcp_alpha_state=gcp_alpha_state,
            ep_alpha_state=ep_alpha_state,
            max_traj_id=jnp.array(config.num_envs * TRAJ_ID_MULTIPLIER, dtype=jnp.int32),  # Start after initial IDs
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
        
        # Initialize goal proposer instances
        gcp_final_rb_proposer = FinalReplayBufferProposer(
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        ep_final_rb_proposer = FinalReplayBufferProposer(
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        env_goals_proposer = RandomEnvironmentGoalProposer()
        nearest_env_goal_proposer = NearestEnvGoalProposer(
            energy_fn_name=self.energy_fn
        )
        ucgr_proposer = UCGRProposer(
            energy_fn_name=self.energy_fn,
            num_rb_samples=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        maxwaypointratio_one_env_proposer = MaxWaypointRatioOneEnvProposer(
            energy_fn_name=self.energy_fn,
            goal_sampling_temperature=self.goal_sampling_temperature,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
            filter_successful_waypoints=self.filter_successful_waypoints
        )
        q_epistemic_proposer = QEpistemicProposer(
            energy_fn_name=self.energy_fn,
            num_ensemble=self.gcp_num_critic_ensemble,
            use_env_goals=False,
            zero_center=False,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        mega_proposer = MEGAProposer(
            bandwidth=0.1,
            use_q_cutoff=True,
            cutoff_percentile=0.3,
            energy_fn_name=self.energy_fn,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        omega_proposer = OMEGAProposer(
            bandwidth=0.1,
            use_q_cutoff=True,
            cutoff_percentile=0.3,
            energy_fn_name=self.energy_fn,
            bias_param=-3.0,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        empowerment_diff_proposer = EmpowermentDifferenceGoalProposer(
            energy_fn_name=self.energy_fn,
            empowerment_num_outer_samples=self.empowerment_num_outer_samples,
            empowerment_num_inner_actions=self.empowerment_num_inner_actions,
            gcp_empowerment_penalty=self.gcp_empowerment_penalty,
            num_rb_goals=self.num_rb_goals,
            discounting=self.discounting,
            goal_sampling_temperature=self.goal_sampling_temperature,
        )
        
        # Create goal proposer functions
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
        elif self.gcp_goal_proposer_name == "mega":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = mega_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "omega":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = omega_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "empowerment_diff":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                # Scores replay-buffer states by E_ep - beta*E_gcp against a random env goal.
                # Uses the EP replay buffer for state sampling (diverse exploration states).
                proposed_goals, updated_ep = empowerment_diff_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        else:
            raise ValueError(f"Unknown gcp_goal_proposer_name: {self.gcp_goal_proposer_name}")
        
        if self.ep_goal_proposer_name == "gcp_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(
                    gcp_replay_buffer, gcp_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params, training_state.ep_critic_state.params,
                    ep_sa_encoder, ep_g_encoder, training_state
                )
                return proposed_goals, updated_gcp, ep_buffer_state
        elif self.ep_goal_proposer_name == "ep_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params, training_state.ep_critic_state.params,
                    ep_sa_encoder, ep_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.ep_goal_proposer_name == "env_goals":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = env_goals_proposer.propose_goals(
                    None, ep_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params, training_state.ep_critic_state.params,
                    ep_sa_encoder, ep_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state
        elif self.ep_goal_proposer_name == "nearest_env_goal":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = nearest_env_goal_proposer.propose_goals(
                    None, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params, training_state.gcp_critic_state.params,
                    gcp_sa_encoder, gcp_g_encoder, training_state
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state
        else:
            raise ValueError(f"Unknown ep_goal_proposer_name: {self.ep_goal_proposer_name}")

        # Used for evaluation
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
        
        def deterministic_ep_actor_step(training_state, env, env_state, extra_fields):
            means, _ = ep_actor.apply(training_state.ep_actor_state.params, env_state.obs)
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

        def deterministic_actor_step_with_proposals(actor_state, env, env_state, proposed_goals, extra_fields):
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

        def actor_step(actor_state, env, env_state, proposed_goals, key, extra_fields):
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)

            means, log_stds = actor_state.apply_fn(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )
        
        @jax.jit
        def get_experience_chunk(gcp_actor_state, ep_actor_state, env_state, training_state,
                                main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            """Collect unroll_length steps with dynamic per-environment phase switching."""
            num_envs = config.num_envs
            goal_indices = train_env.goal_indices
            goal_reach_thresh = train_env.goal_reach_thresh
            
            # Get or initialize per-environment policy phase (True=GCP, False=EP)
            use_gcp = env_state.info.get("policy_phase", jnp.ones((num_envs,), dtype=bool))
            
            # Get current goals (propose if needed)
            key, gc_key, ep_key, rollout_key, reset_key = jax.random.split(key, 5)
            
            # Propose GCP goals if any GCP environment doesn't have goals
            gc_goals = env_state.info.get("gc_proposed_goals", jnp.zeros((num_envs, len(goal_indices))))
            needs_gc_goals = jnp.any(use_gcp) & (jnp.sum(jnp.abs(gc_goals)) < 1e-6)
            
            def propose_gc():
                goals, buf_gc, buf_ep = gcp_propose_goals(
                    gcp_buffer_state, ep_buffer_state, train_env, env_state, gc_key, training_state
                )
                return goals, buf_gc, buf_ep
            
            def keep_gc():
                return gc_goals, gcp_buffer_state, ep_buffer_state
            
            gc_goals, gcp_buffer_state, ep_buffer_state = jax.lax.cond(
                needs_gc_goals,
                propose_gc,
                keep_gc
            )
            
            # Similar for EP goals
            ep_goals = env_state.info.get("ep_proposed_goals", jnp.zeros((num_envs, len(goal_indices))))
            needs_ep_goals = jnp.any(~use_gcp) & (jnp.sum(jnp.abs(ep_goals)) < 1e-6)
            
            def propose_ep():
                goals, buf_gc, buf_ep = ep_propose_goals(
                    gcp_buffer_state, ep_buffer_state, train_env, env_state, ep_key, training_state
                )
                return goals, buf_gc, buf_ep
            
            def keep_ep():
                return ep_goals, gcp_buffer_state, ep_buffer_state
            
            ep_goals, gcp_buffer_state, ep_buffer_state = jax.lax.cond(
                needs_ep_goals,
                propose_ep,
                keep_ep
            )
            
            # Update env_state with all necessary info INCLUDING needs_reset
            new_info = dict(env_state.info)
            new_info["gc_proposed_goals"] = gc_goals
            new_info["ep_proposed_goals"] = ep_goals
            new_info["policy_phase"] = use_gcp
            new_info["needs_reset"] = jnp.zeros((num_envs,), dtype=bool)  # INITIALIZE HERE
            # Initialize step counters if not present
            new_info["gc_steps_taken"] = env_state.info.get("gc_steps_taken", jnp.zeros((num_envs,), dtype=jnp.int32))
            new_info["ep_steps_taken"] = env_state.info.get("ep_steps_taken", jnp.zeros((num_envs,), dtype=jnp.int32))
            env_state = env_state.replace(info=new_info)
            
            # Get step limits
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps
            
            # Unified rollout step
            def rollout_step(carry, _):
                env_st, key_inner = carry
                key_inner, step_key = jax.random.split(key_inner)
                
                # Get current phase, goals, and step counts per environment
                use_gcp_inner = env_st.info["policy_phase"]
                gc_goals_inner = env_st.info["gc_proposed_goals"]
                ep_goals_inner = env_st.info["ep_proposed_goals"]
                gc_steps_taken = env_st.info["gc_steps_taken"]
                ep_steps_taken = env_st.info["ep_steps_taken"]
                
                # Enforce step caps: force switch/reset when limits reached
                gc_at_limit = gc_steps_taken >= num_goal_conditioned_steps
                ep_at_limit = ep_steps_taken >= num_exploratory_steps
                
                # Force GCP→EP switch if GCP has reached its step limit
                use_gcp_inner = use_gcp_inner & ~gc_at_limit
                
                # Select goals based on phase
                selected_goals = jnp.where(use_gcp_inner[:, None], gc_goals_inner, ep_goals_inner)
                
                # Update observations with selected goals
                env_st = env_st.replace(
                    obs=env_st.obs.at[:, -len(goal_indices):].set(selected_goals)
                )
                
                # Step with appropriate policy per environment
                # Vectorized step for GCP environments (deterministic)
                nstate_gcp, trans_gcp = deterministic_actor_step_with_proposals(
                    gcp_actor_state, train_env, env_st, gc_goals_inner, ("truncation", "traj_id")
                )
                
                # Vectorized step for EP environments (stochastic)
                if self.use_same_policy:
                    nstate_ep, trans_ep = actor_step(
                        gcp_actor_state, train_env, env_st, ep_goals_inner, step_key, ("truncation", "traj_id")
                    )
                else:
                    nstate_ep, trans_ep = actor_step(
                        ep_actor_state, train_env, env_st, ep_goals_inner, step_key, ("truncation", "traj_id")
                    )
                
                # Combine results based on phase
                def select_by_phase(gcp_val, ep_val):
                    # Expand mask to match number of dimensions in the values
                    mask = use_gcp_inner
                    for _ in range(len(gcp_val.shape) - 1):
                        mask = mask[..., None]  # Add trailing dimensions
                    return jnp.where(mask, gcp_val, ep_val)
                
                nstate = jax.tree_util.tree_map(select_by_phase, nstate_gcp, nstate_ep)
                transition = jax.tree_util.tree_map(select_by_phase, trans_gcp, trans_ep)
                
                # Check for goal completion
                current_pos = nstate.obs[:, goal_indices]
                gc_dist = jnp.linalg.norm(current_pos - gc_goals_inner, axis=1)
                ep_dist = jnp.linalg.norm(current_pos - ep_goals_inner, axis=1)
                
                gc_goal_reached = (gc_dist < goal_reach_thresh) & use_gcp_inner
                ep_goal_reached = (ep_dist < goal_reach_thresh) & ~use_gcp_inner
                
                # Update step counters: increment based on current phase
                gc_steps_taken_new = gc_steps_taken + use_gcp_inner.astype(jnp.int32)
                ep_steps_taken_new = ep_steps_taken + (~use_gcp_inner).astype(jnp.int32)
                
                # Update phase: GCP→EP when GCP goal reached OR GCP step limit reached
                new_phase = jnp.where(gc_goal_reached | gc_at_limit, False, use_gcp_inner)
                
                # Mark for reset when EP goal reached OR EP step limit reached
                needs_reset_flag = ep_goal_reached | ep_at_limit
                
                # Update info - ALL KEYS MUST BE PRESENT
                new_info_inner = dict(nstate.info)
                new_info_inner["gc_proposed_goals"] = gc_goals_inner
                new_info_inner["ep_proposed_goals"] = ep_goals_inner
                new_info_inner["policy_phase"] = new_phase
                new_info_inner["needs_reset"] = needs_reset_flag
                new_info_inner["gc_steps_taken"] = gc_steps_taken_new
                new_info_inner["ep_steps_taken"] = ep_steps_taken_new
                
                # Add metadata to transition
                terminated = (transition.discount < 1.0) | (transition.extras["state_extras"]["truncation"] > 0.5)
                trans_extras = {
                    **transition.extras["state_extras"],
                    "in_gc_phase": use_gcp_inner.astype(jnp.float32),
                    "in_ep_phase": (~use_gcp_inner).astype(jnp.float32),
                    "gc_proposed_goals": gc_goals_inner,
                    "ep_proposed_goals": ep_goals_inner,
                    "terminated": terminated.astype(jnp.float32),
                }
                transition = transition._replace(extras={"state_extras": trans_extras})
                
                env_st = nstate.replace(info=new_info_inner)
                return (env_st, key_inner), transition
            
            # Run rollout
            (env_state_final, _), transitions = jax.lax.scan(
                rollout_step,
                (env_state, rollout_key),
                None,
                length=unroll_length
            )
            
            # Insert into buffers - insert all actual data
            main_buffer_state = main_replay_buffer.insert(main_buffer_state, transitions)
            gcp_buffer_state = gcp_replay_buffer.insert(gcp_buffer_state, transitions)
            ep_buffer_state = ep_replay_buffer.insert(ep_buffer_state, transitions)
            
            # Handle resets at chunk boundary
            needs_reset = env_state_final.info.get("needs_reset", jnp.zeros((num_envs,), dtype=bool))
            
            def do_resets():
                # Reset environments that completed EP phase
                reset_keys = jax.random.split(reset_key, num_envs)
                reset_state = jax.jit(train_env.reset)(reset_keys)
                
                # Helper to expand mask to match target shape
                def select_with_mask(reset_val, curr_val):
                    """Select reset_val where needs_reset is True, else curr_val."""
                    # Expand mask to match dimensionality of the values
                    mask = needs_reset
                    for _ in range(len(reset_val.shape) - 1):
                        mask = mask[..., None]
                    return jnp.where(mask, reset_val, curr_val)
                
                # Apply mask to all fields in the state
                # obs, reward, done
                combined_obs = select_with_mask(reset_state.obs, env_state_final.obs)
                combined_reward = select_with_mask(reset_state.reward, env_state_final.reward)
                combined_done = select_with_mask(reset_state.done, env_state_final.done)
                
                # pipeline_state - recursively apply to all nested arrays
                combined_pipeline_state = jax.tree_util.tree_map(
                    select_with_mask,
                    reset_state.pipeline_state,
                    env_state_final.pipeline_state
                )
                
                # Handle info dict carefully
                current_traj_ids = env_state_final.info["traj_id"]
                # Use global max_traj_id from training_state to ensure globally unique IDs
                global_max_id = training_state.max_traj_id
                new_traj_ids = jnp.where(needs_reset, global_max_id + jnp.arange(1, num_envs + 1), current_traj_ids)
                # Update global max_traj_id: take max of new IDs (if any resets happened)
                updated_max_id = jnp.max(jnp.where(needs_reset, new_traj_ids, global_max_id))
                
                new_phase = jnp.where(needs_reset, True, env_state_final.info["policy_phase"])
                
                # Build info dict with all keys
                final_info = {}
                
                # Process base environment keys from reset_state
                for key in reset_state.info.keys():
                    if key == "traj_id":
                        final_info["traj_id"] = new_traj_ids
                    elif key in env_state_final.info:
                        reset_val = reset_state.info[key]
                        curr_val = env_state_final.info[key]
                        if isinstance(reset_val, jnp.ndarray):
                            final_info[key] = select_with_mask(reset_val, curr_val)
                        else:
                            # Non-array values - just keep current
                            final_info[key] = curr_val
                    else:
                        final_info[key] = reset_state.info[key]
                
                # Add our custom keys (these aren't in reset_state.info)
                final_info["policy_phase"] = new_phase
                final_info["needs_reset"] = jnp.zeros((num_envs,), dtype=bool)
                final_info["gc_proposed_goals"] = select_with_mask(jnp.zeros_like(gc_goals), gc_goals)
                final_info["ep_proposed_goals"] = select_with_mask(jnp.zeros_like(ep_goals), ep_goals)
                # Reset step counters for environments that reset
                final_info["gc_steps_taken"] = select_with_mask(
                    jnp.zeros((num_envs,), dtype=jnp.int32),
                    env_state_final.info.get("gc_steps_taken", jnp.zeros((num_envs,), dtype=jnp.int32))
                )
                final_info["ep_steps_taken"] = select_with_mask(
                    jnp.zeros((num_envs,), dtype=jnp.int32),
                    env_state_final.info.get("ep_steps_taken", jnp.zeros((num_envs,), dtype=jnp.int32))
                )
                
                # Construct the combined state
                updated_state = env_state_final.replace(
                    obs=combined_obs,
                    reward=combined_reward,
                    done=combined_done,
                    pipeline_state=combined_pipeline_state,
                    info=final_info
                )
                return updated_state, updated_max_id

            def no_resets():
                return env_state_final, training_state.max_traj_id

            env_state_final, updated_max_traj_id = jax.lax.cond(
                jnp.any(needs_reset),
                do_resets,
                no_resets
            )
            return env_state_final, main_buffer_state, gcp_buffer_state, ep_buffer_state, transitions, updated_max_traj_id

        def prefill_replay_buffer(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key = carry
                key, reset_key, new_key = jax.random.split(key, 3)
                
                num_goal_conditioned_steps = config.num_goal_conditioned_steps
                num_exploratory_steps = config.num_exploratory_steps
                total_chunks = (num_goal_conditioned_steps + num_exploratory_steps) // unroll_length
                
                experience_keys = jax.random.split(key, total_chunks)
                
                def collect_chunk(carry_inner, chunk_idx):
                    (env_state_inner, main_buffer_state_inner, gcp_buffer_state_inner, ep_buffer_state_inner, current_max_traj_id) = carry_inner
                    
                    # Create a temporary training_state with current max_traj_id for get_experience_chunk
                    temp_training_state = training_state.replace(max_traj_id=current_max_traj_id)
                    
                    (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new, _, updated_max_traj_id) = get_experience_chunk(
                        temp_training_state.gcp_actor_state,
                        temp_training_state.ep_actor_state,
                        env_state_inner,
                        temp_training_state,
                        main_buffer_state_inner,
                        gcp_buffer_state_inner,
                        ep_buffer_state_inner,
                        experience_keys[chunk_idx]
                    )
                    
                    carry_new = (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new, updated_max_traj_id)
                    return carry_new, ()
                
                initial_carry = (env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, training_state.max_traj_id)
                
                (env_state_final, main_buffer_state_final, gcp_buffer_state_final, ep_buffer_state_final, final_max_traj_id), _ = jax.lax.scan(
                    collect_chunk,
                    initial_carry,
                    jnp.arange(total_chunks)
                )
                
                # Reset environments after collecting experience with globally unique trajectory IDs
                reset_keys = jax.random.split(reset_key, config.num_envs)
                env_state_final = jax.jit(train_env.reset)(reset_keys)
                
                # Generate globally unique trajectory IDs using final_max_traj_id from inner scan
                new_traj_ids = final_max_traj_id + jnp.arange(1, config.num_envs + 1, dtype=jnp.int32)
                new_max_id = new_traj_ids[-1]  # Update max_traj_id
                
                final_info = dict(env_state_final.info)
                final_info["traj_id"] = new_traj_ids
                final_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                final_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
                final_info["policy_phase"] = jnp.ones((config.num_envs,), dtype=bool)
                final_info["needs_reset"] = jnp.zeros((config.num_envs,), dtype=bool)
                final_info["gc_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
                final_info["ep_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
                env_state_final = env_state_final.replace(info=final_info)
                
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + (num_goal_conditioned_steps + num_exploratory_steps) * config.num_envs,
                    max_traj_id=new_max_id,  # Update max_traj_id
                )
                return (training_state, env_state_final, main_buffer_state_final, gcp_buffer_state_final, ep_buffer_state_final, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key),
                (),
                length=int(num_prefill_actor_steps),
            )[0]

        @jax.jit
        def update_networks(carry, gcp_transitions, ep_transitions):
            training_state, key = carry
            
            key, gcp_critic_key, gcp_actor_key, ep_actor_key, ep_critic_key = jax.random.split(key, 5)

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
            ep_networks = dict(
                actor=ep_actor,
                sa_encoder=ep_sa_encoder,
                g_encoder=ep_g_encoder,
            )

            # Update GCP networks
            new_gcp_actor_state, new_gcp_alpha_state, gcp_actor_metrics = update_actor_and_alpha(
                context, gcp_networks, gcp_transitions, 
                training_state.gcp_actor_state, training_state.gcp_critic_state, training_state.gcp_alpha_state,
                gcp_actor_key
            )
            new_gcp_critic_state, gcp_critic_metrics = update_critic(
                context, gcp_networks, gcp_transitions, training_state.gcp_critic_state, gcp_critic_key
            )
            
            # Update EP networks
            new_ep_actor_state, new_ep_alpha_state, ep_actor_metrics = update_actor_and_alpha(
                context, ep_networks, ep_transitions,
                training_state.ep_actor_state, training_state.ep_critic_state, training_state.ep_alpha_state,
                ep_actor_key
            )
            new_ep_critic_state, ep_critic_metrics = update_critic(
                context, ep_networks, ep_transitions, training_state.ep_critic_state, ep_critic_key
            )
            
            # Update training state
            training_state = training_state.replace(
                gcp_actor_state=new_gcp_actor_state,
                gcp_critic_state=new_gcp_critic_state,
                gcp_alpha_state=new_gcp_alpha_state,
                ep_actor_state=new_ep_actor_state,
                ep_critic_state=new_ep_critic_state,
                ep_alpha_state=new_ep_alpha_state,
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
                "ep_entropy": ep_actor_metrics["entropy"],
                "ep_actor_loss": ep_actor_metrics["actor_loss"],
                "ep_alpha_loss": ep_actor_metrics["alpha_loss"],
                "ep_log_alpha": ep_actor_metrics["log_alpha"],
                "ep_categorical_accuracy": ep_critic_metrics["categorical_accuracy"],
                "ep_logits_pos": ep_critic_metrics["logits_pos"],
                "ep_logits_neg": ep_critic_metrics["logits_neg"],
                "ep_logsumexp": ep_critic_metrics["logsumexp"],
                "ep_critic_loss": ep_critic_metrics["critic_loss"],
            }

            return (training_state, key), metrics

        @jax.jit
        def training_step(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            reset_key, experience_keys, sampling_keys, permute_keys, training_keys = jax.random.split(key, 5)
            
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps
            total_chunks = (num_goal_conditioned_steps + num_exploratory_steps) // unroll_length
            
            # Split keys for each chunk
            experience_keys = jax.random.split(experience_keys, total_chunks)
            sampling_keys = jax.random.split(sampling_keys, total_chunks)
            permute_keys = jax.random.split(permute_keys, total_chunks)
            training_keys = jax.random.split(training_keys, total_chunks)
            
            def collect_and_update_chunk(carry, chunk_idx):
                (training_state_inner, env_state_inner, main_buffer_state_inner, 
                 gcp_buffer_state_inner, ep_buffer_state_inner) = carry
                
                # Collect experience chunk
                (env_state_new, main_buffer_state_new, gcp_buffer_state_new, ep_buffer_state_new,
                 collected_transitions, updated_max_traj_id) = get_experience_chunk(
                    training_state_inner.gcp_actor_state,
                    training_state_inner.ep_actor_state,
                    env_state_inner,
                    training_state_inner,
                    main_buffer_state_inner,
                    gcp_buffer_state_inner,
                    ep_buffer_state_inner,
                    experience_keys[chunk_idx]
                )
                
                # Update env_steps and max_traj_id
                training_state_inner = training_state_inner.replace(
                    env_steps=training_state_inner.env_steps + env_steps_per_actor_step,
                    max_traj_id=updated_max_traj_id,
                )
                
                # Sample transitions for training
                main_buffer_state_sampled, gcp_transitions = main_replay_buffer.sample(main_buffer_state_new)
                main_buffer_state_new = main_buffer_state_sampled
                
                if self.train_ep_on_main_buffer:
                    ep_transitions = gcp_transitions
                else:
                    ep_buffer_state_sampled, ep_transitions = ep_replay_buffer.sample(ep_buffer_state_new)
                    ep_buffer_state_new = ep_buffer_state_sampled
                
                # Process transitions
                batch_keys = jax.random.split(sampling_keys[chunk_idx], gcp_transitions.observation.shape[0] + ep_transitions.observation.shape[0])
                gcp_batch_keys = batch_keys[:gcp_transitions.observation.shape[0]]
                ep_batch_keys = batch_keys[gcp_transitions.observation.shape[0]:]
                
                def process_transitions(transitions, batch_keys_inner, permute_key_inner):
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
                
                gcp_last_batch = process_transitions(gcp_transitions, gcp_batch_keys, permute_keys[chunk_idx])
                ep_last_batch = process_transitions(ep_transitions, ep_batch_keys, permute_keys[chunk_idx])
                
                # Update networks (scan over batches)
                ((training_state_updated, _), metrics) = jax.lax.scan(
                    lambda carry, xs: update_networks(carry, xs[0], xs[1]),
                    (training_state_inner, training_keys[chunk_idx]),
                    (gcp_last_batch, ep_last_batch)
                )
                
                carry_new = (training_state_updated, env_state_new, main_buffer_state_new,
                            gcp_buffer_state_new, ep_buffer_state_new)
                
                return carry_new, (metrics, collected_transitions)
            
            # Run collection and updates for all chunks
            initial_carry = (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state)
            
            (training_state_final, env_state_final, main_buffer_state_final,
             gcp_buffer_state_final, ep_buffer_state_final), (all_metrics, all_collected) = jax.lax.scan(
                collect_and_update_chunk,
                initial_carry,
                jnp.arange(total_chunks)
            )
            
            # Reset environments after collecting all experience with globally unique trajectory IDs
            reset_keys = jax.random.split(reset_key, config.num_envs)
            env_state_final = jax.jit(train_env.reset)(reset_keys)
            
            # Generate globally unique trajectory IDs
            current_max_id = training_state_final.max_traj_id
            new_traj_ids = current_max_id + jnp.arange(1, config.num_envs + 1, dtype=jnp.int32)
            new_max_id = new_traj_ids[-1]  # Update max_traj_id
            
            final_info = dict(env_state_final.info)
            final_info["traj_id"] = new_traj_ids
            final_info["gc_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            final_info["ep_proposed_goals"] = jnp.zeros((config.num_envs, len(train_env.goal_indices)))
            final_info["policy_phase"] = jnp.ones((config.num_envs,), dtype=bool)
            final_info["needs_reset"] = jnp.zeros((config.num_envs,), dtype=bool)
            final_info["gc_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
            final_info["ep_steps_taken"] = jnp.zeros((config.num_envs,), dtype=jnp.int32)
            env_state_final = env_state_final.replace(info=final_info)
            
            # Update training_state with new max_traj_id
            training_state_final = training_state_final.replace(max_traj_id=new_max_id)
            
            # Combine all collected transitions for visualization
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
        
        def visualize_goals(train_env, transitions, wandb_key):
            # Convert JAX arrays to numpy
            obs = np.array(transitions.observation)
            last_traj_state = np.array(transitions.extras["last_traj_state"][:, :, :state_size])
            
            # Reshape obs to (total_samples, obs_dim)
            obs_flat = obs.reshape(-1, obs.shape[-1])
            
            # Extract phase and goal data
            in_gc_phase = np.array(transitions.extras["state_extras"]["in_gc_phase"])
            in_ep_phase = np.array(transitions.extras["state_extras"]["in_ep_phase"])
            gc_proposed_goals = np.array(transitions.extras["state_extras"]["gc_proposed_goals"])
            ep_proposed_goals = np.array(transitions.extras["state_extras"]["ep_proposed_goals"])
            traj_ids = np.array(transitions.extras["state_extras"]["traj_id"])
            
            # Reshape to flat
            in_gc_phase_flat = in_gc_phase.reshape(-1)
            in_ep_phase_flat = in_ep_phase.reshape(-1)
            gc_proposed_goals_flat = gc_proposed_goals.reshape(-1, len(train_env.goal_indices))
            ep_proposed_goals_flat = ep_proposed_goals.reshape(-1, len(train_env.goal_indices))
            traj_ids_flat = traj_ids.reshape(-1)
            
            unique_traj_ids = np.unique(traj_ids_flat)
            logging.info(f"Visualization: Found {len(unique_traj_ids)} unique trajectory IDs")
            
            # Extract trajectory data
            gc_final_states = []
            ep_final_states = []
            start_states = []
            gc_goals = []
            ep_goals = []
            gc_intermediate_states_list = []
            ep_intermediate_states_list = []
            gc_step_counts = []  # Track number of GCP steps per trajectory
            ep_step_counts = []  # Track number of EP steps per trajectory
            
            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_flat == traj_id
                traj_indices = np.sort(np.where(traj_mask)[0])
                
                if len(traj_indices) == 0:
                    continue
                
                # Start state
                start_state = obs_flat[traj_indices[0]][train_env.goal_indices]
                start_states.append(start_state)
                
                # GC phase data
                gc_mask = in_gc_phase_flat[traj_indices] > 0.5
                gc_indices = traj_indices[gc_mask]
                gc_step_counts.append(len(gc_indices))  # Track GCP step count
                
                if len(gc_indices) > 0:
                    gc_final_idx = gc_indices[-1]
                    gc_final_state = obs_flat[gc_final_idx][train_env.goal_indices]
                    gc_goal = gc_proposed_goals_flat[gc_final_idx]
                    gc_final_states.append(gc_final_state)
                    gc_goals.append(gc_goal)
                
                    # Intermediate states
                    gc_states = obs_flat[gc_indices][:, train_env.goal_indices]
                    if len(gc_states) > 0:
                        indices = np.linspace(0, len(gc_states) - 1, 6).astype(int)
                        gc_intermediate_states_list.append(gc_states[indices])
                    else:
                        gc_intermediate_states_list.append(np.zeros((6, len(train_env.goal_indices))))
                else:
                    gc_intermediate_states_list.append(np.zeros((6, len(train_env.goal_indices))))
                
                # EP phase data
                ep_mask = in_ep_phase_flat[traj_indices] > 0.5
                ep_indices = traj_indices[ep_mask]
                ep_step_counts.append(len(ep_indices))  # Track EP step count
                
                if len(ep_indices) > 0:
                    ep_final_idx = ep_indices[-1]
                    ep_final_state = obs_flat[ep_final_idx][train_env.goal_indices]
                    ep_goal = ep_proposed_goals_flat[ep_final_idx]
                    ep_final_states.append(ep_final_state)
                    ep_goals.append(ep_goal)
                
                    # Intermediate states
                    ep_states = obs_flat[ep_indices][:, train_env.goal_indices]
                    if len(ep_states) > 0:
                        indices = np.linspace(0, len(ep_states) - 1, 6).astype(int)
                        ep_intermediate_states_list.append(ep_states[indices])
                    else:
                        ep_intermediate_states_list.append(np.zeros((6, len(train_env.goal_indices))))
                else:
                    ep_final_states.append(np.zeros(len(train_env.goal_indices)))
                    ep_goals.append(np.zeros(len(train_env.goal_indices)))
                    ep_intermediate_states_list.append(np.zeros((6, len(train_env.goal_indices))))
            
            # Convert to arrays
            start_states = np.array(start_states)
            gc_final_states = np.array(gc_final_states) if gc_final_states else np.zeros((0, len(train_env.goal_indices)))
            ep_final_states = np.array(ep_final_states) if ep_final_states else np.zeros((0, len(train_env.goal_indices)))
            gc_goals = np.array(gc_goals) if gc_goals else np.zeros((0, len(train_env.goal_indices)))
            ep_goals = np.array(ep_goals) if ep_goals else np.zeros((0, len(train_env.goal_indices)))
            gc_step_counts = np.array(gc_step_counts)
            ep_step_counts = np.array(ep_step_counts)
            
            # Filter to only trajectories with both GCP and EP phases (complete trajectories)
            # This ensures we visualize full trajectories, not partial ones with zeros
            complete_traj_mask = (gc_step_counts > 0) & (ep_step_counts > 0)
            complete_indices = np.where(complete_traj_mask)[0]
            
            # Sample trajectories for visualization (only complete ones)
            if len(complete_indices) > 0:
                num_complete_trajs = len(complete_indices)
                num_viz_trajs = min(4, num_complete_trajs)
                # Sample from complete trajectories only
                sampled_complete_indices = np.random.choice(complete_indices, num_viz_trajs, replace=False)
                
                start_xy = start_states[sampled_complete_indices]
                gc_final_xy = gc_final_states[sampled_complete_indices]
                ep_final_xy = ep_final_states[sampled_complete_indices]
                gc_proposed_goals_xy = gc_goals[sampled_complete_indices]
                ep_proposed_goals_xy = ep_goals[sampled_complete_indices]
                
                gc_intermediate_xy_list = [gc_intermediate_states_list[i] for i in sampled_complete_indices]
                ep_intermediate_xy_list = [ep_intermediate_states_list[i] for i in sampled_complete_indices]
                
                # Get step counts for sampled trajectories
                gc_step_counts_sampled = [gc_step_counts[i] for i in sampled_complete_indices]
                ep_step_counts_sampled = [ep_step_counts[i] for i in sampled_complete_indices]
                
                logging.info(f"Visualization: Plotting {num_viz_trajs} complete trajectories (out of {num_complete_trajs} with both GCP and EP phases)")
            
                # Visualize trajectories
                visualize_dual_crl_trajectories_2d(
                    start_xy, gc_final_xy, ep_final_xy, gc_proposed_goals_xy, ep_proposed_goals_xy,
                    gc_intermediate_xy_list, ep_intermediate_xy_list, f"{wandb_key}/dual_crl_trajectories",
                        x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds,
                        gc_step_counts=gc_step_counts_sampled, ep_step_counts=ep_step_counts_sampled
                )
            else:
                logging.info(f"Visualization: No complete trajectories (with both GCP and EP phases) available yet. Skipping trajectory plot.")
            
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
            
            if len(ep_goals) > 0:
                visualize_scatter_sample(
                    ep_goals, "EP Goal Proposals", f"{wandb_key}/ep_goal_proposals_scatter",
                    train_env.x_bounds, train_env.y_bounds
                )
            
            logging.info(f"Plotted visualizations at env step {training_state.env_steps.item()}")
            
        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, prefill_key
        )

        # Setting up evaluators
        key, eval_gcp_key, eval_ep_key = jax.random.split(key, 3)
        gcp_evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_gcp_key,
        )
        
        ep_evaluator = ActorEvaluator(
            deterministic_ep_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_ep_key,
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

            # Run evaluations
            metrics = gcp_evaluator.run_evaluation(training_state, metrics)
            
            ep_eval_metrics = ep_evaluator.run_evaluation(training_state, {})
            for metric_key, metric_value in ep_eval_metrics.items():
                if metric_key.startswith("eval/"):
                    new_key = metric_key.replace("eval/", "ep_eval/")
                    metrics[new_key] = metric_value
            
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
                    training_state.ep_alpha_state.params,
                    training_state.ep_actor_state.params,
                    training_state.ep_critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
            else:
                params = None

        total_steps = current_step

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics