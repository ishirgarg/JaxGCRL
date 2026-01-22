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
from brax.training import types
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
from .goals import GoalProposer, ReplayBufferGoalProposal, MediumEnergyGoalProposal, MetricPreservationGoalProposal, FisherTraceGoalProposal, QEpistemicGoalProposal, MEGAGoalProposal, OMEGAGoalProposal, UCGRGoalProposal,DISCOVERGoalProposal, mix_goals

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

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len wheree a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index
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
            # Mask for same trajectory AND future timesteps (including current)
            same_traj_mask = traj_ids == traj_ids[i]
            future_mask = jnp.arange(seq_len) >= i
            mask = same_traj_mask & future_mask

            # Get sorted valid indices for future steps
            indices = jnp.where(mask, jnp.arange(seq_len), seq_len)
            sorted_indices = jnp.sort(indices)
            num_future = jnp.sum(mask)

            # Compute evenly spaced fractional positions in (0, 1)
            # e.g. for 2 → [1/3, 2/3], for 3 → [1/4, 1/2, 3/4]
            fractions = (jnp.arange(1, num_intermediate + 1) / (num_intermediate + 1))

            # Map fractions to integer positions within the valid range
            idxs = jnp.floor(fractions * num_future).astype(jnp.int32)
            idxs = jnp.clip(idxs, 0, jnp.maximum(num_future - 1, 0))

            # Gather actual indices in the trajectory
            actual_idxs = sorted_indices[idxs]

            # Get the corresponding future states (with padding for no valid futures)
            def get_state(idx):
                return jnp.where(num_future > 0, obs[idx], jnp.zeros(obs_dim))

            states = jax.vmap(get_state)(actual_idxs)
            return states
                
        return jax.vmap(functools.partial(intermediate_states_for_t, num_intermediate=6))(jnp.arange(seq_len))

    traj_ids = transition.extras["state_extras"]["traj_id"]  # shape (seq_len,)
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
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            "was_proposed_goal_mask": jnp.squeeze(transition.extras["state_extras"]["was_proposed_goal_mask"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "proposed_goals": proposed_goals[:-1],
        "last_traj_state": last_traj_state[:-1],
        "intermediate_traj": intermediate_traj[:-1],
        "last_traj_state_mask": last_traj_state_mask[:-1],
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
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

    goal_proposer_name = "dual_crl"

    # Goal proposer names for go-explore style algorithms
    gcp_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals"] = "gcp_final_rb"
    ep_goal_proposer_name: Literal["gcp_final_rb", "ep_final_rb", "env_goals"] = "ep_final_rb"

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
        key, buffer_key, eval_env_key, env_key, gcp_actor_key, gcp_sa_key, gcp_g_key, ep_actor_key, ep_sa_key, ep_g_key = jax.random.split(key, 10)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)
        
        # Initialize info dict (proposed goals will be set by goal proposers in get_experience)
        initial_info = dict(env_state.info)
        initial_info["was_proposed_goal_mask"] = jnp.zeros((config.num_envs,))
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
                    "was_proposed_goal_mask": 0.0,
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

        from jaxgcrl.agents.crl.proposers import FinalReplayBufferProposer, RandomEnvironmentGoalProposer
        
        # Initialize goal proposer instances
        gcp_final_rb_proposer = FinalReplayBufferProposer()
        ep_final_rb_proposer = FinalReplayBufferProposer()
        env_goals_proposer = RandomEnvironmentGoalProposer()
        
        # Create goal proposer functions that select the right buffer based on name
        # These functions are defined here but will be called inside JIT-compiled code
        # The proposer names are known at Python time, so we can use if/else here
        # Functions return (proposed_goals, updated_gcp_buffer_state, updated_ep_buffer_state)
        # to handle cases where one proposer might update the other's buffer
        if self.gcp_goal_proposer_name == "gcp_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(gcp_replay_buffer, gcp_buffer_state, env, env_state, key)
                return proposed_goals, updated_gcp, ep_buffer_state
        elif self.gcp_goal_proposer_name == "ep_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(ep_replay_buffer, ep_buffer_state, env, env_state, key)
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.gcp_goal_proposer_name == "env_goals":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, _ = env_goals_proposer.propose_goals(None, gcp_buffer_state, env, env_state, key)
                return proposed_goals, gcp_buffer_state, ep_buffer_state
        else:
            raise ValueError(f"Unknown gcp_goal_proposer_name: {self.gcp_goal_proposer_name}")
        
        if self.ep_goal_proposer_name == "gcp_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(gcp_replay_buffer, gcp_buffer_state, env, env_state, key)
                return proposed_goals, updated_gcp, ep_buffer_state
        elif self.ep_goal_proposer_name == "ep_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(ep_replay_buffer, ep_buffer_state, env, env_state, key)
                return proposed_goals, gcp_buffer_state, updated_ep
        elif self.ep_goal_proposer_name == "env_goals":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key):
                proposed_goals, _ = env_goals_proposer.propose_goals(None, ep_buffer_state, env, env_state, key)
                return proposed_goals, gcp_buffer_state, ep_buffer_state
        else:
            raise ValueError(f"Unknown ep_goal_proposer_name: {self.ep_goal_proposer_name}")

        def deterministic_actor_step(actor_state, env, env_state, proposed_goals, extra_fields):
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

            # nstate.obs has shape (batch_size, obs_dim)
            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # Note: We handle early termination by tracking which environments have terminated
        # and filtering them out when inserting into replay buffers, but still completing
        # the rollout for all environments.

        # Debug logging callback - message codes map to log messages
        # Only accepts JAX arrays (message code as int, then values)
        MESSAGE_CODES = {
            0: "=== GET_EXPERIENCE START ===",
            1: "=== GET_EXPERIENCE END ===",
            2: "GC Goal Proposal",
            3: "EP Goal Proposal",
            4: "Starting GC Rollout",
            5: "GC Rollout Complete",
            6: "GC Rollout Step",
            7: "Starting EP Rollout",
            8: "EP Rollout Complete",
            9: "EP Rollout Step",
            10: "Trajectory Filtering",
            11: "Buffer Insertion Complete",
            12: "=== TRAINING STEP START ===",
            13: "=== TRAINING STEP END ===",
            14: "After Experience Collection",
            15: "After Sampling Transitions",
            16: "Before Policy Updates",
            17: "After GC Policy Update",
            18: "After EP Policy Update",
            19: "=== UPDATE_NETWORKS START ===",
            20: "=== UPDATE_NETWORKS END ===",
        }
        
        def debug_log(msg_code, *values):
            """Log debug message with values from JIT-compiled code.
            msg_code: int array (scalar) - message code
            values: JAX arrays to log
            """
            msg_code_int = int(np.array(msg_code))
            msg = MESSAGE_CODES.get(msg_code_int, f"UNKNOWN MESSAGE CODE {msg_code_int}")
            logging.info(f"[JAX DEBUG] {msg}")
            for i, val in enumerate(values):
                val_np = np.array(val)
                if val_np.size <= 10:
                    logging.info(f"  Value {i}: shape={val_np.shape}, value={val_np}")
                else:
                    logging.info(f"  Value {i}: shape={val_np.shape}, min={np.min(val_np):.4f}, max={np.max(val_np):.4f}, mean={np.mean(val_np):.4f}")
        
        @jax.jit
        def get_experience(gcp_actor_state, ep_actor_state, env_state, training_state, 
                          main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            """Collect experience with two sequential rollouts.
            
            First: GC policy (deterministic) for num_goal_conditioned_steps
            Second: EP policy (non-deterministic) for num_exploratory_steps
            Environments may reset during rollout - we track this for logging.
            All trajectories (including incomplete ones) are inserted into buffers.
            """
            num_envs = config.num_envs
            
            # Track initial traj_ids to detect mid-rollout resets
            initial_traj_ids = env_state.info["traj_id"]  # shape: (num_envs,)
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps
            
            # Log start of get_experience
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(0, dtype=jnp.int32),  # Message code 0
                training_state.env_steps
            )
            
            # ===== FIRST ROLLOUT: Goal-Conditioned Policy (Deterministic) =====
            # Propose GC goals at start
            gc_key, ep_key = jax.random.split(key, 2)
            gc_proposed_goals, gcp_buffer_state, ep_buffer_state = gcp_propose_goals(
                gcp_buffer_state, ep_buffer_state, train_env, env_state, gc_key
            )
            
            # Log GC goal proposal
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(2, dtype=jnp.int32),  # Message code 2
                gc_proposed_goals,
                jnp.mean(gc_proposed_goals, axis=0),
                jnp.std(gc_proposed_goals, axis=0)
            )
            
            # Update env_state with GC goals
            new_info = dict(env_state.info)
            new_info["gc_proposed_goals"] = gc_proposed_goals
            env_state = env_state.replace(
                obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(gc_proposed_goals),
                info=new_info
            )

            def gc_rollout_step(carry, unused_t):
                env_state, gcp_actor_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                
                # Get GC proposed goals from env_state.info
                gc_proposed_goals = env_state.info.get("gc_proposed_goals", env_state.obs[:, -len(train_env.goal_indices):])
                
                # Use deterministic actor step
                nstate, transition = deterministic_actor_step(
                    gcp_actor_state, train_env, env_state, gc_proposed_goals, ("truncation", "traj_id")
                )
                
                # Check for early termination (done or truncation) - for logging only
                truncation = transition.extras["state_extras"]["truncation"]
                terminated = (transition.discount < 1.0) | (truncation > 0.5)
                
                # Log GC step details
                jax.experimental.io_callback(
                    debug_log,
                    None,
                    jnp.array(6, dtype=jnp.int32),  # Message code 6
                    jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2],  # First 2 state dims
                    jnp.mean(gc_proposed_goals, axis=0),
                    jnp.sum(terminated),
                    jnp.mean(transition.reward),
                    jnp.mean(transition.discount)
                )
                
                # Update env_state (environment already handles termination correctly)
                new_info = dict(nstate.info)
                new_info["gc_proposed_goals"] = gc_proposed_goals
                env_state = nstate.replace(info=new_info)
                
                # Mark transition as GC phase and include proposed goals and termination info
                # Preserve existing state_extras and add new fields
                existing_state_extras = transition.extras["state_extras"]
                transition_extras = {
                    **existing_state_extras,
                    "in_gc_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                    "in_ep_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                    "gc_proposed_goals": gc_proposed_goals,
                    "ep_proposed_goals": gc_proposed_goals,  # Use GC goals as placeholder (EP goals not set yet)
                    "terminated": terminated.astype(jnp.float32),
                }
                transition = transition._replace(extras={"state_extras": transition_extras})
                
                return (env_state, gcp_actor_state, next_key), transition
            
            # Log before GC rollout
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(4, dtype=jnp.int32),  # Message code 4
                jnp.array(num_goal_conditioned_steps, dtype=jnp.int32),
                jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2]
            )
            
            # Run GC rollout
            (env_state, _, _), gc_transitions = jax.lax.scan(
                gc_rollout_step,
                (env_state, gcp_actor_state, gc_key),
                (),
                length=num_goal_conditioned_steps
            )
            
            # Log after GC rollout
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(5, dtype=jnp.int32),  # Message code 5
                jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2],
                jnp.array(gc_transitions.observation.shape[0], dtype=jnp.int32),  # num_steps
                jnp.array(gc_transitions.observation.shape[1], dtype=jnp.int32),  # num_envs
                jnp.mean(gc_transitions.reward)
            )
            
            # ===== SECOND ROLLOUT: Exploratory Policy (Non-Deterministic) =====
            # Propose EP goals at start of exploratory phase
            ep_proposed_goals, gcp_buffer_state, ep_buffer_state = ep_propose_goals(
                gcp_buffer_state, ep_buffer_state, train_env, env_state, ep_key
            )
            
            # Log EP goal proposal
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(3, dtype=jnp.int32),  # Message code 3
                ep_proposed_goals,
                jnp.mean(ep_proposed_goals, axis=0),
                jnp.std(ep_proposed_goals, axis=0)
            )
            
            # Update env_state with EP goals
            new_info = dict(env_state.info)
            new_info["ep_proposed_goals"] = ep_proposed_goals
            # Preserve GC proposed goals from previous phase (should always exist)
            new_info["gc_proposed_goals"] = env_state.info.get("gc_proposed_goals", ep_proposed_goals)
            env_state = env_state.replace(
                obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(ep_proposed_goals),
                info=new_info
            )

            def ep_rollout_step(carry, unused_t):
                env_state, ep_actor_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                
                # Get EP proposed goals from env_state.info
                ep_proposed_goals = env_state.info.get("ep_proposed_goals", env_state.obs[:, -len(train_env.goal_indices):])
                
                # Use non-deterministic actor step
                nstate, transition = actor_step(
                    ep_actor_state, train_env, env_state, ep_proposed_goals, 
                    current_key, ("truncation", "traj_id")
                )
                
                # Check for early termination (done or truncation) - for logging only
                truncation = transition.extras["state_extras"]["truncation"]
                terminated = (transition.discount < 1.0) | (truncation > 0.5)
                
                # Log EP step details
                jax.experimental.io_callback(
                    debug_log,
                    None,
                    jnp.array(9, dtype=jnp.int32),  # Message code 9
                    jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2],  # First 2 state dims
                    jnp.mean(ep_proposed_goals, axis=0),
                    jnp.sum(terminated),
                    jnp.mean(transition.reward),
                    jnp.mean(transition.discount)
                )
                
                # Update env_state (environment already handles termination correctly)
                new_info = dict(nstate.info)
                new_info["ep_proposed_goals"] = ep_proposed_goals
                # Preserve GC proposed goals if they exist (should always exist)
                new_info["gc_proposed_goals"] = env_state.info.get("gc_proposed_goals", ep_proposed_goals)
                env_state = nstate.replace(info=new_info)
                
                # Mark transition as EP phase and include proposed goals and termination info
                # Preserve existing state_extras and add new fields
                existing_state_extras = transition.extras["state_extras"]
                # Get GC proposed goals (should always exist since GC phase sets them)
                gc_proposed_goals_for_transition = env_state.info.get("gc_proposed_goals", ep_proposed_goals)
                transition_extras = {
                    **existing_state_extras,
                    "in_gc_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                    "in_ep_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                    "gc_proposed_goals": gc_proposed_goals_for_transition,
                    "ep_proposed_goals": ep_proposed_goals,
                    "terminated": terminated.astype(jnp.float32),
                }
                transition = transition._replace(extras={"state_extras": transition_extras})
                
                return (env_state, ep_actor_state, next_key), transition
            
            # Log before EP rollout
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(7, dtype=jnp.int32),  # Message code 7
                jnp.array(num_exploratory_steps, dtype=jnp.int32),
                jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2]
            )
            
            # Run EP rollout
            (env_state, _, _), ep_transitions = jax.lax.scan(
                ep_rollout_step,
                (env_state, ep_actor_state, ep_key),
                (),
                length=num_exploratory_steps
            )
            
            # Log after EP rollout
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(8, dtype=jnp.int32),  # Message code 8
                jnp.mean(env_state.obs[:, :train_env.state_dim], axis=0)[:2],
                jnp.array(ep_transitions.observation.shape[0], dtype=jnp.int32),  # num_steps
                jnp.array(ep_transitions.observation.shape[1], dtype=jnp.int32),  # num_envs
                jnp.mean(ep_transitions.reward)
            )
            
            # ===== COMBINE TRANSITIONS AND INSERT INTO BUFFERS =====
            # Concatenate GC and EP transitions along time dimension
            # gc_transitions: (num_goal_conditioned_steps, num_envs, ...)
            # ep_transitions: (num_exploratory_steps, num_envs, ...)
            # combined: (num_goal_conditioned_steps + num_exploratory_steps, num_envs, ...)
            # The replay buffer expects shape (unroll_length, num_envs, ...) - DO NOT reshape!
            combined_transitions = jax.tree_util.tree_map(
                lambda gc, ep: jnp.concatenate([gc, ep], axis=0),
                gc_transitions, ep_transitions
            )
            
            # Track mid-rollout resets by comparing initial and final traj_ids
            final_traj_ids = env_state.info["traj_id"]  # shape: (num_envs,)
            reset_during_rollout = initial_traj_ids != final_traj_ids  # shape: (num_envs,)
            num_reset_during_rollout = jnp.sum(reset_during_rollout)
            
            # Log reset statistics
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(10, dtype=jnp.int32),  # Message code 10
                num_reset_during_rollout,
                jnp.array(num_envs, dtype=jnp.int32)
            )
            
            # Insert all trajectories into buffers (including incomplete ones)
            # Data shape should be (unroll_length, num_envs, ...) as expected by replay buffer
            main_buffer_state = main_replay_buffer.insert(main_buffer_state, combined_transitions)
            gcp_buffer_state = gcp_replay_buffer.insert(gcp_buffer_state, gc_transitions)
            ep_buffer_state = ep_replay_buffer.insert(ep_buffer_state, ep_transitions)
            
            # Log buffer insertion
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(11, dtype=jnp.int32),  # Message code 11
                main_replay_buffer.size(main_buffer_state),
                gcp_replay_buffer.size(gcp_buffer_state),
                ep_replay_buffer.size(ep_buffer_state),
                jnp.array(num_envs, dtype=jnp.int32)
            )
            
            # Log end of get_experience
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(1, dtype=jnp.int32),  # Message code 1
                training_state.env_steps
            )
            
            return env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state

        def prefill_replay_buffer(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key = carry
                key, reset_key, new_key = jax.random.split(key, 3)
                env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state = get_experience(
                    training_state.gcp_actor_state,
                    training_state.ep_actor_state,
                    env_state,
                    training_state,
                    main_buffer_state,
                    gcp_buffer_state,
                    ep_buffer_state,
                    key,
                )
                
                # Reset environments after collecting experience
                reset_keys = jax.random.split(reset_key, config.num_envs)
                env_state = jax.jit(train_env.reset)(reset_keys)
                # Initialize info dict (proposed goals will be set by goal proposers in get_experience)
                initial_info = dict(env_state.info)
                initial_info["was_proposed_goal_mask"] = jnp.zeros((config.num_envs,))
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
            
            # Log update_networks start
            gcp_shape_0 = jnp.array(gcp_transitions.observation.shape[0] if hasattr(gcp_transitions, 'observation') else 0, dtype=jnp.int32)
            gcp_shape_1 = jnp.array(gcp_transitions.observation.shape[1] if hasattr(gcp_transitions, 'observation') and len(gcp_transitions.observation.shape) > 1 else 0, dtype=jnp.int32)
            ep_shape_0 = jnp.array(ep_transitions.observation.shape[0] if hasattr(ep_transitions, 'observation') else 0, dtype=jnp.int32)
            ep_shape_1 = jnp.array(ep_transitions.observation.shape[1] if hasattr(ep_transitions, 'observation') and len(ep_transitions.observation.shape) > 1 else 0, dtype=jnp.int32)
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(19, dtype=jnp.int32),  # Message code 19
                gcp_shape_0,
                gcp_shape_1,
                ep_shape_0,
                ep_shape_1
            )
            
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

            training_state, gcp_actor_metrics = update_actor_and_alpha(
                context, gcp_networks, gcp_transitions, training_state, gcp_actor_key
            )
            training_state, gcp_critic_metrics = update_critic(
                context, gcp_networks, gcp_transitions, training_state, gcp_critic_key
            )
            training_state, ep_actor_metrics = update_actor_and_alpha(
                context, ep_networks, ep_transitions, training_state, ep_actor_key
            )
            training_state, ep_critic_metrics = update_critic(
                context, ep_networks, ep_transitions, training_state, ep_critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            for key, value in gcp_actor_metrics.items():
                metrics[f"gcp_{key}"] = value
            for key, value in gcp_critic_metrics.items():
                metrics[f"gcp_{key}"] = value
            for key, value in ep_actor_metrics.items():
                metrics[f"ep_{key}"] = value
            for key, value in ep_critic_metrics.items():
                metrics[f"ep_{key}"] = value
            
            # Log update_networks end
            jax.experimental.io_callback(
                debug_log,
                None,
                jnp.array(20, dtype=jnp.int32),  # Message code 20
                training_state.gradient_steps,
                jnp.array(len(metrics), dtype=jnp.int32)
            )

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key):
            experience_key1, reset_key, sampling_key, permute_key, training_key = jax.random.split(key, 5)

            # update buffer
            env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state = get_experience(
                training_state.gcp_actor_state,
                training_state.ep_actor_state,
                env_state,
                training_state,
                main_buffer_state,
                gcp_buffer_state,
                ep_buffer_state,
                experience_key1,
            )
            
            # Reset environments after collecting experience
            reset_keys = jax.random.split(reset_key, config.num_envs)
            env_state = jax.jit(train_env.reset)(reset_keys)
            # Initialize info dict (proposed goals will be set by goal proposers in get_experience)
            initial_info = dict(env_state.info)
            initial_info["was_proposed_goal_mask"] = jnp.zeros((config.num_envs,))
            env_state = env_state.replace(info=initial_info)

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            main_buffer_state, gcp_transitions = main_replay_buffer.sample(main_buffer_state)
            ep_buffer_state, ep_transitions = ep_replay_buffer.sample(ep_buffer_state)
            # transitions.observation has shape (num_envs, episode_length, obs_dim)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, gcp_transitions.observation.shape[0] + ep_transitions.observation.shape[0])
            gcp_batch_keys = batch_keys[:gcp_transitions.observation.shape[0]]
            ep_batch_keys = batch_keys[gcp_transitions.observation.shape[0]:]

            def process_transitions(transitions, batch_keys):
                transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                # transitions.observation has shape (num_envs, episode_length, obs_dim)

                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
                )
                # Shape of obs is (num_envs * episode_length, obs_dim) after flattening

                # permute transitions
                permutation = jax.random.permutation(permute_key, len(transitions.observation))
                transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
                transitions = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                    transitions,
                )
                return transitions
            
            gcp_last_batch = process_transitions(gcp_transitions, gcp_batch_keys)
            ep_last_batch = process_transitions(ep_transitions, ep_batch_keys)

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), gcp_last_batch, ep_last_batch)

            return (
                training_state,
                env_state,
                main_buffer_state,
                gcp_buffer_state,
                ep_buffer_state,
                (gcp_last_batch, ep_last_batch)
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
                        last_batch
                    ),
                    metrics,
                ) = training_step(ts, es, mbs, gcbs, ebs, train_key)
                # Keep last_batch in carry to avoid stacking all batches in memory
                return (ts, es, mbs, gcbs, ebs, k, last_batch), metrics

            # Run one step to get initial last_batch structure for carry
            key, first_key = jax.random.split(key)
            ((training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, init_batch), first_metrics) = training_step(
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, first_key
            )

            (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _, last_batch), rest_metrics = jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, key, init_batch),
                (),
                length=num_training_steps_per_epoch - 1,
            )

            # Combine metrics from first step with rest
            metrics = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a[None], b]),
                first_metrics,
                rest_metrics,
            )
            return training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, last_batch
        
        def visualize_goals(train_env, transitions, actor_state, critic_state, sa_encoder, g_encoder, energy_fn, num_samples, wandb_key):
            # Shape is now (episode_len-1, batch_size, ...) since we only keep the last training step's batch
            obs = transitions.observation # (episode_len-1, batch_size, obs_dim)
            last_traj_state = transitions.extras["last_traj_state"][:, :, :state_size] # (episode_len-1, batch_size, state_size)
            last_traj_state_flat = last_traj_state.reshape(-1, state_size)
            intermediate_traj = transitions.extras["intermediate_traj"] # (episode_len-1, batch_size, num_intermediate_states, obs_dim)
            
            states = obs[:, :, :state_size].reshape(-1, state_size)
            
            # Extract GC and EP specific data
            in_gc_phase = transitions.extras["state_extras"]["in_gc_phase"] # (episode_len-1, batch_size)
            in_ep_phase = transitions.extras["state_extras"]["in_ep_phase"] # (episode_len-1, batch_size)
            gc_proposed_goals = transitions.extras["state_extras"]["gc_proposed_goals"] # (episode_len-1, batch_size, goal_dim)
            ep_proposed_goals = transitions.extras["state_extras"]["ep_proposed_goals"] # (episode_len-1, batch_size, goal_dim)
            traj_ids = transitions.extras["state_extras"]["traj_id"] # (episode_len-1, batch_size)
            
            # Reshape to (total_samples, ...)
            in_gc_phase_flat = in_gc_phase.reshape(-1)
            in_ep_phase_flat = in_ep_phase.reshape(-1)
            gc_proposed_goals_flat = gc_proposed_goals.reshape(-1, len(train_env.goal_indices))
            ep_proposed_goals_flat = ep_proposed_goals.reshape(-1, len(train_env.goal_indices))
            traj_ids_flat = traj_ids.reshape(-1)
            
            # Flatten intermediate trajectories to shape (total_samples, num_intermediate_states, obs_dim)
            intermediate_traj_flat = intermediate_traj.reshape(-1, intermediate_traj.shape[-2], intermediate_traj.shape[-1])
            
            # Extract GC and EP final states for each trajectory
            # For each unique trajectory, find the last GC and EP states
            unique_traj_ids = np.unique(traj_ids_flat)
            gc_final_states = []
            ep_final_states = []
            start_states = []
            gc_goals = []
            ep_goals = []
            intermediate_states_list = []
            
            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_flat == traj_id
                traj_indices = np.sort(np.where(traj_mask)[0])
                
                # Get start state (first observation in trajectory)
                start_state = states[traj_indices[0], train_env.goal_indices]
                start_states.append(start_state)
                
                # Find last GC phase transition (where in_gc_phase > 0.5)
                gc_mask = in_gc_phase_flat[traj_indices] > 0.5
                gc_indices = traj_indices[gc_mask]
                gc_final_idx = gc_indices[-1]
                gc_final_state = states[gc_final_idx, train_env.goal_indices]
                gc_goal = gc_proposed_goals_flat[gc_final_idx]
                gc_final_states.append(gc_final_state)
                gc_goals.append(gc_goal)
                
                # Find last EP phase transition (where in_ep_phase > 0.5)
                ep_mask = in_ep_phase_flat[traj_indices] > 0.5
                ep_indices = traj_indices[ep_mask]
                ep_final_idx = ep_indices[-1]
                ep_final_state = states[ep_final_idx, train_env.goal_indices]
                ep_goal = ep_proposed_goals_flat[ep_final_idx]
                ep_final_states.append(ep_final_state)
                ep_goals.append(ep_goal)
                
                # Get intermediate states for this trajectory
                # The intermediate states are computed from the first transition and span the full trajectory
                # We need to determine which intermediate states belong to GC vs EP phase
                first_transition_intermediates = intermediate_traj_flat[traj_indices[0], :, train_env.goal_indices]  # (num_intermediate, 2)
                
                # The intermediate states correspond to future states in the trajectory
                # We need to map them back to actual trajectory indices to check their phase
                # Since get_intermediate_trajectory_states samples evenly across future trajectory,
                # we can approximate by checking phases at corresponding positions
                # For simplicity, we'll split based on the ratio of GC to EP steps in this trajectory
                num_gc_steps = np.sum(in_gc_phase_flat[traj_indices] > 0.5)
                num_ep_steps = np.sum(in_ep_phase_flat[traj_indices] > 0.5)
                total_steps = len(traj_indices)
                
                # Split intermediate states proportionally
                num_intermediate = first_transition_intermediates.shape[0]
                gc_intermediate_count = int(num_intermediate * num_gc_steps / total_steps)
                gc_intermediate = first_transition_intermediates[:gc_intermediate_count]
                ep_intermediate = first_transition_intermediates[gc_intermediate_count:]
                
                # Store both GC and EP intermediate states
                intermediate_states_list.append({
                    'gc': gc_intermediate,
                    'ep': ep_intermediate
                })
            
            # Convert to numpy arrays
            start_states = np.array(start_states) # (num_trajs, 2)
            gc_final_states = np.array(gc_final_states) # (num_trajs, 2)
            ep_final_states = np.array(ep_final_states) # (num_trajs, 2)
            gc_goals = np.array(gc_goals) # (num_trajs, 2)
            ep_goals = np.array(ep_goals) # (num_trajs, 2)
            intermediate_states = np.array(intermediate_states_list) # (num_trajs, num_intermediate, 2)
            
            # Sample exactly 4 trajectories for 2x2 grid visualization
            num_trajs = start_states.shape[0]
            num_viz_trajs = min(4, num_trajs)
            sample_indices = np.random.choice(num_trajs, num_viz_trajs, replace=False)
            
            start_xy = start_states[sample_indices]
            gc_final_xy = gc_final_states[sample_indices]
            ep_final_xy = ep_final_states[sample_indices]
            gc_proposed_goals_xy = gc_goals[sample_indices]
            ep_proposed_goals_xy = ep_goals[sample_indices]
            intermediate_xy = intermediate_states[sample_indices]
            
            # Visualize trajectories
            visualize_dual_crl_trajectories_2d(
                start_xy, gc_final_xy, ep_final_xy, gc_proposed_goals_xy, ep_proposed_goals_xy,
                intermediate_xy, f"{wandb_key}/dual_crl_trajectories",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # Heatmaps
            # 1. GC final states
            visualize_kde_heatmap(
                gc_final_states, "GC Final States", f"{wandb_key}/gc_final_states_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # 2. EP final states
            visualize_kde_heatmap(
                ep_final_states, "EP Final States", f"{wandb_key}/ep_final_states_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # 3. GC goal proposals
            visualize_kde_heatmap(
                gc_goals, "GC Goal Proposals", f"{wandb_key}/gc_goal_proposals_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )
            
            # 4. EP goal proposals
            visualize_kde_heatmap(
                ep_goals, "EP Goal Proposals", f"{wandb_key}/ep_goal_proposals_heatmap",
                x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds
            )

            logging.info(f"Plotted visualizations at env step {training_state.env_steps.item()}")
            
        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, prefill_key
        )

        """Setting up evaluator"""
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

            training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, metrics, last_batch = training_epoch(
                training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, epoch_key
            )

            visualize_goals(train_env, last_batch, training_state.actor_state, training_state.critic_state, sa_encoder, g_encoder, self.energy_fn, num_samples=5, wandb_key="training")

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
                # Save current policy and critic params.
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
        # assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
