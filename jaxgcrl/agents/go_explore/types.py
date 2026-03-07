from typing import Any, NamedTuple, Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState

@dataclass
class Actor:
    @nn.compact
    def __call__(self, x):
        pass

    def init(self, key, x):
        pass

    def sample_actions(self, params, obs, key, is_deterministic: bool):
        pass

    def update(self,  context, networks, transitions, training_state, actor_key):
        pass

    def process_transitions(self, transitions, process_key, batch_size, discounting, state_size, goal_indices, goal_reach_thresh, use_her):
        pass

class Critic:
    @nn.compact
    def __call__(self, obs, actions):
        pass

    def init(self, key, x):
        pass

    def update(self,  context, networks, transitions, training_state, critic_key):
        pass

@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: Optional[TrainState] = None
    # SAC-specific fields
    target_critic_params: Optional[Any] = None  # Params for target Q-network
    normalizer_params: Optional[Any] = None  # Running statistics for observations
    policy_optimizer_state: Optional[Any] = None  # For SAC's policy optimizer
    q_optimizer_state: Optional[Any] = None  # For SAC's Q-network optimizer
    target_policy_params: Optional[Any] = None  # For SAC's target policy (if TD3-like)


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: Optional[jnp.ndarray] = None  # Required for SAC
    extras: jnp.ndarray = ()