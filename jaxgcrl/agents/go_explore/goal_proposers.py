from typing import Callable, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jaxgcrl.agents.go_explore.utils import sample_trajectories_from_buffer


class GoalProposerState:
    """Container for goal proposer state that can be updated over time.
    
    This allows goal proposers to access training state (buffer, critic params, etc.)
    that changes during training. The state is stored in a mutable dict that closures
    can access.
    
    Usage example for a goal proposer that needs training state:
    
        def create_buffer_based_proposer(env, num_envs, goal_proposer_state):
            def propose_goal(rng: jax.Array) -> jnp.ndarray:
                # Access state from the container
                buffer_state = goal_proposer_state.get('buffer_state')
                critic_params = goal_proposer_state.get('critic_params')
                replay_buffer = goal_proposer_state.get('replay_buffer')
                
                # Use state to propose goal (e.g., sample from buffer, use critic, etc.)
                # ... goal proposal logic ...
                return goal
            
            return propose_goal
        
        # In training loop, update state periodically:
        goal_proposer_state.update(
            buffer_state=buffer_state,
            critic_params=training_state.critic_state.params,
            replay_buffer=replay_buffer,
        )
    
    Note: The goal proposer function will be JIT-compiled, so make sure any state
    accessed from the container is JAX-compatible (JAX arrays, pytrees, etc.).
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
    
    def update(self, **kwargs):
        """Update state values."""
        self._state.update(kwargs)
    
    def get(self, key: str, default: Any = None):
        """Get a state value."""
        return self._state.get(key, default)
    
    def __getitem__(self, key: str):
        return self._state[key]
    
    def __setitem__(self, key: str, value: Any):
        self._state[key] = value


def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
) -> Callable:
    """
    Factory function to create a goal proposer function.
    
    Args:
        goal_proposer_name: Name of the goal proposer to create
        env: The environment instance
        num_envs: Number of parallel environments
        state_size: Size of state dimension (required for random_final_states)
        goal_indices: Indices in state that represent the goal (required for random_final_states)
        
    Returns:
        A goal proposer function that takes (rng, transitions_sample) and returns goal.
        transitions_sample is passed from state.info (Brax pattern, JIT-compatible).
    """
    if goal_proposer_name == "random_env_goals":
        proposer_fn = create_random_env_goals_proposer(env, num_envs)
        # Wrap to take (rng, transitions_sample) and return goal for consistency
        def wrapped_proposer(rng, transitions_sample):
            goal = proposer_fn(rng)
            # For non-buffer proposers, transitions_sample is ignored
            return goal
        return wrapped_proposer
    elif goal_proposer_name == "random_final_states":
        return create_random_final_states_proposer(env, num_envs, state_size, goal_indices)
    else:
        raise ValueError(f"Unknown goal proposer: {goal_proposer_name}")


def create_random_env_goals_proposer(
    env,
    num_envs: int,
) -> Callable[[jax.Array], jnp.ndarray]:
    """
    Creates a goal proposer that samples random goals from the environment's possible_goals.
    
    This is a simple proposer that doesn't need any additional state.
    
    Args:
        env: The environment instance
        num_envs: Number of parallel environments (not used, kept for API consistency)
        
    Returns:
        A function that takes a single rng key and returns a single goal (goal_dim,).
        This will be vmap'd by the training wrapper.
    """
    possible_goals = env.possible_goals  # Shape: (num_goals, goal_dim)
    num_goals = possible_goals.shape[0]  # Use .shape[0] for JIT compatibility
    
    def propose_goal(rng: jax.Array) -> jnp.ndarray:
        """
        Proposes a random goal for a single environment.
        
        Args:
            rng: Single random key (will be vmap'd by training wrapper)
            
        Returns:
            Goal array, shape (goal_dim,)
        """
        idx = jax.random.randint(rng, (), 0, num_goals)
        goal = possible_goals[idx]  # Shape: (goal_dim,)
        return goal
    
    return propose_goal


def create_random_final_states_proposer(
    env,
    num_envs: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
) -> Callable[[jax.Array, Any], jnp.ndarray]:
    """
    Creates a goal proposer that selects random goals from final states in a raw replay buffer sample.
    The sample is provided as an argument (sampled outside this function).
    Falls back to random env goals if sample is not available.
    
    Args:
        env: The environment instance
        num_envs: Number of parallel environments (not used, kept for API consistency)
        state_size: Size of state dimension (required)
        goal_indices: Indices in state that represent the goal (required)
        
    Returns:
        A function that takes (rng, transitions_sample) and returns goal.
        transitions_sample is a raw Transition object from replay_buffer.sample().
    """
    # Create fallback random env goals proposer
    random_env_goals_proposer = create_random_env_goals_proposer(env, num_envs)
    
    # If state_size or goal_indices are not available, return a function that always falls back
    if state_size is None or goal_indices is None:
        def propose_goal(rng: jax.Array, transitions_sample: Any) -> jnp.ndarray:
            return random_env_goals_proposer(rng)
        return propose_goal
    
    def propose_goal(rng: jax.Array, transitions_sample: Any) -> jnp.ndarray:
        """Returns goal.
        
        Args:
            rng: Random key
            transitions_sample: Raw Transition object from replay_buffer.sample()
                            Always provided (even if dummy with all zeros)
        
        The transitions_sample is a raw replay buffer sample with no preprocessing.
        We extract final states from it here. If it's a dummy (all zeros), we'll fallback.
        """
        # Extract final states from raw transitions sample
        # transitions_sample.observation shape: (num_envs, episode_length, obs_size)
        # transitions_sample.extras["state_extras"]["traj_id"] shape: (num_envs, episode_length)
        # transitions_sample.extras["state_extras"]["truncation"] shape: (num_envs, episode_length)
        
        # Flatten to (num_envs * episode_length, obs_size)
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        traj_id_flat = jnp.reshape(transitions_sample.extras["state_extras"]["traj_id"], (-1,))
        truncation_flat = jnp.reshape(transitions_sample.extras["state_extras"]["truncation"], (-1,))
        
        # Check if this is a dummy transition (all zeros) using JAX operations
        # This is JIT-compatible
        is_dummy = jnp.all(obs_flat == 0)
        
        def use_final_states(rng):
            # Extract positions from observations (first state_size elements contain state)
            positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, goal_dim)
            
            # Find final positions using JAX operations (JIT-compatible)
            # Strategy: For each unique trajectory ID, find the first truncation point or last state
            # jnp.unique() needs concrete size when called inside jax.lax.cond()
            # Maximum possible unique values is the length of traj_id_flat
            max_unique_size = traj_id_flat.shape[0]
            # Use fill_value=-1 (invalid trajectory ID) to pad unique_traj_ids
            # This ensures we can distinguish padding from valid trajectory IDs
            fill_value = jnp.array(-1.0, dtype=traj_id_flat.dtype)
            unique_traj_ids = jnp.unique(traj_id_flat, size=max_unique_size, fill_value=fill_value)
            # Count actual number of unique trajectories (non-padding values)
            # Filter out fill_value (-1) which indicates padding
            num_trajs = jnp.sum((unique_traj_ids >= 0).astype(jnp.int32))
            
            # Use scan to process each trajectory (JIT-compatible)
            # goal_indices is a Python tuple, so len() is fine (evaluated at Python level)
            goal_dim = len(goal_indices)
            
            def process_traj(carry, traj_id):
                # Skip processing if traj_id is the fill_value (padding from jnp.unique)
                is_valid_traj_id = traj_id >= 0
                
                # Find all indices for this trajectory
                traj_mask = jnp.where(is_valid_traj_id, traj_id_flat == traj_id, False)
                num_elems = traj_id_flat.shape[0]  # Use .shape[0] instead of len() for JIT compatibility
                indices = jnp.arange(num_elems)
                traj_indices = jnp.where(traj_mask, indices, -1)
                
                # Get valid indices (where mask is True)
                valid_mask = traj_indices >= 0
                num_valid = jnp.sum(valid_mask.astype(jnp.int32))
                has_data = (num_valid > 0) & is_valid_traj_id
                
                # Get valid indices (use first index as fallback)
                first_valid_idx = jnp.argmax(valid_mask.astype(jnp.int32))
                last_valid_idx = num_elems - 1 - jnp.argmax(valid_mask[::-1].astype(jnp.int32))
                
                # Get truncations for this trajectory
                traj_truncations = jnp.where(traj_mask, truncation_flat, False)
                
                # Find first truncation index
                truncation_idx = jnp.where(traj_truncations, indices, -1)
                first_truncation = jnp.max(truncation_idx)
                
                # If truncation found, use it; otherwise use last valid index
                final_idx = jnp.where(
                    first_truncation >= 0,
                    first_truncation,
                    jnp.where(has_data, last_valid_idx, 0)
                )
                final_idx = jnp.clip(final_idx, 0, positions.shape[0] - 1)
                
                # Get final position (zero if no data or invalid traj_id)
                # goal_dim is a Python int constant, so it's fine in JIT
                final_position = jnp.where(
                    has_data,
                    positions[final_idx],
                    jnp.zeros(goal_dim)
                )
                
                return carry, final_position
            
            # Process all trajectories
            _, final_positions = jax.lax.scan(process_traj, None, unique_traj_ids)
            
            # Filter out zero positions (empty trajectories) and sample
            has_data_mask = jnp.any(final_positions != 0, axis=1)
            num_valid = jnp.sum(has_data_mask.astype(jnp.int32))
            
            # Use JAX conditional for JIT compatibility
            # If no valid positions, fallback to random env goals
            # Otherwise, sample from valid final positions
            def sample_from_finals(rng):
                # Create cumulative sum to find idx-th valid position
                # has_data_mask is (num_trajs,), we want to find the idx-th True value
                cumsum = jnp.cumsum(has_data_mask.astype(jnp.int32))
                # Sample random index (0 to num_valid-1)
                idx = jax.random.randint(rng, (), 0, num_valid)
                # Find the first position where cumsum == idx + 1 (the idx-th valid position)
                # Use argmax to find first occurrence
                target = idx + 1
                matches = (cumsum == target).astype(jnp.int32)
                valid_idx = jnp.argmax(matches)
                return final_positions[valid_idx]
            
            def fallback_goal(rng):
                return random_env_goals_proposer(rng)
            
            # Use conditional to choose between sampling from finals or fallback
            goal = jax.lax.cond(
                num_valid > 0,
                sample_from_finals,
                fallback_goal,
                rng
            )
            
            return goal
        
        def fallback_to_random(rng):
            return random_env_goals_proposer(rng)
        

        is_dummy = False
        # Use JAX conditional to check if transitions_sample is a dummy
        # If it's a dummy (all zeros), fallback to random env goals
        goal = jax.lax.cond(
            is_dummy,
            fallback_to_random,
            use_final_states,
            rng
        )
        
        return goal
    
    return propose_goal
