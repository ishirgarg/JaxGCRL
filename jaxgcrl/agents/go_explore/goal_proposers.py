from typing import Callable, Dict, Any, Optional

import jax
import jax.numpy as jnp


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
    goal_proposer_state: Optional[GoalProposerState] = None,
) -> Callable[[jax.Array], jnp.ndarray]:
    """
    Factory function to create a goal proposer function.
    
    Args:
        goal_proposer_name: Name of the goal proposer to create
        env: The environment instance
        num_envs: Number of parallel environments
        goal_proposer_state: Optional state container for goal proposers that need
                           training state (buffer, critic params, etc.). The goal
                           proposer closure will capture this and can access
                           updated state via the container.
        
    Returns:
        A goal proposer function that takes a single rng key and returns a single goal (goal_dim,).
        This will be vmap'd by the training wrapper.
        
        For simple proposers (like random_env_goals), the function signature is:
            (rng: jax.Array) -> jnp.ndarray
            
        For complex proposers that need state, the function is a closure that captures
        goal_proposer_state and accesses it when called:
            def propose_goal(rng: jax.Array) -> jnp.ndarray:
                buffer_state = goal_proposer_state.get('buffer_state')
                critic_params = goal_proposer_state.get('critic_params')
                # ... use state to propose goal
                return goal
    """
    if goal_proposer_name == "random_env_goals":
        return create_random_env_goals_proposer(env, num_envs, goal_proposer_state)
    else:
        raise ValueError(f"Unknown goal proposer: {goal_proposer_name}")


def create_random_env_goals_proposer(
    env,
    num_envs: int,
    goal_proposer_state: Optional[GoalProposerState] = None,
) -> Callable[[jax.Array], jnp.ndarray]:
    """
    Creates a goal proposer that samples random goals from the environment's possible_goals.
    
    This is a simple proposer that doesn't need training state, but accepts
    goal_proposer_state for API consistency.
    
    Args:
        env: The environment instance
        num_envs: Number of parallel environments (not used, kept for API consistency)
        goal_proposer_state: Optional state container (not used for this simple proposer)
        
    Returns:
        A function that takes a single rng key and returns a single goal (goal_dim,).
        This will be vmap'd by the training wrapper.
    """
    possible_goals = env.possible_goals  # Shape: (num_goals, goal_dim)
    
    def propose_goal(rng: jax.Array) -> jnp.ndarray:
        """
        Proposes a random goal for a single environment.
        
        Args:
            rng: Single random key (will be vmap'd by training wrapper)
            
        Returns:
            Goal array, shape (goal_dim,)
        """
        idx = jax.random.randint(rng, (), 0, len(possible_goals))
        goal = possible_goals[idx]  # Shape: (goal_dim,)
        return goal
    
    return propose_goal
