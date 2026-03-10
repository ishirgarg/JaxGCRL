from typing import Callable, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jaxgcrl.agents.go_explore.utils import sample_trajectories_from_buffer



def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
) -> Callable:
    """
    Factory function to create a goal proposer function.
    
    Args:
        goal_proposer_name: Name of the goal proposer to create
        env: The environment instance
        num_envs: Number of parallel environments
        state_size: Size of state dimension (required for rb)
        goal_indices: Indices in state that represent the goal (required for rb)
        num_candidates: Number of candidate goals to filter before final selection
        
    Returns:
        A goal proposer function that takes (rng, start_state, info) and returns goal.
        This will be partialized by the wrapper to only need (rng, start_state).
        The entire info dict is passed through (not vmapped over).
    """
    if goal_proposer_name == "random_env_goals":
        proposer_fn = create_random_env_goals_proposer(env, num_envs)
        # Wrap to take (rng, start_state, info) - start_state and info ignored
        def wrapped_proposer(rng: jax.Array, start_state: jnp.ndarray, info: Dict[str, Any]) -> jnp.ndarray:
            goal = proposer_fn(rng)
            return goal
        return wrapped_proposer
    elif goal_proposer_name == "rb":
        return create_rb_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices)
    else:
        raise ValueError(f"Unknown goal proposer: {goal_proposer_name}")


def create_random_env_goals_proposer(
    env,
    num_envs: int,
) -> Callable[[jax.Array], jnp.ndarray]:
    possible_goals = env.possible_goals  # Shape: (num_goals, goal_dim)
    num_goals = possible_goals.shape[0]  # Use .shape[0] for JIT compatibility
    
    def propose_goal(rng: jax.Array) -> jnp.ndarray:
        idx = jax.random.randint(rng, (), 0, num_goals)
        goal = possible_goals[idx]  # Shape: (goal_dim,)
        return goal
    
    return propose_goal




def create_rb_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
) -> Callable[[jax.Array, jnp.ndarray, Dict[str, Any]], jnp.ndarray]:
    def propose_goal(rng: jax.Array, start_state: jnp.ndarray, info: Dict[str, Any]) -> jnp.ndarray:
        # Extract transitions_sample from info dict
        transitions_sample = info.get('transitions_sample')
        
        # transitions_sample.observation shape: (num_envs, episode_length, obs_size)
        # Flatten to (num_envs * episode_length, obs_size)
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, goal_dim)
        
        # First select num_candidates random states, then randomly select from those
        num_states = positions.shape[0]
        rng1, rng2 = jax.random.split(rng, 2)
        candidate_indices = jax.random.randint(rng1, (num_candidates,), 0, num_states)
        candidate_positions = positions[candidate_indices]  # (num_candidates, goal_dim)
        
        # Randomly select one from candidates
        idx = jax.random.randint(rng2, (), 0, num_candidates)
        goal = candidate_positions[idx]
        
        return goal
    
    return propose_goal
