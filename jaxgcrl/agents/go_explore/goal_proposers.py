from typing import Callable, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jaxgcrl.agents.go_explore.utils import sample_trajectories_from_buffer



def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    num_candidates: int,
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
    elif goal_proposer_name == "rb":
        return create_rb_goal_proposer(env, num_envs, state_size, goal_indices, num_candidates)
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
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    num_candidates: int,
) -> Callable[[jax.Array, Any], jnp.ndarray]:
    def propose_goal(rng: jax.Array, transitions_sample: Any) -> jnp.ndarray:
        # transitions_sample.observation shape: (num_envs, episode_length, obs_size)
        # Flatten to (num_envs * episode_length, obs_size)
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, goal_dim)
        
        # First select num_candidates random states, then randomly select from those
        num_states = positions.shape[0]
        num_to_sample = jnp.minimum(num_candidates, num_states)
        rng1, rng2 = jax.random.split(rng, 2)
        candidate_indices = jax.random.randint(rng1, (num_to_sample,), 0, num_states)
        candidate_positions = positions[candidate_indices]  # (num_candidates, goal_dim)
        
        # Randomly select one from candidates
        idx = jax.random.randint(rng2, (), 0, num_to_sample)
        goal = candidate_positions[idx]
        
        return goal
    
    return propose_goal
