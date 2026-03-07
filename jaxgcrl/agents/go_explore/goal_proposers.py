from typing import Callable

import jax
import jax.numpy as jnp


def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
) -> Callable[[jax.Array], jnp.ndarray]:
    """
    Factory function to create a goal proposer function.
    
    Args:
        goal_proposer_name: Name of the goal proposer to create
        env: The environment instance
        num_envs: Number of parallel environments
        
    Returns:
        A goal proposer function that takes a single rng key and returns a single goal (goal_dim,).
        This will be vmap'd by the training wrapper.
    """
    if goal_proposer_name == "random_env_goals":
        return create_random_env_goals_proposer(env, num_envs)
    else:
        raise ValueError(f"Unknown goal proposer: {goal_proposer_name}")


def create_random_env_goals_proposer(
    env,
    num_envs: int,
) -> Callable[[jax.Array], jnp.ndarray]:
    """
    Creates a goal proposer that samples random goals from the environment's possible_goals.
    
    Args:
        env: The environment instance
        num_envs: Number of parallel environments (not used, kept for API consistency)
        
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
