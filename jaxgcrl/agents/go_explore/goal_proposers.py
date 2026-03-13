from typing import Callable, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxgcrl.agents.go_explore.utils import sample_trajectories_from_buffer
from jaxgcrl.agents.go_explore.algorithms_utils import reconstruct_full_critic_params
from jaxgcrl.agents.go_explore.types import GoalProposerState



def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
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
        actor: Optional actor network object (for goal proposers that need to sample actions)
        critic: Optional critic network object (for goal proposers that need to compute values)
        
    Returns:
        A goal proposer function that takes (rng, start_obs, goal_proposer_state) and returns (goal, updated_state).
        The goal proposer state can be read from and written to.
    """
    if goal_proposer_name == "random_env_goals":
        proposer_fn = create_random_env_goals_proposer(env, num_envs)
        # Wrap to take (rng, start_obs, goal_proposer_state) - start_obs and state ignored
        def wrapped_proposer(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
            goal = proposer_fn(rng)
            # Return empty log_data dict (no visualization for random goals)
            log_data = {}
            return goal, goal_proposer_state, log_data
        return wrapped_proposer
    elif goal_proposer_name == "rb":
        return create_rb_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic)
    elif goal_proposer_name == "q_epistemic":
        return create_q_epistemic_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic)
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
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    def propose_goal(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
        # Extract transitions_sample from goal proposer state
        transitions_sample = goal_proposer_state.transitions_sample
        
        # transitions_sample.observation shape: (num_envs, episode_length, obs_size)
        # Flatten to (num_envs * episode_length, obs_size)
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, jnp.array(goal_indices)]  # (N, goal_dim)
        
        # First select num_candidates random states, then randomly select from those
        num_states = positions.shape[0]
        rng1, rng2 = jax.random.split(rng, 2)
        candidate_indices = jax.random.randint(rng1, (num_candidates,), 0, num_states)
        candidate_positions = positions[candidate_indices]  # (num_candidates, goal_dim)
        
        # Randomly select one from candidates
        idx = jax.random.randint(rng2, (), 0, num_candidates)
        goal = candidate_positions[idx]
        
        # Return empty log_data dict (no visualization for rb proposer)
        log_data = {}
        return goal, goal_proposer_state, log_data
    
    return propose_goal


def create_q_epistemic_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """
    Create a goal proposer that selects goals with highest Q-value variance across the ensemble.
    """
    def propose_goal(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
        # Extract required data from goal proposer state
        transitions_sample = goal_proposer_state.transitions_sample
        actor_params = goal_proposer_state.actor_params
        critic_params = goal_proposer_state.critic_params
        
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, jnp.array(goal_indices)]  # (N, goal_dim)
        
        # Randomly sample num_candidates goals from all states
        num_states = positions.shape[0]
        rng, sample_rng = jax.random.split(rng)
        candidate_indices = jax.random.randint(sample_rng, (num_candidates,), 0, num_states)
        candidate_goals = positions[candidate_indices]  # (num_candidates, goal_dim)
    
        s0 = start_obs[:state_size]  # Shape: (state_size,)
        goal_dim = len(goal_indices)
        
        # Reconstruct full critic params using utility function
        full_critic_params = reconstruct_full_critic_params(critic_params)
        
        # For each candidate goal, compute Q-value mean and std
        def compute_q_stats_for_goal(candidate_goal, rng_key):
            """Compute Q-value mean and std for a single candidate goal."""
            # Construct observation: obs = [s0, g] where s0 is from start_obs and g is candidate_goal
            # Observation structure is [state, goal], so concatenate state with candidate goal
            obs = jnp.concatenate([s0, candidate_goal], axis=-1)  # Shape: (obs_size,)
            
            # Sample action deterministically from policy
            rng_key, action_key = jax.random.split(rng_key)
            action = actor.sample_actions(
                actor_params,
                obs[None, :],  # Add batch dimension: (1, obs_size)
                action_key,
                is_deterministic=True
            )  # Shape: (1, action_size)
            action = action[0]  # Remove batch dimension: (action_size,)
            
            # Compute Q-values using critic
            q_values = critic.apply(
                full_critic_params,
                obs[None, :],  # Add batch dimension: (1, obs_size)
                action[None, :]  # Add batch dimension: (1, action_size)
            )  # Shape: (1, n_critics)
            q_values = q_values[0]  # Remove batch dimension: (n_critics,)
            
            # Compute mean and std across the ensemble
            q_mean = jnp.mean(q_values)
            q_std = jnp.std(q_values)
            
            return q_mean, q_std
        
        # Compute mean and std for all candidate goals
        rng, var_rng = jax.random.split(rng)
        var_keys = jax.random.split(var_rng, num_candidates)
        q_means, q_stds = jax.vmap(compute_q_stats_for_goal)(candidate_goals, var_keys)  # Both shape: (num_candidates,)
        
        # Compute variance from std for selection (variance = std^2)
        variances = q_stds ** 2
        
        # Select goal with highest variance
        best_idx = jnp.argmax(variances)
        selected_goal = candidate_goals[best_idx]  # Shape: (goal_dim,)
        
        # Prepare log_data dict with visualization data
        first_obs_position = s0[jnp.array(goal_indices)]  # Shape: (goal_dim,)
        log_data = {
            'candidate_goals': candidate_goals,        # (num_candidates, goal_dim)
            'first_obs_position': first_obs_position,  # (goal_dim,)
            'q_means': q_means,                        # (num_candidates,)
            'q_stds': q_stds,                          # (num_candidates,)
            'selected_goal': selected_goal,            # (goal_dim,)
        }
        
        return selected_goal, goal_proposer_state, log_data
    
    return propose_goal
