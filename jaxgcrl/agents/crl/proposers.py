"""Goal proposers for CRL agents."""
from flax.struct import dataclass
import jax
import jax.numpy as jnp

from jaxgcrl.agents.crl.goals_utils import get_final_states_from_batch


@dataclass
class FinalReplayBufferProposer:
    """Proposes goals by sampling final states from replay buffer trajectories.
    
    This proposer samples trajectories from the replay buffer and extracts
    the final state of each trajectory as a proposed goal.
    """
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, **kwargs):
        """Propose goals from final states of trajectories in replay buffer.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices attribute)
            env_state: Current environment state (unused, but required by interface)
            key: JAX random key (unused, but required by interface)
            **kwargs: Additional arguments (unused)
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state
        """
        # Sample trajectories from replay buffer
        buffer_state, sampled_transitions = replay_buffer.sample(buffer_state)
        
        # Extract trajectory information
        # sampled_transitions.observation has shape (num_envs, episode_length, obs_dim)
        # sampled_transitions.extras["state_extras"]["traj_id"] has shape (num_envs, episode_length)
        observations = sampled_transitions.observation
        traj_ids = sampled_transitions.extras["state_extras"]["traj_id"]
        
        # Extract final states from each trajectory using utility function
        proposed_goals = get_final_states_from_batch(observations, traj_ids, env.goal_indices)
        
        return proposed_goals, buffer_state


@dataclass
class RandomEnvironmentGoalProposer:
    """Proposes goals by sampling random goals from environment's possible_goals.
    
    This proposer samples a batch of random goals from the environment's
    possible_goals attribute, which contains all valid goal positions.
    """
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, **kwargs):
        """Propose goals by sampling random environment goals.
        
        Args:
            replay_buffer: Replay buffer (unused, but required by interface)
            buffer_state: Current buffer state (returned unchanged)
            env: Training environment (must have possible_goals attribute)
            env_state: Current environment state (used to get batch_size)
            key: JAX random key for sampling
            **kwargs: Additional arguments (unused)
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state (unchanged)
        """
        assert hasattr(env, 'possible_goals'), \
            "Environment must have 'possible_goals' attribute for RandomEnvironmentGoalProposer."
        
        # Get batch size from env_state
        batch_size = env_state.obs.shape[0]
        
        # Get environment goals
        env_goals = env.possible_goals  # (num_env_goals, goal_dim)
        num_env_goals = env_goals.shape[0]
        
        # Sample random indices for each environment in the batch
        indices = jax.random.randint(key, (batch_size,), 0, num_env_goals)
        
        # Extract goals using sampled indices
        proposed_goals = env_goals[indices]  # (batch_size, goal_dim)
        
        return proposed_goals, buffer_state
