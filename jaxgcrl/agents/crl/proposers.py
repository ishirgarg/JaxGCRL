"""Goal proposers for CRL agents."""
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from jaxgcrl.agents.crl.goals_utils import (
    get_final_states_from_batch,
    get_last_state_from_trajectory,
    expand_goal_to_state,
    zero_out_non_goal_indices,
    estimate_log_density_knn,
    should_log_at_interval,
    create_goal_selection_plot,
    create_env_goal_ranking_plot,
)
from jaxgcrl.agents.crl.losses import energy_fn


@dataclass
class FinalReplayBufferProposer:
    """Proposes goals by sampling final states from replay buffer trajectories.
    
    This proposer samples trajectories from the replay buffer and extracts
    the final state of each trajectory as a proposed goal.
    """
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, 
                     actor=None, actor_params=None, critic_params=None, 
                     sa_encoder=None, g_encoder=None, training_state=None):
        """Propose goals from final states of trajectories in replay buffer.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices attribute)
            env_state: Current environment state (unused, but required by interface)
            key: JAX random key (unused, but required by interface)
            actor: Actor network (unused)
            actor_params: Actor parameters (unused)
            critic_params: Critic parameters (unused)
            sa_encoder: State-action encoder network (unused)
            g_encoder: Goal encoder network (unused)
            training_state: Training state (unused)
            
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
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor=None, actor_params=None, critic_params=None,
                     sa_encoder=None, g_encoder=None, training_state=None):
        """Propose goals by sampling random environment goals.
        
        Args:
            replay_buffer: Replay buffer (unused, but required by interface)
            buffer_state: Current buffer state (returned unchanged)
            env: Training environment (must have possible_goals attribute)
            env_state: Current environment state (used to get batch_size)
            key: JAX random key for sampling
            actor: Actor network (unused)
            actor_params: Actor parameters (unused)
            critic_params: Critic parameters (unused)
            sa_encoder: State-action encoder network (unused)
            g_encoder: Goal encoder network (unused)
            training_state: Training state (unused)
            
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


@dataclass
class UCGRProposer:
    """Unsupervised Contrastive Goal-Reaching (UCGR) proposer.
    
    Proposes goals using the MinLSE strategy:
    1. Sample K (s, a) pairs from replay buffer
    2. For each (s, a) pair, find its trajectory's final state as candidate goal
    3. Compute S(g_j) = log Σ_i exp(f(s_i, a_i, g_j)) for each candidate goal
    4. Select g* = argmin_g S(g)
    
    Attributes:
        energy_fn_name: Energy function to use ("dot" for inner product)
        num_samples: Number of (s, a) pairs to sample for MinLSE computation
    """
    energy_fn_name: str = "dot"  # Energy function: f(s,a,g) = φ(s,a)^T ψ(g)
    num_samples: int = 256  # Number of (s, a) pairs to sample
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals using the MinLSE strategy.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices and state_dim attributes)
            env_state: Current environment state (used to get batch_size)
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (must contain 'sa_encoder' and 'g_encoder')
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (unused)
                
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        
        batch_size = env_state.obs.shape[0]
        goal_indices = env.goal_indices
        state_size = env.state_dim
        K = self.num_samples  # Number of (s, a) pairs to sample
        
        # Sample trajectories from replay buffer
        buffer_state, sample_batch = replay_buffer.sample(buffer_state)
        
        # sample_batch.observation has shape (N, ep_len, obs_dim)
        observations = sample_batch.observation  # (N, ep_len, obs_dim)
        actions = sample_batch.action  # (N, ep_len, action_dim)
        traj_ids = sample_batch.extras["state_extras"]["traj_id"]  # (N, ep_len)
        
        N, ep_len = observations.shape[:2]
        
        # Randomly sample K indices from all (trajectory, timestep) pairs
        key, sample_key = jax.random.split(key)
        total_pairs = N * ep_len
        # Sample K random indices (with replacement if K > total_pairs)
        flat_indices = jax.random.randint(sample_key, (K,), 0, total_pairs)
        traj_indices = flat_indices // ep_len  # Which trajectory
        time_indices = flat_indices % ep_len   # Which timestep within trajectory
        
        # Extract the K sampled (s, a) pairs
        sampled_states = observations[traj_indices, time_indices, :state_size]  # (K, state_dim)
        sampled_actions = actions[traj_indices, time_indices]  # (K, action_dim)
        
        # For each sampled (s, a), find the final state of its trajectory
        # First get the traj_id for each sampled pair
        sampled_traj_ids = traj_ids[traj_indices, time_indices]  # (K,)
        
        def get_final_state_for_sample(traj_idx, time_idx, sampled_traj_id):
            """Get the final state of the trajectory containing this (s, a) pair."""
            obs_seq = observations[traj_idx]  # (ep_len, obs_dim)
            traj_id_seq = traj_ids[traj_idx]  # (ep_len,)
            
            # Find the last timestep with the same traj_id
            mask = traj_id_seq == sampled_traj_id
            last_idx = jnp.max(jnp.where(mask, jnp.arange(ep_len), 0))
            return obs_seq[last_idx, goal_indices]  # (goal_dim,)
        
        # Get candidate goals: final states for each sampled (s, a) pair's trajectory
        candidate_goals = jax.vmap(get_final_state_for_sample)(
            traj_indices, time_indices, sampled_traj_ids
        )  # (K, goal_dim)
        
        # Compute MinLSE scores using the K (s, a) pairs
        # For each goal g_j, compute S(g_j) = log Σ_i exp(f(s_i, a_i, g_j))
        
        # Compute state-action encodings φ(s_i, a_i)
        sa_pairs = jnp.concatenate([sampled_states, sampled_actions], axis=-1)  # (K, state_dim + action_dim)
        sa_encodings = sa_encoder.apply(critic_params["sa_encoder"], sa_pairs)  # (K, encoding_dim)
        
        # Compute goal encodings ψ(g_j)
        psi_g = g_encoder.apply(critic_params["g_encoder"], candidate_goals)  # (K, encoding_dim)
        
        # Compute all pairwise energies f(s_i, a_i, g_j) for i, j in [0, K)
        # Result: energies[i, j] = f(s_i, a_i, g_j)
        sa_rep = jnp.repeat(sa_encodings[:, None, :], K, axis=1)  # (K, K, encoding_dim)
        psi_rep = jnp.repeat(psi_g[None, :, :], K, axis=0)  # (K, K, encoding_dim)
        
        sa_flat = sa_rep.reshape(-1, sa_rep.shape[-1])  # (K*K, encoding_dim)
        psi_flat = psi_rep.reshape(-1, psi_rep.shape[-1])  # (K*K, encoding_dim)
        
        energies_flat = energy_fn(self.energy_fn_name, sa_flat, psi_flat)  # (K*K,)
        energies = energies_flat.reshape(K, K)  # (K, K) - energies[i, j] = f(s_i, a_i, g_j)
        
        # Compute scores: S(g_j) = log Σ_i exp(f(s_i, a_i, g_j))
        scores = jax.scipy.special.logsumexp(energies, axis=0)  # (K,)
        
        # Select goal with minimum score: g* = argmin_g S(g)
        min_idx = jnp.argmin(scores)
        proposed_goals = jnp.repeat(
            candidate_goals[min_idx][None, :],
            batch_size,
            axis=0,
        )

        return proposed_goals, buffer_state


@dataclass
class MetricPreservationProposer:
    """Metric Preservation Goal Proposer.
    
    Selects goals that preserve the metric structure by computing energy terms
    for triplets (s, g, h) where s is current state, g is candidate goal, h is environment goal.
    
    Attributes:
        energy_fn_name: Energy function name
        use_one_env_goal: Whether to use one random environment goal per state
        use_kde_correction: Whether to add KDE correction term
        use_waypoint_difficulty: Whether to include waypoint difficulty term
        use_max: If True, simply take max over all (g, h) pairs instead of using logsumexp
        zero_out_cand_goals: Whether to zero out non-goal dimensions in candidate goals
        zero_out_state: Whether to zero out non-goal dimensions in current state
        propose_env_goals: If True, propose environment goals instead of waypoint goals
        goal_sampling_temperature: Temperature for softmax sampling (0 = greedy, >0 = softmax)
        LOG_INTERVAL_STEPS: Log visualizations every N environment steps
    """
    energy_fn_name: str
    use_one_env_goal: bool = False
    use_kde_correction: bool = False
    use_waypoint_difficulty: bool = True
    use_max: bool = False
    zero_out_cand_goals: bool = True
    zero_out_state: bool = False
    propose_env_goals: bool = False
    goal_sampling_temperature: float = 1.0
    LOG_INTERVAL_STEPS: int = 1000000

    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals using metric preservation strategy.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have possible_goals, goal_indices, state_dim attributes)
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (must contain 'sa_encoder' and 'g_encoder')
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (for env_steps logging, optional)
                
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        assert hasattr(env, 'possible_goals'), \
            "Environment must have 'possible_goals' attribute for MetricPreservationProposer."
        
        state_size = env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch, state_dim)

        # --- candidate goals from replay buffer ---
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation

        last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, env.goal_indices]  # (num_candidate_goals, goal_dim)
        candidate_goals_full = last_states[:, :state_size] # Full vector for final states achieved

        if self.zero_out_cand_goals:
            candidate_goals_full = jax.vmap(
                lambda g: expand_goal_to_state(g, state_size, env.goal_indices)
            )(candidate_goals)

        env_goals = env.possible_goals  # (num_env_goals, goal_dim)

        def energy_triplet(state):
            """Compute M[g,h] for a single state and return individual terms."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                state = zero_out_non_goal_indices(state, env.goal_indices)
            
            # Use utility function for KDE estimation
            proposed_goal_densities = estimate_log_density_knn(candidate_goals)
            
            num_cand = candidate_goals.shape[0]
            num_env = env_goals.shape[0]

            # f(s, a1, g)
            s1 = jnp.repeat(state[None, :], num_cand, axis=0)
            obs_sg = jnp.concatenate([s1, candidate_goals], axis=1)
            means, _ = actor.apply(actor_params, obs_sg)
            a1 = jnp.tanh(means)
            phi_sg = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s1, a1], axis=1))
            psi_g = g_encoder.apply(critic_params['g_encoder'], candidate_goals)
            f_sag = energy_fn(self.energy_fn_name, phi_sg, psi_g)  # (num_cand,)

            # f(g, a2, h)
            g_exp = jnp.repeat(candidate_goals_full[:, None, :], num_env, axis=1)  # (num_cand, num_env, state_dim)
            h_exp = jnp.repeat(env_goals[None, :, :], num_cand, axis=0)
            obs_gh = jnp.concatenate([g_exp, h_exp], axis=-1).reshape(num_cand * num_env, -1)
            means2, _ = actor.apply(actor_params, obs_gh)
            a2 = jnp.tanh(means2)
            phi_gh = sa_encoder.apply(critic_params['sa_encoder'],
                                      jnp.concatenate([g_exp.reshape(-1, g_exp.shape[-1]), a2], axis=1))
            psi_h = g_encoder.apply(critic_params['g_encoder'], env_goals)
            psi_h_rep = jnp.repeat(psi_h[None, :, :], num_cand, axis=0).reshape(num_cand * num_env, -1)
            f_gah = energy_fn(self.energy_fn_name, phi_gh, psi_h_rep).reshape(num_cand, num_env)

            # f(s, a3, h)
            s3 = jnp.repeat(state[None, :], num_env, axis=0)
            obs_sh = jnp.concatenate([s3, env_goals], axis=1)
            means3, _ = actor.apply(actor_params, obs_sh)
            a3 = jnp.tanh(means3)
            phi_sh = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s3, a3], axis=1))
            f_sah = energy_fn(self.energy_fn_name, phi_sh, psi_h)  # (num_env,)
            
            # Compute M matrix for goal selection
            term1 = f_sag[:, None]  # f(s, a1, g) - shape (num_cand, 1)
            term2 = f_gah  # f(g, a2, h) - shape (num_cand, num_env)
            term3 = f_sah[None, :]  # -f(s, a3, h) - shape (1, num_env)
            kde_term = proposed_goal_densities[:, None]  # KDE correction - shape (num_cand, 1)
            
            M = term2 - term3
            if self.use_waypoint_difficulty:
                M += term1
            if self.use_kde_correction:
                M += kde_term
            return M
        
        def energy_triplet_with_terms(state):
            """Compute M and all individual terms for visualization (only called for one state)."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                state = zero_out_non_goal_indices(state, env.goal_indices)
            
            num_cand = candidate_goals.shape[0]
            num_env = env_goals.shape[0]

            s1 = jnp.repeat(state[None, :], num_cand, axis=0)
            obs_sg = jnp.concatenate([s1, candidate_goals], axis=1)
            means, _ = actor.apply(actor_params, obs_sg)
            a1 = jnp.tanh(means)
            phi_sg = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s1, a1], axis=1))
            psi_g = g_encoder.apply(critic_params['g_encoder'], candidate_goals)
            f_sag = energy_fn(self.energy_fn_name, phi_sg, psi_g)

            g_exp = jnp.repeat(candidate_goals_full[:, None, :], num_env, axis=1)
            h_exp = jnp.repeat(env_goals[None, :, :], num_cand, axis=0)
            obs_gh = jnp.concatenate([g_exp, h_exp], axis=-1).reshape(num_cand * num_env, -1)
            means2, _ = actor.apply(actor_params, obs_gh)
            a2 = jnp.tanh(means2)
            phi_gh = sa_encoder.apply(critic_params['sa_encoder'],
                                      jnp.concatenate([g_exp.reshape(-1, g_exp.shape[-1]), a2], axis=1))
            psi_h = g_encoder.apply(critic_params['g_encoder'], env_goals)
            psi_h_rep = jnp.repeat(psi_h[None, :, :], num_cand, axis=0).reshape(num_cand * num_env, -1)
            f_gah = energy_fn(self.energy_fn_name, phi_gh, psi_h_rep).reshape(num_cand, num_env)

            s3 = jnp.repeat(state[None, :], num_env, axis=0)
            obs_sh = jnp.concatenate([s3, env_goals], axis=1)
            means3, _ = actor.apply(actor_params, obs_sh)
            a3 = jnp.tanh(means3)
            phi_sh = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s3, a3], axis=1))
            f_sah = energy_fn(self.energy_fn_name, phi_sh, psi_h)

            proposed_goal_densities = estimate_log_density_knn(candidate_goals)
            
            term1 = f_sag[:, None]
            term2 = f_gah
            term3 = f_sah[None, :]
            kde_term = proposed_goal_densities[:, None]
            
            M = term2 - term3
            if self.use_waypoint_difficulty:
                M += term1
            if self.use_kde_correction:
                M += kde_term
            return M, term1, term2, term3, kde_term

        # compute M for all states (only M, not term matrices)
        energy_mats = jax.vmap(energy_triplet)(current_states)  # (batch, num_cand, num_env)
        
        # compute term matrices for ONE state only (for visualization)
        viz_state_idx = 0
        _, term1_single, term2_single, term3_single, kde_single = energy_triplet_with_terms(current_states[viz_state_idx])

        def select_goal_max(M, rand_key):
            """Select goal using softmax sampling over M matrix if temperature > 0, else greedy."""
            if self.goal_sampling_temperature > 0:
                # Softmax sampling: flatten M, compute softmax, sample
                M_flat = M.flatten()
                logits = M_flat / self.goal_sampling_temperature
                probs = jax.nn.softmax(logits)
                idx_flat = jax.random.choice(rand_key, a=M_flat.size, p=probs)
                g_idx, h_idx = jnp.unravel_index(idx_flat, M.shape)
            else:
                # Greedy: take argmax
                idx_flat = jnp.argmax(M)
                g_idx, h_idx = jnp.unravel_index(idx_flat, M.shape)
            return g_idx, h_idx

        def select_goal_minimax(M):
            # Step 1: worst-case slack for each candidate goal over all env goals
            worst_case_slack = jnp.max(M, axis=1)  # shape: (num_candidate_goals,)
            # Step 2: pick the candidate goal with minimal worst-case slack
            g_idx = jnp.argmin(worst_case_slack)
            h_idx = jnp.argmax(M[g_idx, :])
            return g_idx, h_idx
        
        def select_goal_minlogsumexp(M, rand_key):
            score = -jax.scipy.special.logsumexp(M, axis=1)
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmin
                g_idx = jnp.argmin(-score)  # score is negative, so -score is positive
                h_idx = jnp.argmin(M[g_idx])
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key, a=M.shape[0], p=weights)

            h_idx = jnp.argmin(M[g_idx])
            return g_idx, h_idx
        
        def select_goal_minlogsumexp_one_env(M, rand_key):
            """Select one random environment goal and compute weights using only that column."""
            rand_key_h, rand_key_g = jax.random.split(rand_key)

            # Randomly select one environment goal
            num_env_goals = M.shape[1]
            h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
            
            energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
            score = -energies_for_h  # Negative because we want to minimize
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmin
                g_idx = jnp.argmin(-score)  # score is negative, so -score is positive
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx
        
        def select_goal_maxlogsumexp(M, rand_key):
            score = jax.scipy.special.logsumexp(M, axis=1)
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmax
                g_idx = jnp.argmax(score)
                h_idx = jnp.argmax(M[g_idx])
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key, a=M.shape[0], p=weights)

            h_idx = jnp.argmax(M[g_idx])
            return g_idx, h_idx
        
        def select_goal_maxlogsumexp_one_env(M, rand_key):
            """Select one random environment goal and compute weights using only that column."""
            rand_key_h, rand_key_g = jax.random.split(rand_key)

            # Randomly select one environment goal
            num_env_goals = M.shape[1]
            h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
            
            energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
            score = energies_for_h  # Positive because we want to maximize
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmax
                g_idx = jnp.argmax(score)
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx

        # Split key for batch operations
        batch_size = energy_mats.shape[0]
        batch_keys = jax.random.split(key, batch_size + 1)
        key = batch_keys[0]
        batch_keys = batch_keys[1:]

        if self.use_max:
            # Simple max selection over all (g, h) pairs
            best_g_indices, best_h_indices = jax.vmap(select_goal_max)(energy_mats, batch_keys)
        elif self.use_one_env_goal:
            if self.use_waypoint_difficulty:
                best_g_indices, best_h_indices = jax.vmap(select_goal_minlogsumexp_one_env)(energy_mats, batch_keys)
            else:
                best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp_one_env)(energy_mats, batch_keys)
        else:
            if self.use_waypoint_difficulty:
                best_g_indices, best_h_indices = jax.vmap(select_goal_minlogsumexp)(energy_mats, batch_keys) 
            else:
                best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp)(energy_mats, batch_keys)

        # Select proposed goals: either candidate goals (waypoints) or environment goals
        if self.propose_env_goals:
            proposed_goals = env_goals[best_h_indices]  # (batch, goal_dim)
        else:
            proposed_goals = candidate_goals[best_g_indices]      # (batch, goal_dim)

        # Log visualizations only at specified intervals to reduce wandb storage
        if training_state is not None:
            env_steps = training_state.env_steps
        else:
            env_steps = jnp.array(0)  # Default if not provided
            
        jax.experimental.io_callback(
            MetricPreservationProposer._log_goal_selection_viz,
            None,
            current_states,
            candidate_goals,
            env_goals,
            best_g_indices,
            best_h_indices,
            energy_mats,
            term1_single,
            term2_single,
            term3_single,
            kde_single,
            viz_state_idx,
            env_steps,
            env.goal_indices,
            env.x_bounds if hasattr(env, 'x_bounds') else None,
            env.y_bounds if hasattr(env, 'y_bounds') else None,
            self.LOG_INTERVAL_STEPS
        )

        return proposed_goals, buffer_state
    
    # Class variable to track last log step
    @staticmethod
    def _log_goal_selection_viz(current_states, candidate_goals, env_goals, 
                              best_g_indices, best_h_indices, energy_mats, 
                              term1_single, term2_single, term3_single, kde_single, viz_state_idx,
                              env_steps, goal_indices, x_bounds, y_bounds, log_interval_steps):
        """Visualize goal selection showing trajectory from current -> candidate -> env goals."""
        
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'metric_preservation'):
            return
        
        # Use viz_state_idx for env_goal_ranking plot, random for goal_selection plot
        num_states = current_states.shape[0]
        random_state_indices = np.random.choice(num_states, size=min(4, num_states), replace=False)
        # Make sure viz_state_idx is in random_state_indices for consistency
        random_state_indices[0] = int(viz_state_idx)
        
        # Generate both visualizations using shared utilities
        pil_image1 = create_goal_selection_plot(
            current_states, candidate_goals, env_goals, best_g_indices, best_h_indices, energy_mats, 
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        pil_image2 = create_env_goal_ranking_plot(
            current_states, candidate_goals, env_goals, energy_mats, 
            term1_single, term2_single, term3_single, kde_single, viz_state_idx,
            goal_indices, x_bounds, y_bounds
        )
        
        metrics = {
            'metric_preservation/goal_selection_viz': wandb.Image(pil_image1),
            'metric_preservation/env_goal_rankings': wandb.Image(pil_image2),
        }
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class MaxWaypointRatioOneEnvProposer:
    """Max Waypoint Ratio One Environment Goal Proposer.
    
    Similar to MetricPreservationProposer but:
    - Uses one random environment goal per state (use_one_env_goal=True)
    - Does not use waypoint difficulty (use_waypoint_difficulty=False)
    - Uses maxlogsumexp selection strategy
    - Supports temperature-based sampling
    
    Attributes:
        energy_fn_name: Energy function name
        goal_sampling_temperature: Temperature for softmax sampling (0 = greedy, >0 = softmax)
        zero_out_cand_goals: Whether to zero out non-goal dimensions in candidate goals
        zero_out_state: Whether to zero out non-goal dimensions in current state
        propose_env_goals: If True, propose environment goals instead of waypoint goals
        LOG_INTERVAL_STEPS: Log visualizations every N environment steps
    """
    energy_fn_name: str
    goal_sampling_temperature: float = 1.0
    zero_out_cand_goals: bool = True
    zero_out_state: bool = False
    propose_env_goals: bool = False
    LOG_INTERVAL_STEPS: int = 1000000

    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals using max waypoint ratio with one environment goal strategy.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have possible_goals, goal_indices, state_dim attributes)
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (must contain 'sa_encoder' and 'g_encoder')
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (for env_steps logging, optional)
                
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        assert hasattr(env, 'possible_goals'), \
            "Environment must have 'possible_goals' attribute for MaxWaypointRatioOneEnvProposer."
        
        state_size = env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch, state_dim)

        # --- candidate goals from replay buffer ---
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation

        last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, env.goal_indices]  # (num_candidate_goals, goal_dim)
        candidate_goals_full = last_states[:, :state_size] # Full vector for final states achieved

        if self.zero_out_cand_goals:
            candidate_goals_full = jax.vmap(
                lambda g: expand_goal_to_state(g, state_size, env.goal_indices)
            )(candidate_goals)

        env_goals = env.possible_goals  # (num_env_goals, goal_dim)

        def energy_triplet(state):
            """Compute M[g,h] for a single state."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                state = zero_out_non_goal_indices(state, env.goal_indices)
            
            num_cand = candidate_goals.shape[0]
            num_env = env_goals.shape[0]

            # f(g, a2, h) - only term we need (no waypoint difficulty, no KDE)
            g_exp = jnp.repeat(candidate_goals_full[:, None, :], num_env, axis=1)  # (num_cand, num_env, state_dim)
            h_exp = jnp.repeat(env_goals[None, :, :], num_cand, axis=0)
            obs_gh = jnp.concatenate([g_exp, h_exp], axis=-1).reshape(num_cand * num_env, -1)
            means2, _ = actor.apply(actor_params, obs_gh)
            a2 = jnp.tanh(means2)
            phi_gh = sa_encoder.apply(critic_params['sa_encoder'],
                                      jnp.concatenate([g_exp.reshape(-1, g_exp.shape[-1]), a2], axis=1))
            psi_h = g_encoder.apply(critic_params['g_encoder'], env_goals)
            psi_h_rep = jnp.repeat(psi_h[None, :, :], num_cand, axis=0).reshape(num_cand * num_env, -1)
            f_gah = energy_fn(self.energy_fn_name, phi_gh, psi_h_rep).reshape(num_cand, num_env)

            # f(s, a3, h)
            s3 = jnp.repeat(state[None, :], num_env, axis=0)
            obs_sh = jnp.concatenate([s3, env_goals], axis=1)
            means3, _ = actor.apply(actor_params, obs_sh)
            a3 = jnp.tanh(means3)
            phi_sh = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s3, a3], axis=1))
            f_sah = energy_fn(self.energy_fn_name, phi_sh, psi_h)  # (num_env,)
            
            # Compute M matrix: M = f(g, a2, h) - f(s, a3, h)
            M = f_gah - f_sah[None, :]
            return M

        # compute M for all states
        energy_mats = jax.vmap(energy_triplet)(current_states)  # (batch, num_cand, num_env)
        
        def select_goal_maxlogsumexp_one_env(M, rand_key):
            """Select one random environment goal and compute weights using only that column."""
            rand_key_h, rand_key_g = jax.random.split(rand_key)

            # Randomly select one environment goal
            num_env_goals = M.shape[1]
            h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
            
            energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
            score = energies_for_h  # Positive because we want to maximize
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmax
                g_idx = jnp.argmax(score)
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx

        # Split key for batch operations
        batch_size = energy_mats.shape[0]
        batch_keys = jax.random.split(key, batch_size)

        best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp_one_env)(energy_mats, batch_keys)

        # Select proposed goals: either candidate goals (waypoints) or environment goals
        if self.propose_env_goals:
            proposed_goals = env_goals[best_h_indices]  # (batch, goal_dim)
        else:
            proposed_goals = candidate_goals[best_g_indices]      # (batch, goal_dim)

        return proposed_goals, buffer_state
