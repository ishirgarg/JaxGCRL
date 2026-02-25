"""Goal proposers for CRL agents."""
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jaxgcrl.agents.crl.losses import energy_fn as _energy_fn


from jaxgcrl.agents.crl.goals_utils import (
    get_final_states_from_batch,
    get_last_state_from_trajectory,
    expand_goal_to_state,
    zero_out_non_goal_indices,
    estimate_log_density_knn,
    should_log_at_interval,
    create_goal_selection_plot,
    create_env_goal_ranking_plot,
    stack_ensemble_params,
    compute_q_values_ensemble,
    create_2x2_scatter_plot,
    gaussian_kernel_density,
    compute_kl_divergence_empirical,
    compute_energy_for_state_goal_pairs,
    sample_candidate_goals_from_replay_buffer,
    propose_random_environment_goals,
    sample_states_from_replay_buffer,
)
from jaxgcrl.agents.crl.losses import energy_fn


@dataclass
class FinalReplayBufferProposer:
    """Proposes goals by sampling final states from replay buffer trajectories.
    
    This proposer samples trajectories from the replay buffer and extracts
    the final state of each trajectory as a proposed goal.
    
    Attributes:
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
    """
    num_rb_goals: int = 256
    candidate_goals_type: str = "final"
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, 
                     actor=None, actor_params=None, critic_params=None, 
                     sa_encoder=None, g_encoder=None, training_state=None):
        """Propose goals from final states of trajectories in replay buffer.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices attribute)
            env_state: Current environment state (used to get batch_size)
            key: JAX random key
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
        batch_size = env_state.obs.shape[0]
        
        # Sample candidate goals using common function
        candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, env, key, self.num_rb_goals, self.candidate_goals_type
        )
        
        # Sample batch_size goals from candidates (with replacement if needed)
        key, sample_key = jax.random.split(key)
        if candidate_goals.shape[0] >= batch_size:
            indices = jax.random.choice(
                sample_key,
                a=candidate_goals.shape[0],
                shape=(batch_size,),
                replace=False
            )
            proposed_goals = candidate_goals[indices]
        else:
            indices = jax.random.choice(
                sample_key,
                a=candidate_goals.shape[0],
                shape=(batch_size,),
                replace=True
            )
            proposed_goals = candidate_goals[indices]
        
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
        proposed_goals = propose_random_environment_goals(env, env_state.obs.shape[0], key)
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
        num_rb_samples: Number of (s, a) pairs to sample for MinLSE computation
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
        LOG_INTERVAL_STEPS: Log visualizations every N environment steps
    """
    energy_fn_name: str = "dot"  # Energy function: f(s,a,g) = φ(s,a)^T ψ(g)
    num_rb_samples: int = 256  # Number of (s, a) pairs to sample
    candidate_goals_type: str = "final"
    LOG_INTERVAL_STEPS: int = 1000000  # Log visualizations every N environment steps
    
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
        K = self.num_rb_samples  # Number of (s, a) pairs to sample
        
        # Sample candidate goals using common function
        key, sample_key = jax.random.split(key)
        candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, env, sample_key, self.num_samples, self.candidate_goals_type
        )
        
        # Sample trajectories from replay buffer for (s, a) pairs
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
        
        # Use candidate_goals directly (they are already sampled according to candidate_goals_type)
        # If we need K candidate goals but have num_rb_goals, we'll use all candidate_goals
        # and pad/repeat if needed
        num_candidate_goals = candidate_goals.shape[0]
        if num_candidate_goals < K:
            # Repeat candidate goals to get K goals
            num_repeats = (K + num_candidate_goals - 1) // num_candidate_goals
            candidate_goals = jnp.tile(candidate_goals, (num_repeats, 1))[:K]
        elif num_candidate_goals > K:
            # Randomly sample K goals from candidates
            key, sample_key = jax.random.split(key)
            indices = jax.random.choice(
                sample_key,
                a=num_candidate_goals,
                shape=(K,),
                replace=False
            )
            candidate_goals = candidate_goals[indices]
        
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

        # Log UCGR statistics
        env_steps = training_state.env_steps
            
        jax.experimental.io_callback(
            UCGRProposer._log_ucgr_statistics,
            None,
            candidate_goals,
            scores,
            min_idx,
            env.goal_indices,
            env_steps,
            self.LOG_INTERVAL_STEPS
        )

        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_ucgr_statistics(candidate_goals, scores, min_idx, goal_indices, env_steps, log_interval_steps):
        """Log UCGR goal selection statistics."""
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'ucgr'):
            return
        
        # candidate_goals: (K, goal_dim) - already contains goal coordinates
        # scores: (K,) MinLSE scores
        # min_idx: index of selected goal
        
        # Create visualization
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all candidate goals colored by their scores
        # candidate_goals already has shape (K, len(goal_indices))
        scatter = ax.scatter(
            candidate_goals[:, 0], candidate_goals[:, 1],
            c=scores, cmap='viridis_r', s=150, alpha=0.7,
            edgecolors='black', linewidths=0.5, label='Candidate Goals'
        )
        plt.colorbar(scatter, ax=ax, label='MinLSE Score (lower is better)')
        
        # Highlight the selected goal
        selected_goal = candidate_goals[min_idx]
        ax.scatter(
            selected_goal[0], selected_goal[1],
            s=300, marker='*', color='red', edgecolors='black',
            linewidths=1.5, label='Selected Goal', zorder=10
        )
        
        ax.set_xlabel('Goal X')
        ax.set_ylabel('Goal Y')
        ax.set_title(f'UCGR Goal Selection (Step {int(env_steps)})\n'
                    f'Selected goal score: {float(scores[min_idx]):.4f}, '
                    f'Min score: {float(jnp.min(scores)):.4f}, '
                    f'Max score: {float(jnp.max(scores)):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_image.load()
        buf.close()
        plt.close(fig)
        
        metrics = {
            'ucgr/goal_selection_viz': wandb.Image(pil_image),
        }
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class MetricPreservationProposer:
    """Metric Preservation Goal Proposer.
    
    Selects goals that preserve the metric structure by computing energy terms
    for triplets (s, g, h) where s is current state, g is candidate goal, h is environment goal.
    
    Attributes:
        energy_fn_name: Energy function name
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
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
    num_rb_goals: int = 256
    candidate_goals_type: str = "final"
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

        # --- candidate goals from replay buffer using common function ---
        key, sample_key = jax.random.split(key)
        candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, env, sample_key, self.num_rb_goals, self.candidate_goals_type
        )  # (num_rb_goals, goal_dim)
        
        # For candidate_goals_full, we need full state vectors
        # Expand goals to states using expand_goal_to_state
        candidate_goals_full = jax.vmap(
            lambda g: expand_goal_to_state(g, state_size, env.goal_indices)
        )(candidate_goals)  # (num_rb_goals, state_dim)
        
        # Note: If zero_out_cand_goals is False, we'd ideally want the actual full states
        # from the replay buffer, but for simplicity we use the expanded version
        # This should work correctly for most use cases

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
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
        goal_sampling_temperature: Temperature for softmax sampling (0 = greedy, >0 = softmax)
        zero_out_cand_goals: Whether to zero out non-goal dimensions in candidate goals
        zero_out_state: Whether to zero out non-goal dimensions in current state
        propose_env_goals: If True, propose environment goals instead of waypoint goals
        filter_successful_waypoints: If True, filter out candidate goals that are within goal_reach_thresh of any environment goal
        LOG_INTERVAL_STEPS: Log visualizations every N environment steps
    """
    energy_fn_name: str
    num_rb_goals: int = 256
    candidate_goals_type: str = "final"
    goal_sampling_temperature: float = 1.0
    zero_out_cand_goals: bool = True
    zero_out_state: bool = False
    propose_env_goals: bool = False
    filter_successful_waypoints: bool = False
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

        # --- candidate goals from replay buffer using common function ---
        key, sample_key = jax.random.split(key)
        candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, env, sample_key, self.num_rb_goals, self.candidate_goals_type
        )  # (num_rb_goals, goal_dim)
        
        env_goals = env.possible_goals  # (num_env_goals, goal_dim)
        
        # Filter out successful waypoints if requested
        if self.filter_successful_waypoints:
            assert hasattr(env, 'goal_reach_thresh'), \
                "Environment must have 'goal_reach_thresh' attribute for filter_successful_waypoints=True."
            
            # For each candidate goal, check if it's within goal_reach_thresh of any environment goal
            # candidate_goals: (num_rb_goals, goal_dim)
            # env_goals: (num_env_goals, goal_dim)
            
            # Compute distances: (num_rb_goals, num_env_goals)
            candidate_expanded = jnp.repeat(candidate_goals[:, None, :], env_goals.shape[0], axis=1)  # (num_rb_goals, num_env_goals, goal_dim)
            env_expanded = jnp.repeat(env_goals[None, :, :], candidate_goals.shape[0], axis=0)  # (num_rb_goals, num_env_goals, goal_dim)
            distances = jnp.linalg.norm(candidate_expanded - env_expanded, axis=-1)  # (num_rb_goals, num_env_goals)
            
            # Check if any environment goal is within threshold for each candidate
            is_successful = jnp.any(distances < env.goal_reach_thresh, axis=1)  # (num_rb_goals,)
            
            # Keep mask: True for goals we want to keep (not successful)
            keep_mask = ~is_successful  # (num_rb_goals,)
            
            # Check if all goals were filtered out
            all_filtered = jnp.all(~keep_mask)  # True if all goals are filtered
        else:
            # No filtering: all goals are valid
            keep_mask = jnp.ones(candidate_goals.shape[0], dtype=bool)
            all_filtered = jnp.array(False)
        
        # If all goals were filtered, return random environment goals
        def return_random_goals(key):
            sample_key, _ = jax.random.split(key)
            return propose_random_environment_goals(env, current_states.shape[0], sample_key), buffer_state
        
        def compute_goals_normally(key, keep_mask):
            # For candidate_goals_full, we need full state vectors
            # Expand goals to states using expand_goal_to_state
            candidate_goals_full = jax.vmap(
                lambda g: expand_goal_to_state(g, state_size, env.goal_indices)
            )(candidate_goals)  # (num_rb_goals, state_dim)
            
            # Note: If zero_out_cand_goals is False, we'd ideally want the actual full states
            # from the replay buffer, but for simplicity we use the expanded version

            def energy_triplet(state):
                """Compute M[g,h] for a single state."""
                # Optionally zero out everything except goal indices
                if self.zero_out_state:
                    state = zero_out_non_goal_indices(state, env.goal_indices)
                
                num_cand = candidate_goals.shape[0]  # All candidate goals (not filtered)
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
            
            def select_goal_maxlogsumexp_one_env(M, rand_key, mask):
                """Select one random environment goal and compute weights using only that column.
                
                Args:
                    M: Energy matrix (num_cand, num_env)
                    rand_key: Random key
                    mask: Boolean mask (num_cand,) - True for goals to keep
                """
                rand_key_h, rand_key_g = jax.random.split(rand_key)

                # Randomly select one environment goal
                num_env_goals = M.shape[1]
                h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
                
                energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
                score = energies_for_h  # Positive because we want to maximize
                
                # Mask out filtered goals by setting their logits to -inf
                # Set score to -inf for filtered goals so they get zero probability
                score = jnp.where(mask, score, jnp.array(-jnp.inf))
                
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
            
            # Broadcast mask to batch dimension
            keep_mask_batch = jnp.broadcast_to(keep_mask[None, :], (batch_size, keep_mask.shape[0]))

            best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp_one_env)(energy_mats, batch_keys, keep_mask_batch)

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
            
            # Select a state index for visualization (use first state)
            viz_state_idx = jnp.array(0)
            
            jax.experimental.io_callback(
                MaxWaypointRatioOneEnvProposer._log_goal_selection_viz,
                None,
                current_states,
                candidate_goals,
                env_goals,
                best_g_indices,
                best_h_indices,
                energy_mats,
                viz_state_idx,
                env_steps,
                env.goal_indices,
                env.x_bounds if hasattr(env, 'x_bounds') else None,
                env.y_bounds if hasattr(env, 'y_bounds') else None,
                self.LOG_INTERVAL_STEPS
            )
            
            return proposed_goals, buffer_state
        
        # Use conditional to choose between random goals or normal computation
        proposed_goals, buffer_state = jax.lax.cond(
            all_filtered,
            return_random_goals,
            lambda k: compute_goals_normally(k, keep_mask),
            key
        )

        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_goal_selection_viz(current_states, candidate_goals, env_goals, 
                              best_g_indices, best_h_indices, energy_mats, 
                              viz_state_idx, env_steps, goal_indices, x_bounds, y_bounds, log_interval_steps):
        """Visualize goal selection showing trajectory from current -> candidate -> env goals."""
        
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'max_waypoint_ratio'):
            return
        
        # Use viz_state_idx for env_goal_ranking plot, random for goal_selection plot
        num_states = current_states.shape[0]
        random_state_indices = np.random.choice(num_states, size=min(4, num_states), replace=False)
        # Make sure viz_state_idx is in random_state_indices for consistency
        random_state_indices[0] = int(viz_state_idx)
        
        # Generate visualization using shared utility
        pil_image = create_goal_selection_plot(
            current_states, candidate_goals, env_goals, best_g_indices, best_h_indices, energy_mats, 
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        
        metrics = {
            'max_waypoint_ratio/goal_selection_viz': wandb.Image(pil_image),
        }
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class QEpistemicProposer:
    """Proposes goals by selecting those with highest epistemic uncertainty.
    
    Uses an ensemble of critics to estimate uncertainty. For each state in the batch:
    1. Sample candidate goals from replay buffer final states or environment goals
    2. For each (state, candidate_goal) pair, sample an action from the policy
    3. Compute Q-values for the triplet (state, action, goal) across the ensemble
    4. Select the goal with highest standard deviation across the ensemble
    
    This encourages exploration by selecting goals where the agent is most uncertain.
    
    Attributes:
        energy_fn_name: Energy function name
        num_ensemble: Number of critics in the ensemble
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
        use_env_goals: If True, use environment goals; if False, use replay buffer states
        zero_center: If True, center each critic's predictions before computing std
        LOG_INTERVAL_STEPS: Log visualizations every N environment steps
    """
    energy_fn_name: str
    num_ensemble: int = 5
    num_rb_goals: int = 256
    candidate_goals_type: str = "final"
    use_env_goals: bool = False
    zero_center: bool = False
    LOG_INTERVAL_STEPS: int = 1000000

    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals with highest epistemic uncertainty.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices, state_dim attributes, and optionally possible_goals)
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (contains ensemble of sa_encoder and g_encoder params)
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (for env_steps logging, optional)
                
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        # Get current states from env_state
        state_size = env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        batch_size = current_states.shape[0]
        
        # Get candidate goals based on configuration
        if self.use_env_goals:
            assert hasattr(env, 'possible_goals'), \
                "Environment must have 'possible_goals' attribute for QEpistemicProposer with use_env_goals=True."
            candidate_goals = env.possible_goals  # (num_candidate_goals, goal_size)
        else:
            # Sample from replay buffer using common function
            key, sample_key = jax.random.split(key)
            candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
                replay_buffer, buffer_state, env, sample_key, self.num_rb_goals, self.candidate_goals_type
            )  # (num_rb_goals, goal_size)
        
        num_candidates = candidate_goals.shape[0]
        
        # Check if we have an ensemble
        is_ensemble = isinstance(critic_params["sa_encoder"], list)
        if not is_ensemble:
            raise ValueError("QEpistemicProposer requires an ensemble of critics. Set use_gcp_critic_ensemble=True.")
        
        # Stack ensemble parameters into arrays for JAX-compatible indexing
        stacked_sa_params, stacked_g_params = stack_ensemble_params(critic_params)
        
        def compute_q_values_for_state(state):
            """For a single state, compute Q-values across ensemble for all candidate goals.
            
            Args:
                state: (state_dim,) array
                
            Returns:
                all_q_values: (num_ensemble, num_candidates) array of Q-values
            """
            # Expand state to match number of candidates
            state_expanded = jnp.tile(state, (num_candidates, 1))  # (num_candidates, state_dim)
            
            # Compute Q-values using utility function
            all_q_values = compute_q_values_ensemble(
                state_expanded, candidate_goals, actor, actor_params,
                stacked_sa_params, stacked_g_params, sa_encoder, g_encoder,
                self.energy_fn_name, expand_goals=False
            )  # (num_ensemble, num_candidates)
            
            return all_q_values
        
        # Compute Q-values for all states: (batch_size, num_ensemble, num_candidates)
        all_ensemble_q_values = jax.vmap(compute_q_values_for_state)(current_states)
        
        # Optionally center each critic's predictions by subtracting its mean
        if self.zero_center:
            # Compute mean for each critic across all states and candidates
            critic_means = jnp.mean(all_ensemble_q_values, axis=(0, 2), keepdims=True)  # (1, num_ensemble, 1)
            # Subtract the mean from each critic's predictions to remove translational offset
            q_values_for_std = all_ensemble_q_values - critic_means  # (batch_size, num_ensemble, num_candidates)
        else:
            q_values_for_std = all_ensemble_q_values
        
        # Compute standard deviation across ensemble for each (state, candidate) pair
        all_q_stds = jnp.std(q_values_for_std, axis=1)  # (batch_size, num_candidates)
        
        # For each state, select the candidate goal with highest std
        best_goal_indices = jnp.argmax(all_q_stds, axis=1)  # (batch_size,)
        proposed_goals = candidate_goals[best_goal_indices]  # (batch_size, goal_size)
        
        # Log Q-epistemic statistics
        if training_state is not None:
            env_steps = training_state.env_steps
        else:
            env_steps = jnp.array(0)
            
        jax.experimental.io_callback(
            QEpistemicProposer._log_q_epistemic_statistics,
            None,
            all_q_stds,
            all_ensemble_q_values,  # Pass raw Q-values for deviation plotting
            candidate_goals,
            current_states,
            env.goal_indices,
            env_steps,
            self.LOG_INTERVAL_STEPS
        )
        
        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_q_epistemic_statistics(all_q_stds, all_ensemble_q_values, candidate_goals, current_states, goal_indices, env_steps, log_interval_steps):
        """Log Q-epistemic uncertainty statistics."""
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'q_epistemic'):
            return
        
        # all_q_stds: (batch_size, num_candidates)
        # all_ensemble_q_values: (batch_size, num_ensemble, num_candidates)
        max_stds_per_state = jnp.max(all_q_stds, axis=1)  # (batch_size,)
        
        metrics = {
            'q_epistemic/max_std_mean': float(jnp.mean(max_stds_per_state)),
            'q_epistemic/max_std_std': float(jnp.std(max_stds_per_state)),
            'q_epistemic/max_std_max': float(jnp.max(max_stds_per_state)),
            'q_epistemic/max_std_min': float(jnp.min(max_stds_per_state)),
            'q_epistemic/mean_std_across_candidates': float(jnp.mean(all_q_stds)),
        }
        
        # Compute overall mean across all critics, states, and candidates
        # all_ensemble_q_values: (batch_size, num_ensemble, num_candidates)
        overall_mean = jnp.mean(all_ensemble_q_values)  # scalar
        
        # For each critic, compute:
        # 1. Mean absolute deviation from overall mean (scalar)
        # 2. Mean of that critic's predictions (scalar)
        # 3. Ratio of deviation to critic mean
        num_ensemble = all_ensemble_q_values.shape[1]
        
        for critic_idx in range(num_ensemble):
            critic_values = all_ensemble_q_values[:, critic_idx, :]  # (batch_size, num_candidates)
            
            # Mean absolute deviation from overall mean
            deviation = jnp.mean(jnp.abs(critic_values - overall_mean))
            
            # Mean of this critic's predictions
            critic_mean = jnp.mean(critic_values)
            
            # Ratio of deviation to critic mean
            ratio = deviation / (jnp.abs(critic_mean) + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Log as scalar metrics
            metrics[f'q_epistemic/critic_{critic_idx}_deviation'] = float(deviation)
            metrics[f'q_epistemic/critic_{critic_idx}_mean'] = float(critic_mean)
            metrics[f'q_epistemic/critic_{critic_idx}_deviation_ratio'] = float(ratio)
        
        # Create visualization using shared utility
        def title_fn(state_idx, max_val, selected_val):
            max_std_idx = int(np.argmax(all_q_stds[state_idx]))
            return f'State {state_idx}: Max Q-Std = {max_val:.4f} (Goal {max_std_idx})'
        
        pil_image = create_2x2_scatter_plot(
            candidate_goals, current_states, goal_indices, all_q_stds,
            title_fn=title_fn, cmap='hot', color_label='Q-Std'
        )
        metrics['q_epistemic/q_std_heatmaps'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class MEGAProposer:
    """Maximum Entropy Goal Achievement (MEGA) proposer.
    
    Selects goals from low-density regions of the achieved goal distribution
    to maximize exploration at the frontier of achievable goals.
    
    Based on Algorithm 2 from the MEGA paper.
    
    Attributes:
        bandwidth: KDE bandwidth
        use_q_cutoff: Whether to eliminate unachievable goals using Q-values
        cutoff_percentile: Q-value percentile for cutoff (lower = more restrictive)
        energy_fn_name: Energy function to use for Q-value computation
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
    """
    bandwidth: float = 0.1  # KDE bandwidth
    use_q_cutoff: bool = True  # Whether to eliminate unachievable goals using Q-values
    cutoff_percentile: float = 0.3  # Q-value percentile for cutoff (lower = more restrictive)
    energy_fn_name: str = "dot"  # Energy function to use for Q-value computation
    num_rb_goals: int = 256
    candidate_goals_type: str = "any"  # MEGA typically uses "any" to get more diverse goals
    
    def sample_candidate_goals(self, replay_buffer, buffer_state, train_env, key):
        """Sample candidate goals from replay buffer.
        
        Uses the common function to sample candidate goals.
        
        Args:
            replay_buffer: Replay buffer containing past transitions
            buffer_state: Current state of replay buffer
            train_env: Training environment (for goal_indices)
            key: JAX random key
            
        Returns:
            candidate_goals: (num_rb_goals, goal_dim) array of candidate goals
            buffer_state: Updated buffer state
        """
        return sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, train_env, key, self.num_rb_goals, self.candidate_goals_type
        )
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, 
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals by selecting minimum density candidates from replay buffer.
        
        Args:
            replay_buffer: Replay buffer containing past transitions
            buffer_state: Current state of replay buffer
            env: Training environment
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters  
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (unused but required by interface)
            
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        batch_size = env_state.obs.shape[0]
        state_size = env.state_dim
        
        # Sample candidate goals using common function
        key, sample_key = jax.random.split(key)
        candidate_goals, buffer_state = sample_candidate_goals_from_replay_buffer(
            replay_buffer, buffer_state, env, sample_key, self.num_rb_goals, self.candidate_goals_type
        )
        
        # For each environment state, select minimum density goal from candidates
        def select_goal_for_state(current_state):
            """Select minimum density goal for one environment state."""
            # Compute density for each candidate using KDE
            # Normalize for numerical stability
            mean = jnp.mean(candidate_goals, axis=0)
            std = jnp.std(candidate_goals, axis=0) + 1e-6
            
            candidates_normalized = (candidate_goals - mean) / std
            
            # Compute densities using Gaussian KDE
            densities = gaussian_kernel_density(candidates_normalized, candidates_normalized, self.bandwidth)
            
            # Optional: Filter unachievable goals using Q-values
            if self.use_q_cutoff:
                # Vectorized computation using utility function
                s_rep = jnp.repeat(current_state[None, :], len(candidate_goals), axis=0)
                q_values = compute_energy_for_state_goal_pairs(
                    s_rep, candidate_goals, actor, actor_params,
                    critic_params, sa_encoder, g_encoder, self.energy_fn_name
                )
                
                # Compute adaptive cutoff (percentile of Q-values)
                cutoff_value = jnp.percentile(q_values, self.cutoff_percentile * 100)
                
                # Set density of unachievable goals to infinity (so they won't be selected)
                densities = jnp.where(q_values >= cutoff_value, densities, jnp.inf)
            
            # Select minimum density candidate
            min_idx = jnp.argmin(densities)
            return candidate_goals[min_idx]
    
        
        # Process all states in batch
        current_states = env_state.obs[:, :state_size]
        proposed_goals = jax.vmap(select_goal_for_state)(current_states)
        
        return proposed_goals, buffer_state


@dataclass
class OMEGAProposer:
    """OMEGA (annealing MEGA to desired goals) proposer.
    
    Anneals from MEGA exploration to desired goal distribution using α parameter
    that depends on KL divergence between desired and achieved distributions.
    
    α = 1 / max(b + D_KL(p_dg || p_ag), 1)
    
    With probability α: sample from desired goal distribution
    With probability 1-α: use MEGA to explore low-density regions
    
    Based on Algorithm 2 from the MEGA paper.
    
    Attributes:
        bandwidth: KDE bandwidth
        use_q_cutoff: Whether to eliminate unachievable goals using Q-values
        cutoff_percentile: Q-value percentile for cutoff
        energy_fn_name: Energy function to use for Q-value computation
        bias_param: 'b' in paper, controls annealing speed (-3 recommended)
        num_rb_goals: Number of candidate goals to sample from replay buffer
        candidate_goals_type: Either "final" (use final trajectory states) or "any" (use any trajectory state)
    """
    bandwidth: float = 0.1  
    use_q_cutoff: bool = True
    cutoff_percentile: float = 0.3
    energy_fn_name: str = "dot"
    bias_param: float = -3.0  # 'b' in paper, controls annealing speed (-3 recommended)
    num_rb_goals: int = 256
    candidate_goals_type: str = "any"  # OMEGA typically uses "any" like MEGA
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose goals by annealing between MEGA and desired goals.
        
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        assert hasattr(env, 'possible_goals'), \
            "Environment must store property `possible_goals` for OMEGAProposer."
        
        batch_size = env_state.obs.shape[0]
        
        # Get desired goals from environment
        desired_goals = env.possible_goals  # (num_env_goals, goal_dim)
        
        # Create MEGA proposer (used for both sampling and goal selection)
        mega_proposer = MEGAProposer(
            bandwidth=self.bandwidth,
            use_q_cutoff=self.use_q_cutoff,
            cutoff_percentile=self.cutoff_percentile,
            energy_fn_name=self.energy_fn_name,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type
        )
        
        # Sample candidate goals once - used for both KL divergence and MEGA
        key, sample_key = jax.random.split(key)
        achieved_goals, buffer_state = mega_proposer.sample_candidate_goals(
            replay_buffer, buffer_state, env, sample_key
        )  # (num_envs * episode_length, goal_dim)
        
        # Compute α based on KL divergence between desired and achieved goal distributions
        kl_div = compute_kl_divergence_empirical(desired_goals, achieved_goals, self.bandwidth)
        alpha = 1.0 / jnp.maximum(self.bias_param + kl_div, 1.0)
        
        # Log alpha value using wandb
        if training_state is not None:
            env_steps = training_state.env_steps
        else:
            env_steps = jnp.array(0)
            
        jax.experimental.io_callback(
            OMEGAProposer._log_alpha_callback,
            None,
            alpha,
            env_steps
        )
        
        # Decide whether to use MEGA or environment goals
        key, choice_key, mega_key = jax.random.split(key, 3)
        use_env_goals = jax.random.uniform(choice_key, (batch_size,)) < alpha
        
        # Get MEGA goals - pass achieved_goals as candidate_goals to avoid resampling
        # Note: We need to modify mega_proposer.propose_goals to accept pre-sampled candidates
        # For now, we'll just call it normally (will resample internally)
        mega_goals, buffer_state = mega_proposer.propose_goals(
            replay_buffer, buffer_state, env, env_state,
            mega_key, actor, actor_params, critic_params, sa_encoder, g_encoder, training_state
        )
        
        # Sample from desired goals for environments that should use env goals
        key, sample_key = jax.random.split(key)
        env_goal_indices = jax.random.randint(sample_key, (batch_size,), 0, len(desired_goals))
        sampled_env_goals = desired_goals[env_goal_indices]
        
        # Mix goals based on α
        proposed_goals = jnp.where(
            use_env_goals[:, None],
            sampled_env_goals,
            mega_goals
        )
        
        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_alpha_callback(alpha_val, env_steps):
        """Log alpha to wandb."""
        metrics = {
            'omega/alpha': float(alpha_val),
        }
        wandb.log(metrics, step=int(env_steps))


@dataclass
class EmpowermentDifferenceGoalProposer:
    """Empowerment Difference Goal Proposer.

    Scores replay-buffer states by the difference between exploratory-policy
    empowerment and goal-conditioned-policy empowerment for a *single* randomly
    sampled environment goal ``g``:

        E_diff(s, g) = E_ep(s, g)  -  beta * E_gcp(s, g)

    where empowerment is estimated via a Monte Carlo InfoNCE estimator:

        E_hat(s, g) = (1/N) * sum_k [
            f(s, a_k, s+_k)
            - logsumexp_m( f(s, a_m', s+_k) ) + log(M)
        ]

    N outer samples produce (a_k, s+_k) pairs; M separate inner actions a_m'
    (never reused from the outer loop) form the contrastive denominator.
    Future states s+_k are drawn by truncated geometric horizon sampling from
    the same trajectory as the current state.

    Attributes:
        energy_fn_name: Energy function name (e.g. "dot", "norm").
        empowerment_num_outer_samples: N – outer MC samples per state.
        empowerment_num_inner_actions: M – contrastive actions per outer sample.
        gcp_empowerment_penalty: beta – weight on GCP empowerment in E_diff.
        num_rb_goals: Number of replay-buffer states to score per proposal step.
        discounting: Discount factor gamma; geometric horizon p = 1 - gamma.
        goal_sampling_temperature: Softmax temperature for goal selection.
            0 → greedy (argmax).
    """

    energy_fn_name: str = "dot"
    empowerment_num_outer_samples: int = 10
    empowerment_num_inner_actions: int = 10
    gcp_empowerment_penalty: float = 1.0
    num_rb_goals: int = 256
    discounting: float = 0.99
    goal_sampling_temperature: float = 1.0
    LOG_INTERVAL_STEPS: int = 1000000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_future_states(
        all_observations, all_traj_ids,
        traj_indices, time_indices,
        state_dim, num_outer_samples, discounting, key,
    ):
        """Sample ``num_outer_samples`` future states for each batch state.

        For the state at position ``(traj_indices[i], time_indices[i])``, sample
        ``num_outer_samples`` horizons K from a truncated geometric distribution
        (p = 1 - discounting, truncated to the remaining same-trajectory steps)
        and return the corresponding future observations.

        Args:
            all_observations: ``(N_traj, ep_len, obs_dim)`` observation block.
            all_traj_ids: ``(N_traj, ep_len)`` trajectory IDs.
            traj_indices: ``(batch_size,)`` trajectory index per sampled state.
            time_indices: ``(batch_size,)`` timestep index per sampled state.
            state_dim: First ``state_dim`` features of an observation are state.
            num_outer_samples: N – number of future states to draw per state.
            discounting: Discount factor gamma.
            key: JAX random key.

        Returns:
            future_states: ``(batch_size, num_outer_samples, state_dim)``.
                Gradients are stopped; sampling is not differentiable.
        """
        ep_len = all_observations.shape[1]

        def _sample_for_one(traj_i, time_t, rng):
            traj_obs = all_observations[traj_i]       # (ep_len, obs_dim)
            traj_id_seq = all_traj_ids[traj_i]        # (ep_len,)

            # Mask: same trajectory id AND strictly in the future
            curr_id = traj_id_seq[time_t]
            same_traj = traj_id_seq == curr_id                   # (ep_len,)
            is_future = jnp.arange(ep_len) > time_t             # (ep_len,)
            valid_mask = same_traj & is_future                   # (ep_len,)

            # Log-geometric weights: P(K=k) ∝ gamma^k, k = j - time_t ≥ 1
            k_vals = jnp.arange(ep_len, dtype=jnp.float32) - time_t.astype(jnp.float32)
            log_probs = k_vals * jnp.log(jnp.clip(discounting, 1e-6, 1.0 - 1e-7))
            log_probs = jnp.where(valid_mask, log_probs, -jnp.inf)

            # Tiny self-probability as fallback for terminal states where no
            # valid future exists (prevents all-inf softmax).
            self_mask = jnp.arange(ep_len) == time_t
            log_self = jnp.where(self_mask, jnp.log(1e-5), -jnp.inf)
            log_probs = jnp.logaddexp(log_probs, log_self)

            # Draw num_outer_samples indices from the categorical distribution
            rngs = jax.random.split(rng, num_outer_samples)
            future_idx = jax.vmap(
                lambda k: jax.random.categorical(k, log_probs)
            )(rngs)  # (num_outer_samples,)

            # Stop gradient: future-state sampling is non-differentiable
            return jax.lax.stop_gradient(
                traj_obs[future_idx, :state_dim]
            )  # (num_outer_samples, state_dim)

        batch_size = traj_indices.shape[0]
        rngs = jax.random.split(key, batch_size)
        return jax.vmap(_sample_for_one)(traj_indices, time_indices, rngs)
        # (batch_size, num_outer_samples, state_dim)

    def _compute_empowerment(
        self, policy, policy_params, critic_params,
        sa_encoder, g_encoder,
        states, goal, future_states, goal_indices, key,
        actual_actions, trajectory_goals,
    ):
        """Estimate empowerment for a batch of states via Monte Carlo InfoNCE.

        The estimator is fully vectorised: no Python loops over the batch.

        Args:
            policy: Policy network (actor).
            policy_params: Corresponding policy parameters.
            critic_params: Critic parameters dict with keys
                ``'sa_encoder'`` and ``'g_encoder'``.  Single (non-ensemble)
                critic only; ensemble GCP critics are averaged before use
                (see :meth:`propose_goals`).
            sa_encoder: State-action encoder network.
            g_encoder: Goal encoder network.
            states: ``(batch_size, state_dim)`` current states.
            goal: ``(goal_dim,)`` single environment goal (same for all states).
            future_states: ``(batch_size, N, state_dim)`` pre-sampled future
                states (N = empowerment_num_outer_samples).
            goal_indices: Indices to extract goal features from a full state.
            key: JAX random key.
            actual_actions: ``(batch_size, N, action_dim)`` actual actions from
                trajectory. These are used for the outer actions.
            trajectory_goals: ``(batch_size, goal_dim)`` goals that were used
                for the trajectory. Inner actions are conditioned on these goals.

        Returns:
            empowerment: ``(batch_size,)`` empowerment estimate per state.
        """
        N = self.empowerment_num_outer_samples
        M = self.empowerment_num_inner_actions
        batch_size = states.shape[0]
        action_dim = None  # inferred from policy output

        # ---- Use actual actions from trajectory for outer actions -----------
        # Use the actual actions that were sampled at state s in the trajectory
        actions_outer = actual_actions  # (B, N, A)

        # ---- Sample M inner actions, conditioned on trajectory goals --------
        # Condition inner actions on the goal that was used for that trajectory
        goals_for_inner = jnp.tile(trajectory_goals[:, None, None, :], (1, N, M, 1))  # (B, N, M, goal_dim)
        states_inner = jnp.tile(states[:, None, None, :], (1, N, M, 1))  # (B, N, M, state_dim)
        obs_inner = jnp.concatenate([
            states_inner.reshape(batch_size * N * M, -1),  # (B*N*M, state_dim)
            goals_for_inner.reshape(batch_size * N * M, -1)  # (B*N*M, goal_dim)
        ], axis=-1)  # (B*N*M, obs_dim)
        
        means_i, log_stds_i = policy.apply(policy_params, obs_inner)
        key, k_in = jax.random.split(key)
        noise_i = jax.random.normal(k_in, means_i.shape)
        actions_inner = jnp.tanh(means_i + jnp.exp(log_stds_i) * noise_i)
        actions_inner = actions_inner.reshape(batch_size, N, M, -1)  # (B, N, M, A)

        # ---- Compute outer energies: f(s_i, a_k, s+_{i,k}) -----------------
        # future_states: (B, N, state_dim) -> goal features: (B, N, goal_dim)
        future_goals = future_states[:, :, goal_indices]          # (B, N, G)

        states_outer_rep = jnp.tile(states[:, None, :], (1, N, 1))  # (B, N, S)
        sa_outer = jnp.concatenate(
            [states_outer_rep.reshape(batch_size * N, -1),
             actions_outer.reshape(batch_size * N, -1)],
            axis=1,
        )  # (B*N, S+A)
        phi_outer = sa_encoder.apply(
            critic_params['sa_encoder'], sa_outer
        )  # (B*N, repr_dim)
        psi_future = g_encoder.apply(
            critic_params['g_encoder'],
            future_goals.reshape(batch_size * N, -1),
        )  # (B*N, repr_dim)

        energies_outer = _energy_fn(
            self.energy_fn_name, phi_outer, psi_future
        ).reshape(batch_size, N)  # (B, N)

        # ---- Compute inner energies: f(s_i, a_m', s+_{i,k}) ----------------
        # Same future state s+_{i,k} but different actions a_m'
        states_inner_rep = jnp.tile(states[:, None, None, :], (1, N, M, 1))  # (B,N,M,S)
        sa_inner = jnp.concatenate(
            [states_inner_rep.reshape(batch_size * N * M, -1),
             actions_inner.reshape(batch_size * N * M, -1)],
            axis=1,
        )  # (B*N*M, S+A)
        phi_inner = sa_encoder.apply(
            critic_params['sa_encoder'], sa_inner
        )  # (B*N*M, repr_dim)

        # Expand future_goals to (B, N, M, G) then flatten
        future_goals_inner = jnp.tile(future_goals[:, :, None, :], (1, 1, M, 1))
        psi_inner = g_encoder.apply(
            critic_params['g_encoder'],
            future_goals_inner.reshape(batch_size * N * M, -1),
        )  # (B*N*M, repr_dim)

        energies_inner = _energy_fn(
            self.energy_fn_name, phi_inner, psi_inner
        ).reshape(batch_size, N, M)  # (B, N, M)

        # ---- InfoNCE per outer sample --------------------------------------
        # log((1/M) * sum_m exp(f_inner)) = logsumexp(f_inner) - log(M)
        logsumexp_inner = jax.scipy.special.logsumexp(
            energies_inner, axis=2
        )  # (B, N)
        per_outer = energies_outer - logsumexp_inner + jnp.log(M)  # (B, N)

        # Average over outer samples
        return jnp.mean(per_outer, axis=1)  # (B,)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propose_goals(
        self, replay_buffer, buffer_state, env, env_state, key,
        actor, actor_params, critic_params,
        sa_encoder, g_encoder, training_state,
        gcp_replay_buffer, gcp_buffer_state,
        ep_replay_buffer, ep_buffer_state,
    ):
        """Propose goals scored by the empowerment difference E_ep - beta*E_gcp.

        One random environment goal ``g`` is sampled and held fixed for the
        entire batch.  Replay-buffer states are sampled and scored:

            E_diff(s_i, g) = E_ep(s_i, g) - beta * E_gcp(s_i, g)

        Each environment independently samples a proposed goal from the
        softmax distribution over E_diff scores.

        Args:
            replay_buffer: Replay buffer (unused, kept for interface compatibility).
            buffer_state: Buffer state (unused, kept for interface compatibility).
            env: Training environment (must have ``possible_goals``,
                ``goal_indices``, ``state_dim``).
            env_state: Current environment state (used for batch_size).
            key: JAX random key.
            actor: Actor network – used for *both* GCP and EP evaluation
                (same architecture, different params supplied via
                ``actor_params`` / ``training_state.ep_actor_state.params``).
            actor_params: GCP actor parameters (= training_state.gcp_actor_state.params).
            critic_params: GCP critic parameters dict.
            sa_encoder: State-action encoder network (shared architecture).
            g_encoder: Goal encoder network (shared architecture).
            training_state: Training state; used to access EP actor/critic
                parameters via ``training_state.ep_actor_state.params`` and
                ``training_state.ep_critic_state.params``.
            gcp_replay_buffer: GCP replay buffer for GCP empowerment calculation.
            gcp_buffer_state: GCP buffer state.
            ep_replay_buffer: EP replay buffer for EP empowerment calculation.
            ep_buffer_state: EP buffer state.

        Returns:
            proposed_goals: ``(batch_size, goal_dim)`` proposed goals.
            buffer_state: Updated EP buffer state.
        """
        assert hasattr(env, 'possible_goals'), (
            "EmpowermentDifferenceGoalProposer requires env.possible_goals."
        )
        assert training_state is not None, (
            "EmpowermentDifferenceGoalProposer requires training_state to access "
            "both EP and GCP actor/critic parameters."
        )

        batch_size = env_state.obs.shape[0]
        state_dim = env.state_dim
        goal_indices = env.goal_indices
        N = self.empowerment_num_outer_samples

        # ---- 1. Sample one random environment goal g ----------------------
        key, g_key = jax.random.split(key)
        env_goals = env.possible_goals  # (num_env_goals, goal_dim)
        g_idx = jax.random.randint(g_key, (), 0, env_goals.shape[0])
        goal = env_goals[g_idx]  # (goal_dim,)

        # ---- 2. Sample batch of states from EP replay buffer (for diversity) ---
        # We'll compute both empowerments on the same states
        key, ep_sample_key = jax.random.split(key)
        (
            states, ep_traj_indices, ep_time_indices,
            ep_all_observations, ep_all_traj_ids,
            ep_all_actions, ep_all_gc_goals, ep_all_ep_goals,
            ep_bs,
        ) = sample_states_from_replay_buffer(
            ep_replay_buffer, ep_buffer_state, state_dim,
            self.num_rb_goals, ep_sample_key,
        )
        # states: (num_rb_goals, state_dim)
        
        # ---- 2b. Also sample from GCP replay buffer to get GCP actions/goals ---
        # We'll use the same number of samples, but from GCP buffer
        key, gcp_sample_key = jax.random.split(key)
        (
            _, gcp_traj_indices, gcp_time_indices,
            gcp_all_observations, gcp_all_traj_ids,
            gcp_all_actions, gcp_all_gc_goals, gcp_all_ep_goals,
            gcp_bs,
        ) = sample_states_from_replay_buffer(
            gcp_replay_buffer, gcp_buffer_state, state_dim,
            self.num_rb_goals, gcp_sample_key,
        )
        # Note: We don't use gcp_states, we use the same states from EP buffer

        # ---- 3. Sample N future states per replay-buffer state (geometric) -
        # Sample futures from EP trajectories (since we use EP states)
        key, fs_key = jax.random.split(key)
        future_states = EmpowermentDifferenceGoalProposer._sample_future_states(
            ep_all_observations, ep_all_traj_ids,
            ep_traj_indices, ep_time_indices,
            state_dim, N, self.discounting, fs_key,
        )
        # future_states: (num_rb_goals, N, state_dim)

        # ---- 4. Extract actual actions and goals from trajectories -------
        # For EP empowerment: get actions and goals from EP trajectories
        ep_actions_at_states = ep_all_actions[ep_traj_indices, ep_time_indices]  # (num_rb_goals, action_dim)
        ep_actions_outer = jnp.tile(ep_actions_at_states[:, None, :], (1, N, 1))  # (num_rb_goals, N, action_dim)
        ep_traj_goals = ep_all_ep_goals[ep_traj_indices, ep_time_indices]  # (num_rb_goals, goal_dim)
        
        # For GCP empowerment: get actions and goals from GCP trajectories
        gcp_actions_at_states = gcp_all_actions[gcp_traj_indices, gcp_time_indices]  # (num_rb_goals, action_dim)
        gcp_actions_outer = jnp.tile(gcp_actions_at_states[:, None, :], (1, N, 1))  # (num_rb_goals, N, action_dim)
        gcp_traj_goals = gcp_all_gc_goals[gcp_traj_indices, gcp_time_indices]  # (num_rb_goals, goal_dim)
        
        # Sample futures from GCP trajectories for GCP empowerment
        # (We use the same future states for both, sampled from EP trajectories)
        # Actually, we should sample from GCP trajectories for GCP empowerment
        key, gcp_fs_key = jax.random.split(key)
        gcp_future_states = EmpowermentDifferenceGoalProposer._sample_future_states(
            gcp_all_observations, gcp_all_traj_ids,
            gcp_traj_indices, gcp_time_indices,
            state_dim, N, self.discounting, gcp_fs_key,
        )
        # gcp_future_states: (num_rb_goals, N, state_dim)

        # ---- 5. Resolve EP / GCP actor and critic params -------------------
        ep_actor_params = training_state.ep_actor_state.params
        ep_critic_params = training_state.ep_critic_state.params

        # If GCP critic is an ensemble, collapse to a single set of params by
        # averaging (only for proposer scoring; training uses the full ensemble)
        if isinstance(critic_params['sa_encoder'], list):
            gcp_sa_params = jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs, axis=0), axis=0),
                *critic_params['sa_encoder'],
            )
            gcp_g_params = jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs, axis=0), axis=0),
                *critic_params['g_encoder'],
            )
            gcp_critic_single = {'sa_encoder': gcp_sa_params, 'g_encoder': gcp_g_params}
        else:
            gcp_critic_single = critic_params

        # ---- 6. Compute E_ep and E_gcp ------------------------------------
        key, ep_key, gcp_key = jax.random.split(key, 3)

        # Compute EP empowerment using same states, but EP actions and goals
        e_ep = self._compute_empowerment(
            actor, ep_actor_params, ep_critic_params,
            sa_encoder, g_encoder,
            states, goal, future_states, goal_indices, ep_key,
            actual_actions=ep_actions_outer,
            trajectory_goals=ep_traj_goals,
        )  # (num_rb_goals,)

        # Compute GCP empowerment using same states, but GCP actions and goals
        e_gcp = self._compute_empowerment(
            actor, actor_params, gcp_critic_single,
            sa_encoder, g_encoder,
            states, goal, gcp_future_states, goal_indices, gcp_key,
            actual_actions=gcp_actions_outer,
            trajectory_goals=gcp_traj_goals,
        )  # (num_rb_goals,)

        # ---- 7. E_diff = E_ep - beta * E_gcp ------------------------------
        # Note: e_ep and e_gcp are computed on different sets of states
        # We need to align them - use EP states for goal selection
        e_diff = e_ep - self.gcp_empowerment_penalty * e_gcp  # (num_rb_goals,)

        # ---- 8. Select one goal per environment via softmax / argmax -------
        key, sel_key = jax.random.split(key)
        candidate_goals = states[:, goal_indices]  # (num_rb_goals, goal_dim)

        if self.goal_sampling_temperature > 0:
            logits = e_diff / self.goal_sampling_temperature
            probs = jax.nn.softmax(logits)  # (num_rb_goals,)
            # Each environment draws independently
            sel_keys = jax.random.split(sel_key, batch_size)
            selected_indices = jax.vmap(
                lambda k: jax.random.choice(k, a=self.num_rb_goals, p=probs)
            )(sel_keys)  # (batch_size,)
        else:
            # Greedy: every environment gets the same top-ranked state
            best_idx = jnp.argmax(e_diff)
            selected_indices = jnp.full((batch_size,), best_idx, dtype=jnp.int32)

        proposed_goals = candidate_goals[selected_indices]  # (batch_size, goal_dim)

        # ---- 9. Visualise at logging interval ------------------------------
        env_steps = training_state.env_steps if training_state is not None else jnp.array(0)
        
        # For visualization, use the first selected index (or best if greedy)
        viz_selected_idx = selected_indices[0] if batch_size > 0 else jnp.argmax(e_diff)

        jax.experimental.io_callback(
            EmpowermentDifferenceGoalProposer._visualize,
            None,
            e_ep, e_gcp, e_diff,
            states, goal, goal_indices, viz_selected_idx,
            env_steps,
            self.LOG_INTERVAL_STEPS,
        )

        # Return ep_buffer_state since we use EP states for goal selection
        return proposed_goals, ep_bs

    @staticmethod
    def _visualize(
        e_ep, e_gcp, e_diff,
        states, goal, goal_indices, selected_idx,
        env_steps, log_interval_steps,
    ):
        """Log scatter plots of candidate goals colored by empowerment values.

        Generates three subplots showing candidate goals as scatter points:
            - Colored by E_ep (exploratory empowerment)
            - Colored by E_gcp (goal-conditioned empowerment)
            - Colored by E_diff (empowerment difference)
        
        The selected goal is highlighted with a green circle in each plot.

        Args:
            e_ep: ``(num_rb_goals,)`` exploratory empowerment scores.
            e_gcp: ``(num_rb_goals,)`` GCP empowerment scores.
            e_diff: ``(num_rb_goals,)`` empowerment difference scores.
            states: ``(num_rb_goals, state_dim)`` replay-buffer states.
            goal: ``(goal_dim,)`` the fixed environment goal used for scoring.
            goal_indices: Indices to extract goal coordinates from a state.
            selected_idx: Index of the selected goal to highlight.
            env_steps: Current environment step count (for wandb x-axis).
            log_interval_steps: Minimum steps between consecutive log calls.
        """
        if not should_log_at_interval(env_steps, log_interval_steps, 'empowerment_diff'):
            return

        import io
        import matplotlib.pyplot as plt
        from PIL import Image

        e_ep_np = np.array(e_ep)
        e_gcp_np = np.array(e_gcp)
        e_diff_np = np.array(e_diff)
        states_np = np.array(states)
        selected_idx = int(selected_idx)

        # Extract goal coordinates from states
        candidate_goals = states_np[:, goal_indices]  # (num_rb_goals, goal_dim)
        selected_goal = candidate_goals[selected_idx]  # (goal_dim,)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, scores, label, cmap in zip(
            axes,
            [e_ep_np, e_gcp_np, e_diff_np],
            ['E_ep (exploratory)', 'E_gcp (goal-conditioned)', 'E_diff = E_ep - β·E_gcp'],
            ['viridis', 'plasma', 'coolwarm'],
        ):
            # Scatter plot of candidate goals colored by empowerment values
            scatter = ax.scatter(
                candidate_goals[:, 0], candidate_goals[:, 1],
                c=scores, cmap=cmap, s=150, alpha=0.8,
                edgecolors='black', linewidths=0.5,
            )
            plt.colorbar(scatter, ax=ax, label='Empowerment value')
            
            # Highlight selected goal with green circle
            ax.scatter(
                selected_goal[0], selected_goal[1],
                s=400, marker='o', facecolors='none',
                edgecolors='green', linewidths=3, zorder=10,
                label='Selected Goal'
            )
            
            ax.set_xlabel('Goal X', fontsize=11)
            ax.set_ylabel('Goal Y', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')
            ax.set_aspect('equal', adjustable='box')

        goal_np = np.array(goal)
        fig.suptitle(
            f'Empowerment Difference Goal Proposer  |  step {int(env_steps)}'
            f'  |  env goal = ({goal_np[0]:.2f}, {goal_np[1]:.2f})',
            fontsize=13, fontweight='bold',
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_image.load()
        buf.close()
        plt.close(fig)

        wandb.log(
            {'empowerment_diff/scatter_plots': wandb.Image(pil_image)},
            step=int(env_steps),
        )


@dataclass
class NearestEnvGoalProposer:
    """Proposes goals by selecting the environment goal with maximum critic value.
    
    For each environment state, this proposer:
    1. Gets all possible environment goals from env.possible_goals
    2. Computes Q-value for each (current_state, env_goal) pair using GCP critic
    3. Selects the env_goal with the maximum Q-value (highest critic value)
    
    This encourages the EP policy to practice reaching goals with high critic values.
    
    Attributes:
        energy_fn_name: Energy function to use for Q-value computation
    """
    energy_fn_name: str = "norm"  # Energy function: typically "norm", "dot", "l2", or "cosine"
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose environment goals based on maximum GCP critic values.
        
        Args:
            replay_buffer: Replay buffer (unused but required by interface)
            buffer_state: Current buffer state (returned unchanged)
            env: Training environment (must have possible_goals attribute)
            env_state: Current environment state
            key: JAX random key (unused but required by interface)
            actor: Actor network (GCP actor for computing Q-values)
            actor_params: Actor parameters
            critic_params: Critic parameters (GCP critic)
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (unused)
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state (unchanged)
        """
        batch_size = env_state.obs.shape[0]
        state_size = env.state_dim
        
        # Get all possible environment goals
        env_goals = jnp.array(env.possible_goals)  # (num_env_goals, goal_dim)
        num_env_goals = env_goals.shape[0]
        
        # Get current states for all environments
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        
        def select_goal_for_state(state):
            """For a single state, select the environment goal with maximum Q-value.
            
            Args:
                state: (state_dim,) current state
                
            Returns:
                selected_goal: (goal_dim,) environment goal with maximum Q-value
            """
            # Expand state to match number of environment goals
            states_expanded = jnp.tile(state, (num_env_goals, 1))  # (num_env_goals, state_dim)
            
            # Compute Q-values for all (state, env_goal) pairs
            q_values = compute_energy_for_state_goal_pairs(
                states_expanded, 
                env_goals,
                actor,
                actor_params,
                critic_params,
                sa_encoder,
                g_encoder,
                self.energy_fn_name
            )  # (num_env_goals,)
            
            # Select goal with maximum Q-value (highest critic value)
            max_idx = jnp.argmax(q_values)
            return env_goals[max_idx]
        
        # Process all states in batch
        proposed_goals = jax.vmap(select_goal_for_state)(current_states)
        
        return proposed_goals, buffer_state


@dataclass
class NearestEnvGoalToGCPGoalProposer:
    """Proposes goals by selecting the nearest environment goal to the GCP-proposed goal.
    
    For each environment, this proposer:
    1. Gets the GCP-proposed goal from env_state.info["gc_proposed_goals"]
    2. Gets all possible environment goals from env.possible_goals
    3. Computes Q-value for each (gcp_goal, env_goal) pair using GCP critic
    4. Selects the env_goal with the maximum Q-value (nearest via critic)
    
    This encourages the EP policy to practice reaching environment goals that are
    "nearest" (via maximum critic value) to the goals proposed for the GCP policy.
    
    Attributes:
        energy_fn_name: Energy function to use for Q-value computation
    """
    energy_fn_name: str = "norm"  # Energy function: typically "norm", "dot", "l2", or "cosine"
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder, training_state=None):
        """Propose environment goals nearest to GCP-proposed goals via maximum critic.
        
        Args:
            replay_buffer: Replay buffer (unused but required by interface)
            buffer_state: Current buffer state (returned unchanged)
            env: Training environment (must have possible_goals attribute)
            env_state: Current environment state (must have gc_proposed_goals in info)
            key: JAX random key (unused but required by interface)
            actor: Actor network (GCP actor for computing Q-values)
            actor_params: Actor parameters
            critic_params: Critic parameters (GCP critic)
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            training_state: Training state (unused)
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state (unchanged)
        """
        batch_size = env_state.obs.shape[0]
        state_size = env.state_dim
        goal_indices = env.goal_indices
        
        # Get GCP-proposed goals from environment state info
        gc_proposed_goals = env_state.info.get("gc_proposed_goals", None)
        if gc_proposed_goals is None:
            raise ValueError(
                "NearestEnvGoalToGCPGoalProposer requires gc_proposed_goals in env_state.info. "
                "Make sure GCP goals are proposed before EP goals."
            )
        gc_proposed_goals = jnp.array(gc_proposed_goals)  # (batch_size, goal_dim)
        
        # Get all possible environment goals
        env_goals = jnp.array(env.possible_goals)  # (num_env_goals, goal_dim)
        num_env_goals = env_goals.shape[0]
        
        def select_nearest_env_goal_to_gcp_goal(gcp_goal):
            """For a single GCP-proposed goal, select the nearest env goal via max critic.
            
            Args:
                gcp_goal: (goal_dim,) GCP-proposed goal
                
            Returns:
                selected_goal: (goal_dim,) environment goal with maximum Q-value
            """
            # Expand GCP goal to a full state vector (goal at goal_indices, zeros elsewhere)
            gcp_state = expand_goal_to_state(gcp_goal, state_size, goal_indices)  # (state_dim,)
            
            # Expand state to match number of environment goals
            states_expanded = jnp.tile(gcp_state, (num_env_goals, 1))  # (num_env_goals, state_dim)
            
            # Compute Q-values for all (gcp_state, env_goal) pairs
            q_values = compute_energy_for_state_goal_pairs(
                states_expanded, 
                env_goals,
                actor,
                actor_params,
                critic_params,
                sa_encoder,
                g_encoder,
                self.energy_fn_name
            )  # (num_env_goals,)
            
            # Select environment goal with maximum Q-value (nearest via critic)
            max_idx = jnp.argmax(q_values)
            return env_goals[max_idx]
        
        # Process all GCP-proposed goals in batch
        proposed_goals = jax.vmap(select_nearest_env_goal_to_gcp_goal)(gc_proposed_goals)
        
        return proposed_goals, buffer_state
