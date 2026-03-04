#!/usr/bin/env python3
"""CLI script to plot Q ensemble statistics as heatmaps.

This script loads saved Q parameters and replay buffer samples, computes
Q-values for all transitions, and visualizes mean and standard deviation
across the ensemble as heatmaps.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jaxgcrl.agents.sac import networks
from brax.training.acme import running_statistics
from brax.training.acme import specs


def load_data(q_params_path: str, replay_sample_path: str) -> Dict[str, Any]:
    """Load Q parameters and replay buffer sample from pickle files."""
    print(f"Loading Q parameters from {q_params_path}...")
    with open(q_params_path, "rb") as f:
        q_params_data = pickle.load(f)
    
    print(f"Loading replay buffer sample from {replay_sample_path}...")
    with open(replay_sample_path, "rb") as f:
        replay_sample = pickle.load(f)
    
    return {
        "q_params": q_params_data["q_params"],
        "target_q_params": q_params_data.get("target_q_params"),
        "n_critics": q_params_data["n_critics"],
        "transitions": replay_sample,
    }


def reconstruct_q_network(
    obs_size: int,
    action_size: int,
    n_critics: int,
    hidden_sizes: tuple = (256, 256),
    layer_norm: bool = False,
):
    """Reconstruct the Q network architecture."""
    return networks.make_q_network(
        obs_size=obs_size,
        action_size=action_size,
        n_critics=n_critics,
        hidden_layer_sizes=hidden_sizes,
        layer_norm=layer_norm,
    )


def compute_q_values(
    q_network,
    q_params: Any,
    normalizer_params: Any,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Compute Q-values for given observations and actions.
    
    Returns:
        Q-values of shape (batch_size, n_critics)
    """
    # Convert to JAX arrays
    obs_jax = jnp.asarray(observations)
    actions_jax = jnp.asarray(actions)
    
    # Apply Q network
    q_values = q_network.apply(normalizer_params, q_params, obs_jax, actions_jax)
    
    # Handle shape: might be (batch_size, n_critics) or (batch_size, n_critics, 1)
    if q_values.ndim == 3 and q_values.shape[-1] == 1:
        q_values = jnp.squeeze(q_values, axis=-1)
    
    return np.asarray(q_values)


def compute_ensemble_stats(q_values: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute mean and std across ensemble.
    
    Args:
        q_values: Shape (batch_size, n_critics)
    
    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (batch_size,)
    """
    return {
        "mean": np.mean(q_values, axis=-1),
        "std": np.std(q_values, axis=-1),
        "min": np.min(q_values, axis=-1),
        "max": np.max(q_values, axis=-1),
    }


def create_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    cmap: str = "viridis",
):
    """Create a scatter plot with color-coded values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=20, alpha=0.7, edgecolors="none")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(scatter, ax=ax, label="Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Q ensemble statistics as heatmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--q_params_path",
        type=str,
        required=True,
        help="Path to Q ensemble parameters pickle file",
    )
    parser.add_argument(
        "--replay_sample_path",
        type=str,
        required=True,
        help="Path to replay buffer sample pickle file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./q_ensemble_plots",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        help="Observation size (will try to infer from data if not provided)",
    )
    parser.add_argument(
        "--action_size",
        type=int,
        help="Action size (will try to infer from data if not provided)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for Q network",
    )
    parser.add_argument(
        "--layer_norm",
        action="store_true",
        help="Whether Q network uses layer norm",
    )
    parser.add_argument(
        "--use_target_params",
        action="store_true",
        help="Use target Q parameters instead of current Q parameters",
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=None,
        help="Number of random trajectories to plot. If None, plot all trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trajectory selection",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help="State dimension (observation size without goal). If None, will try to infer from data.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only take every stride-th state from each trajectory (default: 1, take all states)",
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.q_params_path, args.replay_sample_path)
    
    transitions = data["transitions"]
    n_critics = data["n_critics"]
    
    # Extract observations and actions
    observations = transitions.observation
    actions = transitions.action
    rewards = transitions.reward
    
    # Extract trajectory IDs if available (for Q(s0, a0, s) computation)
    traj_ids_for_computation = None
    if hasattr(transitions, 'extras') and transitions.extras is not None:
        if isinstance(transitions.extras, dict) and "state_extras" in transitions.extras:
            traj_ids_for_computation = transitions.extras["state_extras"].get("traj_id", None)
    
    # Infer sizes if not provided
    if args.obs_size is None:
        args.obs_size = observations.shape[-1]
        print(f"Inferred obs_size: {args.obs_size}")
    if args.action_size is None:
        args.action_size = actions.shape[-1]
        print(f"Inferred action_size: {args.action_size}")
    
    # Handle trajectory filtering
    if args.num_traj is not None:
        print(f"Randomly selecting {args.num_traj} trajectories...")
        np.random.seed(args.seed)
        
        # Check if we have trajectory IDs in extras
        if hasattr(transitions, 'extras') and transitions.extras is not None:
            if isinstance(transitions.extras, dict) and "state_extras" in transitions.extras:
                traj_ids = transitions.extras["state_extras"].get("traj_id", None)
            else:
                traj_ids = None
        else:
            traj_ids = None
        
        if traj_ids is not None:
            # We have trajectory IDs - randomly select trajectories
            traj_ids = np.asarray(traj_ids)
            
            # Flatten if needed
            if traj_ids.ndim > 1:
                traj_ids = traj_ids.flatten()
            if observations.ndim > 2:
                original_shape = observations.shape
                observations = observations.reshape(-1, observations.shape[-1])
                actions = actions.reshape(-1, actions.shape[-1])
                rewards = rewards.flatten() if rewards.ndim > 1 else rewards
                print(f"Flattened transitions from {original_shape} to {observations.shape}")
            
            # Get unique trajectory IDs
            unique_traj_ids = np.unique(traj_ids)
            
            # Randomly select trajectories
            num_available = len(unique_traj_ids)
            num_to_select = min(args.num_traj, num_available)
            selected_traj_ids = np.random.choice(unique_traj_ids, size=num_to_select, replace=False)
            
            # Filter to selected trajectories
            mask = np.isin(traj_ids, selected_traj_ids)
            observations = observations[mask]
            actions = actions[mask]
            rewards = rewards[mask]
            if traj_ids_for_computation is not None:
                traj_ids_for_computation = traj_ids_for_computation[mask] if traj_ids_for_computation.ndim == 1 else np.asarray(traj_ids_for_computation).flatten()[mask]
            
            print(f"Randomly selected {len(selected_traj_ids)} trajectories from {num_available} available")
            print(f"Filtered to {len(observations)} transitions from selected trajectories")
        else:
            # No trajectory IDs - assume structure is (num_envs, episode_length, ...)
            # and randomly select trajectories
            if observations.ndim == 3:
                # Shape: (num_envs, episode_length, obs_dim)
                num_envs, episode_length = observations.shape[:2]
                
                # Randomly select trajectory indices
                num_to_select = min(args.num_traj, num_envs)
                selected_indices = np.random.choice(num_envs, size=num_to_select, replace=False)
                
                # Filter observations and actions
                observations = observations[selected_indices].reshape(-1, observations.shape[-1])
                actions = actions[selected_indices].reshape(-1, actions.shape[-1])
                rewards = rewards.reshape(num_envs, episode_length)[selected_indices].flatten()
                
                # Create trajectory IDs for filtered data
                traj_ids_for_computation = np.repeat(selected_indices, episode_length)
                
                print(f"Randomly selected {len(selected_indices)} trajectories from {num_envs} available (indices: {selected_indices})")
                print(f"Filtered to {len(observations)} transitions from selected trajectories")
            else:
                print("Warning: Cannot filter by trajectory - no trajectory IDs found and data is not in trajectory format. Plotting all data.")
    else:
        # No filtering - just flatten if needed
        if observations.ndim > 2:
            # Reshape from (num_envs, episode_length, ...) to (num_envs * episode_length, ...)
            original_shape = observations.shape
            observations = observations.reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            rewards = rewards.flatten() if rewards.ndim > 1 else rewards
            print(f"Flattened transitions from {original_shape} to {observations.shape}")
    
    # Apply stride filtering if specified
    if args.stride > 1:
        print(f"Applying stride={args.stride} to filter states...")
        
        # Check if we have trajectory IDs to apply stride per-trajectory
        if traj_ids_for_computation is not None:
            traj_ids_for_computation = np.asarray(traj_ids_for_computation)
            if traj_ids_for_computation.ndim > 1:
                traj_ids_for_computation = traj_ids_for_computation.flatten()
            
            # Apply stride per trajectory
            unique_traj_ids = np.unique(traj_ids_for_computation)
            stride_mask = np.zeros(len(observations), dtype=bool)
            
            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_for_computation == traj_id
                traj_indices = np.where(traj_mask)[0]
                # Take every stride-th index within this trajectory
                stride_indices = traj_indices[::args.stride]
                stride_mask[stride_indices] = True
            
            observations = observations[stride_mask]
            actions = actions[stride_mask]
            rewards = rewards[stride_mask]
            traj_ids_for_computation = traj_ids_for_computation[stride_mask]
            
            print(f"After stride filtering: {len(observations)} transitions")
        else:
            # No trajectory IDs - apply stride globally
            stride_mask = np.arange(len(observations)) % args.stride == 0
            observations = observations[stride_mask]
            actions = actions[stride_mask]
            rewards = rewards[stride_mask]
            print(f"After stride filtering: {len(observations)} transitions (global stride)")
    
    batch_size = observations.shape[0]
    print(f"Processing {batch_size} transitions with {n_critics} critics")
    
    # Reconstruct Q network
    print("Reconstructing Q network...")
    q_network = reconstruct_q_network(
        obs_size=args.obs_size,
        action_size=args.action_size,
        n_critics=n_critics,
        hidden_sizes=tuple(args.hidden_sizes),
        layer_norm=args.layer_norm,
    )
    
    # Use identity normalizer (assuming observations are already normalized or we don't have normalizer params)
    # We'll use a dummy normalizer that just passes through the observations
    normalizer_params = running_statistics.init_state(
        specs.Array((args.obs_size,), jnp.dtype("float32"))
    )
    
    # Select which Q params to use
    q_params = data["target_q_params"] if args.use_target_params else data["q_params"]
    
    # Compute Q-values
    print("Computing Q-values...")
    q_values = compute_q_values(
        q_network,
        q_params,
        normalizer_params,
        observations,
        actions,
    )
    
    print(f"Q-values shape: {q_values.shape}")
    
    # Compute ensemble statistics
    print("Computing ensemble statistics...")
    stats = compute_ensemble_stats(q_values)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract x, y coordinates (first two dimensions of observations)
    x_coords = observations[:, 0]
    y_coords = observations[:, 1]
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Mean Q-value scatter plot
    create_scatter_plot(
        x_coords,
        y_coords,
        stats["mean"],
        title=f"Q Ensemble Mean (n_critics={n_critics})",
        xlabel="X coordinate (state[0])",
        ylabel="Y coordinate (state[1])",
        output_path=str(output_dir / "q_ensemble_mean.png"),
        cmap="viridis",
    )
    
    # 2. Std Q-value scatter plot
    create_scatter_plot(
        x_coords,
        y_coords,
        stats["std"],
        title=f"Q Ensemble Std (n_critics={n_critics})",
        xlabel="X coordinate (state[0])",
        ylabel="Y coordinate (state[1])",
        output_path=str(output_dir / "q_ensemble_std.png"),
        cmap="coolwarm",
    )
    
    # 3. Compute Q(s0, a0, s) where s0 is first state in trajectory, a0 is action at s0, s is goal
    print("Computing Q(s0, a0, s) for each state...")
    
    # Infer state_dim if not provided
    if args.state_dim is None:
        # Try to infer from observation structure
        # Observation is [state | goal], so state_dim is typically obs_size - goal_dim
        # For most environments, goal is 2D (x, y), so state_dim = obs_size - 2
        # But we can't be sure, so we'll use a heuristic or require it to be specified
        # For now, assume goal is last 2 dimensions (common for x, y goals)
        if args.obs_size >= 2:
            args.state_dim = args.obs_size - 2
            print(f"Inferred state_dim: {args.state_dim} (assuming 2D goal)")
        else:
            raise ValueError("Cannot infer state_dim. Please provide --state_dim explicitly.")
    
    # Build trajectory mapping: for each trajectory, store s0 and a0
    traj_s0 = {}
    traj_a0 = {}
    
    if traj_ids_for_computation is not None:
        # We have trajectory IDs for filtered data
        traj_ids_for_computation = np.asarray(traj_ids_for_computation)
        if traj_ids_for_computation.ndim > 1:
            traj_ids_for_computation = traj_ids_for_computation.flatten()
        
        unique_traj_ids = np.unique(traj_ids_for_computation)
        
        # For each trajectory, find s0 and a0
        for traj_id in unique_traj_ids:
            mask = traj_ids_for_computation == traj_id
            traj_obs = observations[mask]
            traj_actions = actions[mask]
            if len(traj_obs) > 0:
                # Extract state from first observation
                traj_s0[traj_id] = traj_obs[0, :args.state_dim]
                traj_a0[traj_id] = traj_actions[0]
    elif observations.ndim > 0:
        # Try to infer trajectory structure from data
        # If we don't have explicit IDs, we can't reliably compute this
        print("Warning: No trajectory IDs available. Cannot compute Q(s0, a0, s). Skipping these plots.")
        traj_ids_for_computation = None
    
    # For each filtered observation, compute Q(s0, a0, s) - BATCHED VERSION
    if traj_ids_for_computation is not None and len(traj_s0) > 0:
        # Build batched inputs
        obs_s0_goal_s_batch = []
        a0_batch = []
        goal_states_for_plot = []
        valid_indices = []
        
        for i in range(len(observations)):
            # Extract state from observation
            state_full = observations[i, :args.state_dim]
            # s is the first two elements of the state
            s = state_full[:2]
            traj_id = traj_ids_for_computation[i]
            
            if traj_id in traj_s0:
                s0 = traj_s0[traj_id]
                a0 = traj_a0[traj_id]
                
                # Construct observation [s0 | s] where s is the first two elements of the state
                obs_s0_goal_s = np.concatenate([s0, s])
                
                obs_s0_goal_s_batch.append(obs_s0_goal_s)
                a0_batch.append(a0)
                goal_states_for_plot.append(s)
                valid_indices.append(i)
        
        if len(obs_s0_goal_s_batch) > 0:
            # Batch all computations at once
            obs_s0_goal_s_batch = np.array(obs_s0_goal_s_batch)
            a0_batch = np.array(a0_batch)
            
            # Compute Q([s0 | s], a0) for all at once
            q_values_s0_a0_s = compute_q_values(
                q_network,
                q_params,
                normalizer_params,
                obs_s0_goal_s_batch,
                a0_batch,
            )
        else:
            q_values_s0_a0_s = None
            goal_states_for_plot = None
    
    if q_values_s0_a0_s is not None and len(q_values_s0_a0_s) > 0 and goal_states_for_plot is not None:
        q_values_s0_a0_s = np.array(q_values_s0_a0_s)
        goal_states_for_plot = np.array(goal_states_for_plot)
        
        # Compute ensemble statistics
        stats_s0_a0_s = compute_ensemble_stats(q_values_s0_a0_s)
        
        # Extract x, y coordinates of goal states
        x_coords_goal = goal_states_for_plot[:, 0]
        y_coords_goal = goal_states_for_plot[:, 1]
        
        # 3. Mean Q(s0, a0, s) scatter plot
        create_scatter_plot(
            x_coords_goal,
            y_coords_goal,
            stats_s0_a0_s["mean"],
            title=f"Q(s0, a0, s) Ensemble Mean (n_critics={n_critics})",
            xlabel="X coordinate (goal[0])",
            ylabel="Y coordinate (goal[1])",
            output_path=str(output_dir / "q_s0_a0_s_mean.png"),
            cmap="viridis",
        )
        
        # 4. Std Q(s0, a0, s) scatter plot
        create_scatter_plot(
            x_coords_goal,
            y_coords_goal,
            stats_s0_a0_s["std"],
            title=f"Q(s0, a0, s) Ensemble Std (n_critics={n_critics})",
            xlabel="X coordinate (goal[0])",
            ylabel="Y coordinate (goal[1])",
            output_path=str(output_dir / "q_s0_a0_s_std.png"),
            cmap="coolwarm",
        )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Mean Q-value: {np.mean(stats['mean']):.4f} ± {np.std(stats['mean']):.4f}")
    print(f"Std Q-value: {np.mean(stats['std']):.4f} ± {np.std(stats['std']):.4f}")
    print(f"Min Q-value: {np.min(stats['min']):.4f}")
    print(f"Max Q-value: {np.max(stats['max']):.4f}")
    print("=" * 60)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
