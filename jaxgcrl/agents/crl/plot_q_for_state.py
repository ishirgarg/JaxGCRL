#!/usr/bin/env python3
"""CLI script to plot Q-values for a given state across replay buffer states.

This script loads saved CRL parameters and replay buffer states, computes
Q(s, pi(s, g), g) for a given state s and all replay buffer states g,
and visualizes mean and standard deviation across the ensemble.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import heapq

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.agents.crl.losses import energy_fn
from jaxgcrl.agents.crl.goals_utils import stack_ensemble_params


def load_params(step_pkl_path: str) -> Dict[str, Any]:
    """Load parameters from step_X.pkl file."""
    print(f"Loading parameters from {step_pkl_path}...")
    with open(step_pkl_path, "rb") as f:
        params = pickle.load(f)
    
    # Params is a tuple: (alpha_params, actor_params, critic_params)
    alpha_params, actor_params, critic_params = params
    
    return {
        "alpha_params": alpha_params,
        "actor_params": actor_params,
        "critic_params": critic_params,
    }


def load_replay_states(replay_path: str) -> np.ndarray:
    """Load replay buffer states from pickle file.
    
    Expected format: either a Transition object or a dict with 'observation' key.
    """
    print(f"Loading replay buffer states from {replay_path}...")
    with open(replay_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle different formats
    if hasattr(data, 'observation'):
        # Transition object
        observations = data.observation
    elif isinstance(data, dict):
        if 'observation' in data:
            observations = data['observation']
        elif 'states' in data:
            observations = data['states']
        else:
            raise ValueError(f"Unknown data format. Keys: {data.keys()}")
    else:
        observations = data
    
    # Convert to numpy if needed
    if isinstance(observations, jnp.ndarray):
        observations = np.asarray(observations)
    
    # Handle different shapes
    if observations.ndim == 3:
        # (num_envs, episode_length, obs_dim) -> flatten
        observations = observations.reshape(-1, observations.shape[-1])
    elif observations.ndim == 1:
        observations = observations[None, :]
    
    return observations


def reconstruct_networks(
    state_size: int,
    action_size: int,
    goal_size: int,
    repr_dim: int = 64,
    network_width: int = 256,
    network_depth: int = 4,
    skip_connections: int = 0,
    use_relu: bool = False,
    use_ln: bool = False,
):
    """Reconstruct the CRL networks."""
    actor = Actor(
        action_size=action_size,
        network_width=network_width,
        network_depth=network_depth,
        skip_connections=skip_connections,
        use_relu=use_relu,
        use_ln=use_ln,
    )
    
    sa_encoder = Encoder(
        repr_dim=repr_dim,
        network_width=network_width,
        network_depth=network_depth,
        skip_connections=skip_connections,
        use_relu=use_relu,
        use_ln=use_ln,
    )
    
    g_encoder = Encoder(
        repr_dim=repr_dim,
        network_width=network_width,
        network_depth=network_depth,
        skip_connections=skip_connections,
        use_relu=use_relu,
        use_ln=use_ln,
    )
    
    return actor, sa_encoder, g_encoder


def compute_q_values_for_state(
    state: np.ndarray,
    replay_states: np.ndarray,
    actor,
    actor_params: Any,
    critic_params: Any,
    sa_encoder,
    g_encoder,
    energy_fn_name: str,
    state_size: int,
    goal_indices: np.ndarray,
    obs_dim: Optional[int] = None,
) -> np.ndarray:
    """Compute Q(s, pi(s, g), g) for all replay buffer states g.
    
    Args:
        state: (state_dim,) the input state s
        replay_states: (num_replay, obs_dim) or (num_replay, state_dim) replay buffer states
        actor: Actor network
        actor_params: Actor parameters
        critic_params: Critic parameters (may be ensemble)
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function
        state_size: Size of state dimension
        goal_indices: Indices of goal coordinates in state
        obs_dim: Optional observation dimension (if replay_states are observations)
        
    Returns:
        q_values: (num_replay, num_ensemble) Q-values
    """
    num_replay = replay_states.shape[0]
    
    # Determine if replay_states are observations or just states
    # If obs_dim is provided and matches replay_states.shape[1], treat as observations
    if obs_dim is not None and replay_states.shape[1] == obs_dim:
        # replay_states are observations [state | goal]
        # Extract state portion and goal portion
        replay_state_portion = replay_states[:, :state_size]  # (num_replay, state_dim)
        replay_goals = replay_states[:, state_size:]  # (num_replay, goal_dim)
    else:
        # replay_states are just states, extract goal portion using goal_indices
        replay_state_portion = replay_states  # (num_replay, state_dim)
        replay_goals = replay_states[:, goal_indices]  # (num_replay, goal_dim)
    
    # Expand state to match number of replay states
    state_expanded = jnp.repeat(state[None, :], num_replay, axis=0)  # (num_replay, state_dim)
    
    # Check if ensemble
    is_ensemble = isinstance(critic_params["sa_encoder"], list)
    
    if is_ensemble:
        # Ensemble case
        stacked_sa_params, stacked_g_params = stack_ensemble_params(critic_params)
        num_ensemble = len(critic_params["sa_encoder"])
        
        # Create observations [s | g]
        obs = jnp.concatenate([state_expanded, replay_goals], axis=1)  # (num_replay, obs_dim)
        
        # Get actions from policy
        means, _ = actor.apply(actor_params, obs)
        actions = jnp.tanh(means)  # (num_replay, action_dim)
        
        # Compute state-action pairs
        sa_pairs = jnp.concatenate([state_expanded, actions], axis=1)  # (num_replay, state_dim + action_dim)
        
        # Compute Q-values for all ensemble members
        def compute_q_single_critic(sa_p, g_p):
            phi_sa = sa_encoder.apply(sa_p, sa_pairs)  # (num_replay, repr_dim)
            psi_g = g_encoder.apply(g_p, replay_goals)  # (num_replay, repr_dim)
            q_vals = energy_fn(energy_fn_name, phi_sa, psi_g)  # (num_replay,)
            return q_vals
        
        all_q_values = jax.vmap(compute_q_single_critic)(
            stacked_sa_params, stacked_g_params
        )  # (num_ensemble, num_replay)
        
        q_values = all_q_values.T  # (num_replay, num_ensemble)
    else:
        # Single critic case
        # Create observations [s | g]
        obs = jnp.concatenate([state_expanded, replay_goals], axis=1)  # (num_replay, obs_dim)
        
        # Get actions from policy
        means, _ = actor.apply(actor_params, obs)
        actions = jnp.tanh(means)  # (num_replay, action_dim)
        
        # Compute state-action pairs
        sa_pairs = jnp.concatenate([state_expanded, actions], axis=1)  # (num_replay, state_dim + action_dim)
        
        # Compute Q-values
        phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pairs)  # (num_replay, repr_dim)
        psi_g = g_encoder.apply(critic_params['g_encoder'], replay_goals)  # (num_replay, repr_dim)
        q_values_single = energy_fn(energy_fn_name, phi_sa, psi_g)  # (num_replay,)
        q_values = q_values_single[:, None]  # (num_replay, 1) for consistency
    
    return np.asarray(q_values)


def compute_q_matrix(
    all_states: np.ndarray,
    actor,
    actor_params: Any,
    critic_params: Any,
    sa_encoder,
    g_encoder,
    energy_fn_name: str,
    state_size: int,
    goal_indices: np.ndarray,
    use_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Q(s, pi(s, w), w) for all pairs of states (s, w).
    
    Args:
        all_states: (num_states, state_dim) all states (replay + initial + final)
        actor: Actor network
        actor_params: Actor parameters
        critic_params: Critic parameters (may be ensemble)
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function
        state_size: Size of state dimension
        goal_indices: Indices of goal coordinates in state
        use_mean: If True, use mean across ensemble; if False, use minimum
        
    Returns:
        q_matrix_mean: (num_states, num_states) Mean Q-values across ensemble
        q_matrix_std: (num_states, num_states) Std Q-values across ensemble  
        q_matrix_all: (num_states, num_states, num_ensemble) All Q-values (for ensemble case)
    """
    num_states = all_states.shape[0]
    
    # Extract goal portions from all states
    all_goals = all_states[:, goal_indices]  # (num_states, goal_dim)
    
    # Check if ensemble
    is_ensemble = isinstance(critic_params["sa_encoder"], list)
    
    if is_ensemble:
        stacked_sa_params, stacked_g_params = stack_ensemble_params(critic_params)
        num_ensemble = len(critic_params["sa_encoder"])
        
        # For each state s, compute Q(s, pi(s, w), w) for all w
        def compute_q_for_state_s(state_s):
            state_s_expanded = jnp.repeat(state_s[None, :], num_states, axis=0)  # (num_states, state_dim)
            
            # Create observations [s | w] for all w
            obs = jnp.concatenate([state_s_expanded, all_goals], axis=1)  # (num_states, obs_dim)
            
            # Get actions from policy
            means, _ = actor.apply(actor_params, obs)
            actions = jnp.tanh(means)  # (num_states, action_dim)
            
            # Compute state-action pairs
            sa_pairs = jnp.concatenate([state_s_expanded, actions], axis=1)  # (num_states, state_dim + action_dim)
            
            # Compute Q-values for all ensemble members
            def compute_q_single_critic(sa_p, g_p):
                phi_sa = sa_encoder.apply(sa_p, sa_pairs)  # (num_states, repr_dim)
                psi_g = g_encoder.apply(g_p, all_goals)  # (num_states, repr_dim)
                q_vals = energy_fn(energy_fn_name, phi_sa, psi_g)  # (num_states,)
                return q_vals
            
            all_q_values = jax.vmap(compute_q_single_critic)(
                stacked_sa_params, stacked_g_params
            )  # (num_ensemble, num_states)
            
            q_values = all_q_values.T  # (num_states, num_ensemble)
            
            # Return mean, std, and all values
            q_mean = jnp.mean(q_values, axis=1)  # (num_states,)
            q_std = jnp.std(q_values, axis=1)  # (num_states,)
            return q_mean, q_std, q_values
        
        # Compute for all states s
        q_means, q_stds, q_all = jax.vmap(compute_q_for_state_s, out_axes=(0, 0, 0))(all_states)
        # q_means: (num_states, num_states)
        # q_stds: (num_states, num_states)
        # q_all: (num_states, num_states, num_ensemble)
        
        return np.asarray(q_means), np.asarray(q_stds), np.asarray(q_all)
    else:
        # Single critic case - return same values for mean and std
        def compute_q_for_state_s(state_s):
            state_s_expanded = jnp.repeat(state_s[None, :], num_states, axis=0)  # (num_states, state_dim)
            
            # Create observations [s | w] for all w
            obs = jnp.concatenate([state_s_expanded, all_goals], axis=1)  # (num_states, obs_dim)
            
            # Get actions from policy
            means, _ = actor.apply(actor_params, obs)
            actions = jnp.tanh(means)  # (num_states, action_dim)
            
            # Compute state-action pairs
            sa_pairs = jnp.concatenate([state_s_expanded, actions], axis=1)  # (num_states, state_dim + action_dim)
            
            # Compute Q-values
            phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pairs)  # (num_states, repr_dim)
            psi_g = g_encoder.apply(critic_params['g_encoder'], all_goals)  # (num_states, repr_dim)
            q_values = energy_fn(energy_fn_name, phi_sa, psi_g)  # (num_states,)
            # For single critic, return mean (q_values), std (zeros), and all (expanded to 1 ensemble member)
            return q_values, jnp.zeros_like(q_values), q_values[:, None]  # mean, std (0 for single), all (num_states, 1)
        
        q_means, q_stds, q_all = jax.vmap(compute_q_for_state_s, out_axes=(0, 0, 0))(all_states)
        # q_means: (num_states, num_states)
        # q_stds: (num_states, num_states)  
        # q_all: (num_states, num_states, 1)
        return np.asarray(q_means), np.asarray(q_stds), np.asarray(q_all)


def dijkstra_path(
    q_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    exponentiate: bool = False,
    negate_weights: bool = True,
) -> Tuple[Optional[List[int]], float]:
    """Find shortest path from start to end using Dijkstra's algorithm.
    
    Args:
        q_matrix: (num_states, num_states) Q-value matrix (edge weights)
        start_idx: Index of start state
        end_idx: Index of end state
        exponentiate: If True, exponentiate Q-values before using as weights
        negate_weights: If True, negate weights (for Q-values where higher is better).
                       If False, use weights directly (for costs where lower is better).
        
    Returns:
        Tuple of (path, total_cost) where path is list of state indices or None, 
        and total_cost is the sum of edge weights along the path
    """
    num_states = q_matrix.shape[0]
    
    # Exponentiate if requested
    if exponentiate:
        q_matrix = np.exp(q_matrix)
    
    # Convert Q-values to edge weights
    # Higher Q-values are better, so we negate them for Dijkstra (which finds minimum)
    # We'll use -Q as the weight, so Dijkstra minimizes -Q (which maximizes Q)
    # But if negate_weights is False, the input is already a cost matrix
    if negate_weights:
        weights = -q_matrix  # (num_states, num_states)
    else:
        weights = q_matrix  # (num_states, num_states) - already costs
    
    # Initialize distances
    distances = np.full(num_states, np.inf)
    distances[start_idx] = 0.0
    
    # Initialize previous nodes
    previous = np.full(num_states, -1, dtype=int)
    
    # Priority queue: (distance, node_idx)
    pq = [(0.0, start_idx)]
    visited = set()
    
    while pq:
        current_dist, current_idx = heapq.heappop(pq)
        
        if current_idx in visited:
            continue
        
        visited.add(current_idx)
        
        if current_idx == end_idx:
            # Reconstruct path and compute total cost
            path = []
            node = end_idx
            total_cost = 0.0
            prev_node = -1
            while node != -1:
                path.append(node)
                if prev_node != -1:
                    total_cost += weights[prev_node, node]
                prev_node = node
                node = previous[node]
            path.reverse()
            return path, total_cost
        
        # Check all neighbors
        for neighbor_idx in range(num_states):
            if neighbor_idx == current_idx:
                continue
            
            edge_weight = weights[current_idx, neighbor_idx]
            if np.isinf(edge_weight) or np.isnan(edge_weight):
                continue
            
            new_dist = current_dist + edge_weight
            
            if new_dist < distances[neighbor_idx]:
                distances[neighbor_idx] = new_dist
                previous[neighbor_idx] = current_idx
                heapq.heappush(pq, (new_dist, neighbor_idx))
    
    # No path found
    return None, np.inf


def plot_q_values(
    replay_states: np.ndarray,
    q_values: np.ndarray,
    state: np.ndarray,
    goal_indices: np.ndarray,
    state_size: int,
    obs_dim: Optional[int] = None,
    final_state: Optional[np.ndarray] = None,
    path: Optional[List[int]] = None,
    all_states: Optional[np.ndarray] = None,
    exponentiate: bool = False,
    best_path: Optional[List[int]] = None,
    best_lambda: Optional[float] = None,
    alpha: Optional[float] = None,
    output_path: Optional[str] = None,
):
    """Plot mean, std Q-values, and path.
    
    Args:
        replay_states: (num_replay, obs_dim) or (num_replay, state_dim) replay buffer states
        q_values: (num_replay, num_ensemble) Q-values
        state: (state_dim,) the input state s
        goal_indices: Indices of goal coordinates in state
        state_size: Size of state dimension
        obs_dim: Optional observation dimension (if replay_states are observations)
        final_state: (state_dim,) optional final state
        path: Optional list of state indices representing the path
        all_states: (num_states, state_dim) all states if path is provided
        exponentiate: If True, exponentiate Q-values before computing statistics
        output_path: Optional path to save plot
    """
    # Exponentiate if requested
    if exponentiate:
        q_values = np.exp(q_values)
    
    # Compute statistics
    q_means = np.mean(q_values, axis=1)  # (num_replay,)
    q_stds = np.std(q_values, axis=1)  # (num_replay,)
    
    # Extract goal coordinates for plotting
    # Determine if replay_states are observations or just states
    if obs_dim is not None and replay_states.shape[1] == obs_dim:
        # replay_states are observations [state | goal]
        # Extract goal portion
        goal_coords = replay_states[:, state_size:]  # (num_replay, goal_dim)
    else:
        # replay_states are just states, extract goal portion using goal_indices
        goal_coords = replay_states[:, goal_indices]  # (num_replay, goal_dim)
    
    x_coords = goal_coords[:, 0]
    y_coords = goal_coords[:, 1]
    
    # Extract state coordinates for plotting
    state_goal = state[goal_indices]
    state_x = state_goal[0]
    state_y = state_goal[1]
    
    # Create plots (4 subplots if best_path is provided, 3 if path, 2 otherwise)
    num_plots = 2
    if path is not None and all_states is not None:
        num_plots = 3
    if best_path is not None and all_states is not None:
        num_plots = 4
    
    if num_plots == 4:
        fig, axes = plt.subplots(1, 4, figsize=(32, 6))
    elif num_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean plot
    scatter1 = axes[0].scatter(x_coords, y_coords, c=q_means, cmap='viridis', s=50, alpha=0.7)
    axes[0].scatter(state_x, state_y, c='red', marker='*', s=300, edgecolors='black', 
                    linewidths=1.5, label='Input State s', zorder=10)
    if final_state is not None:
        final_goal = final_state[goal_indices]
        axes[0].scatter(final_goal[0], final_goal[1], c='blue', marker='s', s=300, 
                       edgecolors='black', linewidths=1.5, label='Final State', zorder=10)
    plt.colorbar(scatter1, ax=axes[0], label='Mean Q-value')
    axes[0].set_xlabel('Goal X coordinate')
    axes[0].set_ylabel('Goal Y coordinate')
    axes[0].set_title(f'Q(s, pi(s, g), g) Mean (n_critics={q_values.shape[1]})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Std plot
    scatter2 = axes[1].scatter(x_coords, y_coords, c=q_stds, cmap='coolwarm', s=50, alpha=0.7)
    axes[1].scatter(state_x, state_y, c='red', marker='*', s=300, edgecolors='black', 
                    linewidths=1.5, label='Input State s', zorder=10)
    if final_state is not None:
        final_goal = final_state[goal_indices]
        axes[1].scatter(final_goal[0], final_goal[1], c='blue', marker='s', s=300, 
                       edgecolors='black', linewidths=1.5, label='Final State', zorder=10)
    plt.colorbar(scatter2, ax=axes[1], label='Std Q-value')
    axes[1].set_xlabel('Goal X coordinate')
    axes[1].set_ylabel('Goal Y coordinate')
    axes[1].set_title(f'Q(s, pi(s, g), g) Std (n_critics={q_values.shape[1]})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Path plot (if path is provided)
    if path is not None and all_states is not None:
        # Extract goal coordinates for all states
        all_goal_coords = all_states[:, goal_indices]  # (num_states, goal_dim)
        all_x = all_goal_coords[:, 0]
        all_y = all_goal_coords[:, 1]
        
        # Plot all states
        axes[2].scatter(all_x, all_y, c='lightgray', s=30, alpha=0.3, label='All States')
        
        # Plot path
        path_x = all_x[path]
        path_y = all_y[path]
        axes[2].plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=8, 
                     label='Path', zorder=5)
        
        # Plot start and end
        axes[2].scatter(path_x[0], path_y[0], c='green', marker='*', s=400, 
                       edgecolors='black', linewidths=2, label='Start', zorder=10)
        axes[2].scatter(path_x[-1], path_y[-1], c='blue', marker='s', s=400, 
                       edgecolors='black', linewidths=2, label='End', zorder=10)
        
        axes[2].set_xlabel('Goal X coordinate')
        axes[2].set_ylabel('Goal Y coordinate')
        axes[2].set_title(f'Dijkstra Path (length={len(path)})')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_aspect('equal', adjustable='box')
    
    # Best path plot (if provided)
    if best_path is not None and all_states is not None:
        # Extract goal coordinates for all states
        all_goal_coords = all_states[:, goal_indices]  # (num_states, goal_dim)
        all_x = all_goal_coords[:, 0]
        all_y = all_goal_coords[:, 1]
        
        # Plot all states
        axes[3].scatter(all_x, all_y, c='lightgray', s=30, alpha=0.3, label='All States')
        
        # Plot best path
        path_x = all_x[best_path]
        path_y = all_y[best_path]
        axes[3].plot(path_x, path_y, 'purple', linewidth=2.5, marker='o', markersize=8, 
                     label=f'Best Path (λ={best_lambda:.4f})', zorder=5)
        
        # Plot start and end
        axes[3].scatter(path_x[0], path_y[0], c='green', marker='*', s=400, 
                       edgecolors='black', linewidths=2, label='Start', zorder=10)
        axes[3].scatter(path_x[-1], path_y[-1], c='blue', marker='s', s=400, 
                       edgecolors='black', linewidths=2, label='End', zorder=10)
        
        axes[3].set_xlabel('Goal X coordinate')
        axes[3].set_ylabel('Goal Y coordinate')
        if alpha is not None:
            axes[3].set_title(f'Best Configuration Path (λ={best_lambda:.4f}, α={alpha:.4f})')
        else:
            axes[3].set_title(f'Best Configuration Path (λ={best_lambda:.4f})')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        axes[3].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Q-values for a given state across replay buffer states"
    )
    parser.add_argument(
        "--step-pkl",
        type=str,
        required=True,
        help="Path to step_X.pkl file containing saved parameters"
    )
    parser.add_argument(
        "--replay-path",
        type=str,
        required=True,
        help="Path to replay buffer states (pickle file)"
    )
    parser.add_argument(
        "--x",
        type=float,
        required=True,
        help="X coordinate of input state"
    )
    parser.add_argument(
        "--y",
        type=float,
        required=True,
        help="Y coordinate of input state"
    )
    parser.add_argument(
        "--final-x",
        type=float,
        default=None,
        help="X coordinate of final goal state (for pathfinding)"
    )
    parser.add_argument(
        "--final-y",
        type=float,
        default=None,
        help="Y coordinate of final goal state (for pathfinding)"
    )
    parser.add_argument(
        "--state-size",
        type=int,
        required=True,
        help="Size of state dimension"
    )
    parser.add_argument(
        "--action-size",
        type=int,
        required=True,
        help="Size of action dimension"
    )
    parser.add_argument(
        "--goal-indices",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Indices of goal coordinates in state (default: [0, 1])"
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=None,
        help="Observation dimension (if replay states are observations, not just states)"
    )
    parser.add_argument(
        "--energy-fn",
        type=str,
        default="norm",
        choices=["norm", "dot", "cosine", "l2"],
        help="Energy function name (default: norm)"
    )
    parser.add_argument(
        "--repr-dim",
        type=int,
        default=64,
        help="Representation dimension (default: 64)"
    )
    parser.add_argument(
        "--network-width",
        type=int,
        default=256,
        help="Network width (default: 256)"
    )
    parser.add_argument(
        "--network-depth",
        type=int,
        default=4,
        help="Network depth (default: 4)"
    )
    parser.add_argument(
        "--skip-connections",
        type=int,
        default=0,
        help="Skip connections frequency (default: 0)"
    )
    parser.add_argument(
        "--use-relu",
        action="store_true",
        help="Use ReLU activation (default: Swish)"
    )
    parser.add_argument(
        "--use-ln",
        action="store_true",
        help="Use layer normalization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: show interactively)"
    )
    parser.add_argument(
        "--exponentiate",
        action="store_true",
        help="Exponentiate Q-values before computing statistics and pathfinding"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha parameter for configuration cost (default: 0.1)"
    )
    parser.add_argument(
        "--lambda-vals",
        type=float,
        nargs="+",
        default=None,
        dest="lambda_vals",
        help="List of lambda values to sweep over for configuration cost planning"
    )
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.step_pkl)
    
    # Load replay buffer states
    replay_states = load_replay_states(args.replay_path)
    
    # Construct input state from x, y coordinates
    # We need to create a full state vector with x, y at goal_indices
    state = np.zeros(args.state_size)
    state[args.goal_indices[0]] = args.x
    state[args.goal_indices[1]] = args.y
    
    # Construct final state if provided
    final_state = None
    if args.final_x is not None and args.final_y is not None:
        final_state = np.zeros(args.state_size)
        final_state[args.goal_indices[0]] = args.final_x
        final_state[args.goal_indices[1]] = args.final_y
    
    # Determine goal size from replay states
    goal_size = len(args.goal_indices)
    
    # Reconstruct networks
    actor, sa_encoder, g_encoder = reconstruct_networks(
        state_size=args.state_size,
        action_size=args.action_size,
        goal_size=goal_size,
        repr_dim=args.repr_dim,
        network_width=args.network_width,
        network_depth=args.network_depth,
        skip_connections=args.skip_connections,
        use_relu=args.use_relu,
        use_ln=args.use_ln,
    )
    
    # Compute Q-values for initial state
    print("Computing Q-values for initial state...")
    q_values = compute_q_values_for_state(
        state=state,
        replay_states=replay_states,
        actor=actor,
        actor_params=params["actor_params"],
        critic_params=params["critic_params"],
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        energy_fn_name=args.energy_fn,
        state_size=args.state_size,
        goal_indices=np.array(args.goal_indices),
        obs_dim=args.obs_dim,
    )
    
    print(f"Computed Q-values: shape {q_values.shape}")
    print(f"Mean Q-value: {np.mean(q_values):.4f}")
    print(f"Std Q-value: {np.std(q_values):.4f}")
    
    # Compute path if final state is provided
    path = None
    all_states = None
    if final_state is not None:
        print("Computing Q-matrix for all state pairs...")
        
        # Determine if replay_states are observations or just states
        if args.obs_dim is not None and replay_states.shape[1] == args.obs_dim:
            # Extract state portion from observations
            replay_state_portion = replay_states[:, :args.state_size]
        else:
            replay_state_portion = replay_states
        
        # Combine all states: replay + initial + final
        all_states = np.vstack([
            replay_state_portion,
            state[None, :],
            final_state[None, :],
        ])  # (num_replay + 2, state_dim)
        
        num_replay = replay_state_portion.shape[0]
        start_idx = num_replay  # Initial state index
        end_idx = num_replay + 1  # Final state index
        
        # Compute Q-matrix for all pairs (returns mean, std, and all values)
        q_matrix_mean, q_matrix_std, q_matrix_all = compute_q_matrix(
            all_states=all_states,
            actor=actor,
            actor_params=params["actor_params"],
            critic_params=params["critic_params"],
            sa_encoder=sa_encoder,
            g_encoder=g_encoder,
            energy_fn_name=args.energy_fn,
            state_size=args.state_size,
            goal_indices=np.array(args.goal_indices),
            use_mean=True,
        )
        
        print(f"Computed Q-matrix: mean shape {q_matrix_mean.shape}, std shape {q_matrix_std.shape}")
        
        # Run standard Dijkstra's algorithm
        print(f"Running Dijkstra's algorithm from state {start_idx} to {end_idx}...")
        path, path_cost = dijkstra_path(q_matrix_mean, start_idx, end_idx, exponentiate=args.exponentiate)
        
        if path is not None:
            print(f"Found path with {len(path)} states: {path}")
        else:
            print("No path found!")
        
        # Run configuration cost planning if lambda values are provided
        best_path = None
        best_lambda = None
        best_config_cost = np.inf
        
        if args.lambda_vals is not None and len(args.lambda_vals) > 0:
            print(f"\nRunning configuration cost planning with lambda values: {args.lambda_vals}")
            
            # Compute C = 2 * log(1 / alpha)
            C = 2 * np.log(1.0 / args.alpha)
            sqrt_C = np.sqrt(C)
            
            # Compute mean and std properly
            # f = mean(exp(Q)) - mean of exponentiated Q-values
            # s = std(Q) - standard deviation BEFORE exponentiating
            if q_matrix_all.ndim == 3 and q_matrix_all.shape[2] > 1:  # Ensemble case
                # Exponentiate all Q-values first
                q_exp_all = np.exp(q_matrix_all)  # (num_states, num_states, num_ensemble)
                # f = mean(exp(Q))
                q_exp_mean = np.mean(q_exp_all, axis=2)  # (num_states, num_states)
                # s = std(Q) - compute std BEFORE exponentiating
                q_exp_std = np.std(q_matrix_all, axis=2)  # (num_states, num_states)
            else:
                # Single critic - f = exp(q), s = 0
                q_exp_mean = np.exp(q_matrix_mean)
                q_exp_std = np.zeros_like(q_exp_mean)
            
            for lam in args.lambda_vals:
                print(f"  Testing lambda={lam:.4f}...")
                
                # Compute edge weights: -log f + (sqrt(C) / (2 lambda)) * (s^2 / f^2)
                # Avoid division by zero
                f = q_exp_mean
                s = q_exp_std
                epsilon = 1e-8
                
                # Compute s^2 / f^2, handling zeros
                f_safe = np.maximum(f, epsilon)
                s_safe = np.maximum(s, 0.0)
                ratio = (s_safe ** 2) / (f_safe ** 2)
                
                # Edge weight formula
                edge_weights = -np.log(f_safe) + (sqrt_C / (2 * lam)) * ratio
                
                # Run Dijkstra (edge_weights are already costs, so don't negate)
                path_lam, path_cost_lam = dijkstra_path(edge_weights, start_idx, end_idx, exponentiate=False, negate_weights=False)
                
                if path_lam is not None:
                    # Total configuration cost = path edge weight cost + (lambda * sqrt(C) / 2)
                    config_cost = path_cost_lam + (lam * sqrt_C / 2)
                    print(f"    Path length: {len(path_lam)}, Path cost: {path_cost_lam:.4f}, Config cost: {config_cost:.4f}")
                    
                    if config_cost < best_config_cost:
                        best_config_cost = config_cost
                        best_path = path_lam
                        best_lambda = lam
                        print(f"    -> New best! (config cost: {config_cost:.4f})")
                else:
                    print(f"    No path found for lambda={lam:.4f}")
            
            if best_path is not None:
                print(f"\nBest configuration: lambda={best_lambda:.4f}, config cost={best_config_cost:.4f}")
            else:
                print("\nNo valid path found for any lambda value")
    
    # Plot
    plot_q_values(
        replay_states=replay_states,
        q_values=q_values,
        state=state,
        goal_indices=np.array(args.goal_indices),
        state_size=args.state_size,
        obs_dim=args.obs_dim,
        final_state=final_state,
        path=path,
        all_states=all_states,
        exponentiate=args.exponentiate,
        best_path=best_path if 'best_path' in locals() else None,
        best_lambda=best_lambda if 'best_lambda' in locals() else None,
        alpha=args.alpha if args.lambda_vals is not None and len(args.lambda_vals) > 0 else None,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
