#!/usr/bin/env python3
"""CLI script to plot Q-values for a given state across replay buffer states.

This script loads saved CRL parameters and replay buffer states, computes
Q(s, pi(s, g), g) for a given state s and all replay buffer states g,
and visualizes mean and standard deviation across the ensemble.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

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


def plot_q_values(
    replay_states: np.ndarray,
    q_values: np.ndarray,
    state: np.ndarray,
    goal_indices: np.ndarray,
    state_size: int,
    obs_dim: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """Plot mean and std Q-values.
    
    Args:
        replay_states: (num_replay, obs_dim) or (num_replay, state_dim) replay buffer states
        q_values: (num_replay, num_ensemble) Q-values
        state: (state_dim,) the input state s
        goal_indices: Indices of goal coordinates in state
        state_size: Size of state dimension
        obs_dim: Optional observation dimension (if replay_states are observations)
        output_path: Optional path to save plot
    """
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
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean plot
    scatter1 = axes[0].scatter(x_coords, y_coords, c=q_means, cmap='viridis', s=50, alpha=0.7)
    axes[0].scatter(state_x, state_y, c='red', marker='*', s=300, edgecolors='black', 
                    linewidths=1.5, label='Input State s', zorder=10)
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
    plt.colorbar(scatter2, ax=axes[1], label='Std Q-value')
    axes[1].set_xlabel('Goal X coordinate')
    axes[1].set_ylabel('Goal Y coordinate')
    axes[1].set_title(f'Q(s, pi(s, g), g) Std (n_critics={q_values.shape[1]})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
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
    
    # Compute Q-values
    print("Computing Q-values...")
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
    
    # Plot
    plot_q_values(
        replay_states=replay_states,
        q_values=q_values,
        state=state,
        goal_indices=np.array(args.goal_indices),
        state_size=args.state_size,
        obs_dim=args.obs_dim,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
