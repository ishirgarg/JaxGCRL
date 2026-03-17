"""Visualization functions for trajectory analysis."""

import logging
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import wandb
from .utils import sample_trajectories_from_buffer, sample_trajectory_sequences


def create_kde_heatmap(
    positions: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    grid_resolution: int = 100,
    bandwidth: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a KDE heatmap for 2D positions.
    
    Args:
        positions: (N, 2) array of [x, y] positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        grid_resolution: Resolution of the heatmap grid
        bandwidth: Bandwidth for KDE (None for automatic)
        
    Returns:
        Tuple of (X, Y, Z) where X, Y are meshgrids and Z is the density
    """
    if len(positions) == 0:
        # Return empty heatmap
        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        x = np.linspace(x_min, x_max, grid_resolution)
        y = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        return X, Y, Z
    
    # Convert bounds to numpy
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Create grid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute KDE
    try:
        kde = gaussian_kde(positions.T, bw_method=bandwidth)
        positions_grid = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions_grid).reshape(X.shape)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback if KDE fails (e.g., not enough points or singular matrix)
        Z = np.zeros_like(X)
    
    return X, Y, Z


def plot_positions_with_heatmap(
    positions: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    title: str,
    ax: plt.Axes = None,
    alpha_points: float = 0.3,
    alpha_heatmap: float = 0.5,
    point_size: float = 1.0,
    grid_resolution: int = 100,
) -> plt.Axes:
    """
    Plot positions as scatter points with KDE heatmap overlay.
    
    Args:
        positions: (N, 2) array of [x, y] positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        alpha_points: Transparency for scatter points
        alpha_heatmap: Transparency for heatmap
        point_size: Size of scatter points
        grid_resolution: Resolution of the heatmap grid
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create KDE heatmap
    X, Y, Z = create_kde_heatmap(positions, x_bounds, y_bounds, grid_resolution)
    
    # Plot heatmap
    if np.any(Z > 0):
        ax.contourf(X, Y, Z, levels=20, alpha=alpha_heatmap, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot scatter points
    if len(positions) > 0:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=point_size,
            alpha=alpha_points,
            c='red',
            edgecolors='darkred',
            linewidths=0.5,
        )
    
    # Set bounds and labels
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_trajectory_sequences(
    trajectory_states: np.ndarray,
    trajectory_goals: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Plot trajectory sequences in a 2x2 grid, showing start, intermediate states, final state, and goal.
    
    Args:
        trajectory_states: (num_trajectories, 8, 2) array of [x, y] positions
                         [start, 6 intermediate states, final]
        trajectory_goals: (num_trajectories, 2) array of [x, y] goal positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        fig: Matplotlib figure (creates new if None)
        
    Returns:
        Matplotlib figure
    """
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    else:
        # Extract axes from existing figure if it has a 2x2 grid
        axes = fig.subplots(2, 2) if len(fig.axes) == 0 else np.array(fig.axes).reshape(2, 2)
    
    if len(trajectory_states) == 0:
        return fig
    
    # Colors for different trajectories
    colors = ['blue', 'green', 'red', 'purple']
    
    # Set bounds
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Plot each trajectory in its own subplot
    num_trajectories = min(len(trajectory_states), 4)
    for i in range(num_trajectories):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        states = trajectory_states[i]
        goal = trajectory_goals[i]
        color = colors[i % len(colors)]
        
        # Plot trajectory path: start -> intermediate -> final
        # States shape: (8, 2) = [start, 6 intermediate, final]
        ax.plot(states[:, 0], states[:, 1], 'o-', color=color, 
                linewidth=2, markersize=6, alpha=0.7)
        
        # Mark start state
        ax.plot(states[0, 0], states[0, 1], 'o', color=color, 
                markersize=10, markeredgecolor='black', markeredgewidth=2, label='Start')
        
        # Mark final state
        ax.plot(states[-1, 0], states[-1, 1], 's', color=color, 
                markersize=10, markeredgecolor='black', markeredgewidth=2, label='Final')
        
        # Plot line from final state to goal (different style)
        ax.plot([states[-1, 0], goal[0]], [states[-1, 1], goal[1]], 
                '--', color=color, linewidth=2, alpha=0.5, label='To Goal')
        
        # Mark goal
        ax.plot(goal[0], goal[1], '*', color=color, 
                markersize=15, markeredgecolor='black', markeredgewidth=1, label='Goal')
        
        # Set bounds and labels for each subplot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots if we have fewer than 4 trajectories
    for i in range(num_trajectories, 4):
        row = i // 2
        col = i % 2
        axes[row, col].axis('off')
    
    return fig


def visualize_trajectories(
    all_positions: np.ndarray,
    final_positions: np.ndarray,
    goal_positions: np.ndarray,
    trajectory_states: np.ndarray,
    trajectory_goals: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    save_path: str = None,
    figsize: Tuple[int, int] = (24, 16),
) -> plt.Figure:
    """
    Create visualization with three plots in first row and 2x2 grid of trajectories in second row.
    
    Args:
        all_positions: (N, 2) array of [x, y] positions from all states
        final_positions: (M, 2) array of [x, y] positions from final states
        goal_positions: (N, 2) array of [x, y] goal positions from all observations
        trajectory_states: (num_trajectories, 8, 2) array of trajectory sequences
        trajectory_goals: (num_trajectories, 2) array of goal positions for trajectories
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # First row: 3 plots (all states, final states, goals)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot 1: All states
    num_all_points = len(all_positions)
    plot_positions_with_heatmap(
        all_positions,
        x_bounds,
        y_bounds,
        title=f'All States in Trajectories (n={num_all_points})',
        ax=ax1,
        alpha_points=0.2,
        alpha_heatmap=0.4,
        point_size=0.5,
    )
    
    # Plot 2: Final states
    num_final_points = len(final_positions)
    plot_positions_with_heatmap(
        final_positions,
        x_bounds,
        y_bounds,
        title=f'Final States of Trajectories (n={num_final_points})',
        ax=ax2,
        alpha_points=0.4,
        alpha_heatmap=0.5,
        point_size=2.0,
    )
    
    # Plot 3: Goals
    num_goal_points = len(goal_positions)
    plot_positions_with_heatmap(
        goal_positions,
        x_bounds,
        y_bounds,
        title=f'Goals in Trajectories (n={num_goal_points})',
        ax=ax3,
        alpha_points=0.3,
        alpha_heatmap=0.5,
        point_size=1.0,
    )
    
    # Second row: 2x2 grid for trajectory sequences (spans columns 0-1)
    traj_gs = gs[1, :2].subgridspec(2, 2, hspace=0.3, wspace=0.3)
    traj_axes = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(fig.add_subplot(traj_gs[i, j]))
        traj_axes.append(row)
    traj_axes = np.array(traj_axes)
    
    # Plot trajectories in 2x2 grid
    if len(trajectory_states) > 0:
        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        
        colors = ['blue', 'green', 'red', 'purple']
        num_trajectories = min(len(trajectory_states), 4)
        
        for i in range(num_trajectories):
            row = i // 2
            col = i % 2
            ax = traj_axes[row, col]
            
            states = trajectory_states[i]
            goal = trajectory_goals[i]
            color = colors[i % len(colors)]
            
            # Plot trajectory path: start -> intermediate -> final
            ax.plot(states[:, 0], states[:, 1], 'o-', color=color, 
                    linewidth=2, markersize=6, alpha=0.7)
            
            # Mark start state
            ax.plot(states[0, 0], states[0, 1], 'o', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=2, label='Start')
            
            # Mark final state
            ax.plot(states[-1, 0], states[-1, 1], 's', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=2, label='Final')
            
            # Plot line from final state to goal (different style)
            ax.plot([states[-1, 0], goal[0]], [states[-1, 1], goal[1]], 
                    '--', color=color, linewidth=2, alpha=0.5, label='To Goal')
            
            # Mark goal
            ax.plot(goal[0], goal[1], '*', color=color, 
                    markersize=15, markeredgecolor='black', markeredgewidth=1, label='Goal')
            
            # Set bounds and labels for each subplot
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused subplots if we have fewer than 4 trajectories
        for i in range(num_trajectories, 4):
            row = i // 2
            col = i % 2
            traj_axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def all_visualizations(
    replay_buffer: Any,
    buffer_state: Any,
    env: Any,
    state_size: int,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
) -> Any:
    """
    Create all trajectory visualizations from the replay buffer.
    
    This function handles sampling trajectories, creating visualizations,
    and logging to wandb.
    
    Args:
        replay_buffer: The replay buffer instance
        buffer_state: Current buffer state
        env: Environment instance (must have x_bounds and y_bounds attributes)
        state_size: Size of state dimension
        goal_indices: Indices for x, y positions (typically [0, 1])
        rng_key: Random key for sampling
        
    Returns:
        Updated buffer_state after sampling operations
    """
    # Check if environment has bounds (maze environments)
    if not (hasattr(env, 'x_bounds') and hasattr(env, 'y_bounds')):
        return buffer_state
    
    # Sample trajectories from buffer
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return buffer_state
    
    buffer_state, all_positions, final_positions, goal_positions = sample_trajectories_from_buffer(
        replay_buffer,
        buffer_state,
        state_size=state_size,
        goal_indices=goal_indices,
        rng_key=rng_key,
    )
    
    # Sample trajectory sequences for detailed plotting
    buffer_state, trajectory_states, trajectory_goals = sample_trajectory_sequences(
        replay_buffer,
        buffer_state,
        state_size=state_size,
        goal_indices=goal_indices,
        rng_key=rng_key,
        num_trajectories=4,
    )
    
    # Only visualize if we have data
    if len(all_positions) == 0 and len(final_positions) == 0 and len(goal_positions) == 0:
        return buffer_state
    
    # Create visualization (don't save to file)
    fig = visualize_trajectories(
        all_positions,
        final_positions,
        goal_positions,
        trajectory_states,
        trajectory_goals,
        env.x_bounds,
        env.y_bounds,
        save_path=None,
    )
    
    wandb.log({"trajectory_visualization": wandb.Image(fig)})
    plt.close(fig)
    
    return buffer_state

# Module-level variable to track last visualized env_steps (for go_explore only)
_last_viz_env_steps = -1

def handle_goal_proposer_visualization(
    log_data: dict,
    goal_proposer_name: str,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    env_steps: int = -1,
) -> None:
    """
    Generic handler for goal proposer visualization.
    Dispatches to appropriate visualization function based on goal_proposer_name.
    
    Args:
        log_data: Dictionary with visualization data from goal proposer
        goal_proposer_name: Name of the goal proposer (e.g., "q_epistemic", "rb")
        x_bounds: Environment x bounds [x_min, x_max]
        y_bounds: Environment y bounds [y_min, y_max]
        env_steps: Current environment steps (for go_explore: only visualize if >= 1M steps since last)
    """
    global _last_viz_env_steps
    
    if not log_data:  # Empty dict means no visualization
        return
    
    # For go_explore: only visualize if env_steps is provided and it's been >= 1M steps since last viz
    if env_steps >= 0:
        if _last_viz_env_steps >= 0 and (env_steps - _last_viz_env_steps) < 1_000_000:
            return
        _last_viz_env_steps = env_steps
    
    if goal_proposer_name == "q_epistemic":
        # Extract data for q_epistemic visualization
        candidate_goals = log_data["candidate_goals"]
        first_obs_position = log_data["first_obs_position"]
        q_means = log_data["q_means"]
        q_stds = log_data["q_stds"]
        selected_goal = log_data.get("selected_goal")

        visualize_q_epistemic_candidates(
            candidate_goals,
            first_obs_position,
            q_means,
            q_stds,
            x_bounds,
            y_bounds,
            selected_goal,
        )
    elif goal_proposer_name == "ucgr":
        visualize_ucgr_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            minlse_scores=log_data["minlse_scores"],
            selected_goal=log_data["selected_goal"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    elif goal_proposer_name == "max_critic_to_env":
        visualize_max_critic_to_env_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            q_means=log_data["q_means"],
            env_goal=log_data["selected_goal"],
            selected_state_goal=log_data["selected_state_goal"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    # Add more goal proposer visualizations here as needed


def visualize_q_epistemic_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    q_means: np.ndarray,
    q_stds: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    selected_goal: np.ndarray,
) -> None:
    """
    Visualize Q-epistemic goal proposer candidates with mean and std Q-values.
    
    Creates two plots side by side:
    - Left: candidate states and first observation colored by mean Q-value
    - Right: candidate states and first observation colored by std Q-value
    
    Args:
        candidate_goals: (num_candidates, 2) array of [x, y] candidate goal positions
        first_obs_position: (2,) array of [x, y] position from first observation
        q_means: (num_candidates,) array of mean Q-values for each candidate
        q_stds: (num_candidates,) array of std Q-values for each candidate
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Ensure numpy arrays
    candidate_goals = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    q_means = np.asarray(q_means)
    q_stds = np.asarray(q_stds)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)
    selected_goal = np.asarray(selected_goal)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Left plot: colored by mean Q-value
    if len(candidate_goals) > 0:
        scatter1 = ax1.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_means,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter1, ax=ax1, label='Mean Q-value')
    
    # Plot first observation position
    ax1.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Q-Epistemic Candidates (Mean Q-value)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()
    
    ax1.scatter(
        selected_goal[0],
        selected_goal[1],
        c="lime",
        s=150,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Selected Goal",
        zorder=11,
    )

    # Right plot: colored by std Q-value
    if len(candidate_goals) > 0:
        scatter2 = ax2.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_stds,
            cmap='plasma',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter2, ax=ax2, label='Std Q-value')
    
    # Plot first observation position
    ax2.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    # Optionally plot selected goal on right subplot
    if selected_goal is not None and selected_goal.size > 0:
        ax2.scatter(
            selected_goal[0],
            selected_goal[1],
            c="lime",
            s=150,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            label="Selected Goal",
            zorder=11,
        )

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_title('Q-Epistemic Candidates (Std Q-value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"q_epistemic_candidates": wandb.Image(fig)})
    plt.close(fig)


def visualize_max_critic_to_env_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    q_means: np.ndarray,
    env_goal: np.ndarray,
    selected_state_goal: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """
    Visualize max_critic_to_env goal proposer candidates colored by mean Q-value.
    
    Shows:
    - Candidate states colored by their mean Q-value (with the random env goal g)
    - The random environment goal g (selected goal)
    - The state that maximizes Q(w, g) (selected_state_goal)
    - The first observation position
    
    Args:
        candidate_goals: (num_candidates, 2) array of [x, y] candidate goal positions
        first_obs_position: (2,) array of [x, y] position from first observation
        q_means: (num_candidates,) array of mean Q-values for each candidate
        env_goal: (2,) array of [x, y] position of the random environment goal g
        selected_state_goal: (2,) array of [x, y] position of the state that maximizes Q
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Ensure numpy arrays
    candidate_goals = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    q_means = np.asarray(q_means)
    env_goal = np.asarray(env_goal)
    selected_state_goal = np.asarray(selected_state_goal)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Plot candidate states colored by mean Q-value
    if len(candidate_goals) > 0:
        scatter = ax.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_means,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label='Mean Q-value')
    
    # Plot first observation position
    ax.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    # Plot the random environment goal g (selected goal)
    ax.scatter(
        env_goal[0],
        env_goal[1],
        c="lime",
        s=150,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Selected Goal (Env Goal g)",
        zorder=11,
    )
    
    # Plot the state that maximizes Q(w, g)
    ax.scatter(
        selected_state_goal[0],
        selected_state_goal[1],
        c="orange",
        s=150,
        marker="D",
        edgecolors="black",
        linewidths=1.5,
        label="Max Q State w",
        zorder=11,
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Max Critic to Env: Candidates Colored by Q(w, g)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"max_critic_to_env_candidates": wandb.Image(fig)})
    plt.close(fig)


def visualize_ucgr_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    minlse_scores: np.ndarray,
    selected_goal: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """
    Visualize UCGR goal-proposer candidates coloured by their MinLSE score.

    Two side-by-side panels:
    - Left:  candidates coloured by raw MinLSE score (lower = harder = frontier).
             Cool colours → hard; warm colours → easy.  Selected goal marked in lime.
    - Right: candidates coloured by normalised *difficulty* (1 − normalised score),
             so the selected (hardest) goal is the brightest point.  Useful for
             seeing at a glance where the frontier of the agent's capability lies.

    Both panels also show the agent's start position as a red star.

    Args:
        candidate_goals:    (num_candidates, 2) [x, y] positions of candidate goals.
        first_obs_position: (2,) [x, y] of the agent's start position this step.
        minlse_scores:      (num_candidates,) MinLSE reachability score per candidate.
                            Lower score → harder to reach → more useful for training.
        selected_goal:      (2,) [x, y] of the chosen (lowest-score) goal.
        x_bounds:           [x_min, x_max] for the environment.
        y_bounds:           [y_min, y_max] for the environment.
    """
    # ── Ensure plain numpy ───────────────────────────────────────────────────
    candidate_goals    = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    minlse_scores      = np.asarray(minlse_scores)
    selected_goal      = np.asarray(selected_goal)
    x_bounds           = np.asarray(x_bounds)
    y_bounds           = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # ── Normalised difficulty (inverted score, 0-1) ──────────────────────────
    score_min, score_max = minlse_scores.min(), minlse_scores.max()
    score_range = score_max - score_min
    if score_range > 0:
        # difficulty ∈ [0, 1]; 1 = hardest (lowest MinLSE score)
        difficulty = 1.0 - (minlse_scores - score_min) / score_range
    else:
        difficulty = np.ones_like(minlse_scores) * 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ── Shared helper ────────────────────────────────────────────────────────
    def _setup_ax(ax, title):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # ── Left panel: raw MinLSE score (cool = hard, warm = easy) ─────────────
    if len(candidate_goals) > 0:
        sc1 = ax1.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=minlse_scores,
            cmap="coolwarm_r",   # reversed: blue (cool) = low score = hard
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("MinLSE Score  (↓ harder to reach)", fontsize=10)

    # Start position
    ax1.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    # Selected goal
    ax1.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=180, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (hardest)", zorder=11,
    )
    _setup_ax(ax1, "UCGR Candidates — MinLSE Score")
    ax1.legend(fontsize=10, loc="best")

    # ── Right panel: normalised difficulty (bright = hardest) ────────────────
    if len(candidate_goals) > 0:
        sc2 = ax2.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=difficulty,
            cmap="plasma",
            vmin=0.0, vmax=1.0,
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Normalised Difficulty  (↑ harder)", fontsize=10)

    # Start position
    ax2.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    # Selected goal
    ax2.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=180, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (hardest)", zorder=11,
    )
    _setup_ax(ax2, "UCGR Candidates — Normalised Difficulty")
    ax2.legend(fontsize=10, loc="best")

    plt.tight_layout()
    wandb.log({"ucgr_candidates": wandb.Image(fig)})
    plt.close(fig)