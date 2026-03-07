"""Visualization functions for trajectory analysis."""

import logging
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import wandb
from .utils import sample_trajectories_from_buffer


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


def visualize_trajectories(
    all_positions: np.ndarray,
    final_positions: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Create visualization with two plots: all states and final states.
    
    Args:
        all_positions: (N, 2) array of [x, y] positions from all states
        final_positions: (M, 2) array of [x, y] positions from final states
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: All states
    num_all_points = len(all_positions)
    plot_positions_with_heatmap(
        all_positions,
        x_bounds,
        y_bounds,
        title=f'All States in Trajectories (n={num_all_points})',
        ax=axes[0],
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
        ax=axes[1],
        alpha_points=0.4,
        alpha_heatmap=0.5,
        point_size=2.0,
    )
    
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
) -> None:
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
    """
    # Check if environment has bounds (maze environments)
    if not (hasattr(env, 'x_bounds') and hasattr(env, 'y_bounds')):
        return
    
    # Sample trajectories from buffer
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return
    
    all_positions, final_positions = sample_trajectories_from_buffer(
        replay_buffer,
        buffer_state,
        state_size=state_size,
        goal_indices=goal_indices,
        rng_key=rng_key,
    )
    
    # Only visualize if we have data
    if len(all_positions) == 0 and len(final_positions) == 0:
        return
    
    # Create visualization (don't save to file)
    fig = visualize_trajectories(
        all_positions,
        final_positions,
        env.x_bounds,
        env.y_bounds,
        save_path=None,
    )
    
    wandb.log({"trajectory_visualization": wandb.Image(fig)})
    plt.close(fig)