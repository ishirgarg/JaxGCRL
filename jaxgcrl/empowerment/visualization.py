"""Adapted visualization functions for empowerment-based SAC training.

These are copies of functions from jaxgcrl/agents/go_explore/visualization.py,
adapted to display BOTH ant positions AND ball positions (via goal_indices)
in addition to the original state-coverage plots.

Do NOT modify the originals in go_explore/; these are intentional copies.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import wandb


# ---------------------------------------------------------------------------
# Unchanged helpers (copied verbatim from go_explore/visualization.py)
# ---------------------------------------------------------------------------

def create_kde_heatmap(
    positions: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    grid_resolution: int = 100,
    bandwidth: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a KDE heatmap for 2D positions."""
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    if len(positions) == 0:
        return X, Y, np.zeros_like(X)
    try:
        kde = gaussian_kde(positions.T, bw_method=bandwidth)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except (np.linalg.LinAlgError, ValueError):
        Z = np.zeros_like(X)
    return X, Y, Z


def plot_positions_with_heatmap(
    positions: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    title: str,
    ax: plt.Axes = None,
    alpha_points: float = 0.3,
    alpha_heatmap: float = 0.5,
    point_size: float = 1.0,
    grid_resolution: int = 100,
    point_color: str = "red",
) -> plt.Axes:
    """Plot positions as scatter points with KDE heatmap overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    X, Y, Z = create_kde_heatmap(positions, x_bounds, y_bounds, grid_resolution)
    if np.any(Z > 0):
        ax.contourf(X, Y, Z, levels=20, alpha=alpha_heatmap, cmap="viridis")
        ax.contour(X, Y, Z, levels=10, colors="black", alpha=0.3, linewidths=0.5)

    if len(positions) > 0:
        ax.scatter(
            positions[:, 0], positions[:, 1],
            s=point_size, alpha=alpha_points,
            c=point_color, edgecolors="dark" + point_color if point_color == "red" else "black",
            linewidths=0.5,
        )

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    return ax


# ---------------------------------------------------------------------------
# Adapted: extract buffer data (ant + ball positions) for non-GC SAC
# ---------------------------------------------------------------------------

def sample_buffer_data_with_ball(
    replay_buffer: Any,
    buffer_state: Any,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
    max_points: int = 512,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Sample transitions from the replay buffer and extract both ant and ball positions.

    For empowerment SAC, the buffer stores plain state observations (no goal
    target appended).  Ant x,y are at indices 0,1; ball x,y are at goal_indices.

    Returns:
        buffer_state: updated buffer state
        ant_positions: (N, 2) array of [ant_x, ant_y]
        ball_positions: (N, 2) array of [ball_x, ball_y]
    """
    if replay_buffer.size(buffer_state) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return buffer_state, empty, empty

    buffer_state, transitions = replay_buffer.sample(buffer_state)

    # transitions.observation shape: (num_envs, episode_length, state_size)
    obs_flat = np.array(
        jnp.reshape(transitions.observation, (-1, transitions.observation.shape[-1]))
    )

    ant_positions = obs_flat[:, :2]  # ant x, y are always at indices 0, 1
    ball_positions = obs_flat[:, list(goal_indices)]  # ball xy via goal_indices

    if len(ant_positions) > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(len(ant_positions), max_points, replace=False)
        ant_positions = ant_positions[idx]
        ball_positions = ball_positions[idx]

    return buffer_state, ant_positions, ball_positions


# ---------------------------------------------------------------------------
# Adapted: all_visualizations (shows ant AND ball positions)
# ---------------------------------------------------------------------------

def all_visualizations_with_ball(
    replay_buffer: Any,
    buffer_state: Any,
    env: Any,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
    emp_map: np.ndarray = None,
    emp_extent: Tuple[float, float, float, float] = None,
) -> Any:
    """
    Create all trajectory visualizations, plotting both ant and ball positions.

    Logs a single figure to wandb with:
      - Row 0: [Ant KDE, Ball KDE, Empowerment map (optional)]
      - Row 1: [Ant final positions, Ball final positions, Ant+Ball scatter]

    Args:
        replay_buffer: The replay buffer instance.
        buffer_state: Current buffer state.
        env: Unwrapped environment (must have x_bounds and y_bounds).
        goal_indices: Indices for ball x,y within state obs (e.g. (28, 29)).
        rng_key: Random key for sampling.
        emp_map: Optional precomputed empowerment map (grid_res, grid_res).
        emp_extent: Optional (x_low, x_high, y_low, y_high) for emp_map extent.

    Returns:
        Updated buffer_state.
    """
    if not (hasattr(env, "x_bounds") and hasattr(env, "y_bounds")):
        return buffer_state
    if replay_buffer.size(buffer_state) == 0:
        return buffer_state

    buffer_state, ant_pos, ball_pos = sample_buffer_data_with_ball(
        replay_buffer, buffer_state, goal_indices, rng_key
    )

    if len(ant_pos) == 0:
        return buffer_state

    x_bounds = env.x_bounds
    y_bounds = env.y_bounds

    # Also extract final ant positions (last obs in each trajectory)
    _, transitions_raw = replay_buffer.sample(buffer_state)
    obs_raw = np.array(
        jnp.reshape(transitions_raw.observation, (-1, transitions_raw.observation.shape[-1]))
    )
    trunc_raw = np.array(
        jnp.reshape(transitions_raw.extras["state_extras"]["truncation"], (-1,))
    )
    final_mask = trunc_raw > 0.5
    ant_final = obs_raw[final_mask, :2] if np.any(final_mask) else obs_raw[:1, :2]
    ball_final = obs_raw[final_mask][:, list(goal_indices)] if np.any(final_mask) else obs_raw[:1, list(goal_indices)]

    # Build figure
    n_cols = 3 if emp_map is not None else 2
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))

    # Row 0, col 0: Ant positions KDE
    plot_positions_with_heatmap(
        ant_pos, x_bounds, y_bounds,
        title=f"Ant positions (n={len(ant_pos)})",
        ax=axes[0, 0], alpha_points=0.2, alpha_heatmap=0.5, point_size=0.8,
        point_color="steelblue",
    )

    # Row 0, col 1: Ball positions KDE
    plot_positions_with_heatmap(
        ball_pos, x_bounds, y_bounds,
        title=f"Ball positions (n={len(ball_pos)})",
        ax=axes[0, 1], alpha_points=0.2, alpha_heatmap=0.5, point_size=0.8,
        point_color="darkorange",
    )

    # Row 0, col 2 (optional): Empowerment map overlay
    if emp_map is not None and emp_extent is not None:
        ax_emp = axes[0, 2]
        x_low, x_high, y_low, y_high = emp_extent
        im = ax_emp.imshow(
            emp_map,
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="equal",
            cmap="viridis",
            alpha=0.7,
        )
        # Overlay current ant positions
        if len(ant_pos) > 0:
            ax_emp.scatter(
                ant_pos[:, 0], ant_pos[:, 1],
                s=1, c="white", alpha=0.4, edgecolors="none",
            )
        fig.colorbar(im, ax=ax_emp, fraction=0.046, pad=0.04)
        ax_emp.set_title("Empowerment + Ant coverage", fontsize=13, fontweight="bold")
        ax_emp.set_xlabel("X"); ax_emp.set_ylabel("Y")
        ax_emp.set_xlim(x_low, x_high); ax_emp.set_ylim(y_low, y_high)
        ax_emp.set_aspect("equal", adjustable="box")
        ax_emp.grid(True, alpha=0.2)

    # Row 1, col 0: Ant final positions
    plot_positions_with_heatmap(
        ant_final, x_bounds, y_bounds,
        title=f"Ant final positions (n={len(ant_final)})",
        ax=axes[1, 0], alpha_points=0.5, alpha_heatmap=0.5, point_size=2.0,
        point_color="steelblue",
    )

    # Row 1, col 1: Ball final positions
    plot_positions_with_heatmap(
        ball_final, x_bounds, y_bounds,
        title=f"Ball final positions (n={len(ball_final)})",
        ax=axes[1, 1], alpha_points=0.5, alpha_heatmap=0.5, point_size=2.0,
        point_color="darkorange",
    )

    # Row 1, col 2 (or second slot if no emp_map): Ant + Ball scatter together
    ax_both = axes[1, 2] if emp_map is not None else axes[1, 1]
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # Only draw combined scatter if we haven't overwritten it
    if emp_map is not None or n_cols == 2:
        ax_both = axes[1, n_cols - 1]
        ax_both.scatter(
            ant_pos[:, 0], ant_pos[:, 1],
            s=1, alpha=0.3, c="steelblue", label="Ant",
        )
        ax_both.scatter(
            ball_pos[:, 0], ball_pos[:, 1],
            s=1, alpha=0.3, c="darkorange", label="Ball",
        )
        ax_both.set_xlim(x_min, x_max)
        ax_both.set_ylim(y_min, y_max)
        ax_both.set_aspect("equal", adjustable="box")
        ax_both.set_title("Ant + Ball positions", fontsize=13, fontweight="bold")
        ax_both.set_xlabel("X"); ax_both.set_ylabel("Y")
        ax_both.legend(fontsize=10, loc="upper right")
        ax_both.grid(True, alpha=0.3)

    plt.suptitle("State coverage — Empowerment SAC", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    wandb.log({"trajectory_visualization": wandb.Image(fig)})
    plt.close(fig)

    return buffer_state
