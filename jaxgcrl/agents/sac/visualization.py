import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import wandb

from typing import Any, Tuple

from jaxgcrl.envs.ant_ball_maze import BIG_MAZE, SQUARE_MAZE, U_MAZE


def log_dist_vs_reward_scatter(
    replay_buffer: Any,
    buffer_state: Any,
    goal_indices: Tuple[int, int],
    empowerment_reward_scaling: float,
    current_step: int,
) -> Any:
    """Logs a scatter plot of distance to ball vs unscaled empowerment reward. Returns updated buffer_state."""
    buffer_state, vis_transitions = replay_buffer.sample(buffer_state)
    obs_flat = np.array(jnp.reshape(vis_transitions.observation, (-1, vis_transitions.observation.shape[-1])))
    rew_flat = np.array(jnp.reshape(vis_transitions.reward, (-1,)))
    # Unscale to show raw empowerment values
    scale = empowerment_reward_scaling if empowerment_reward_scaling != 0.0 else 1.0
    rew_unscaled = rew_flat / float(scale)
    # With new relative observation: last 4 dims are [ball_rel(2), goal_rel(2)]
    ball_rel = obs_flat[:, -4:-2]
    dists = np.linalg.norm(ball_rel, axis=1)
    n = len(dists)
    if n == 0:
        return buffer_state
    sel = np.random.choice(n, size=min(512, n), replace=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(dists[sel], rew_unscaled[sel], s=8, alpha=0.6, c="tab:blue")
    ax.set_xlabel("Distance to ball")
    ax.set_ylabel("Empowerment reward (unscaled)")
    ax.set_title("Distance-to-ball vs empowerment reward")
    plt.tight_layout()
    wandb.log({"empowerment/dist_vs_reward": wandb.Image(fig)}, step=current_step)
    plt.close(fig)
    return buffer_state


def log_empowerment_map(
    replay_buffer: Any,
    buffer_state: Any,
    unwrapped_env: Any,
    goal_indices: Tuple[int, int],
    goal_target_indices: Tuple[int, int],
    obs_size_net: int,
    empowerment_reward_with_key,  # Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    current_step: int,
    env_name: str,
    rng_key: jnp.ndarray,
) -> Any:
    """Logs an empowerment heatmap in the same style as the standalone plotting script. Returns updated buffer_state."""
    # Layout and bounds
    if "square" in env_name:
        layout = np.array(SQUARE_MAZE, dtype=object)
        layout_name = "square_maze"
    elif "u_maze" in env_name:
        layout = np.array(U_MAZE, dtype=object)
        layout_name = "u_maze"
    elif "big_maze" in env_name:
        layout = np.array(BIG_MAZE, dtype=object)
        layout_name = "big_maze"
    else:
        layout = np.array(SQUARE_MAZE, dtype=object)
        layout_name = "square_maze"
    wall_mask = np.equal(layout, 1)
    x_low = float(unwrapped_env.x_bounds[0])
    x_high = float(unwrapped_env.x_bounds[1])
    y_low = float(unwrapped_env.y_bounds[0])
    y_high = float(unwrapped_env.y_bounds[1])
    grid_res = 80
    xs = np.linspace(x_low, x_high, grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    maze_size_scaling = (x_high - x_low) / float(layout.shape[0])
    half = 0.5 * maze_size_scaling

    # Generate a random ball and goal world positions within bounds.
    rng_np = np.random.default_rng()
    ball_world = np.array([
        float(rng_np.uniform(low=x_low, high=x_high)),
        float(rng_np.uniform(low=y_low, high=y_high)),
    ], dtype=np.float32)
    goal_world = np.array([
        float(rng_np.uniform(low=x_low, high=x_high)),
        float(rng_np.uniform(low=y_low, high=y_high)),
    ], dtype=np.float32)

    # Build grid obs and compute empowerment map
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    # Sample a base observation to copy non-relative features; fall back to zeros if buffer empty.
    buffer_state, vis_transitions = replay_buffer.sample(buffer_state)
    obs_flat = np.array(jnp.reshape(vis_transitions.observation, (-1, vis_transitions.observation.shape[-1])))
    if len(obs_flat) == 0:
        return buffer_state
    ref = obs_flat[0].astype(np.float32, copy=True)
    obs_batch = np.repeat(ref[None, :obs_size_net], flat_x.shape[0], axis=0)
    # Sweep ant positions and set relative terms from sampled ball/goal
    obs_batch[:, 0] = flat_x
    obs_batch[:, 1] = flat_y
    obs_batch[:, -4:-2] = np.stack([ball_world[0] - flat_x, ball_world[1] - flat_y], axis=1)
    goal_rel = (goal_world - ball_world).astype(np.float32)
    obs_batch[:, -2:] = goal_rel[None, :]

    emp_vals = np.asarray(empowerment_reward_with_key(jnp.asarray(obs_batch), rng_key))
    emp_map = emp_vals.reshape(grid_res, grid_res)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        emp_map,
        origin="lower",
        extent=[x_low, x_high, y_low, y_high],
        cmap="viridis",
        aspect="equal",
    )
    for i in range(layout.shape[0]):
        for j in range(layout.shape[1]):
            if wall_mask[i, j]:
                ax.add_patch(
                    Rectangle(
                        (i * maze_size_scaling - half, j * maze_size_scaling - half),
                        maze_size_scaling,
                        maze_size_scaling,
                        facecolor=(0.35, 0.35, 0.35, 0.28),
                        edgecolor=(0.2, 0.2, 0.2, 0.5),
                        linewidth=0.5,
                    )
                )
    # Overlay fixed ball and goal
    ax.scatter([ball_world[0]], [ball_world[1]], c="red", s=55, marker="o", edgecolors="white", linewidths=0.8)
    ax.scatter([goal_world[0]], [goal_world[1]], c="cyan", s=70, marker="*", edgecolors="black", linewidths=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Empowerment")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(x_low, x_high); ax.set_ylim(y_low, y_high)
    ax.set_title(f"Empowerment map on AntBallMaze ({layout_name}) | step={current_step}")
    plt.tight_layout()
    wandb.log({"empowerment/empowerment_map": wandb.Image(fig)}, step=current_step)
    plt.close(fig)
    return buffer_state

