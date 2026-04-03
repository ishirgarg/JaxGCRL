import argparse
import glob
import json
import os
import re
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _latest_run_dir(ckpt_root: str) -> str:
    run_dirs = [p for p in glob.glob(os.path.join(ckpt_root, "*")) if os.path.isdir(p)]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {ckpt_root}")
    run_dirs.sort(key=os.path.getmtime)
    return run_dirs[-1]


def _latest_epoch(run_dir: str) -> int:
    ckpts = glob.glob(os.path.join(run_dir, "params_*.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl found in {run_dir}")
    epochs = []
    for path in ckpts:
        m = re.search(r"params_(\d+)\.pkl$", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f"Could not parse checkpoint epochs in {run_dir}")
    return max(epochs)


def _parse_int_pair(text: str) -> tuple[int, int]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated values, got: {text}")
    return int(parts[0]), int(parts[1])


def _setup_external_imports(ogbench_root: str):
    impls_root = os.path.join(ogbench_root, "impls")
    for p in (impls_root, ogbench_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    from agents import agents as agent_registry
    from utils.env_utils import make_env_and_datasets
    from utils.flax_utils import restore_agent

    return agent_registry, make_env_and_datasets, restore_agent


def _choose_layout(layout_name: str, square, u_maze, big_maze):
    if layout_name == "square_maze":
        return square
    if layout_name == "u_maze":
        return u_maze
    if layout_name == "big_maze":
        return big_maze
    raise ValueError(f"Unsupported layout_name: {layout_name}")


def _match_obs_dim_with_stack(obs_batch: np.ndarray, target_dim: int) -> np.ndarray:
    """Match target dim by repeating the base observation for frame-stacked checkpoints."""
    cur_dim = int(obs_batch.shape[-1])
    if cur_dim == target_dim:
        return obs_batch
    if target_dim > cur_dim and target_dim % cur_dim == 0:
        stack = target_dim // cur_dim
        return np.concatenate([obs_batch] * stack, axis=-1)
    raise ValueError(
        f"Observation dimension mismatch after stack handling: got {cur_dim}, expected {target_dim}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot empowerment on jaxgcrl AntBallMaze using OGBench checkpoints and empowerment computation."
    )
    parser.add_argument("--ogbench_root", type=str, default="/home/ishir/ogbench")
    parser.add_argument("--ckpt_root", type=str, default="/home/ishir/ogbench/impls/ckpts")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit OGBench run dir.")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch.")
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png). Defaults to run dir.")
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--grid_res", type=int, default=40)
    parser.add_argument("--layout_name", type=str, default="square_maze", choices=["square_maze", "u_maze", "big_maze"])
    parser.add_argument("--maze_size_scaling", type=float, default=4.0)
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed. If omitted, uses unseeded randomness for goal/ball sampling.",
    )
    args = parser.parse_args()

    from jaxgcrl.envs.ant_ball_maze import BIG_MAZE, SQUARE_MAZE, U_MAZE, AntBallMaze

    agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(args.ogbench_root)

    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"flags.json not found in {run_dir}")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = flags["agent"]
    agent_cfg["num_splus_samples"] = int(args.num_splus_samples)
    env_name = flags["env_name"]

    env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=agent_cfg.get("frame_stack"))
    base_env = env.unwrapped
    example_batch = train_dataset.sample(1)
    # Debug: print example shapes expected by the checkpoint
    print(
        "OGBench example shapes:",
        "observations", example_batch["observations"].shape,
        "actions", example_batch["actions"].shape,
        flush=True,
    )
    if agent_cfg.get("discrete"):
        example_batch["actions"] = np.full_like(example_batch["actions"], env.action_space.n - 1)

    agent_class = agent_registry[agent_cfg["agent_name"]]
    agent = agent_class.create(
        seed=flags.get("seed", 0),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        config=agent_cfg,
    )
    agent = restore_agent(agent, run_dir, epoch)

    jax_env = AntBallMaze(
        backend=args.backend,
        maze_layout_name=args.layout_name,
        maze_size_scaling=args.maze_size_scaling,
    )
    if args.seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        seed = int(args.seed)

    jax_env.reset(jax.random.PRNGKey(seed))
    ex_obs_dim = int(example_batch["observations"].shape[-1])
    obs0_og, _ = env.reset()
    obs0_og = np.asarray(obs0_og, dtype=np.float32)

    layout = np.array(_choose_layout(args.layout_name, SQUARE_MAZE, U_MAZE, BIG_MAZE), dtype=object)
    wall_mask = np.equal(layout, 1)
    half = 0.5 * args.maze_size_scaling
    x_low = -half
    x_high = (layout.shape[0] - 1) * args.maze_size_scaling + half
    y_low = -half
    y_high = (layout.shape[1] - 1) * args.maze_size_scaling + half
    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    def compute_empowerment_map(fixed_ball_world: tuple[float, float],
                                fixed_goal_world: tuple[float, float]) -> np.ndarray:
        if not hasattr(base_env, "set_agent_ball_xy") or not (hasattr(base_env, "get_ob") or hasattr(base_env, "_get_obs")):
            raise RuntimeError("OGBench env-native observation path is unavailable for this environment.")
        # Construct observations through the OGBench env itself (parity path).
        if hasattr(base_env, "set_goal"):
            base_env.set_goal(goal_xy=np.array(fixed_goal_world, dtype=np.float64))
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        obs_list = []
        for x, y in zip(flat_x, flat_y):
            base_env.set_agent_ball_xy(
                np.array([x, y], dtype=np.float64),
                np.array(fixed_ball_world, dtype=np.float64),
            )
            if hasattr(base_env, "get_ob"):
                obs_single_og = np.asarray(base_env.get_ob(), dtype=np.float32)
            else:
                obs_single_og = np.asarray(base_env._get_obs(), dtype=np.float32)
            if obs_single_og.shape[0] != obs0_og.shape[0]:
                stack = obs0_og.shape[0] // obs_single_og.shape[0]
                if stack > 1 and stack * obs_single_og.shape[0] == obs0_og.shape[0]:
                    obs_single_og = np.concatenate([obs_single_og] * stack, axis=-1)
            obs_list.append(obs_single_og)
        og_obs = np.stack(obs_list, axis=0)
        og_obs = _match_obs_dim_with_stack(og_obs, ex_obs_dim)
        if int(og_obs.shape[-1]) != ex_obs_dim:
            raise ValueError(f"Final observation dim mismatch: got {og_obs.shape[-1]}, expected {ex_obs_dim}")
        print("Passing in obs shapes:", og_obs.shape, flush=True)
        # OOM-safe chunked evaluation over the grid with adaptive fallback.
        og_obs_np = og_obs.astype(np.float32, copy=False)
        chunk_size = min(128, int(og_obs_np.shape[0]))
        while chunk_size >= 1:
            try:
                emps = []
                for start in range(0, og_obs_np.shape[0], chunk_size):
                    end = min(start + chunk_size, og_obs_np.shape[0])
                    emps.append(
                        np.asarray(agent.empowerment(jnp.asarray(og_obs_np[start:end]), rng=jax.random.PRNGKey(0)))
                    )
                emp = np.concatenate(emps, axis=0)
                return emp.reshape(args.grid_res, args.grid_res)
            except Exception as exc:
                msg = str(exc)
                if ("RESOURCE_EXHAUSTED" not in msg and "Out of memory" not in msg) or chunk_size == 1:
                    raise
                next_chunk = max(1, chunk_size // 2)
                print(f"OOM at chunk_size={chunk_size}; retrying with chunk_size={next_chunk}", flush=True)
                chunk_size = next_chunk

    # Build a 3x3 grid of empowerment maps at different ball and goal locations (OGBench-style, with explicit goals).
    rng = np.random.default_rng(None if args.seed is None else seed)
    ball_world_list = [
        (float(rng.uniform(low=x_low, high=x_high)), float(rng.uniform(low=y_low, high=y_high)))
        for _ in range(9)
    ]
    goal_world_list = [
        (float(rng.uniform(low=x_low, high=x_high)), float(rng.uniform(low=y_low, high=y_high)))
        for _ in range(9)
    ]
    maps = [compute_empowerment_map(bp, gp) for bp, gp in zip(ball_world_list, goal_world_list)]
    default_out_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = (
        args.output
        if args.output is not None
        else os.path.join(default_out_dir, f"empowerment_map_jaxgcrl_e{epoch}.png")
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    def _draw_walls(ax):
        # For strict parity with OGBench plot, do not draw walls.
        return

    # Plot 3x3 grid (OGBench parity)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flatten()
    for ax, emp_map, bp, gp in zip(axes, maps[:9], ball_world_list[:9], goal_world_list[:9]):
        im = ax.imshow(
            emp_map,
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="auto",
            cmap="viridis",
        )
        _draw_walls(ax)
        ax.scatter([bp[0]], [bp[1]], c="red", s=55, marker="o", edgecolors="white", linewidths=0.8)
        ax.scatter([gp[0]], [gp[1]], c="cyan", s=70, marker="*", edgecolors="black", linewidths=0.7)
        ax.set_title(f"Ball=({bp[0]:.2f}, {bp[1]:.2f})  Goal=({gp[0]:.2f}, {gp[1]:.2f})", fontsize=9)
        ax.set_xlabel("Ant x"); ax.set_ylabel("Ant y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Empowerment maps on jaxgcrl AntBallMaze ({args.layout_name}) | epoch={epoch}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_img, dpi=180)
    np.save(out_npy, np.stack(maps, axis=0))
    print(f"Saved image: {out_img}")
    print(f"Saved array stack: {out_npy}")


if __name__ == "__main__":
    main()
