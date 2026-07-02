import argparse
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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


def _setup_external_imports(ogbench_root: str):
    import sys
    impls_root = os.path.join(ogbench_root, "impls")
    for p in (impls_root, ogbench_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agents import agents as agent_registry
    from utils.env_utils import make_env_and_datasets
    from utils.flax_utils import restore_agent
    return agent_registry, make_env_and_datasets, restore_agent


def _choose_layout(name: str):
    from jaxgcrl.envs.ant_maze import U_MAZE, U_MAZE_HARD, U_MAZE_EVAL, BIG_MAZE, BIG_MAZE_HARD, BIG_MAZE_EVAL, HARDEST_MAZE, OGBENCH_MEDIUM_NAVIGATE
    if name == "u_maze": return np.array(U_MAZE, dtype=object)
    if name == "u_maze_hard": return np.array(U_MAZE_HARD, dtype=object)
    if name == "u_maze_eval": return np.array(U_MAZE_EVAL, dtype=object)
    if name == "maze_ogbench_medium_navigate": return np.array(OGBENCH_MEDIUM_NAVIGATE, dtype=object)
    if name == "big_maze": return np.array(BIG_MAZE, dtype=object)
    if name == "big_maze_hard": return np.array(BIG_MAZE_HARD, dtype=object)
    if name == "big_maze_eval": return np.array(BIG_MAZE_EVAL, dtype=object)
    if name == "hardest_maze": return np.array(HARDEST_MAZE, dtype=object)
    raise ValueError(f"Unknown layout: {name}")


def _overlay_maze(ax, layout: np.ndarray, maze_size_scaling: float):
    # AntMaze uses a corner-origin frame with xy_offset = one cell (= maze_size_scaling)
    half = 0.5 * maze_size_scaling
    xy_offset = float(maze_size_scaling)
    rows, cols = layout.shape
    for i in range(rows):
        for j in range(cols):
            if layout[i, j] == 1:
                llx = i * maze_size_scaling - xy_offset - half
                lly = j * maze_size_scaling - xy_offset - half
                ax.add_patch(plt.Rectangle(
                    (llx, lly),
                    maze_size_scaling,
                    maze_size_scaling,
                    facecolor=(0.35, 0.35, 0.35, 0.28),
                    edgecolor=(0.2, 0.2, 0.2, 0.5),
                    linewidth=0.5,
                ))


def _build_heads(env, num_heads: int, ex_obs_dim: int, seed: int) -> np.ndarray:
    # Build num_heads fixed heads by resetting and taking env obs; tile to match ex_obs_dim if needed.
    heads = []
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_heads)
    for k in keys:
        state = env.reset(k)
        base = np.asarray(state.obs, dtype=np.float32)
        if int(base.shape[-1]) != int(ex_obs_dim):
            stack = ex_obs_dim // int(base.shape[-1])
            if stack > 1 and stack * int(base.shape[-1]) == int(ex_obs_dim):
                base = np.concatenate([base] * stack, axis=-1)
            elif int(base.shape[-1]) > int(ex_obs_dim):
                base = base[:ex_obs_dim]
            else:
                pad = ex_obs_dim - int(base.shape[-1])
                base = np.pad(base, (0, pad))
        heads.append(base)
    return np.stack(heads, axis=0)  # (H, D)


def main():
    parser = argparse.ArgumentParser(description="Plot empowerment map on AntMaze (no ball): 2x2 grid with distinct heads.")
    parser.add_argument("--ogbench_root", type=str, default="/home/ishir/ogbench")
    parser.add_argument("--ckpt_root", type=str, default="/home/ishir/ogbench/impls/ckpts")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--layout_name", type=str, default="maze_ogbench_medium_navigate",
                        choices=["maze_ogbench_medium_navigate","u_maze","u_maze_hard","u_maze_eval","big_maze","big_maze_hard","big_maze_eval","hardest_maze"])
    parser.add_argument("--maze_size_scaling", type=float, default=4.0)
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument("--grid_res", type=int, default=80)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load empowerment agent from OGBench
    agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(args.ogbench_root)
    run_dir = args.run_dir if args.run_dir else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)
    flags_path = os.path.join(run_dir, "flags.json")
    with open(flags_path, "r") as f:
        flags = json.load(f)
    agent_cfg = flags["agent"]
    env_name_og = flags["env_name"]
    env_og, train_dataset, _ = make_env_and_datasets(env_name_og, frame_stack=agent_cfg.get("frame_stack"))
    example_batch = train_dataset.sample(1)
    if agent_cfg.get("discrete"):
        example_batch["actions"] = np.full_like(example_batch["actions"], env_og.action_space.n - 1)
    ex_obs_dim = int(example_batch["observations"].shape[-1])
    agent_class = agent_registry[agent_cfg["agent_name"]]
    agent = agent_class.create(
        seed=flags.get("seed", 0),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        config=agent_cfg,
    )
    agent = restore_agent(agent, run_dir, epoch)

    # Build AntMaze env (Brax) and layout/bounds
    from jaxgcrl.envs.ant_maze import AntMaze
    env = AntMaze(backend=args.backend, maze_layout_name=args.layout_name, maze_size_scaling=args.maze_size_scaling)
    layout = _choose_layout(args.layout_name)
    x_low, x_high = float(env.x_bounds[0]), float(env.x_bounds[1])
    y_low, y_high = float(env.y_bounds[0]), float(env.y_bounds[1])

    # Heads: distinct per subplot
    heads = _build_heads(env, num_heads=args.num_heads, ex_obs_dim=ex_obs_dim, seed=args.seed)

    # Grid
    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)

    # Build maps
    maps = []
    for h in heads[:4]:
        obs_batch = np.repeat(h[None, :], flat_x.shape[0], axis=0)
        # Overwrite only Ant x,y at obs[0], obs[1]
        obs_batch[:, 0] = flat_x
        obs_batch[:, 1] = flat_y
        # Independent RNG per point
        root_key = jax.random.PRNGKey(np.uint32(np.random.randint(0, 2**32 - 1)))
        keys = jax.random.split(root_key, obs_batch.shape[0])
        emp = np.asarray(
            jax.vmap(
                lambda ob, key: agent.empowerment(jnp.asarray(ob[None, :]), rng=key).squeeze(),
                in_axes=(0, 0),
            )(jnp.asarray(obs_batch), keys)
        ).reshape(args.grid_res, args.grid_res)
        maps.append(emp)

    # Plot 2x2
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = args.output if args.output else os.path.join(out_dir, f"empowerment_map_ant_maze_e{epoch}.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, emp_map in zip(axes, maps[:4]):
        im = ax.imshow(
            emp_map,
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="equal",
            cmap="viridis",
        )
        _overlay_maze(ax, layout, args.maze_size_scaling)
        ax.set_xlabel("Ant x"); ax.set_ylabel("Ant y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Empowerment maps on AntMaze ({args.layout_name}) | epoch={epoch}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_img, dpi=180)
    print(f"Saved image: {out_img}")


if __name__ == "__main__":
    main()

