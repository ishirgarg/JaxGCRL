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
    impls_root = os.path.join(ogbench_root, "impls")
    for p in (impls_root, ogbench_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    from agents import agents as agent_registry
    from utils.env_utils import make_env_and_datasets
    from utils.flax_utils import restore_agent

    return agent_registry, make_env_and_datasets, restore_agent


def _match_obs_dim_with_stack(obs: np.ndarray, target_dim: int) -> np.ndarray:
    cur_dim = int(obs.shape[-1])
    if cur_dim == target_dim:
        return obs
    if target_dim > cur_dim and target_dim % cur_dim == 0:
        stack = target_dim // cur_dim
        return np.concatenate([obs] * stack, axis=-1)
    raise ValueError(
        f"Observation dimension mismatch after stack handling: got {cur_dim}, expected {target_dim}"
    )


def _get_base_obs(base_env, ant_xy: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    """Place ant+ball, return the resulting environment observation as float32."""
    base_env.set_agent_ball_xy(
        ant_xy.astype(np.float64),
        ball_xy.astype(np.float64),
    )
    if hasattr(base_env, "get_ob"):
        return np.asarray(base_env.get_ob(), dtype=np.float32)
    return np.asarray(base_env._get_obs(), dtype=np.float32)


def compute_empowerment_map(
    agent,
    env,
    base_env,
    grid_res: int,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    *,
    # "Head": the ant+ball placement used to build the base observation template.
    # The env is reset first, then get_ob() is called once with these coords;
    # the resulting obs is the template for every grid point (only indices
    # 0,1,15,16 are overwritten per point).
    head_ant_xy: np.ndarray,
    head_ball_xy: np.ndarray,
    # Ball x,y baked into every grid-point observation (indices 15, 16).
    fixed_ball_xy: np.ndarray,
    # Optional goal (passed to env.set_goal if supported).
    goal_xy: np.ndarray | None,
    ex_obs_dim: int,
    obs0_dim: int,
) -> np.ndarray:
    if not hasattr(base_env, "set_agent_ball_xy"):
        raise RuntimeError("base_env must support set_agent_ball_xy")
    if not (hasattr(base_env, "get_ob") or hasattr(base_env, "_get_obs")):
        raise RuntimeError("base_env must support get_ob or _get_obs")

    if goal_xy is not None and hasattr(base_env, "set_goal"):
        base_env.set_goal(goal_xy=goal_xy.astype(np.float64))

    # Reset to a clean physics state before sampling the head observation.
    env.reset()

    # Build the base observation template from the head placement (one env call).
    base_obs = _get_base_obs(base_env, head_ant_xy, head_ball_xy)

    # Match frame-stacking dimension.
    if base_obs.shape[0] != obs0_dim:
        stack = obs0_dim // base_obs.shape[0]
        if stack > 1 and stack * base_obs.shape[0] == obs0_dim:
            base_obs = np.concatenate([base_obs] * stack, axis=-1)
    base_obs = _match_obs_dim_with_stack(base_obs[None, :], ex_obs_dim)[0]

    # Build obs batch: copy template, overwrite ant x/y and ball x/y per point.
    xs = np.linspace(x_low, x_high, grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    N = flat_x.shape[0]

    obs_batch = np.broadcast_to(base_obs[None, :], (N, base_obs.shape[0])).copy()
    obs_batch[:, 0]  = flat_x
    obs_batch[:, 1]  = flat_y
    obs_batch[:, 15] = float(fixed_ball_xy[0])  # ball x (OGBench qpos layout)
    obs_batch[:, 16] = float(fixed_ball_xy[1])  # ball y

    print(
        f"  head ant=({head_ant_xy[0]:.2f},{head_ant_xy[1]:.2f}) "
        f"ball=({head_ball_xy[0]:.2f},{head_ball_xy[1]:.2f}) | "
        f"obs batch: {obs_batch.shape}",
        flush=True,
    )

    # OOM-safe chunked empowerment evaluation.
    chunk_size = min(128, N)
    while chunk_size >= 1:
        try:
            emps = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                emps.append(
                    np.asarray(
                        agent.empowerment(
                            jnp.asarray(obs_batch[start:end]),
                            rng=jax.random.PRNGKey(0),
                        )
                    )
                )
            return np.concatenate(emps, axis=0).reshape(grid_res, grid_res)
        except Exception as exc:
            msg = str(exc)
            if ("RESOURCE_EXHAUSTED" not in msg and "Out of memory" not in msg) or chunk_size == 1:
                raise
            next_chunk = max(1, chunk_size // 2)
            print(f"  OOM at chunk_size={chunk_size}; retrying with {next_chunk}", flush=True)
            chunk_size = next_chunk


def main():
    parser = argparse.ArgumentParser(
        description="Plot empowerment on jaxgcrl AntBallMaze using OGBench checkpoints."
    )
    parser.add_argument("--ogbench_root", type=str, default="/home/ishir/ogbench")
    parser.add_argument("--ckpt_root", type=str, default="/home/ishir/ogbench/impls/ckpts")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--grid_res", type=int, default=40)
    parser.add_argument("--layout_name", type=str, default="square_maze",
                        choices=["square_maze", "easy_square_maze", "small_square_maze", "u_maze", "big_maze"])
    parser.add_argument("--maze_size_scaling", type=float, default=4.0)
    parser.add_argument("--backend", type=str, default="spring")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fix_ball",
        action="store_true",
        help=(
            "Sample one random ball position and reuse it for every subplot's grid observations. "
            "Each subplot still uses its own independently sampled head observation. "
            "When omitted, each subplot samples its own ball position."
        ),
    )
    args = parser.parse_args()

    agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(args.ogbench_root)

    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch   = args.epoch  if args.epoch  is not None else _latest_epoch(run_dir)

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
    seed = int(np.random.SeedSequence().generate_state(1)[0]) if args.seed is None else int(args.seed)
    jax_env.reset(jax.random.PRNGKey(seed))

    ex_obs_dim = int(example_batch["observations"].shape[-1])
    obs0_og, _ = env.reset()
    obs0_dim   = int(np.asarray(obs0_og).shape[0])

    x_low,  x_high = float(jax_env.x_bounds[0]), float(jax_env.x_bounds[1])
    y_low,  y_high = float(jax_env.y_bounds[0]), float(jax_env.y_bounds[1])

    NUM_PLOTS = 9
    rng = np.random.default_rng(seed)

    def _sample_xy() -> np.ndarray:
        return np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=np.float32)

    # Each subplot gets its own randomly sampled head (ant + ball for get_ob call).
    head_ant_xys  = [_sample_xy() for _ in range(NUM_PLOTS)]
    head_ball_xys = [_sample_xy() for _ in range(NUM_PLOTS)]

    # Ball baked into every grid-point obs: one shared random pos or one per subplot.
    if args.fix_ball:
        shared_ball = _sample_xy()
        print(f"Grid ball position fixed for all subplots: ({shared_ball[0]:.2f}, {shared_ball[1]:.2f})", flush=True)
        grid_ball_xys = [shared_ball] * NUM_PLOTS
    else:
        grid_ball_xys = [_sample_xy() for _ in range(NUM_PLOTS)]

    goal_xys = [_sample_xy() for _ in range(NUM_PLOTS)]

    maps = []
    for i, (h_ant, h_ball, g_ball, goal) in enumerate(
        zip(head_ant_xys, head_ball_xys, grid_ball_xys, goal_xys)
    ):
        print(f"\n--- Subplot {i} ---", flush=True)
        emp_map = compute_empowerment_map(
            agent=agent,
            env=env,
            base_env=base_env,
            grid_res=args.grid_res,
            x_low=x_low, x_high=x_high,
            y_low=y_low, y_high=y_high,
            head_ant_xy=h_ant,
            head_ball_xy=h_ball,
            fixed_ball_xy=g_ball,
            goal_xy=goal,
            ex_obs_dim=ex_obs_dim,
            obs0_dim=obs0_dim,
        )
        maps.append(emp_map)

    default_out_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = (
        args.output if args.output is not None
        else os.path.join(default_out_dir, f"empowerment_map_jaxgcrl_e{epoch}.png")
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flatten()
    for ax, emp_map, h_ant, h_ball, g_ball, goal in zip(
        axes, maps, head_ant_xys, head_ball_xys, grid_ball_xys, goal_xys
    ):
        im = ax.imshow(
            emp_map,
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="auto",
            cmap="viridis",
        )
        ax.scatter([g_ball[0]], [g_ball[1]], c="red",  s=55, marker="o",
                   edgecolors="white", linewidths=0.8)
        ax.scatter([goal[0]],   [goal[1]],   c="cyan", s=70, marker="*",
                   edgecolors="black", linewidths=0.7)
        ax.set_title(
            f"Head ant=({h_ant[0]:.1f},{h_ant[1]:.1f}) "
            f"ball=({h_ball[0]:.1f},{h_ball[1]:.1f})\n"
            f"Grid ball=({g_ball[0]:.1f},{g_ball[1]:.1f})  "
            f"Goal=({goal[0]:.1f},{goal[1]:.1f})",
            fontsize=7,
        )
        ax.set_xlabel("Ant x")
        ax.set_ylabel("Ant y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ball_label = (
        f"fixed grid ball=({shared_ball[0]:.2f},{shared_ball[1]:.2f})" if args.fix_ball
        else "grid ball sampled per subplot"
    )
    fig.suptitle(
        f"Empowerment — jaxgcrl AntBallMaze ({args.layout_name}) | epoch={epoch} | {ball_label}\n"
        "(each subplot uses a distinct randomly-sampled head observation)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_img, dpi=180)
    np.save(out_npy, np.stack(maps, axis=0))
    print(f"\nSaved image: {out_img}")
    print(f"Saved array stack: {out_npy}")


if __name__ == "__main__":
    main()