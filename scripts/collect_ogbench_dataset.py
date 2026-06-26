"""Collect an OGBench-format dataset from a trained go_explore_simple policy.

Loads a goal-conditioned actor checkpoint, rolls out many episodes (each
conditioned on a proposed goal), and writes the trajectories in the exact
OGBench dataset format (``<save_path>.npz`` + ``<save_path>-val.npz``).

Example (scale2 arena ant-soccer, goals along the ant->goal line):

    python scripts/collect_ogbench_dataset.py \
        --env ant_ball_4d_ogbench_arena_1g_scale2 \
        --checkpoint runs/run_antsoccer_scale2_s_1/ckpt/final \
        --save_path datasets/antsoccer_scale2_line.npz \
        --goal_proposer line_to_goal --noise_scale 1.0 \
        --num_episodes 1000 --episode_length 250 --num_envs 256 --action_noise 0.1

Network flags default to the GoExploreSimple dataclass values; override them if
the policy was trained with different architecture hyperparameters.
"""

import argparse
import os
import pickle

import jax
import numpy as np

from jaxgcrl.agents.go_explore.algorithms import get_algorithm
from jaxgcrl.utils.collect_ogbench import (
    GOAL_PROPOSERS,
    collect_dataset,
    make_goal_proposer,
    save_ogbench_dataset,
)
from jaxgcrl.utils.env import create_env


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def load_actor_params(path):
    """Load the goal-conditioned actor params from a checkpoint.

    Checkpoints are a 3-tuple ``(alpha_params, actor_params, critic_params)`` saved
    either via ``brax.io.model.save_params`` (run.py's ``ckpt/final``, flax msgpack)
    or pickled (go_explore_simple's ``step_*.pkl``).  We try both; the actor is
    element index 1.
    """
    params = None
    try:
        from brax.io import model
        params = model.load_params(path)
    except Exception:
        with open(path, "rb") as f:
            params = pickle.load(f)
    if not isinstance(params, (tuple, list)) or len(params) < 2:
        raise ValueError(
            f"Unexpected checkpoint structure at {path}: expected a 3-tuple "
            f"(alpha, actor, critic), got {type(params)} len "
            f"{len(params) if hasattr(params, '__len__') else 'n/a'}."
        )
    return params[1]


# Architecture fields that must match the trained network for the loaded actor
# params to be valid.  These are read from the run's saved training config when
# available so the user does not have to re-type them (and cannot silently
# mis-type the ones a shape-only forward-pass probe can't catch, e.g.
# skip_connections / use_relu).
ARCH_KEYS = ("agent_type", "h_dim", "n_hidden", "skip_connections", "use_relu",
             "use_ln", "repr_dim", "discounting", "energy_fn", "n_critics")


def find_train_args_pkl(checkpoint, explicit):
    """Locate the run's ``args.pkl`` (saved by run.py next to the checkpoint)."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--train_args_pkl not found: {explicit}")
        return explicit
    ckpt = os.path.abspath(checkpoint)
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(ckpt)), "args.pkl"),  # run_dir/ckpt/final
        os.path.join(os.path.dirname(ckpt), "args.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_train_arch(path):
    """Return a dict of architecture kwargs from a run's pickled training config."""
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    agent = cfg.get("agent") if isinstance(cfg, dict) else getattr(cfg, "agent", None)
    if agent is None:
        return None
    arch = {k: getattr(agent, k) for k in ARCH_KEYS if hasattr(agent, k)}
    return arch or None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # env / checkpoint / output
    p.add_argument("--env", default="ant_ball_4d_ogbench_arena_1g_scale2")
    p.add_argument("--backend", default=None,
                   help="Override env backend (default: env's native, usually mjx).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the (alpha, actor, critic) checkpoint.")
    p.add_argument("--save_path", required=True,
                   help="Output .npz path; val written to <save_path>-val.npz.")

    # actor architecture. By default these are read from the run's args.pkl
    # (saved next to the checkpoint by run.py) so they always match training;
    # the flags below are only fallbacks / overrides.
    p.add_argument("--train_args_pkl", default=None,
                   help="Path to the run's args.pkl (default: auto-detect near --checkpoint).")
    p.add_argument("--ignore_train_args", type=str2bool, default=False,
                   help="Ignore args.pkl and use the architecture flags below verbatim.")
    p.add_argument("--agent_type", default="crl", choices=["crl", "sac"])
    p.add_argument("--h_dim", type=int, default=512)
    p.add_argument("--n_hidden", type=int, default=4)
    p.add_argument("--skip_connections", type=int, default=4)
    p.add_argument("--use_relu", type=str2bool, default=False)
    p.add_argument("--use_ln", type=str2bool, default=True)
    p.add_argument("--repr_dim", type=int, default=64)
    p.add_argument("--discounting", type=float, default=0.99)
    p.add_argument("--energy_fn", default="norm")
    p.add_argument("--n_critics", type=int, default=1)

    # goal proposer
    p.add_argument("--goal_proposer", default="line_to_goal", choices=list(GOAL_PROPOSERS))
    p.add_argument("--t_min", type=float, default=0.0)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Stddev (world units) of isotropic noise around the waypoint.")
    p.add_argument("--clip_to_bounds", type=str2bool, default=True)

    # collection
    p.add_argument("--num_episodes", type=int, default=1000,
                   help="Number of TRAIN episodes (val adds num_episodes//10 unless --val_episodes).")
    p.add_argument("--val_episodes", type=int, default=None,
                   help="Number of val episodes (default max(1, num_episodes//10); "
                        "0 = train-only, no -val.npz).")
    p.add_argument("--episode_length", type=int, default=250)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--action_noise", type=float, default=0.1)
    p.add_argument("--eps_random", type=float, default=0.0)
    p.add_argument("--deterministic", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()

    # Authoritative source for the actor architecture is the training config, so
    # re-typed flags can't silently mis-specify the network (a shape-only probe
    # cannot catch skip_connections / use_relu mismatches).
    if not args.ignore_train_args:
        pkl = find_train_args_pkl(args.checkpoint, args.train_args_pkl)
        if pkl is not None:
            try:
                arch = load_train_arch(pkl)
            except Exception as e:
                arch = None
                print(f"[collect] WARNING: could not read training config {pkl}: {e}")
            if arch:
                for k, v in arch.items():
                    setattr(args, k, v)
                print(f"[collect] actor architecture loaded from {pkl}: "
                      + ", ".join(f"{k}={arch[k]}" for k in arch))
                missing = [k for k in ARCH_KEYS if k not in arch]
                if missing:
                    print(f"[collect] WARNING: training config is missing {missing}; "
                          "falling back to CLI defaults for those (may not match training).")
            else:
                print(f"[collect] WARNING: {pkl} has no usable agent config; "
                      "using --agent_type/--h_dim/... flags (must match training).")
        else:
            print("[collect] WARNING: no training args.pkl found near --checkpoint; "
                  "using --agent_type/--h_dim/... flags verbatim (MUST match training, "
                  "incl. --skip_connections/--use_relu which the forward-pass probe cannot verify).")

    if args.agent_type not in ("crl", "sac"):
        raise ValueError(
            f"agent_type={args.agent_type!r} is not supported by this collector, which "
            "rebuilds a single goal-conditioned 'crl'/'sac' actor. The checkpoint appears "
            "to be from a different agent (e.g. a hierarchical skill controller)."
        )

    env = create_env(env_name=args.env, backend=args.backend)
    state_size = int(env.state_dim)
    goal_dim = int(len(env.goal_indices))
    obs_size = state_size + goal_dim
    action_size = int(env.action_size)
    print(f"[collect] env={args.env} state_size={state_size} goal_dim={goal_dim} "
          f"obs_size={obs_size} action_size={action_size}")

    actor, _ = get_algorithm(
        agent_type=args.agent_type,
        action_size=action_size,
        obs_size=obs_size,
        state_size=state_size,
        goal_indices=env.goal_indices,
        h_dim=args.h_dim,
        n_hidden=args.n_hidden,
        skip_connections=args.skip_connections,
        use_relu=args.use_relu,
        use_ln=args.use_ln,
        repr_dim=args.repr_dim,
        discounting=args.discounting,
        energy_fn=args.energy_fn,
        n_critics=args.n_critics,
    )

    actor_params = load_actor_params(args.checkpoint)

    # Sanity: one forward pass must succeed with the loaded params.
    probe = jax.numpy.ones((2, obs_size))
    try:
        out = actor.sample_actions(actor_params, probe, jax.random.PRNGKey(0), is_deterministic=True)
        assert out.shape == (2, action_size), f"actor output shape {out.shape} != (2, {action_size})"
    except Exception as e:
        raise RuntimeError(
            "Loaded actor params are incompatible with the rebuilt actor network. The "
            "architecture must match how the policy was trained; pass --train_args_pkl "
            "(or keep the run's args.pkl next to the checkpoint) so it is read from the "
            f"training config rather than re-typed flags. Underlying error: {e}"
        )
    print(f"[collect] actor loaded; forward pass OK "
          f"(agent_type={args.agent_type} h_dim={args.h_dim} n_hidden={args.n_hidden} "
          f"skip_connections={args.skip_connections} use_relu={args.use_relu}).")

    goal_proposer = make_goal_proposer(
        args.goal_proposer, env,
        t_min=args.t_min, t_max=args.t_max,
        noise_scale=args.noise_scale, clip_to_bounds=args.clip_to_bounds,
    )

    # Default to ~10% val but never 0 (OGBench's make_env_and_datasets requires a
    # -val.npz).  An explicit --val_episodes 0 is honored (train-only output).
    if args.val_episodes is not None:
        val_episodes = args.val_episodes
    else:
        val_episodes = max(1, args.num_episodes // 10)
    total_episodes = args.num_episodes + val_episodes
    print(f"[collect] collecting {args.num_episodes} train + {val_episodes} val episodes "
          f"x {args.episode_length} steps ({total_episodes * args.episode_length} transitions)")

    data = collect_dataset(
        env, actor, actor_params, goal_proposer,
        n_episodes=total_episodes,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        action_noise=args.action_noise,
        eps_random=args.eps_random,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    save_ogbench_dataset(
        args.save_path, data,
        train_episodes=args.num_episodes,
        val_episodes=val_episodes,
        episode_length=args.episode_length,
    )

    # Quick self-check of the produced format.
    obs = data["observations"]
    term = data["terminals"]
    n_eps = int(term.sum())
    qv = np.concatenate([data["qpos"], data["qvel"]], axis=-1)
    obs_match = obs.shape == qv.shape and bool(np.allclose(obs, qv))
    print(f"[collect] DONE. observations={obs.shape} actions={data['actions'].shape} "
          f"qpos={data['qpos'].shape} qvel={data['qvel'].shape}")
    print(f"[collect] terminals: {n_eps} episode ends (expected {total_episodes}); "
          f"obs==concat(qpos,qvel): {obs_match}")


if __name__ == "__main__":
    main()
