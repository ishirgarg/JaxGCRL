from __future__ import annotations

import glob
import json
import os
import re
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np

from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_ball_maze import AntBallMaze
from jaxgcrl.envs.ant_ball_ogbench import AntBallOGBench
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.humanoid_maze import HumanoidMaze

# Same override layout as ``plot_empowerment_ant_soccer`` (AntBall / AntBallMaze) and
# ``plot_empowerment_ant_maze`` (AntMaze only overwrites ant x,y at obs[0:2]).
_BALL_SOCCER_ENV_TYPES: Tuple[Type, ...] = (AntBall, AntBallMaze)


def infer_empowerment_override_indices_from_env(
    jax_env,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return ``(ogbench_obs_indices, jaxgcrl_state_indices)`` for offline empowerment scoring.

    Ant maze: OGBench / template obs indices ``0, 1`` ← jax state ``0, 1`` (ant root xy), matching
    ``plot_empowerment_ant_maze.py``.

    Ant soccer (``AntBall``) and maze + ball (``AntBallMaze``): indices ``0, 1, 15, 16`` ←
    ``0, 1, -4, -3`` (ant xy + object xy), matching ``plot_empowerment_ant_soccer.py``.
    """
    if isinstance(jax_env, AntBallOGBench):
        # OGBench-compatible layout: ant x,y at state[0:2], ball x,y at state[15:17].
        # State indices map 1:1 to OGBench obs indices (same structure).
        return (0, 1, 15, 16), (0, 1, 15, 16)
    if isinstance(jax_env, _BALL_SOCCER_ENV_TYPES):
        return (0, 1, 15, 16), (0, 1, -4, -3)
    if isinstance(jax_env, AntMaze):
        return (0, 1), (0, 1)
    if isinstance(jax_env, HumanoidMaze):
        if not getattr(jax_env, "_is_ogbench_layout", False):
            raise ValueError(
                "Empowerment mapping for HumanoidMaze requires the OGBench obs "
                "layout (state[0:2] = root xy)."
            )
        return (0, 1), (0, 1)
    raise ValueError(
        "Offline empowerment mapping is only defined for AntMaze, AntBall, AntBallMaze, "
        "AntBallOGBench, and HumanoidMaze (OGBench layout); "
        f"got env type {type(jax_env).__name__!r}."
    )


def _jax_obs_resize_to_ex_dim(base: np.ndarray, ex_obs_dim: int) -> np.ndarray:
    """Match a single flat jax observation to ``ex_obs_dim`` (same rules as ``plot_empowerment_ant_maze``)."""
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    cur = int(base.shape[-1])
    if cur == int(ex_obs_dim):
        return base
    stack = int(ex_obs_dim) // cur
    if stack > 1 and stack * cur == int(ex_obs_dim):
        return np.concatenate([base] * stack, axis=-1)
    if cur > int(ex_obs_dim):
        return base[: int(ex_obs_dim)]
    pad = int(ex_obs_dim) - cur
    return np.pad(base, (0, pad))


def build_empowerment_base_obs_template(
    jax_env,
    ogbench_env,
    base_env,
    ex_obs_dim: int,
    template_rng: jax.Array,
) -> np.ndarray:
    """One full OGBench-shaped template vector for offline empowerment scoring.

    Ball envs (``AntBallOGBench``, ``AntBall``, ``AntBallMaze``) use the fixed
    head ``ant_xy=(0,0)``, ``ball_xy=(5,0)`` — matching ``EmpowermentSAC`` — so
    the non-xy slots of the template (velocities, quaternions) are reproducible
    across runs and independent of ``template_rng``. ``AntMaze`` has no ball;
    its template is still seeded from one jax ``reset`` since there is no
    corresponding fixed-head convention.

    Candidate-state positions are **not** baked in here; ``make_empowerment_obs_builder``
    overwrites the slots returned by ``infer_empowerment_override_indices_from_env``.
    """
    if isinstance(jax_env, (AntBallOGBench,) + _BALL_SOCCER_ENV_TYPES):
        return build_ogbench_empowerment_obs_template(
            ogbench_env,
            base_env,
            ex_obs_dim,
            head_ant_xy=(0.0, 0.0),
            head_ball_xy=(5.0, 0.0),
        )
    if isinstance(jax_env, AntMaze):
        state = jax_env.reset(template_rng)
        return _jax_obs_resize_to_ex_dim(state.obs, ex_obs_dim)
    if isinstance(jax_env, HumanoidMaze):
        if not getattr(jax_env, "_is_ogbench_layout", False):
            raise ValueError(
                "Empowerment template for HumanoidMaze requires the OGBench obs layout."
            )
        state = jax_env.reset(template_rng)
        return _jax_obs_resize_to_ex_dim(state.obs, ex_obs_dim)
    raise ValueError(
        "Empowerment base template is only defined for AntMaze, AntBall, AntBallMaze, "
        "AntBallOGBench, and HumanoidMaze (OGBench layout); "
        f"got env type {type(jax_env).__name__!r}."
    )


def infer_ogbench_root_from_run_dir(run_dir: str) -> str:
    """Infer OGBench repo root (the directory that contains ``impls/``).

    Handles:
    - Checkpoints under ``<root>/impls/ckpts/...`` (walk hits ``impls``; root is its parent).
    - Any ancestor ``<root>`` with a child directory named ``impls``.
    """
    path = os.path.abspath(run_dir)
    current = path
    while True:
        if os.path.basename(current) == "impls":
            parent = os.path.dirname(current)
            if parent != current:
                return parent
        impls = os.path.join(current, "impls")
        if os.path.isdir(impls):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise ValueError(
                f"Could not infer OGBench repo root from empowerment run_dir={run_dir!r}: "
                "no ancestor directory contains an 'impls' subdirectory, and the path is not under "
                "a directory named 'impls'."
            )
        current = parent


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


def _latest_epoch(run_dir: str) -> int:
    ckpts = glob.glob(os.path.join(run_dir, "params_*.pkl"))
    epochs = []
    for p in ckpts:
        m = re.search(r"params_(\d+)\.pkl$", os.path.basename(p))
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No params_*.pkl found in {run_dir}")
    return max(epochs)


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


def _get_base_obs_from_ogbench_env(
    base_env,
    head_ant_xy: Tuple[float, float],
    head_ball_xy: Tuple[float, float],
) -> np.ndarray:
    if hasattr(base_env, "set_agent_ball_xy"):
        base_env.set_agent_ball_xy(
            np.asarray(head_ant_xy, dtype=np.float64),
            np.asarray(head_ball_xy, dtype=np.float64),
        )
    if hasattr(base_env, "get_ob"):
        return np.asarray(base_env.get_ob(), dtype=np.float32)
    return np.asarray(base_env._get_obs(), dtype=np.float32)


def build_ogbench_empowerment_obs_template(
    env,
    base_env,
    ex_obs_dim: int,
    head_ant_xy: Tuple[float, float] = (0.0, 0.0),
    head_ball_xy: Tuple[float, float] = (5.0, 0.0),
) -> np.ndarray:
    """Single env reset + head placement, then stack to match checkpoint ``ex_obs_dim``."""
    obs0_og, _ = env.reset()
    obs0_dim = int(np.asarray(obs0_og).shape[0])
    env.reset()
    base_obs = _get_base_obs_from_ogbench_env(base_env, head_ant_xy, head_ball_xy)
    if base_obs.shape[0] != obs0_dim:
        stack = obs0_dim // base_obs.shape[0]
        if stack > 1 and stack * base_obs.shape[0] == obs0_dim:
            base_obs = np.concatenate([base_obs] * stack, axis=-1)
    base_obs = _match_obs_dim_with_stack(base_obs[None, :], ex_obs_dim)[0]
    return base_obs


def load_offline_empowerment_agent(
    run_dir: str,
    jax_env,
    template_rng: jax.Array,
    epoch: Optional[int] = None,
    num_splus_samples: int = 128,
    *,
    use_full_obs: bool = False,
    ogbench_root: Optional[str] = None,
):
    """Loads an OGBench empowerment agent checkpoint for offline scoring.

    OGBench import paths are inferred from ``run_dir`` (ancestor with ``impls/``).
    The observation template is built from ``jax_env`` and ``template_rng`` via
    ``build_empowerment_base_obs_template`` (no separate ant/ball arguments).

    When ``use_full_obs`` is True the empowerment net is fed the full state row
    directly, so the OGBench-shaped template is unused; we skip both the
    ``make_env_and_datasets`` call (which loads the multi-GB offline dataset)
    and the template construction. ``ex_obs_dim`` is derived from
    ``jax_env.state_dim`` (must equal the checkpoint's obs dim, or the
    agent restore will fail downstream) and the example batch is built from
    zero arrays.

    Returns:
        ``emp_agent``, ``ex_obs_dim``, ``base_obs_template`` (numpy, shape
        (ex_obs_dim,), or ``None`` when ``use_full_obs`` is True).
    """
    # Explicit override (e.g. checkpoints stored outside the OGBench tree); else
    # infer the repo root (dir containing impls/) from the checkpoint run_dir.
    if ogbench_root is None:
        ogbench_root = infer_ogbench_root_from_run_dir(run_dir)
    agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(ogbench_root)
    flags_path = os.path.join(run_dir, "flags.json")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = dict(flags["agent"])
    agent_cfg["num_splus_samples"] = int(num_splus_samples)
    env_name_og = flags["env_name"]

    if use_full_obs:
        if agent_cfg.get("discrete"):
            raise NotImplementedError(
                "use_full_obs=True does not support discrete OGBench checkpoints "
                "(action_space.n is only known after make_env_and_datasets)."
            )
        ex_obs_dim = int(jax_env.state_dim)
        action_size = int(jax_env.action_size)
        example_batch = {
            "observations": np.zeros((1, ex_obs_dim), dtype=np.float32),
            "actions": np.zeros((1, action_size), dtype=np.float32),
        }
        base_obs_template = None
    else:
        env_og, train_dataset, _ = make_env_and_datasets(env_name_og, frame_stack=agent_cfg.get("frame_stack"))
        example_batch = train_dataset.sample(1)
        if agent_cfg.get("discrete"):
            example_batch["actions"] = np.full_like(example_batch["actions"], env_og.action_space.n - 1)
        ex_obs_dim = int(example_batch["observations"].shape[-1])
        base_env = env_og.unwrapped if hasattr(env_og, "unwrapped") else env_og
        base_obs_template = build_empowerment_base_obs_template(
            jax_env,
            env_og,
            base_env,
            ex_obs_dim,
            template_rng,
        )

    agent_class = agent_registry[agent_cfg["agent_name"]]
    emp_agent = agent_class.create(
        seed=flags.get("seed", 0),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        config=agent_cfg,
    )
    use_epoch = _latest_epoch(run_dir) if epoch is None else int(epoch)
    emp_agent = restore_agent(emp_agent, run_dir, use_epoch)

    return emp_agent, ex_obs_dim, base_obs_template


def make_empowerment_obs_builder(
    base_obs: jnp.ndarray,
    ogbench_obs_indices: Tuple[int, ...],
    jaxgcrl_state_indices: Tuple[int, ...],
    *,
    state_size: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Maps jaxgcrl state rows to OGBench empowerment observations.

    Broadcasts the template to batch size, then copies
    ``states[:, jax_cols[k]]`` into ``out[:, ogbench_cols[k]]`` for each ``k``.
    Negative entries in ``jaxgcrl_state_indices`` are resolved with ``state_size``
    (same as Python indexing). Pairs come from ``infer_empowerment_override_indices_from_env``.
    """
    if len(ogbench_obs_indices) != len(jaxgcrl_state_indices):
        raise ValueError(
            "ogbench_obs_indices and jaxgcrl_state_indices must have the same length; "
            f"got {len(ogbench_obs_indices)} vs {len(jaxgcrl_state_indices)}."
        )
    base_obs = jnp.asarray(base_obs)
    ex_obs_dim = int(base_obs.shape[-1])
    jax_cols = jnp.array(
        [i if i >= 0 else state_size + i for i in jaxgcrl_state_indices],
        dtype=jnp.int32,
    )
    ogbench_cols = jnp.array(ogbench_obs_indices, dtype=jnp.int32)

    def _builder(states: jnp.ndarray) -> jnp.ndarray:
        out = jnp.broadcast_to(base_obs, (states.shape[0], ex_obs_dim))
        return out.at[:, ogbench_cols].set(states[:, jax_cols])

    return _builder


def make_empowerment_full_obs_builder() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Pass the full state row through as the empowerment observation.

    Used when ``use_full_empowerment`` is enabled: instead of overwriting a few
    slots of a cached OGBench template (per ``infer_empowerment_override_indices_from_env``),
    the whole ``state`` row (obs with the goal sliced off) is fed to the
    empowerment network. Assumes ``state_size`` equals the checkpoint's
    ``ex_obs_dim`` — the empowerment agent must have been trained with a
    matching observation layout.
    """
    def _builder(states: jnp.ndarray) -> jnp.ndarray:
        return states

    return _builder


def make_offline_empowerment_scorer(
    emp_agent,
    obs_builder: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    chunk_size: int = 32,
    mean: float = 0.0,
    scale: float = 1.0,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Returns scorer(states, rng) -> empowerment score per state row.

    Runs ``emp_agent.empowerment`` on batches of at most ``chunk_size`` rows so peak
    activation memory scales with ``chunk_size`` instead of ``states.shape[0]`` (e.g. all
    ``num_candidates`` at once). Uses ``lax.fori_loop`` so the trace stays JIT-friendly.

    When ``mean``/``scale`` differ from the defaults, the returned scorer emits
    ``(raw_empowerment - mean) / scale``, giving callers a single source of truth
    for the normalization rather than reapplying it at every site.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    mean_f = jnp.asarray(mean, dtype=jnp.float32)
    scale_f = jnp.asarray(scale, dtype=jnp.float32)

    def _score(states: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        n = states.shape[0]
        pad = (chunk_size - (n % chunk_size)) % chunk_size
        states_pad = jnp.pad(states, ((0, pad), (0, 0)))
        total = states_pad.shape[0]
        n_chunks = total // chunk_size
        acc0 = jnp.zeros((total,), dtype=jnp.float32)

        def body(i, acc):
            chunk = jax.lax.dynamic_slice_in_dim(
                states_pad, i * chunk_size, chunk_size, axis=0
            )
            emp_obs = obs_builder(chunk)
            ki = jax.random.fold_in(rng, i)
            s = emp_agent.empowerment(emp_obs, rng=ki)
            s = jnp.reshape(s, (chunk_size,)).astype(jnp.float32)
            return jax.lax.dynamic_update_slice(acc, s, (i * chunk_size,))

        acc = jax.lax.fori_loop(0, n_chunks, body, acc0)
        return (acc[:n] - mean_f) / scale_f

    return _score
