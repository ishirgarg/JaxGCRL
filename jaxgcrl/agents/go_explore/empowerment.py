from __future__ import annotations

import glob
import json
import os
import re
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


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


def load_offline_empowerment_agent(
    ogbench_root: str,
    run_dir: str,
    epoch: Optional[int] = None,
    num_splus_samples: int = 128,
):
    """Loads an OGBench empowerment agent checkpoint for offline scoring."""
    agent_registry, make_env_and_datasets, restore_agent = _setup_external_imports(ogbench_root)
    flags_path = os.path.join(run_dir, "flags.json")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = dict(flags["agent"])
    agent_cfg["num_splus_samples"] = int(num_splus_samples)
    env_name_og = flags["env_name"]
    env_og, train_dataset, _ = make_env_and_datasets(env_name_og, frame_stack=agent_cfg.get("frame_stack"))
    example_batch = train_dataset.sample(1)
    if agent_cfg.get("discrete"):
        example_batch["actions"] = np.full_like(example_batch["actions"], env_og.action_space.n - 1)
    ex_obs_dim = int(example_batch["observations"].shape[-1])

    agent_class = agent_registry[agent_cfg["agent_name"]]
    emp_agent = agent_class.create(
        seed=flags.get("seed", 0),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        config=agent_cfg,
    )
    use_epoch = _latest_epoch(run_dir) if epoch is None else int(epoch)
    emp_agent = restore_agent(emp_agent, run_dir, use_epoch)
    return emp_agent, ex_obs_dim


def make_empowerment_obs_builder(
    ex_obs_dim: int,
    overwrite_map: Optional[dict[int, int]] = None,
    base_obs: Optional[jnp.ndarray] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Creates a state->empowerment-observation adapter.

    - If `base_obs` is None, states are padded/truncated to `ex_obs_dim`.
    - If `base_obs` is provided, it is broadcast and overwritten at indices
      given by `overwrite_map` where map is {target_idx: source_state_idx}.
    """

    if base_obs is not None:
        base_obs = jnp.asarray(base_obs)

    def _builder(states: jnp.ndarray) -> jnp.ndarray:
        if base_obs is None:
            d = states.shape[-1]
            if d == ex_obs_dim:
                return states
            if d > ex_obs_dim:
                return states[:, :ex_obs_dim]
            pad = ex_obs_dim - d
            return jnp.pad(states, ((0, 0), (0, pad)))

        out = jnp.broadcast_to(base_obs, (states.shape[0], ex_obs_dim))
        if overwrite_map:
            for tgt_idx, src_idx in overwrite_map.items():
                out = out.at[:, int(tgt_idx)].set(states[:, int(src_idx)])
        return out

    return _builder


def make_offline_empowerment_scorer(
    emp_agent,
    obs_builder: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Returns scorer(states, rng)->empowerment score per state."""

    def _score(states: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        emp_obs = obs_builder(states)
        return emp_agent.empowerment(emp_obs, rng=rng)

    return _score

