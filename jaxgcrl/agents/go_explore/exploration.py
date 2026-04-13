"""Exploration bonus functions for reward shaping.

Each bonus function has the signature:
    bonus_fn(transitions: Transition, key: jnp.ndarray) -> jnp.ndarray
where transitions have shape (num_batches, batch_size, ...) and the
return value has shape (num_batches, batch_size).

Use ``create_exploration_bonus`` as the single entry point.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


def create_exploration_bonus(
    bonus_type: str,
    *,
    env,
    state_size: int,
    key: jax.Array,
    empowerment_run_dir: Optional[str] = None,
    empowerment_epoch: Optional[int] = None,
    empowerment_num_splus_samples: int = 128,
    empowerment_score_chunk_size: int = 32,
) -> Callable:
    """Factory that returns an exploration bonus function.

    Args:
        bonus_type: Type of bonus. Currently supported: "empowerment".
        env: Unwrapped JaxGCRL environment (for inferring obs mappings).
        state_size: State dimension (obs_size - goal_dim).
        key: JAX random key consumed during initialization.
        empowerment_run_dir: Path to OGBench empowerment checkpoint.
        empowerment_epoch: Checkpoint epoch (None = latest).
        empowerment_num_splus_samples: Agent sampling diversity parameter.
        empowerment_score_chunk_size: Batch chunk size for memory efficiency.

    Returns:
        A function ``bonus_fn(transitions, key) -> (num_batches, batch_size)``
        of per-transition bonus values.
    """
    if bonus_type == "empowerment":
        return _create_empowerment_bonus(
            env=env,
            state_size=state_size,
            key=key,
            run_dir=empowerment_run_dir,
            epoch=empowerment_epoch,
            num_splus_samples=empowerment_num_splus_samples,
            chunk_size=empowerment_score_chunk_size,
        )
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type!r}")


def _create_empowerment_bonus(
    *,
    env,
    state_size: int,
    key: jax.Array,
    run_dir: Optional[str],
    epoch: Optional[int],
    num_splus_samples: int,
    chunk_size: int,
) -> Callable:
    """Build an empowerment-based exploration bonus."""
    if run_dir is None:
        raise ValueError(
            "empowerment_run_dir must be set when exploration_bonus_type='empowerment'."
        )

    from .empowerment import (
        infer_empowerment_override_indices_from_env,
        load_offline_empowerment_agent,
        make_empowerment_obs_builder,
        make_offline_empowerment_scorer,
    )

    ogbench_idx, jax_idx = infer_empowerment_override_indices_from_env(env)
    emp_agent, _ex_obs_dim, base_obs = load_offline_empowerment_agent(
        run_dir=run_dir,
        jax_env=env,
        template_rng=key,
        epoch=epoch,
        num_splus_samples=num_splus_samples,
    )
    obs_builder = make_empowerment_obs_builder(
        jnp.asarray(base_obs),
        ogbench_idx,
        jax_idx,
        state_size=state_size,
    )
    scorer = make_offline_empowerment_scorer(
        emp_agent, obs_builder, chunk_size=chunk_size
    )

    def empowerment_bonus(transitions, bonus_key):
        states = transitions.observation[..., :state_size]
        shape = states.shape[:2]
        flat_states = states.reshape(-1, state_size)
        scores = scorer(flat_states, bonus_key)
        return scores.reshape(shape)

    return empowerment_bonus
