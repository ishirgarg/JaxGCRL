"""DADS evaluation utilities.

Contains:
- Policy wrapper for evaluation (replaces the goal with a skill).
- Multi-skill success evaluation.
- Multi-skill rendering.
- Post-training empowerment heatmap over the visited x-y region.
"""

import functools
import logging
import os
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.types import PolicyParams

from jaxgcrl.utils.evaluator import Evaluator

from . import losses as dads_losses


def make_dads_eval_policy(
    base_make_policy: Callable,
    state_dim: int,
    num_skills: int,
    skill_idx: int,
) -> Callable:
    """Wrap ``make_policy`` so it accepts raw environment observations.

    During evaluation the env returns ``[state | goal]`` observations.  This
    wrapper discards the goal and appends a fixed one-hot skill so the trained
    DADS policy (which expects ``[state | z]``) can be used unchanged.
    """
    fixed_skill = jax.nn.one_hot(skill_idx, num_skills, dtype=jnp.float32)

    def wrapped_make_policy(params: PolicyParams, deterministic: bool = False):
        base_policy = base_make_policy(params, deterministic=deterministic)

        def augmented_policy(obs, key):
            state_part = obs[..., :state_dim]
            batch_shape = obs.shape[:-1]
            skill = jnp.broadcast_to(fixed_skill, batch_shape + (num_skills,))
            aug_obs = jnp.concatenate([state_part, skill], axis=-1)
            return base_policy(aug_obs, key)

        return augmented_policy

    return wrapped_make_policy


def run_multi_skill_evaluation(
    make_policy: Callable,
    state_dim: int,
    num_skills: int,
    eval_env_wrapped,
    deterministic_eval: bool,
    num_eval_envs: int,
    episode_length: int,
    action_repeat: int,
    eval_key: jax.Array,
    params: PolicyParams,
    training_metrics: Dict,
) -> Dict:
    """Run evaluation for each skill separately and aggregate metrics."""
    all_metrics = {}
    skill_metric_values: Dict[str, list] = {}
    per_skill_success: Dict[str, float] = {}

    for skill_idx in range(num_skills):
        skill_eval_make_policy = make_dads_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=skill_idx
        )
        skill_evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(skill_eval_make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=jax.random.fold_in(eval_key, skill_idx),
        )
        skill_metrics = skill_evaluator.run_evaluation(params, training_metrics)

        for key, value in skill_metrics.items():
            if key.startswith("eval/episode_success"):
                success_type = key.replace("eval/episode_", "")
                per_skill_success[f"skill_{skill_idx}_eval/{success_type}"] = value
        for key, value in skill_metrics.items():
            if (
                key.startswith("eval/episode_")
                and not key.endswith("_std")
                and not key.startswith("eval/episode_success")
            ):
                base_key = key.replace("eval/episode_", "")
                skill_metric_values.setdefault(base_key, []).append(value)

    for base_key, values in skill_metric_values.items():
        if values:
            all_metrics[f"eval/episode_{base_key}"] = np.mean(values)
    all_metrics.update(per_skill_success)
    return all_metrics


def render_all_skills(
    make_policy: Callable,
    state_dim: int,
    num_skills: int,
    params: PolicyParams,
    unwrapped_env,
    render_dir: str,
    exp_name: str,
    step: int,
):
    """Render a rollout video for each skill."""
    from jaxgcrl.utils.env import render

    for skill_idx in range(num_skills):
        skill_eval_make_policy = make_dads_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=skill_idx
        )
        render(
            skill_eval_make_policy, params, unwrapped_env,
            render_dir, f"{exp_name}_skill_{skill_idx}", step,
        )


# ===========================================================================
# Empowerment heatmap
# ===========================================================================

def _where_done(done, x, y):
    """Select ``x`` where done else ``y`` (broadcasting done over a pytree leaf)."""
    d = done.reshape([done.shape[0]] + [1] * (x.ndim - 1))
    return jnp.where(d, x, y)


def _collect_visited_xy(
    env,
    make_policy,
    params,
    state_dim: int,
    num_skills: int,
    collect_envs: int,
    collect_steps: int,
    key: jax.Array,
) -> np.ndarray:
    """Roll out the policy for all skills (auto-reset) and return visited (x,y).

    Returns an array of shape (collect_envs * collect_steps, 2).
    """
    base_policy = make_policy(
        (params["normalizer_params"], params["policy_params"]), deterministic=False
    )
    reset_default = jax.jit(jax.vmap(lambda rng: env.reset(rng)))
    step_fn = jax.jit(jax.vmap(env.step))

    # Round-robin skill assignment so every skill is represented.
    skill_idx = jnp.arange(collect_envs) % num_skills
    skills_onehot = jax.nn.one_hot(skill_idx, num_skills, dtype=jnp.float32)

    key, reset_key = jax.random.split(key)
    state = reset_default(jax.random.split(reset_key, collect_envs))
    # Reuse this initial reset state for auto-reset (the arena reset is
    # ~deterministic: one R/B/G cell), avoiding a full env reset every step.
    reset_template = state

    def f(carry, _):
        state, key = carry
        key, ak = jax.random.split(key, 2)
        aug_obs = jnp.concatenate([state.obs[:, :state_dim], skills_onehot], axis=-1)
        action, _ = base_policy(aug_obs, ak)
        nstate = step_fn(state, action)
        # Auto-reset terminated envs to the initial state so trajectories stay valid.
        nstate = jax.tree_util.tree_map(
            lambda r, n: _where_done(nstate.done, r, n), reset_template, nstate
        )
        return (nstate, key), nstate.obs[:, :2]

    _, xy = jax.lax.scan(f, (state, key), (), length=collect_steps)
    xy = np.asarray(xy).reshape(-1, 2)  # (collect_steps * collect_envs, 2)
    return xy


def _empowerment_for_cells(
    env,
    dads_network,
    make_policy,
    params,
    cell_xy: np.ndarray,            # (C, 2) ant placement (cell centres)
    ball_spawn_xy: np.ndarray,      # (2,)
    goal_arr: jnp.ndarray,          # goal passed to env.reset
    state_dim: int,
    num_skills: int,
    future_discount: float,
    rollout_horizon: int,
    num_future_samples: int,
    use_xy_prior: bool,
    goal_indices,
    non_goal_indices,
    chunk_cells: int,
    key: jax.Array,
) -> np.ndarray:
    """Empowerment at each cell: place the ant there, roll every skill, score s+.

    For each cell and skill z the policy is rolled out for ``rollout_horizon``
    steps; ``num_future_samples`` Geometric(1-future_discount) future states s+
    are drawn from the rollout and scored with the DADS reward
    r(s,z,s+).  Empowerment(cell) = mean over (z, s+) of r.
    """
    base_policy = make_policy(
        (params["normalizer_params"], params["policy_params"]), deterministic=False
    )
    dynamics_params = params["dynamics_params"]
    dyn_input_norm = params["dyn_input_normalizer_params"]
    dyn_delta_norm = params["dyn_delta_normalizer_params"]

    ball_spawn = jnp.asarray(ball_spawn_xy, dtype=jnp.float32)

    def _reset_at(rng, ant_xy):
        start = jnp.concatenate([ant_xy, ball_spawn])
        return env.reset(rng, goal=goal_arr, start=start)

    reset_at = jax.jit(jax.vmap(_reset_at))
    step_fn = jax.jit(jax.vmap(env.step))

    all_skills = jnp.eye(num_skills, dtype=jnp.float32)  # (K, K)
    # log(gamma) for the geometric offset; for future_discount >= 1 the geometric
    # is undefined (infinite mean), so fall back to a uniform offset over [1, L].
    geometric_offsets = future_discount < 1.0
    log_gamma = float(np.log(future_discount)) if geometric_offsets else 0.0

    @jax.jit
    def _cell_chunk_emp(ant_xys, key):
        # ant_xys: (Cc, 2)
        Cc = ant_xys.shape[0]
        key, rkey, okey, pkey = jax.random.split(key, 4)

        # Base states (one physics state per cell), replicated across skills.
        base_state = reset_at(jax.random.split(rkey, Cc), ant_xys)  # batched State (Cc)
        # Tile each cell across the K skills -> batch B = Cc * K.
        base_state = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, num_skills, axis=0), base_state
        )
        skills_b = jnp.tile(all_skills, (Cc, 1))              # (Cc*K, K)
        s0 = base_state.obs[:, :state_dim]                    # (Cc*K, state_dim)
        B = Cc * num_skills

        # Roll out each (cell, skill) for rollout_horizon steps, recording done.
        def f(carry, _):
            state, k = carry
            k, ak = jax.random.split(k)
            aug_obs = jnp.concatenate([state.obs[:, :state_dim], skills_b], axis=-1)
            action, _ = base_policy(aug_obs, ak)
            nstate = step_fn(state, action)
            return (nstate, k), (nstate.obs[:, :state_dim], nstate.done)

        _, (states_traj, done_traj) = jax.lax.scan(
            f, (base_state, pkey), (), length=rollout_horizon
        )  # states_traj (L, B, state_dim), done_traj (L, B)

        # The training-time future distribution only contains within-episode
        # states (it ends at termination). The unwrapped rollout keeps
        # integrating post-termination physics, which is out-of-distribution for
        # the discriminator. states_traj[t] is the state after t+1 steps; the
        # first termination is at index first_done (rollout_horizon-1 if never).
        # Clamp every offset so s+ never lands past the terminal state.
        done_bool = done_traj > 0
        ever_done = jnp.any(done_bool, axis=0)                              # (B,)
        first_done = jnp.where(
            ever_done, jnp.argmax(done_bool, axis=0), rollout_horizon - 1
        )                                                                   # (B,)
        max_offset = (first_done + 1)[None, :]                             # (1, B)

        # Sample M Geometric(1-gamma) offsets in [1, L] per rollout (uniform
        # fallback when future_discount >= 1).
        if geometric_offsets:
            u = jax.random.uniform(okey, (num_future_samples, B), minval=1e-6, maxval=1.0)
            offsets = 1 + jnp.floor(jnp.log(u) / log_gamma).astype(jnp.int32)
        else:
            offsets = jax.random.randint(okey, (num_future_samples, B), 1, rollout_horizon + 1)
        offsets = jnp.clip(offsets, 1, jnp.minimum(rollout_horizon, max_offset))  # (M, B)

        def gather_one(off_b, traj_b):                       # (M,), (L, sd)
            return traj_b[off_b - 1]
        s_plus = jax.vmap(gather_one, in_axes=(1, 1))(offsets, states_traj)  # (B, M, sd)

        # Flatten (B, M) -> compute reward for every (cell, skill, sample).
        s0_bm = jnp.repeat(s0, num_future_samples, axis=0)            # (B*M, sd)
        skills_bm = jnp.repeat(skills_b, num_future_samples, axis=0)  # (B*M, K)
        s_plus_bm = jnp.reshape(s_plus, (B * num_future_samples, state_dim))

        r = dads_losses.dads_reward_from_arrays(
            dads_network, state_dim, num_skills, dynamics_params,
            s0_bm, skills_bm, s_plus_bm, dyn_input_norm, dyn_delta_norm,
            use_xy_prior=use_xy_prior, goal_indices=goal_indices,
            non_goal_indices=non_goal_indices,
        )  # (B*M,)
        r = jnp.mean(jnp.reshape(r, (B, num_future_samples)), axis=-1)  # (B,)
        emp = jnp.mean(jnp.reshape(r, (Cc, num_skills)), axis=-1)       # (Cc,)
        return emp

    emps = []
    C = cell_xy.shape[0]
    for start in range(0, C, chunk_cells):
        end = min(start + chunk_cells, C)
        chunk = jnp.asarray(cell_xy[start:end], dtype=jnp.float32)
        key, ck = jax.random.split(key)
        emps.append(np.asarray(_cell_chunk_emp(chunk, ck)))
        logging.info("  empowerment cells %d/%d", end, C)
    return np.concatenate(emps, axis=0)


def compute_and_log_empowerment_heatmap(
    dads_network,
    make_policy,
    unwrapped_env,
    params,
    state_dim: int,
    num_skills: int,
    future_discount: float,
    goal_indices,
    non_goal_indices,
    use_xy_prior: bool,
    grid_spacing: float,
    rollout_horizon: int,
    num_future_samples: int,
    collect_envs: int,
    collect_steps: int,
    max_cells: int,
    ball_spawn_xy,
    exp_name: str,
    step: int,
    seed: int,
    save_dir: Optional[str] = None,
):
    """Compute and log the DADS empowerment heatmap over the visited x-y region.

    1. Roll out the policy (all skills, auto-reset) to find visited (x,y).
    2. Bin to a ``grid_spacing`` grid; for each occupied cell, place the ant at
       the cell centre (ball fixed at its spawn) and estimate empowerment via
       per-skill rollouts + s+ sampling.
    3. Plot the heatmap (unvisited cells blank) and log to wandb.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    env = unwrapped_env

    # The heatmap places the ant at a cell with the ball fixed at its spawn,
    # which requires a ball env (4D reset start = [ant_xy, ball_xy]). Cleanly
    # skip ball-less envs (e.g. antmaze) instead of failing deep in the rollout.
    if ball_spawn_xy is None and not hasattr(env, "possible_balls"):
        logging.info(
            "Empowerment heatmap skipped: env %s has no ball (no possible_balls).",
            type(env).__name__,
        )
        return

    x_low, x_high = (float(env.x_bounds[0]), float(env.x_bounds[1])) if hasattr(env, "x_bounds") else (-6.0, 26.0)
    y_low, y_high = (float(env.y_bounds[0]), float(env.y_bounds[1])) if hasattr(env, "y_bounds") else (-6.0, 26.0)

    if ball_spawn_xy is None:
        ball_spawn_xy = [float(v) for v in jax.device_get(env.possible_balls[0])]
    ball_spawn_xy = np.asarray(ball_spawn_xy, dtype=np.float32)

    # Goal passed to env.reset (irrelevant to dynamics; just sets the marker).
    if hasattr(env, "possible_goals") and env.possible_goals.shape[0] > 0:
        goal_arr = jnp.asarray(env.possible_goals[0], dtype=jnp.float32)
    else:
        goal_arr = None

    key = jax.random.PRNGKey(seed + 12345)
    key, collect_key, emp_key = jax.random.split(key, 3)

    # ---- 1. Visited (x,y) ----------------------------------------------------
    logging.info("Empowerment heatmap: collecting visited states (%d envs x %d steps)",
                 collect_envs, collect_steps)
    visited_xy = _collect_visited_xy(
        env, make_policy, params, state_dim, num_skills,
        collect_envs, collect_steps, collect_key,
    )

    # ---- 2. Bin to grid ------------------------------------------------------
    nx = max(1, int(round((x_high - x_low) / grid_spacing)))
    ny = max(1, int(round((y_high - y_low) / grid_spacing)))
    ix = np.floor((visited_xy[:, 0] - x_low) / grid_spacing).astype(np.int64)
    iy = np.floor((visited_xy[:, 1] - y_low) / grid_spacing).astype(np.int64)
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[inb], iy[inb]
    occupied = np.unique(np.stack([ix, iy], axis=1), axis=0)  # (C, 2) [ix, iy]
    if occupied.shape[0] == 0:
        logging.warning("Empowerment heatmap: no visited cells found; skipping.")
        return

    if occupied.shape[0] > max_cells:
        sel = np.random.default_rng(seed).choice(occupied.shape[0], size=max_cells, replace=False)
        occupied = occupied[sel]
    logging.info("Empowerment heatmap: %d occupied cells (grid %dx%d)",
                 occupied.shape[0], nx, ny)

    cell_centre_xy = np.stack(
        [x_low + (occupied[:, 0] + 0.5) * grid_spacing,
         y_low + (occupied[:, 1] + 0.5) * grid_spacing],
        axis=1,
    ).astype(np.float32)

    # ---- 3. Empowerment per occupied cell ------------------------------------
    emp_vals = _empowerment_for_cells(
        env, dads_network, make_policy, params, cell_centre_xy, ball_spawn_xy,
        goal_arr, state_dim, num_skills, future_discount, rollout_horizon,
        num_future_samples, use_xy_prior, goal_indices, non_goal_indices,
        chunk_cells=max(1, 768 // max(1, num_skills)), key=emp_key,
    )

    # ---- 4. Build grid + plot ------------------------------------------------
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[occupied[:, 1], occupied[:, 0]] = emp_vals

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid, origin="lower", extent=[x_low, x_high, y_low, y_high],
        aspect="equal", cmap="viridis",
    )
    ax.scatter([ball_spawn_xy[0]], [ball_spawn_xy[1]], c="red", s=60, marker="o",
               edgecolors="white", linewidths=0.8, label="ball spawn")
    ax.set_xlabel("ant x")
    ax.set_ylabel("ant y")
    ax.set_title(
        f"DADS empowerment (visited region) | {exp_name}\n"
        f"step={step} | K={num_skills} skills | gamma_f={future_discount} | "
        f"H={rollout_horizon}, M={num_future_samples}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="empowerment (MI lower bound)")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"empowerment_{exp_name}_{step}.png")
        npy_path = os.path.join(save_dir, f"empowerment_{exp_name}_{step}.npy")
        fig.savefig(png_path, dpi=160)
        np.save(npy_path, grid)
        logging.info("Saved empowerment heatmap: %s", png_path)

    finite = emp_vals[np.isfinite(emp_vals)]
    log_dict = {"empowerment/heatmap": wandb.Image(fig)}
    if finite.size:
        log_dict.update({
            "empowerment/mean": float(np.mean(finite)),
            "empowerment/min": float(np.min(finite)),
            "empowerment/max": float(np.max(finite)),
            "empowerment/num_cells": int(finite.size),
        })
    try:
        wandb.log(log_dict, step=int(step))
    except Exception:
        logging.exception("wandb.log of empowerment heatmap failed")
    plt.close(fig)
