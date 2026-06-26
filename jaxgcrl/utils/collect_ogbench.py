"""Collect OGBench-format offline datasets from a trained goal-conditioned policy.

This module rolls out a frozen goal-conditioned actor (e.g. one trained with the
``go_explore_simple`` agent) for many episodes, each conditioned on a *proposed*
goal, and writes the resulting state/action trajectories to disk in the **exact**
OGBench dataset format so they can be fed straight into OGBench tooling.

OGBench dataset format (see ``ogbench/data_gen_scripts/generate_antsoccer.py``):
a single ``np.savez_compressed`` with keys

  observations  (N, state_dim)   float32   state only -- NO goal (HER-relabeled later)
  actions       (N, action_dim)  float32
  terminals     (N,)             bool      True only on each episode's last step
  qpos          (N, nq)          float32   MuJoCo qpos before the step
  qvel          (N, nv)          float32   MuJoCo qvel before the step

``observations[t]`` / ``qpos[t]`` / ``qvel[t]`` are recorded *before* ``action[t]``.
Train / val are split by step count into ``<path>.npz`` and ``<path>-val.npz``.

Design notes
------------
* The proposed goal is injected **purely into the policy's observation**, never
  into the environment.  Each episode resets the env to its natural scene (ant at
  R, ball at B, real goal G), a goal proposer reads that scene and returns a
  conditioning goal ``p``, and every step feeds the actor
  ``[state[:state_size], p]`` while the real env is stepped normally.  The env's
  own reward / target is irrelevant because OGBench data stores no goal.
* The collection driver is fully generic: any Brax-style env exposing
  ``state_dim`` / ``goal_indices`` and a deterministic ``step(state, action)``
  works, and goal proposers are pluggable via :func:`make_goal_proposer`.
"""

import math
import os
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------- #
# OGBench .npz writer
# --------------------------------------------------------------------------- #
def save_ogbench_dataset(
    save_path: str,
    data: Dict[str, np.ndarray],
    train_episodes: int,
    val_episodes: int,
    episode_length: int,
):
    """Split a flat collected dataset into train/val and save in OGBench format.

    Args:
        save_path: Output path for the train split (``.npz``).  The val split is
            written to the same path with ``.npz`` -> ``-val.npz``.
        data: Dict with keys ``observations, actions, terminals, qpos, qvel``,
            each a flat array whose first axis is total steps (episodes stored
            end-to-end, every episode exactly ``episode_length`` long).
        train_episodes: Number of leading episodes that form the train split.
        val_episodes: Number of trailing episodes that form the val split.
        episode_length: Steps per episode.

    Returns:
        Tuple of the paths actually written (train, and val if ``val_episodes > 0``).
    """
    if train_episodes < 1:
        raise ValueError(f"train_episodes must be >= 1, got {train_episodes}")
    if val_episodes < 0:
        raise ValueError(f"val_episodes must be >= 0, got {val_episodes}")

    n_rows = int(data["observations"].shape[0])
    for k, v in data.items():
        if int(np.asarray(v).shape[0]) != n_rows:
            raise ValueError(
                f"data['{k}'] has {np.asarray(v).shape[0]} rows but "
                f"data['observations'] has {n_rows}; all keys must share the row count."
            )
    expected = (train_episodes + val_episodes) * episode_length
    if n_rows != expected:
        raise ValueError(
            f"data has {n_rows} rows but expected "
            f"({train_episodes} + {val_episodes}) * {episode_length} = {expected}; "
            f"the train/val boundary would not land on an episode end."
        )
    train_steps = train_episodes * episode_length

    def split(sl: slice) -> Dict[str, np.ndarray]:
        return {
            k: (np.asarray(v[sl], dtype=bool) if k == "terminals"
                else np.asarray(v[sl], dtype=np.float32))
            for k, v in data.items()
        }

    # Derive the val path from the basename only.  ``str.replace`` would corrupt a
    # path whose *directory* contains ".npz", and would be a no-op (colliding
    # train and val onto one file) when there is no extension at all.
    base, ext = os.path.splitext(save_path)
    if ext != ".npz":
        base, ext = save_path, ".npz"
    train_path, val_path = base + ext, base + "-val" + ext
    os.makedirs(os.path.dirname(os.path.abspath(train_path)) or ".", exist_ok=True)

    train = split(slice(0, train_steps))
    np.savez_compressed(train_path, **train)
    print(f"[collect] wrote {train['observations'].shape[0]} train steps -> {train_path}")

    written = [train_path]
    if val_episodes > 0:
        val = split(slice(train_steps, train_steps + val_episodes * episode_length))
        np.savez_compressed(val_path, **val)
        print(f"[collect] wrote {val['observations'].shape[0]} val steps   -> {val_path}")
        written.append(val_path)
    else:
        print("[collect] val_episodes == 0; no val file written.")
    return tuple(written)


# --------------------------------------------------------------------------- #
# Per-env OGBench state extractor
# --------------------------------------------------------------------------- #
def extract_ogbench_state(env, pipeline_state, obs, state_size):
    """Return ``(observation, qpos, qvel)`` for a (possibly batched) timestep.

    The OGBench observation is the env's *state* portion only (``obs[:state_size]``,
    goal sliced off).  ``qpos`` / ``qvel`` are the underlying MuJoCo arrays with
    any trailing non-physical target-marker joints removed.

    Works batched: pass ``state.obs`` / ``state.pipeline_state`` of a vmapped env
    and the leading axis is preserved.
    """
    ob = obs[..., :state_size]
    target_q = int(getattr(env, "_TARGET_Q", 0))
    target_qd = int(getattr(env, "_TARGET_QD", 0))
    q = pipeline_state.q
    qd = pipeline_state.qd
    if target_q > 0:
        q = q[..., :-target_q]
    if target_qd > 0:
        qd = qd[..., :-target_qd]
    return ob, q, qd


# --------------------------------------------------------------------------- #
# Goal-proposer registry
# --------------------------------------------------------------------------- #
GOAL_PROPOSERS = ("env_goal", "line_to_goal")


def make_goal_proposer(name: str, env, **params) -> Callable:
    """Build a goal proposer ``fn(rng, init_state) -> goal (goal_dim,)``.

    The proposer operates on a *single* env's reset ``State`` (it is vmapped over
    the batch by the collection driver).  It reads whatever it needs from
    ``init_state.obs``; for the ant-soccer envs the obs layout is
    ``[..., ant_xy@0:2, ..., ball_target@-2:]`` and the goal slot the policy
    consumes is the trailing ``goal_dim`` entries.

    Proposers
    ---------
    ``env_goal``     : identity -- return the env's own sampled goal (baseline).
    ``line_to_goal`` : sample ``p = ant_xy + t*(G - ant_xy) + noise`` with
                       ``t ~ U(t_min, t_max)`` and tile the 2D waypoint to fill
                       ``goal_dim`` (so a 4D ``[ant_xy, ball_xy]`` goal becomes
                       ``[p, p]``).  Params: ``t_min, t_max, noise_scale,
                       clip_to_bounds``.
    """
    goal_dim = int(len(env.goal_indices))

    if name == "env_goal":
        def proposer(rng, init_state):
            return init_state.obs[-goal_dim:]
        return proposer

    if name == "line_to_goal":
        t_min = float(params.get("t_min", 0.0))
        t_max = float(params.get("t_max", 1.0))
        noise_scale = float(params.get("noise_scale", 1.0))
        clip_to_bounds = bool(params.get("clip_to_bounds", True))
        if goal_dim % 2 != 0:
            raise ValueError(
                f"line_to_goal expects an even goal_dim (xy pairs); got {goal_dim}."
            )
        n_copies = goal_dim // 2
        x_lo, x_hi = (float(env.x_bounds[0]), float(env.x_bounds[1]))
        y_lo, y_hi = (float(env.y_bounds[0]), float(env.y_bounds[1]))

        def proposer(rng, init_state):
            obs = init_state.obs
            a_xy = obs[0:2]      # ant root xy (state[0:2])
            g_xy = obs[-2:]      # ball-target xy == goal cell G
            t_key, n_key = jax.random.split(rng)
            t = jax.random.uniform(t_key, (), minval=t_min, maxval=t_max)
            p = a_xy + t * (g_xy - a_xy) + noise_scale * jax.random.normal(n_key, (2,))
            if clip_to_bounds:
                p = jnp.stack([jnp.clip(p[0], x_lo, x_hi), jnp.clip(p[1], y_lo, y_hi)])
            return jnp.tile(p, n_copies)
        return proposer

    raise ValueError(f"Unknown goal proposer '{name}'. Known: {GOAL_PROPOSERS}")


# --------------------------------------------------------------------------- #
# Collection driver
# --------------------------------------------------------------------------- #
def collect_dataset(
    env,
    actor,
    actor_params,
    goal_proposer: Callable,
    *,
    n_episodes: int,
    episode_length: int,
    num_envs: int,
    action_noise: float = 0.0,
    eps_random: float = 0.0,
    deterministic: bool = True,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Roll out ``n_episodes`` goal-conditioned episodes and return OGBench arrays.

    Episodes are collected in rounds of ``num_envs`` parallel envs.  Each episode
    is exactly ``episode_length`` steps; only its final step is marked terminal.
    Episodes are laid out end-to-end so the returned arrays are directly
    savable as an OGBench dataset.

    Args:
        env: A Brax-style env with ``reset(rng)`` and ``step(state, action)``
            (deterministic, no rng) and attributes ``state_dim`` / ``goal_indices``.
        actor: Object with ``sample_actions(params, obs, key, is_deterministic)``.
        actor_params: The actor parameters (the goal-conditioned policy).
        goal_proposer: ``fn(rng, init_state) -> goal`` (single-env; vmapped here).
        n_episodes: Total episodes to collect (train + val).
        episode_length: Steps per episode.
        num_envs: Parallel envs per round.
        action_noise: Stddev of Gaussian noise added to actions (then clipped).
        eps_random: Probability of replacing an action with uniform[-1, 1].
        deterministic: Sample the policy mode (True) vs. stochastically (False).
        seed: PRNG seed.

    Returns:
        Dict with ``observations, actions, terminals, qpos, qvel`` (flat, host numpy).
    """
    state_size = int(env.state_dim)

    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step)
    proposer_fn = jax.vmap(goal_proposer)

    last_t = episode_length - 1

    def run_round(round_key):
        reset_key, prop_key, scan_key = jax.random.split(round_key, 3)
        scene = reset_fn(jax.random.split(reset_key, num_envs))
        goals = proposer_fn(jax.random.split(prop_key, num_envs), scene)  # (num_envs, goal_dim)

        def scan_step(carry, t):
            state, key = carry
            key, akey, nkey, ekey, ukey = jax.random.split(key, 5)

            raw_state = state.obs[:, :state_size]
            policy_obs = jnp.concatenate([raw_state, goals], axis=-1)
            action = actor.sample_actions(
                actor_params, policy_obs, akey, is_deterministic=deterministic
            )
            action = action + action_noise * jax.random.normal(nkey, action.shape)
            rand_action = jax.random.uniform(ukey, action.shape, minval=-1.0, maxval=1.0)
            use_rand = jax.random.uniform(ekey, (num_envs, 1)) < eps_random
            action = jnp.where(use_rand, rand_action, action)
            action = jnp.clip(action, -1.0, 1.0)

            ob, qpos, qvel = extract_ogbench_state(
                env, state.pipeline_state, state.obs, state_size
            )
            terminal = jnp.full((num_envs,), t == last_t)

            nstate = step_fn(state, action)
            return (nstate, key), (ob, action, terminal, qpos, qvel)

        (_, _), recs = jax.lax.scan(
            scan_step, (scene, scan_key), jnp.arange(episode_length)
        )
        # recs leaves: (episode_length, num_envs, ...)
        return recs

    run_round_jit = jax.jit(run_round)

    num_rounds = math.ceil(n_episodes / num_envs)
    round_keys = jax.random.split(jax.random.PRNGKey(seed), num_rounds)

    obs_chunks, act_chunks, term_chunks, qpos_chunks, qvel_chunks = [], [], [], [], []
    for r in range(num_rounds):
        ob, action, terminal, qpos, qvel = jax.device_get(run_round_jit(round_keys[r]))
        # (L, num_envs, dim) -> (num_envs, L, dim) so each env's episode is contiguous.
        obs_chunks.append(np.asarray(ob).swapaxes(0, 1))
        act_chunks.append(np.asarray(action).swapaxes(0, 1))
        term_chunks.append(np.asarray(terminal).swapaxes(0, 1))
        qpos_chunks.append(np.asarray(qpos).swapaxes(0, 1))
        qvel_chunks.append(np.asarray(qvel).swapaxes(0, 1))
        print(f"[collect] round {r + 1}/{num_rounds} done "
              f"({(r + 1) * num_envs} episodes collected)")

    def stack_trim(chunks):
        arr = np.concatenate(chunks, axis=0)[:n_episodes]   # (n_episodes, L, ...)
        return arr.reshape(n_episodes * episode_length, *arr.shape[2:])

    return {
        "observations": stack_trim(obs_chunks),
        "actions": stack_trim(act_chunks),
        "terminals": stack_trim(term_chunks).astype(bool),
        "qpos": stack_trim(qpos_chunks),
        "qvel": stack_trim(qvel_chunks),
    }
