"""Fully online empowerment estimator for the empowerment goal proposers.

The offline empowerment goal proposers (``create_empowerment_goal_proposer`` and
friends in :mod:`goal_proposers`) score replay-buffer candidate states with a
*pretrained* OGBench ``empowerment_skill`` checkpoint. This module provides the
**online** counterpart: instead of loading a frozen checkpoint, we instantiate a
fresh OGBench ``empowerment_skill`` agent and train it in lockstep with the main
Go-Explore agent on batches drawn from the replay buffer. The proposer logic is
otherwise unchanged — the *only* difference is where the empowerment estimate
comes from.

Exact-match guarantee
─────────────────────
To guarantee the online training matches OGBench ``empowerment_skill`` *exactly*,
we do not reimplement the agent — we import and use the real
``agents.empowerment_skill.EmpowermentAgent`` class from the OGBench tree (the
same tree the offline path imports from). The agent is a ``flax.struct.PyTreeNode``
(its ``network`` is a ``flax.struct.PyTreeNode`` ``TrainState``), so it threads
cleanly through ``jax.lax.scan`` as carry, and its jitted ``update`` / ``empowerment``
methods compose with the surrounding jit.

The agent config is built to match the provided ``flags.json`` exactly
(``antsoccer-arena-navigate-v0`` empowerment_skill run): ``num_skills=15``,
``num_splus_samples=1``, ``value_latent_dim=256``, ``separate_qv=True``,
``use_self_q_loss=True``, ``use_self_v_loss=True``, ``no_target_q_for_policy=True``,
``sample_z=True``, ``hidden_dims=(512, 512, 512)``, ``actor_hidden_dims=(512, 512, 512)``,
``lr=3e-4``, ``batch_size=1024``, ``discount=0.99``, ``tau=0.005``,
``layer_norm=True``, ``const_std=True``, ``bc_alpha=0.01``, ``anneal_alpha=False``.

Goal sampling for the value loss mirrors the dataset config
(``value_p_randomgoal=1.0``, ``value_p_curgoal=0.0``, ``value_p_trajgoal=0.0``):
the value goal for each row is a uniformly random state from the sampled batch.
The batch is itself a fresh uniform draw from the replay buffer, so this closely
approximates OGBench's "uniform random state from the dataset" goal — it differs
only in that the support is the current mini-pool rather than the whole buffer
(and the pool is held fixed across inner steps when ``num_grad_steps > 1``).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ── OGBench agent import ────────────────────────────────────────────────────


def _load_empowerment_agent_class(ogbench_root: str):
    """Import OGBench's ``empowerment_skill.EmpowermentAgent`` class.

    ``ogbench_root`` must be the OGBench repo root (the directory containing
    ``impls/``). Mirrors :func:`empowerment._setup_external_imports` so the
    online path uses the exact same agent code as the offline path.
    """
    impls_root = os.path.join(ogbench_root, "impls")
    for p in (impls_root, ogbench_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agents import agents as agent_registry

    return agent_registry["empowerment_skill"]


def build_online_empowerment_config(
    *,
    num_skills: int = 15,
    num_splus_samples: int = 1,
    value_latent_dim: int = 256,
    separate_qv: bool = True,
    use_self_q_loss: bool = True,
    use_self_v_loss: bool = True,
    no_target_q_for_policy: bool = True,
    sample_z: bool = True,
    discount: float = 0.99,
    tau: float = 0.005,
    lr: float = 3e-4,
    batch_size: int = 1024,
    hidden_dims: Sequence[int] = (512, 512, 512),
    actor_hidden_dims: Sequence[int] = (512, 512, 512),
    value_hidden_dims: Optional[Sequence[int]] = None,
    layer_norm: bool = True,
    const_std: bool = True,
    bc_alpha: float = 0.01,
    anneal_alpha: bool = False,
    log_interval: int = 5000,
) -> dict:
    """Build the ``empowerment_skill`` agent config dict.

    Every field is transcribed from the provided ``flags.json`` ``agent`` block
    so the online agent is configured identically to the OGBench reference run.
    The keyword defaults are exactly those config values; callers override only
    the knobs they expose.
    """
    return dict(
        agent_name="empowerment_skill",
        lr=float(lr),
        batch_size=int(batch_size),
        hidden_dims=tuple(int(h) for h in hidden_dims),
        # None -> EmpowermentAgent falls back to hidden_dims for the value nets,
        # matching value_hidden_dims=null in the reference config.
        value_hidden_dims=(None if value_hidden_dims is None
                           else tuple(int(h) for h in value_hidden_dims)),
        value_latent_dim=int(value_latent_dim),
        actor_hidden_dims=tuple(int(h) for h in actor_hidden_dims),
        layer_norm=bool(layer_norm),
        discount=float(discount),
        tau=float(tau),
        num_skills=int(num_skills),
        num_splus_samples=int(num_splus_samples),
        obs_indices=None,
        bc_alpha=float(bc_alpha),
        anneal_alpha=bool(anneal_alpha),
        separate_qv=bool(separate_qv),
        use_self_v_loss=bool(use_self_v_loss),
        use_self_q_loss=bool(use_self_q_loss),
        no_target_q_for_policy=bool(no_target_q_for_policy),
        sample_z=bool(sample_z),
        log_interval=int(log_interval),
        discrete=False,
        const_std=bool(const_std),
        encoder=None,
        dataset_class="GCDataset",
        value_p_curgoal=0.0,
        value_p_trajgoal=0.0,
        value_p_randomgoal=1.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=None,
    )


def create_online_empowerment_agent(
    *,
    ogbench_root: str,
    state_size: int,
    action_size: int,
    seed: int = 0,
    **config_overrides,
):
    """Instantiate a fresh (untrained) OGBench ``empowerment_skill`` agent.

    The agent observes raw states of dimension ``state_size`` (the replay-buffer
    observation with the appended goal sliced off) and continuous actions of
    dimension ``action_size``. Returns the ``EmpowermentAgent`` (a pytree) ready
    to be threaded through the training loop.
    """
    agent_class = _load_empowerment_agent_class(ogbench_root)
    config = build_online_empowerment_config(**config_overrides)
    ex_observations = np.zeros((1, int(state_size)), dtype=np.float32)
    ex_actions = np.zeros((1, int(action_size)), dtype=np.float32)
    agent = agent_class.create(
        seed=int(seed),
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )
    return agent


# ── batch construction + training ──────────────────────────────────────────


def _flatten_transitions(transitions, state_size: int, action_size: int):
    """Flatten replay-buffer transitions to ``(B, ·)`` state/action/next arrays.

    ``observation`` / ``next_observation`` are sliced to ``state_size`` (drop the
    appended goal conditioning), matching the offline empowerment scorer.
    """
    obs = transitions.observation
    obs_flat = jnp.reshape(obs, (-1, obs.shape[-1]))
    states = obs_flat[:, :state_size]

    if transitions.next_observation is None:
        # Go-Explore rollouts always store next_observation; if a caller passes
        # transitions without it, the Q future-loss would silently regress on a
        # self-transition (s'==s) and corrupt training. Fail loudly instead.
        raise ValueError(
            "online empowerment requires transitions.next_observation to be populated "
            "(the Q future loss regresses on s'); got None."
        )
    nxt = transitions.next_observation
    nxt_flat = jnp.reshape(nxt, (-1, nxt.shape[-1]))
    next_states = nxt_flat[:, :state_size]

    act = transitions.action
    actions = jnp.reshape(act, (-1, act.shape[-1]))[:, :action_size]
    return states, actions, next_states


def make_online_empowerment_train_fn(
    *, state_size: int, action_size: int, batch_size: int,
) -> Callable:
    """Return ``train_fn(agent, transitions, rng, num_grad_steps) -> (agent, metrics)``.

    Each gradient step draws ``batch_size`` rows (with replacement) from the
    flattened replay-buffer sample as the observation/action/next batch, and an
    *independent* set of ``batch_size`` rows as the value goals
    (``value_p_randomgoal=1.0``). It then calls the OGBench agent's own
    ``update`` — so the per-step computation is identical to OGBench
    ``empowerment_skill`` training.
    """
    B = int(batch_size)

    def train_fn(agent, transitions, rng, num_grad_steps: int = 1):
        # num_grad_steps is a static Python int; a 0-length scan would yield
        # NaN-averaged metrics. Disable the feature via online_empowerment=False.
        if int(num_grad_steps) < 1:
            raise ValueError(
                f"num_grad_steps must be >= 1, got {num_grad_steps}; "
                "disable online empowerment via online_empowerment=False."
            )
        states, actions, next_states = _flatten_transitions(
            transitions, state_size, action_size
        )
        pool = states.shape[0]

        def step(carry, _):
            ag, key = carry
            idx_key, goal_key, next_key = jax.random.split(key, 3)
            idx = jax.random.randint(idx_key, (B,), 0, pool)
            goal_idx = jax.random.randint(goal_key, (B,), 0, pool)
            batch = {
                "observations": states[idx],
                "actions": actions[idx],
                "next_observations": next_states[idx],
                # value_p_randomgoal=1.0: uniformly random state from the buffer.
                "value_goals": states[goal_idx],
            }
            ag, info = ag.update(batch)
            return (ag, next_key), info

        (agent, _), info_seq = jax.lax.scan(
            step, (agent, rng), (), length=num_grad_steps
        )
        metrics = jax.tree_util.tree_map(jnp.mean, info_seq)
        return agent, metrics

    return train_fn


# ── scoring (for the goal proposer + heatmap) ───────────────────────────────


def make_online_empowerment_scorer(
    *, mean: float = 0.0, scale: float = 1.0, chunk_size: int = 32,
) -> Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return ``score(agent, states, rng) -> normalized empowerment per state``.

    Mirrors the offline scorer (:func:`empowerment.make_offline_empowerment_scorer`):
    emits ``(raw - mean) / scale`` and evaluates ``agent.empowerment`` on
    candidate rows in batches of at most ``chunk_size`` via ``lax.fori_loop``, so
    peak activation memory scales with ``chunk_size`` instead of the full
    candidate count. This matters because the proposer runs the scorer under a
    ``vmap`` over environments — without chunking, ``num_candidates`` × ``num_envs``
    × K skills × N MC samples can OOM.

    Variance of the per-state estimate is governed by the agent's own
    ``num_splus_samples`` (the MC draws inside ``empowerment``); no extra
    call-averaging is done here.
    """
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    mean_f = jnp.asarray(mean, dtype=jnp.float32)
    scale_f = jnp.asarray(scale, dtype=jnp.float32)
    cs = int(chunk_size)

    def score(agent, states: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        n = states.shape[0]
        pad = (cs - (n % cs)) % cs
        states_pad = jnp.pad(states, ((0, pad), (0, 0)))
        total = states_pad.shape[0]
        n_chunks = total // cs
        acc0 = jnp.zeros((total,), dtype=jnp.float32)

        def body(i, acc):
            chunk = jax.lax.dynamic_slice_in_dim(states_pad, i * cs, cs, axis=0)
            ki = jax.random.fold_in(rng, i)
            s = agent.empowerment(chunk, rng=ki).astype(jnp.float32)
            s = jnp.reshape(s, (cs,))
            return jax.lax.dynamic_update_slice(acc, s, (i * cs,))

        acc = jax.lax.fori_loop(0, n_chunks, body, acc0)
        return (acc[:n] - mean_f) / scale_f

    return score
