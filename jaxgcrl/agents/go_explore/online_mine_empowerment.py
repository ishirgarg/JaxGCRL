"""Online MINE-based empowerment exploration bonus.

Implements ``Empowerment-driven Exploration using Mutual Information Estimation``
(Madhu Kumar 2018, ``mine.pdf``). Empowerment of a state ``E(s) = max_pi I(S';A|s)``
is approximated via the Mutual Information Neural Estimator (MINE) using the
Donsker-Varadhan lower bound, with a learned forward dynamics model used to
sample the marginal ``p(s'|s)`` (by integrating actions through the dynamics).

The bonus owns two small networks trained jointly on online transitions:

* ``f``  forward dynamics ``f(s, a) -> s'``, trained by MSE.
* ``T``  statistics network ``T(s, a, s') -> R``, trained by maximising the DV
  lower bound ``E_joint[T] - log E_marg[exp(T)]``.

Marginal sampling uses the **agent's main goal-conditioned actor** (passed in
at factory time as ``actor_sample_fn`` and called with current ``actor_params``
threaded through ``train_fn`` / ``bonus_fn``). Two i.i.d. action samples are
drawn from ``pi(.|obs)``: ``a*`` for the action coordinate of the marginal,
``a**`` for the dynamics input so ``s'_marg = f(s, a**)``. Two independent
draws are required so that ``a_marg`` and ``s'_marg`` are independent given
``s`` -- the definition of the MINE marginal ``pi(a|s) p(s'|s)``.

Per-transition bonus is the per-sample DV contribution
``T(s, a, s') - log E_marg[exp(T)]`` (the log-partition is estimated from a
batch of marginal samples and is shared across the batch). Then recentred /
rescaled by ``(raw - bonus_mean) / bonus_scale`` like every other bonus.

Online-only: the bonus is registered in ``_ONLINE_ONLY_BONUS_TYPES`` so its
contribution is masked to zero on offline rows.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import variance_scaling
from flax.struct import dataclass as fdataclass
from jax.scipy.special import logsumexp


# ── networks ────────────────────────────────────────────────────────────────


class _MLPHead(nn.Module):
    hidden_dims: Tuple[int, ...]
    out_dim: int
    layer_norm: bool

    @nn.compact
    def __call__(self, x):
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        for h in self.hidden_dims:
            x = nn.Dense(h, kernel_init=init, bias_init=bias_init)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.swish(x)
        return nn.Dense(self.out_dim, kernel_init=init, bias_init=bias_init)(x)


class _ForwardDynamics(nn.Module):
    state_size: int
    hidden_dims: Tuple[int, ...]
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        return _MLPHead(
            hidden_dims=self.hidden_dims,
            out_dim=self.state_size,
            layer_norm=self.layer_norm,
        )(x)


class _StatisticsT(nn.Module):
    hidden_dims: Tuple[int, ...]
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, actions, next_observations):
        x = jnp.concatenate([observations, actions, next_observations], axis=-1)
        out = _MLPHead(
            hidden_dims=self.hidden_dims,
            out_dim=1,
            layer_norm=self.layer_norm,
        )(x)
        return out.squeeze(-1)


# ── state ─────────────────────────────────────────────────────────────────


@fdataclass
class OnlineMineEmpowermentBonusState:
    """Mutable state for the online MINE empowerment bonus."""

    t_params: Any
    t_opt_state: Any
    dyn_params: Any
    dyn_opt_state: Any
    key: jax.Array
    step: jnp.ndarray  # int32 scalar


# ── factory ───────────────────────────────────────────────────────────────


def _create_online_mine_empowerment_bonus(
    *,
    state_size: int,
    action_size: int,
    actor_sample_fn: Callable[[Any, jnp.ndarray, jax.Array], jnp.ndarray],
    key: jax.Array,
    lr_dyn: float,
    lr_t: float,
    dyn_hidden_dims: Sequence[int],
    t_hidden_dims: Sequence[int],
    layer_norm: bool,
    bonus_mean: float,
    bonus_scale: float,
):
    """Returns ``(bonus_fn, initial_state, init_from_states_fn, train_fn)``.

    ``actor_sample_fn(actor_params, observations, key) -> actions`` is the
    main agent's stochastic policy sampler, baked in at factory time and
    invoked at each train / bonus step on the *full* observation (state +
    goal) carried by ``transitions.observation``. ``actor_params`` itself is
    threaded through at runtime via the extended ``bonus_fn`` / ``train_fn``
    signatures so the marginal sampler always uses the current actor.
    """
    dyn_hidden_dims_tuple = tuple(int(h) for h in dyn_hidden_dims)
    t_hidden_dims_tuple = tuple(int(h) for h in t_hidden_dims)

    dyn_net = _ForwardDynamics(
        state_size=state_size,
        hidden_dims=dyn_hidden_dims_tuple,
        layer_norm=layer_norm,
    )
    t_net = _StatisticsT(
        hidden_dims=t_hidden_dims_tuple, layer_norm=layer_norm,
    )

    init_dyn_key, init_t_key, train_key = jax.random.split(key, 3)
    dummy_obs = jnp.zeros((1, state_size), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, action_size), dtype=jnp.float32)
    dummy_next = jnp.zeros((1, state_size), dtype=jnp.float32)

    dyn_params0 = dyn_net.init(init_dyn_key, dummy_obs, dummy_act)
    t_params0 = t_net.init(init_t_key, dummy_obs, dummy_act, dummy_next)

    dyn_optimizer = optax.adam(learning_rate=lr_dyn)
    t_optimizer = optax.adam(learning_rate=lr_t)

    initial_state = OnlineMineEmpowermentBonusState(
        t_params=t_params0,
        t_opt_state=t_optimizer.init(t_params0),
        dyn_params=dyn_params0,
        dyn_opt_state=dyn_optimizer.init(dyn_params0),
        key=train_key,
        step=jnp.asarray(0, dtype=jnp.int32),
    )

    bonus_mean_f = jnp.asarray(bonus_mean, dtype=jnp.float32)
    bonus_scale_f = jnp.asarray(bonus_scale, dtype=jnp.float32)

    # ── losses ──────────────────────────────────────────────────────────────

    def _dyn_loss(dyn_params, obs_state, actions, next_obs_state):
        pred = dyn_net.apply(dyn_params, obs_state, actions)
        loss = jnp.mean((pred - next_obs_state) ** 2)
        return loss, {
            "dyn_loss": loss,
            "dyn_pred_mean": jnp.mean(pred),
            "dyn_target_mean": jnp.mean(next_obs_state),
        }

    def _t_marg_samples(t_params, dyn_params_frozen, actor_params,
                        obs_full, obs_state, key):
        """Sample (a*, s'_marg) per row and apply T to (s, a*, s'_marg).

        Two independent action samples from pi(.|obs_full): ``a*`` is the
        marginal action, ``a**`` is consumed by forward dynamics so
        ``s'_marg = f(s, a**)``. Two independent draws give the marginal
        joint ``pi(a|s) p(s'|s)``.
        """
        a_star_key, a_dyn_key = jax.random.split(key)
        a_star = actor_sample_fn(actor_params, obs_full, a_star_key)
        a_dyn = actor_sample_fn(actor_params, obs_full, a_dyn_key)
        s_marg = dyn_net.apply(dyn_params_frozen, obs_state, a_dyn)
        return t_net.apply(t_params, obs_state, a_star, s_marg)

    def _mine_loss(t_params, dyn_params_frozen, actor_params,
                   obs_full, obs_state, actions, next_obs_state, key):
        """Negated Donsker-Varadhan lower bound: minimised by optax.adam."""
        t_joint = t_net.apply(t_params, obs_state, actions, next_obs_state)
        t_marg = _t_marg_samples(
            t_params, dyn_params_frozen, actor_params,
            obs_full, obs_state, key,
        )
        n = jnp.asarray(t_marg.shape[0], dtype=t_marg.dtype)
        log_partition = logsumexp(t_marg) - jnp.log(n)
        bound = jnp.mean(t_joint) - log_partition
        loss = -bound
        return loss, {
            "mine_loss": loss,
            "mine_lower_bound": bound,
            "mine_t_joint_mean": jnp.mean(t_joint),
            "mine_t_marg_mean": jnp.mean(t_marg),
            "mine_log_partition": log_partition,
        }

    # ── bonus scoring ──────────────────────────────────────────────────────

    def _mine_score(state: OnlineMineEmpowermentBonusState,
                    actor_params,
                    obs_full: jnp.ndarray,
                    obs_state: jnp.ndarray,
                    actions: jnp.ndarray,
                    next_obs_state: jnp.ndarray,
                    key: jax.Array) -> jnp.ndarray:
        """Per-transition DV contribution. Output shape: (batch,)."""
        t_p = jax.lax.stop_gradient(state.t_params)
        dyn_p = jax.lax.stop_gradient(state.dyn_params)
        actor_p = jax.lax.stop_gradient(actor_params)
        t_joint = t_net.apply(t_p, obs_state, actions, next_obs_state)
        t_marg = _t_marg_samples(
            t_p, dyn_p, actor_p, obs_full, obs_state, key,
        )
        n = jnp.asarray(t_marg.shape[0], dtype=t_marg.dtype)
        log_partition = logsumexp(t_marg) - jnp.log(n)
        return t_joint - log_partition

    # ── train_fn ────────────────────────────────────────────────────────────

    def train_fn(state: OnlineMineEmpowermentBonusState,
                 online_transitions, num_grad_steps: int = 1,
                 *, actor_params: Any):
        """One or more grad steps on (forward dynamics, T).

        Each scan step:
          1. Update forward dynamics by MSE.
          2. Update T by maximising DV bound, with the just-updated dyn
             frozen via stop_gradient inside the loss. Marginal actions are
             drawn from ``pi(.|obs_full; actor_params)``.
        """

        def step(carry, _):
            st: OnlineMineEmpowermentBonusState = carry
            mine_k, k_next = jax.random.split(st.key)

            obs_full = online_transitions.observation
            obs_state = obs_full[..., :state_size]
            if online_transitions.next_observation is not None:
                next_state = online_transitions.next_observation[..., :state_size]
            else:
                next_state = jnp.concatenate(
                    [obs_state[..., 1:, :], obs_state[..., -1:, :]], axis=-2,
                )
            actions = online_transitions.action

            obs_size = obs_full.shape[-1]
            flat_full = obs_full.reshape(-1, obs_size)
            flat_state = obs_state.reshape(-1, state_size)
            flat_next = next_state.reshape(-1, state_size)
            flat_action = actions.reshape(-1, action_size)

            # ── forward dynamics update ──────────────────────────────────
            (_, dyn_metrics), dyn_grads = jax.value_and_grad(
                _dyn_loss, has_aux=True,
            )(st.dyn_params, flat_state, flat_action, flat_next)
            dyn_updates, new_dyn_opt = dyn_optimizer.update(
                dyn_grads, st.dyn_opt_state, st.dyn_params,
            )
            new_dyn_params = optax.apply_updates(st.dyn_params, dyn_updates)

            # ── MINE statistics update (uses freshly updated dyn) ────────
            (_, t_metrics), t_grads = jax.value_and_grad(
                _mine_loss, has_aux=True,
            )(
                st.t_params, new_dyn_params, actor_params,
                flat_full, flat_state, flat_action, flat_next, mine_k,
            )
            t_updates, new_t_opt = t_optimizer.update(
                t_grads, st.t_opt_state, st.t_params,
            )
            new_t_params = optax.apply_updates(st.t_params, t_updates)

            new_st = OnlineMineEmpowermentBonusState(
                t_params=new_t_params, t_opt_state=new_t_opt,
                dyn_params=new_dyn_params, dyn_opt_state=new_dyn_opt,
                key=k_next, step=st.step + 1,
            )
            merged = {}
            for k, v in dyn_metrics.items():
                merged[f"dyn/{k}"] = v
            for k, v in t_metrics.items():
                merged[f"t/{k}"] = v
            return new_st, merged

        new_state, metrics_seq = jax.lax.scan(
            step, state, (), length=num_grad_steps,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics_seq)
        return new_state, metrics

    # ── bonus_fn ───────────────────────────────────────────────────────────

    def bonus_fn(state: OnlineMineEmpowermentBonusState, transitions, bonus_key,
                 *, actor_params: Any):
        """Per-transition DV bonus, shape matching ``transitions.reward``."""
        obs_full = transitions.observation
        obs_state = obs_full[..., :state_size]
        if transitions.next_observation is not None:
            next_state = transitions.next_observation[..., :state_size]
        else:
            next_state = jnp.concatenate(
                [obs_state[..., 1:, :], obs_state[..., -1:, :]], axis=-2,
            )
        actions = transitions.action
        shape = obs_state.shape[:-1]

        obs_size = obs_full.shape[-1]
        flat_full = obs_full.reshape(-1, obs_size)
        flat_state = obs_state.reshape(-1, state_size)
        flat_next = next_state.reshape(-1, state_size)
        flat_action = actions.reshape(-1, action_size)

        raw = _mine_score(
            state, actor_params,
            flat_full, flat_state, flat_action, flat_next, bonus_key,
        ).reshape(shape)
        bonus = (raw - bonus_mean_f) / bonus_scale_f
        metrics = {
            "mine_raw_mean": jnp.mean(raw),
            "mine_raw_min": jnp.min(raw),
            "mine_raw_max": jnp.max(raw),
            "mine_shifted_mean": jnp.mean(bonus),
        }
        return bonus, state, metrics

    # Expose the per-transition scorer for visualisation callers.
    bonus_fn.score_states = _mine_score  # type: ignore[attr-defined]

    return bonus_fn, initial_state, None, train_fn
