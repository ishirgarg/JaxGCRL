"""Online empowerment exploration bonus.

Trains a skill-conditioned policy + Q + (optional) V on samples from the
replay buffer in parallel with the main agent. Per-state empowerment
I(Z; S+ | s) is the bonus signal, normalized by ``(raw - mean) / scale``
before being added to the env reward (matching the offline empowerment bonus
path in :func:`_create_empowerment_bonus`).

Algorithmic source: OGBench ``empowerment_skill.py`` (Myers 2025), simplified
to the JaxGCRL setting:

- s+ for the Q loss is sampled with ``value_p_randomgoal=1.0``: a uniform
  random state from the replay buffer (implemented as a within-batch random
  permutation; each train batch is itself a fresh uniform sample from the
  buffer).
- BC loss is exposed but defaults off (alpha=0).
- Skills are discrete one-hots, sampled uniformly per gradient step.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import variance_scaling
from flax.struct import dataclass as fdataclass
from jax.scipy.special import logsumexp


# ── numerics helpers (port of empowerment_skill.py:34-77) ──────────────────


def _log1mexp(x):
    log_half = -0.6931471805599453
    return jnp.where(
        x < log_half,
        jnp.log1p(-jnp.exp(x)),
        jnp.log(-jnp.expm1(x)),
    )


def _log_diff_exp(log_total, log_part):
    return log_total + _log1mexp(log_part - log_total)


def _clipped_linexp_loss(target, pred, gamma, t=5.0):
    target = jax.lax.stop_gradient(target)
    t_ = jax.lax.stop_gradient(jnp.asarray(t, dtype=pred.dtype))
    p0 = target + t_
    value_p0 = gamma * jnp.exp(t_) - (target + t_)
    slope_p0 = gamma * jnp.exp(t_) - 1.0
    true_loss = gamma * jnp.exp(pred - target) - pred
    linear_loss = value_p0 + slope_p0 * (pred - p0)
    return jnp.where(pred - target < t_, true_loss, linear_loss).mean()


def _logits_from_embeddings(phi_emb, psi_emb, latent_dim):
    return -jnp.sum((phi_emb - psi_emb) ** 2, axis=-1) / latent_dim


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


class _OnlineEmpowermentQ(nn.Module):
    hidden_dims: Tuple[int, ...]
    latent_dim: int
    layer_norm: bool

    def setup(self):
        self.phi_net = _MLPHead(self.hidden_dims, self.latent_dim, self.layer_norm)
        self.psi_net = _MLPHead(self.hidden_dims, self.latent_dim, self.layer_norm)

    def phi(self, observations, actions, skills_onehot):
        return self.phi_net(jnp.concatenate([observations, actions, skills_onehot], axis=-1))

    def psi(self, future_states):
        return self.psi_net(future_states)

    def __call__(self, observations, actions, skills_onehot, future_states):
        return _logits_from_embeddings(
            self.phi(observations, actions, skills_onehot),
            self.psi(future_states),
            self.latent_dim,
        )


class _OnlineEmpowermentV(nn.Module):
    hidden_dims: Tuple[int, ...]
    latent_dim: int
    layer_norm: bool

    def setup(self):
        self.phi_net = _MLPHead(self.hidden_dims, self.latent_dim, self.layer_norm)
        self.psi_net = _MLPHead(self.hidden_dims, self.latent_dim, self.layer_norm)

    def phi(self, observations, skills_onehot):
        return self.phi_net(jnp.concatenate([observations, skills_onehot], axis=-1))

    def psi(self, future_states):
        return self.psi_net(future_states)

    def __call__(self, observations, skills_onehot, future_states):
        return _logits_from_embeddings(
            self.phi(observations, skills_onehot),
            self.psi(future_states),
            self.latent_dim,
        )


class _SkillConditionedActor(nn.Module):
    """Gaussian actor with tanh squash, conditioned on (obs, skill_onehot)."""

    action_size: int
    hidden_dims: Tuple[int, ...]
    layer_norm: bool
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, observations, skills_onehot):
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = jnp.concatenate([observations, skills_onehot], axis=-1)
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(h, kernel_init=init, bias_init=bias_init, name=f"hidden_{i}")(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = nn.swish(x)
        mean = nn.Dense(self.action_size, kernel_init=init, bias_init=bias_init, name="mean")(x)
        log_std = nn.Dense(self.action_size, kernel_init=init, bias_init=bias_init, name="log_std")(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


def _actor_action_mode(actor, params, observations, skills_onehot):
    mean, _ = actor.apply(params, observations, skills_onehot)
    return jnp.tanh(mean)


def _actor_log_prob(actor, params, observations, skills_onehot, actions):
    """log pi(a | s, z) for tanh-squashed Gaussian."""
    mean, log_std = actor.apply(params, observations, skills_onehot)
    std = jnp.exp(log_std)
    raw = jnp.arctanh(jnp.clip(actions, -0.999999, 0.999999))
    base_logp = -0.5 * (((raw - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    base_logp = jnp.sum(base_logp, axis=-1)
    log_det = jnp.sum(jnp.log(1.0 - actions ** 2 + 1e-6), axis=-1)
    return base_logp - log_det


# ── state ─────────────────────────────────────────────────────────────────


@fdataclass
class OnlineEmpowermentBonusState:
    """Mutable state for the online empowerment bonus.

    ``v_params`` / ``v_opt_state`` / ``target_v_params`` are always allocated
    even when ``separate_qv=False`` (they hold a fresh-init V network that is
    never updated). Keeping the pytree shape static simplifies jit-tracing
    across separate_qv values.
    """

    q_params: Any
    q_opt_state: Any
    target_q_params: Any
    v_params: Any
    v_opt_state: Any
    target_v_params: Any
    policy_params: Any
    policy_opt_state: Any
    key: jax.Array
    step: jnp.ndarray  # int32 scalar


# ── factory ───────────────────────────────────────────────────────────────


def _create_online_empowerment_bonus(
    *,
    state_size: int,
    action_size: int,
    key: jax.Array,
    lr: float,
    value_hidden_dims: Sequence[int],
    actor_hidden_dims: Sequence[int],
    value_latent_dim: int,
    num_skills: int,
    num_splus_samples: int,
    discount: float,
    tau: float,
    separate_qv: bool,
    use_self_q_loss: bool,
    layer_norm: bool,
    bc_alpha: float,
    bonus_mean: float,
    bonus_scale: float,
):
    """Returns ``(bonus_fn, initial_state, init_from_states_fn, train_fn)``.

    Matches the contract of :func:`_create_misc_bonus` so the existing
    ExplorationBonuses dispatcher accepts it.

    ``bonus_fn`` carries an extra ``score_states`` Python attribute exposing
    the per-state empowerment estimator for visualization callers.
    """
    value_hidden_dims_tuple = tuple(int(h) for h in value_hidden_dims)
    actor_hidden_dims_tuple = tuple(int(h) for h in actor_hidden_dims)
    K = int(num_skills)
    N = int(num_splus_samples)
    d = int(value_latent_dim)
    log_K = jnp.log(jnp.asarray(K, dtype=jnp.float32))

    q_net = _OnlineEmpowermentQ(
        hidden_dims=value_hidden_dims_tuple, latent_dim=d, layer_norm=layer_norm,
    )
    v_net = _OnlineEmpowermentV(
        hidden_dims=value_hidden_dims_tuple, latent_dim=d, layer_norm=layer_norm,
    )
    actor = _SkillConditionedActor(
        action_size=action_size, hidden_dims=actor_hidden_dims_tuple, layer_norm=layer_norm,
    )

    init_q_key, init_v_key, init_pi_key, train_key = jax.random.split(key, 4)
    dummy_obs = jnp.zeros((1, state_size), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, action_size), dtype=jnp.float32)
    dummy_z = jnp.zeros((1, K), dtype=jnp.float32)
    dummy_fut = jnp.zeros((1, state_size), dtype=jnp.float32)

    q_params0 = q_net.init(init_q_key, dummy_obs, dummy_act, dummy_z, dummy_fut)
    v_params0 = v_net.init(init_v_key, dummy_obs, dummy_z, dummy_fut)
    pi_params0 = actor.init(init_pi_key, dummy_obs, dummy_z)

    q_optimizer = optax.adam(learning_rate=lr)
    v_optimizer = optax.adam(learning_rate=lr)
    pi_optimizer = optax.adam(learning_rate=lr)

    initial_state = OnlineEmpowermentBonusState(
        q_params=q_params0,
        q_opt_state=q_optimizer.init(q_params0),
        target_q_params=q_params0,
        v_params=v_params0,
        v_opt_state=v_optimizer.init(v_params0),
        target_v_params=v_params0,
        policy_params=pi_params0,
        policy_opt_state=pi_optimizer.init(pi_params0),
        key=train_key,
        step=jnp.asarray(0, dtype=jnp.int32),
    )

    bonus_mean_f = jnp.asarray(bonus_mean, dtype=jnp.float32)
    bonus_scale_f = jnp.asarray(bonus_scale, dtype=jnp.float32)
    bc_alpha_f = jnp.asarray(bc_alpha, dtype=jnp.float32)
    discount_f = jnp.asarray(discount, dtype=jnp.float32)
    tau_f = jnp.asarray(tau, dtype=jnp.float32)
    enable_bc = bc_alpha > 0.0

    # ── value-modulation interface ─────────────────────────────────────────

    def _v_phi(v_or_q_params, obs, skills_onehot, pi_params):
        """φ_V(s, z). separate=V phi, shared=Q phi at policy actions."""
        if separate_qv:
            return v_net.apply(v_or_q_params, obs, skills_onehot, method=v_net.phi)
        actions = _actor_action_mode(actor, pi_params, obs, skills_onehot)
        return q_net.apply(
            v_or_q_params, obs, actions, skills_onehot, method=q_net.phi,
        )

    def _v_phi_all(v_or_q_params, obs, pi_params):
        """φ_V(s, z) for every skill — returns ``[K, batch, d]``."""
        eye = jnp.eye(K, dtype=obs.dtype)

        def per_skill(z_onehot):
            z_batch = jnp.broadcast_to(z_onehot, (obs.shape[0], K))
            return _v_phi(v_or_q_params, obs, z_batch, pi_params)

        return jax.vmap(per_skill)(eye)

    def _empowerment_score(state: OnlineEmpowermentBonusState,
                            states: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Per-state empowerment via Monte Carlo. Output shape: (batch,)."""
        v_or_q = jax.lax.stop_gradient(
            state.target_v_params if separate_qv else state.target_q_params
        )
        pi_params = jax.lax.stop_gradient(state.policy_params)

        skill_keys = jax.random.split(key, K)
        phi_all = _v_phi_all(v_or_q, states, pi_params)  # [K, batch, d]

        def per_skill(phi_z, skill_key):
            noise = jax.random.normal(skill_key, (N, *phi_z.shape))
            psi_samples = phi_z[None] + noise * jnp.sqrt(d / 2.0)

            def contribution(psi):
                log_v = _logits_from_embeddings(phi_z, psi, d)         # [batch]
                log_v_all = _logits_from_embeddings(phi_all, psi, d)   # [K, batch]
                log_denom = logsumexp(log_v_all, axis=0) - log_K
                return log_v - log_denom

            return jax.vmap(contribution)(psi_samples).mean(axis=0)    # [batch]

        return jax.vmap(per_skill)(phi_all, skill_keys).mean(axis=0)   # [batch]

    # ── losses ──────────────────────────────────────────────────────────────

    def _q_loss(q_params, target_q_params, target_v_params, policy_params,
                obs, actions, next_obs, future, skills_onehot):
        log_q = q_net.apply(q_params, obs, actions, skills_onehot, future)

        if separate_qv:
            log_v_next_future = v_net.apply(
                target_v_params, next_obs, skills_onehot, future,
            )
            loss_future = _clipped_linexp_loss(
                target=-log_v_next_future, pred=-log_q, gamma=discount_f,
            )
            loss = loss_future
            metrics = {
                "q_loss": loss,
                "q_loss_future": loss_future,
                "q_log_mean": log_q.mean(),
                "v_next_future_log_mean": log_v_next_future.mean(),
            }
            if use_self_q_loss:
                log_q_current = q_net.apply(
                    q_params, obs, actions, skills_onehot, obs,
                )
                log_v_next_current = v_net.apply(
                    target_v_params, next_obs, skills_onehot, obs,
                )
                log_curr_target = jnp.logaddexp(
                    jnp.log(1.0 - discount_f),
                    jnp.log(discount_f) + log_v_next_current,
                )
                loss_current = _clipped_linexp_loss(
                    target=-log_curr_target, pred=-log_q_current, gamma=1.0,
                )
                loss = loss + loss_current
                metrics.update({
                    "q_loss": loss,
                    "q_loss_current": loss_current,
                    "q_log_current_mean": log_q_current.mean(),
                    "v_next_current_log_mean": log_v_next_current.mean(),
                })
            return loss, metrics

        # Shared mode: V ≡ Q(π); recompute target Q via current policy on s'.
        actions_next = _actor_action_mode(actor, policy_params, next_obs, skills_onehot)
        log_q_next_future = q_net.apply(
            target_q_params, next_obs, actions_next, skills_onehot, future,
        )
        loss_future = _clipped_linexp_loss(
            target=-log_q_next_future, pred=-log_q, gamma=discount_f,
        )
        loss = loss_future
        metrics = {
            "q_loss": loss,
            "q_loss_future": loss_future,
            "q_log_mean": log_q.mean(),
            "q_log_next_future_mean": log_q_next_future.mean(),
        }
        if use_self_q_loss:
            log_q_current = q_net.apply(
                q_params, obs, actions, skills_onehot, next_obs,
            )
            log_q_next_current = q_net.apply(
                target_q_params, next_obs, actions_next, skills_onehot, next_obs,
            )
            log_curr_target = jnp.logaddexp(
                jnp.log(1.0 - discount_f),
                jnp.log(discount_f) + log_q_next_current,
            )
            loss_current = _clipped_linexp_loss(
                target=-log_curr_target, pred=-log_q_current, gamma=1.0,
            )
            loss = loss + loss_current
            metrics.update({
                "q_loss": loss,
                "q_loss_current": loss_current,
                "q_log_current_mean": log_q_current.mean(),
                "q_log_next_current_mean": log_q_next_current.mean(),
            })
        return loss, metrics

    def _v_loss(v_params, target_q_params, policy_params,
                obs, future, skills_onehot):
        if not separate_qv:
            zero = jnp.zeros((), dtype=obs.dtype)
            return zero, {"v_loss": zero, "v_loss_future": zero}
        actions_pi = _actor_action_mode(actor, policy_params, obs, skills_onehot)
        log_q_pi = q_net.apply(
            target_q_params, obs, actions_pi, skills_onehot, future,
        )
        log_v = v_net.apply(v_params, obs, skills_onehot, future)
        loss_future = _clipped_linexp_loss(
            target=-log_q_pi, pred=-log_v, gamma=1.0,
        )
        return loss_future, {
            "v_loss": loss_future,
            "v_loss_future": loss_future,
            "v_log_mean": log_v.mean(),
            "q_pi_log_mean": log_q_pi.mean(),
            "v_max": jnp.exp(log_v).max(),
            "v_min": jnp.exp(log_v).min(),
        }

    def _bc_loss(policy_params, obs, actions, skills_onehot):
        log_prob = _actor_log_prob(actor, policy_params, obs, skills_onehot, actions)
        loss = -log_prob.mean()
        return loss, {
            "bc_loss": loss,
            "bc_log_prob_mean": log_prob.mean(),
            "bc_log_prob_max": log_prob.max(),
            "bc_log_prob_min": log_prob.min(),
        }

    def _policy_loss(policy_params, q_params_frozen, v_params_frozen,
                     obs, skills, skills_onehot, key):
        """Empowerment-gradient policy loss (mixture-denominator estimator).

        Uses the *main* Q (and V, in separate-qv mode) networks frozen via
        stop_gradient — NOT the target networks. The caller must pass the
        just-updated main params.
        """
        batch_size = obs.shape[0]
        sample_key, _ = jax.random.split(key)

        # phi_z_v: frozen w.r.t. policy (matches OGBench `policy_params=None`).
        if separate_qv:
            v_or_q_for_v = jax.lax.stop_gradient(v_params_frozen)
            phi_z_v = v_net.apply(
                v_or_q_for_v, obs, skills_onehot, method=v_net.phi,
            )
            v_or_q_for_all = v_or_q_for_v
            pi_for_all = None
        else:
            v_or_q_for_v = jax.lax.stop_gradient(q_params_frozen)
            phi_z_v = _v_phi(
                v_or_q_for_v, obs, skills_onehot,
                jax.lax.stop_gradient(policy_params),
            )
            v_or_q_for_all = v_or_q_for_v
            # Shared mode: gradient flows through policy actions for all K.
            pi_for_all = policy_params

        psi = phi_z_v[None] + jax.random.normal(
            sample_key, (N, *phi_z_v.shape),
        ) * jnp.sqrt(d / 2.0)  # [N, batch, d]

        # phi_z_q with grad through policy actions.
        policy_actions = _actor_action_mode(actor, policy_params, obs, skills_onehot)
        phi_z_q = q_net.apply(
            jax.lax.stop_gradient(q_params_frozen),
            obs, policy_actions, skills_onehot, method=q_net.phi,
        )  # [batch, d]
        log_q = _logits_from_embeddings(phi_z_q[None], psi, d)  # [N, batch]

        phi_all = _v_phi_all(v_or_q_for_all, obs, pi_for_all)  # [K, batch, d]
        log_v_all = _logits_from_embeddings(phi_all[None], psi[:, None], d)
        log_v_z = log_v_all[:, skills, jnp.arange(batch_size)]
        log_v_all_lse = logsumexp(log_v_all, axis=1)

        log_c_sg = jax.lax.stop_gradient(_log_diff_exp(log_v_all_lse, log_v_z))
        log_m_bar = logsumexp(jnp.stack([log_q, log_c_sg], axis=0), axis=0) - log_K

        log_q_over_v_z = log_q - jax.lax.stop_gradient(log_v_z)
        q_term = jnp.exp(log_q_over_v_z) * (log_q - log_m_bar)

        log_v_ratio = log_v_all - jax.lax.stop_gradient(log_v_z[:, None, :])
        v_ratio = jnp.exp(log_v_ratio)
        v_log_v_sum = jax.lax.stop_gradient((v_ratio * log_v_all).sum(axis=1))
        v_all_sum_weighted = jax.lax.stop_gradient(jnp.exp(log_v_all_lse - log_v_z))
        v_log_m_sum = v_all_sum_weighted * log_m_bar
        last_term = jax.lax.stop_gradient(log_v_z) - log_m_bar
        v_others_term = (v_log_v_sum - v_log_m_sum) - last_term

        e_delta = ((q_term + v_others_term) / K).sum(axis=0)  # [batch]
        loss = -e_delta.mean()
        return loss, {
            "policy_loss": loss,
            "e_delta_mean": e_delta.mean(),
            "e_delta_max": e_delta.max(),
            "e_delta_min": e_delta.min(),
        }

    def _policy_loss_with_bc(policy_params, q_params_frozen, v_params_frozen,
                              obs, actions, skills, skills_onehot, key):
        pi_loss, pi_metrics = _policy_loss(
            policy_params, q_params_frozen, v_params_frozen,
            obs, skills, skills_onehot, key,
        )
        bc_loss_val, bc_metrics = _bc_loss(policy_params, obs, actions, skills_onehot)
        total = pi_loss + bc_alpha_f * bc_loss_val
        merged = {**pi_metrics, **bc_metrics, "bc_alpha": bc_alpha_f}
        return total, merged

    # ── train_fn ────────────────────────────────────────────────────────────

    def train_fn(state: OnlineEmpowermentBonusState,
                 online_transitions, num_grad_steps: int = 1):
        """One or more grad steps on Q, V (optional), policy.

        Each scan step samples skills uniformly and a within-batch random
        permutation as s+ (= ``value_p_randomgoal=1.0``). Soft-updates target
        networks at the end of each step.
        """

        def step(carry, _):
            st: OnlineEmpowermentBonusState = carry
            skill_k, perm_k, pi_k, k_next = jax.random.split(st.key, 4)

            states = online_transitions.observation[..., :state_size]
            if online_transitions.next_observation is not None:
                next_states = online_transitions.next_observation[..., :state_size]
            else:
                # Fall back to time-shifted observation; clamp last step.
                next_states = jnp.concatenate(
                    [states[..., 1:, :], states[..., -1:, :]], axis=-2,
                )
            actions = online_transitions.action

            flat_s = states.reshape(-1, state_size)
            flat_s_next = next_states.reshape(-1, state_size)
            flat_a = actions.reshape(-1, action_size)

            B = flat_s.shape[0]
            # value_p_randomgoal=1.0: s+ is a uniformly random state from
            # the batch (the batch is itself a uniform sample from the buffer).
            perm = jax.random.permutation(perm_k, B)
            flat_future = flat_s[perm]

            skills = jax.random.randint(skill_k, (B,), 0, K)
            skills_onehot = jnp.eye(K, dtype=flat_s.dtype)[skills]

            # ── Q update ─────────────────────────────────────────────────
            (_, q_metrics), q_grads = jax.value_and_grad(
                _q_loss, has_aux=True,
            )(
                st.q_params, st.target_q_params, st.target_v_params, st.policy_params,
                flat_s, flat_a, flat_s_next, flat_future, skills_onehot,
            )
            q_updates, new_q_opt = q_optimizer.update(
                q_grads, st.q_opt_state, st.q_params,
            )
            new_q_params = optax.apply_updates(st.q_params, q_updates)

            # ── V update (separate_qv only; otherwise no-op) ──────────────
            (_, v_metrics), v_grads = jax.value_and_grad(
                _v_loss, has_aux=True,
            )(
                st.v_params, st.target_q_params, st.policy_params,
                flat_s, flat_future, skills_onehot,
            )
            if separate_qv:
                v_updates, new_v_opt = v_optimizer.update(
                    v_grads, st.v_opt_state, st.v_params,
                )
                new_v_params = optax.apply_updates(st.v_params, v_updates)
            else:
                new_v_params, new_v_opt = st.v_params, st.v_opt_state

            # ── Policy update (Q and V params already updated this step) ──
            # Use the *main* freshly-updated Q/V networks (frozen via
            # stop_gradient inside the loss) — NOT the slow targets.
            if enable_bc:
                (_, pi_metrics), pi_grads = jax.value_and_grad(
                    _policy_loss_with_bc, has_aux=True,
                )(
                    st.policy_params, new_q_params, new_v_params,
                    flat_s, flat_a, skills, skills_onehot, pi_k,
                )
            else:
                (_, pi_metrics), pi_grads = jax.value_and_grad(
                    _policy_loss, has_aux=True,
                )(
                    st.policy_params, new_q_params, new_v_params,
                    flat_s, skills, skills_onehot, pi_k,
                )
            pi_updates, new_pi_opt = pi_optimizer.update(
                pi_grads, st.policy_opt_state, st.policy_params,
            )
            new_pi_params = optax.apply_updates(st.policy_params, pi_updates)

            # ── target soft-updates ───────────────────────────────────────
            new_target_q = jax.tree_util.tree_map(
                lambda p, tp: tau_f * p + (1.0 - tau_f) * tp,
                new_q_params, st.target_q_params,
            )
            if separate_qv:
                new_target_v = jax.tree_util.tree_map(
                    lambda p, tp: tau_f * p + (1.0 - tau_f) * tp,
                    new_v_params, st.target_v_params,
                )
            else:
                new_target_v = st.target_v_params

            new_st = OnlineEmpowermentBonusState(
                q_params=new_q_params, q_opt_state=new_q_opt,
                target_q_params=new_target_q,
                v_params=new_v_params, v_opt_state=new_v_opt,
                target_v_params=new_target_v,
                policy_params=new_pi_params, policy_opt_state=new_pi_opt,
                key=k_next, step=st.step + 1,
            )
            merged = {}
            for k, v in q_metrics.items():
                merged[f"q/{k}"] = v
            for k, v in v_metrics.items():
                merged[f"v/{k}"] = v
            for k, v in pi_metrics.items():
                merged[f"policy/{k}"] = v
            return new_st, merged

        new_state, metrics_seq = jax.lax.scan(
            step, state, (), length=num_grad_steps,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics_seq)
        return new_state, metrics

    # ── bonus_fn ───────────────────────────────────────────────────────────

    def bonus_fn(state: OnlineEmpowermentBonusState, transitions, bonus_key):
        """Per-state empowerment of next_observation; matches offline path."""
        states = transitions.next_observation[..., :state_size]
        shape = states.shape[:-1]
        flat = states.reshape(-1, state_size)
        raw = _empowerment_score(state, flat, bonus_key).reshape(shape)
        bonus = (raw - bonus_mean_f) / bonus_scale_f
        metrics = {
            "empowerment_raw_mean": jnp.mean(raw),
            "empowerment_shifted_mean": jnp.mean(bonus),
            "empowerment_raw_min": jnp.min(raw),
            "empowerment_raw_max": jnp.max(raw),
        }
        return bonus, state, metrics

    # Expose the per-state scorer for visualization callers (no normalization,
    # accepts any (B, state_size) tensor). Stashed as a Python attribute so
    # callers can fetch it via ``exploration_bonuses.fns[idx].score_states``
    # without changing the bonus dispatch contract.
    bonus_fn.score_states = _empowerment_score  # type: ignore[attr-defined]

    return bonus_fn, initial_state, None, train_fn
