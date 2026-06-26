"""DADS losses and reward computation (future-state variant).

Dynamics-Aware Discovery of Skills (DADS): https://arxiv.org/abs/1907.01657

This repo models a *gamma-discounted future state* ``s+`` instead of the one-step
next state ``s'``.  The skill dynamics model is

    q_phi(s+ | s, z) = N( mean_phi(s, z), I )   over   delta = s+ - s

trained by maximum likelihood, and the DADS intrinsic reward is

    r(s, z, s+) = log q(s+|s,z) - log[ (1/K) * Sum_{z'} q(s+|s,z') ]

which lower-bounds the mutual information I(s+; z | s) (the "empowerment" of s).

Normalization
-------------
Both the network input ([s|z]) and the delta target (s+ - s) are normalized with
*persisted running statistics* (``running_statistics``), NOT per-batch statistics.
This is important for two reasons:
  * The same normalization is applied to the numerator q(s+|s,z) and to every
    term q(s+|s,z') in the marginal, so their ratio is a valid MI estimate
    (per-batch input statistics differ between the B-sample numerator and the
    B*K-sample denominator, silently corrupting the estimate).
  * Running statistics are stationary, so the post-training empowerment map is
    well-defined and comparable across grid cells.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.types import Params

from .networks import DADSNetworks

_PMAP_AXIS_NAME = "i"

# Clip normalized inputs/targets to +/- this many std. Without it, a delta
# dimension with tiny-but-nonzero variance (brax floors std at 1e-6) produces
# astronomically large normalized targets, an unbounded-below intrinsic reward,
# and destabilizing critic gradients. Clipping is a fixed deterministic map
# applied identically to the numerator and every marginal term, so the reward
# stays a valid (slightly looser) lower bound on I(s+; z | s).
_NORM_CLIP = 10.0

# Type alias for transitions (avoids a circular import with dads.py).
Transition = Any


def _normalize(x, normalizer_params):
    return running_statistics.normalize(x, normalizer_params, max_abs_value=_NORM_CLIP)


def _gaussian_log_prob_identity_cov(x: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """Log prob of ``x`` under N(mean, I), summed over the last dim.

    Args:
        x:    Target values, shape (..., D).
        mean: Predicted mean,  shape (..., D).

    Returns:
        Per-sample log probability, shape (...,).
    """
    return -0.5 * jnp.sum((x - mean) ** 2, axis=-1) - 0.5 * x.shape[-1] * jnp.log(2.0 * jnp.pi)


def _prepare_delta_and_input(
    transitions: Transition,
    state_dim: int,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the dynamics target ``delta = s+ - s`` and network input ``[s|z]``.

    ``s+`` is the (geometric-future) state stored in
    ``transitions.extras["future_state"]``; ``s`` is the raw state portion of
    ``transitions.observation``.

    Returns:
        delta:         (B, output_dim) displacement to the future state.
        dynamics_input:(B, input_dim)  network input [s (minus goal idx) | z].
        states:        (B, state_dim)  raw states.
        skills:        (B, num_skills) one-hot skills.
    """
    states = transitions.observation[:, :state_dim]          # (B, state_dim)
    future_states = transitions.extras["future_state"]       # (B, state_dim)
    skills = transitions.extras["state_extras"]["skill"]     # (B, num_skills)

    if use_xy_prior:
        state_without_xy = jnp.take(states, non_goal_indices, axis=1)
        delta = future_states[:, goal_indices] - states[:, goal_indices]
        dynamics_input = jnp.concatenate([state_without_xy, skills], axis=-1)
    else:
        delta = future_states - states
        dynamics_input = jnp.concatenate([states, skills], axis=-1)

    return delta, dynamics_input, states, skills


def compute_per_sample_dynamics_nll(
    dads_network: DADSNetworks,
    state_dim: int,
    dyn_params: Params,
    transitions: Transition,
    dyn_input_normalizer_params: running_statistics.RunningStatisticsState,
    dyn_delta_normalizer_params: running_statistics.RunningStatisticsState,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-sample NLL of q_phi(s+|s,z); shape (B,)."""
    delta, dynamics_input, _, _ = _prepare_delta_and_input(
        transitions, state_dim, use_xy_prior, goal_indices, non_goal_indices
    )
    input_norm = _normalize(dynamics_input, dyn_input_normalizer_params)
    target = _normalize(delta, dyn_delta_normalizer_params)
    pred = dads_network.skill_dynamics_network.apply(None, dyn_params, input_norm)
    log_prob = _gaussian_log_prob_identity_cov(target, pred)
    return -log_prob


def skill_dynamics_loss_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dyn_params: Params,
    transitions: Transition,
    dyn_input_normalizer_params: running_statistics.RunningStatisticsState,
    dyn_delta_normalizer_params: running_statistics.RunningStatisticsState,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean NLL loss for the skill dynamics model q_phi(s+|s,z).

    Rows whose timestep had no valid in-episode future (terminal steps, where
    s+ degenerates to s and delta=0) are masked out via ``future_valid``.
    """
    per_sample_nll = compute_per_sample_dynamics_nll(
        dads_network, state_dim, dyn_params, transitions,
        dyn_input_normalizer_params, dyn_delta_normalizer_params,
        use_xy_prior=use_xy_prior, goal_indices=goal_indices,
        non_goal_indices=non_goal_indices,
    )
    valid = transitions.extras.get("future_valid", None)
    if valid is None:
        return jnp.mean(per_sample_nll)
    return jnp.sum(per_sample_nll * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def dads_reward_from_arrays(
    dads_network: DADSNetworks,
    state_dim: int,
    num_skills: int,
    dyn_params: Params,
    states: jnp.ndarray,
    skills: jnp.ndarray,
    future_states: jnp.ndarray,
    dyn_input_normalizer_params: running_statistics.RunningStatisticsState,
    dyn_delta_normalizer_params: running_statistics.RunningStatisticsState,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """DADS intrinsic reward r(s,z,s+) for explicit arrays; shape (B,).

    r = log q(s+|s,z) - log[ (1/K) Sum_{z'} q(s+|s,z') ]

    Args:
        states:        (B, state_dim) base states s.
        skills:        (B, num_skills) one-hot skills z (the realized skill).
        future_states: (B, state_dim) future states s+.
    """
    B = states.shape[0]
    output_size = len(goal_indices) if use_xy_prior else state_dim

    if use_xy_prior:
        state_part = jnp.take(states, non_goal_indices, axis=1)
        delta = future_states[:, goal_indices] - states[:, goal_indices]
    else:
        state_part = states
        delta = future_states - states
    P = state_part.shape[-1]

    # Target is shared across all skills (the realized future displacement).
    target = _normalize(delta, dyn_delta_normalizer_params)  # (B, output)

    # ---- numerator: log q(s+|s,z) for the actual skill ----------------------
    dynamics_input = jnp.concatenate([state_part, skills], axis=-1)
    input_norm = _normalize(dynamics_input, dyn_input_normalizer_params)
    pred_z = dads_network.skill_dynamics_network.apply(None, dyn_params, input_norm)
    log_q_z = _gaussian_log_prob_identity_cov(target, pred_z)  # (B,)

    # ---- denominator: log q(s+|s,z') for every skill z' ---------------------
    all_skills = jnp.eye(num_skills, dtype=jnp.float32)  # (K, K)
    state_bk = jnp.broadcast_to(state_part[:, None, :], (B, num_skills, P))
    skills_bk = jnp.broadcast_to(all_skills[None, :, :], (B, num_skills, num_skills))
    flat_input = jnp.reshape(
        jnp.concatenate([state_bk, skills_bk], axis=-1), (B * num_skills, P + num_skills)
    )
    # Same running normalizer as the numerator -> consistent q across skills.
    flat_input_norm = _normalize(flat_input, dyn_input_normalizer_params)
    flat_pred = dads_network.skill_dynamics_network.apply(None, dyn_params, flat_input_norm)

    target_bk = jnp.reshape(
        jnp.broadcast_to(target[:, None, :], (B, num_skills, output_size)),
        (B * num_skills, output_size),
    )
    flat_log_probs = _gaussian_log_prob_identity_cov(target_bk, flat_pred)  # (B*K,)
    all_log_probs = jnp.reshape(flat_log_probs, (B, num_skills))            # (B, K)

    log_marginal = jax.scipy.special.logsumexp(all_log_probs, axis=-1) - jnp.log(num_skills)
    return log_q_z - log_marginal  # (B,)


def compute_dads_reward(
    dads_network: DADSNetworks,
    state_dim: int,
    num_skills: int,
    dyn_params: Params,
    transitions: Transition,
    dyn_input_normalizer_params: running_statistics.RunningStatisticsState,
    dyn_delta_normalizer_params: running_statistics.RunningStatisticsState,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """DADS intrinsic reward r(s,z,s+) from a transition batch; shape (B,).

    Terminal-step rows (no valid in-episode future; s+ == s) are zeroed via
    ``future_valid`` so the policy gets no spurious signal from delta=0 samples.
    """
    states = transitions.observation[:, :state_dim]
    future_states = transitions.extras["future_state"]
    skills = transitions.extras["state_extras"]["skill"]
    reward = dads_reward_from_arrays(
        dads_network, state_dim, num_skills, dyn_params,
        states, skills, future_states,
        dyn_input_normalizer_params, dyn_delta_normalizer_params,
        use_xy_prior=use_xy_prior, goal_indices=goal_indices,
        non_goal_indices=non_goal_indices,
    )
    valid = transitions.extras.get("future_valid", None)
    if valid is not None:
        reward = reward * valid
    return reward


def make_skill_dynamics_update_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dynamics_optimizer: optax.GradientTransformation,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
):
    """Create the skill dynamics update function.

    Returns an update function
    ``(dyn_params, transitions, dyn_input_norm, dyn_delta_norm, optimizer_state)``
    -> ``(loss, new_params, new_optimizer_state)``.
    """
    def loss_fn(dyn_params, transitions, dyn_input_norm, dyn_delta_norm):
        return skill_dynamics_loss_fn(
            dads_network, state_dim, dyn_params, transitions,
            dyn_input_norm, dyn_delta_norm,
            use_xy_prior=use_xy_prior, goal_indices=goal_indices,
            non_goal_indices=non_goal_indices,
        )

    return gradients.gradient_update_fn(
        loss_fn,
        dynamics_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
