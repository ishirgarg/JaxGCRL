"""DADS losses and reward computation.

Dynamics-Aware Discovery of Skills (DADS): https://arxiv.org/abs/1907.01657

Contains:
- Skill dynamics loss (negative log-likelihood of a diagonal Gaussian)
- DADS intrinsic reward: r(s, z, s') = log q(s'|s,z) - log[(1/K) Σ_z' q(s'|s,z')]
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients
from brax.training.types import Params

from .networks import DADSNetworks

_PMAP_AXIS_NAME = "i"

# Type alias for transitions (to avoid circular import)
Transition = Any  # Will be the Transition NamedTuple from dads.py


def _gaussian_log_prob_sum(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
) -> jnp.ndarray:
    """Log probability of ``x`` under N(mean, exp(log_std)^2), summed over the last dim.

    Args:
        x:       Target values, shape (..., D).
        mean:    Predicted mean,  shape (..., D).
        log_std: Predicted log-std, shape (..., D).

    Returns:
        Scalar-per-sample log probability, shape (...,).
    """
    return -0.5 * jnp.sum(
        ((x - mean) ** 2) * jnp.exp(-2.0 * log_std) + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def skill_dynamics_loss_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dyn_params: Params,
    transitions: Transition,
) -> jnp.ndarray:
    """Negative log-likelihood loss for the skill dynamics model.

    Trains q_phi(s' | s, z) = N(s + mean_phi(s,z), diag(exp(log_std_phi(s,z))^2)).

    Args:
        dads_network: DADS networks container.
        state_dim:    Dimension of the raw state (no skill appended).
        dyn_params:   Skill dynamics network parameters.
        transitions:  Batch of transitions with skill labels in extras.

    Returns:
        Scalar mean NLL loss.
    """
    states      = transitions.observation[:, :state_dim]       # (B, state_dim)
    next_states = transitions.next_observation[:, :state_dim]  # (B, state_dim)
    skills      = transitions.extras["state_extras"]["skill"]  # (B, num_skills)
    delta_s     = next_states - states                         # (B, state_dim)

    dynamics_input = jnp.concatenate([states, skills], axis=-1)  # (B, state_dim+num_skills)
    output  = dads_network.skill_dynamics_network.apply(None, dyn_params, dynamics_input)
    mean    = output[:, :state_dim]   # (B, state_dim)
    log_std = output[:, state_dim:]   # (B, state_dim)

    log_prob = _gaussian_log_prob_sum(delta_s, mean, log_std)  # (B,)
    return -jnp.mean(log_prob)


def compute_dads_reward(
    dads_network: DADSNetworks,
    state_dim: int,
    num_skills: int,
    dyn_params: Params,
    transitions: Transition,
) -> jnp.ndarray:
    """Compute DADS intrinsic reward.

    r(s, z, s') = log q(s'|s,z) - log[ (1/K) Σ_{z'} q(s'|s,z') ]

    Equivalently:
        = log q(s'|s,z) - logsumexp_{z'}[log q(s'|s,z')] + log(K)

    This maximises the mutual information I(s'; z | s).

    Args:
        dads_network: DADS networks container.
        state_dim:    Dimension of the raw state (no skill appended).
        num_skills:   Number of discrete skills K.
        dyn_params:   Skill dynamics network parameters.
        transitions:  Batch of transitions with skill labels in extras.

    Returns:
        Array of intrinsic rewards, shape (batch_size,).
    """
    states      = transitions.observation[:, :state_dim]       # (B, state_dim)
    next_states = transitions.next_observation[:, :state_dim]  # (B, state_dim)
    skills      = transitions.extras["state_extras"]["skill"]  # (B, num_skills)
    delta_s     = next_states - states                         # (B, state_dim)

    B = states.shape[0]

    # ---- log q(s'|s,z) for the *actual* skill --------------------------------
    dynamics_input = jnp.concatenate([states, skills], axis=-1)  # (B, state_dim+K)
    output  = dads_network.skill_dynamics_network.apply(None, dyn_params, dynamics_input)
    mean    = output[:, :state_dim]
    log_std = output[:, state_dim:]
    log_q_z = _gaussian_log_prob_sum(delta_s, mean, log_std)  # (B,)

    # ---- log q(s'|s,z') for *all* skills z' -----------------------------------
    # Build (B, K) grid: each sample paired with every skill.
    all_skills = jnp.eye(num_skills, dtype=jnp.float32)  # (K, num_skills)

    # Broadcast to (B, K, state_dim) and (B, K, num_skills)
    states_bk  = jnp.broadcast_to(states[:, None, :],     (B, num_skills, state_dim))
    skills_bk  = jnp.broadcast_to(all_skills[None, :, :], (B, num_skills, num_skills))
    delta_s_bk = jnp.broadcast_to(delta_s[:, None, :],    (B, num_skills, state_dim))

    # Flatten to (B*K, ...)
    flat_input   = jnp.reshape(jnp.concatenate([states_bk, skills_bk], axis=-1),
                               (B * num_skills, state_dim + num_skills))
    flat_delta_s = jnp.reshape(delta_s_bk, (B * num_skills, state_dim))

    flat_output    = dads_network.skill_dynamics_network.apply(None, dyn_params, flat_input)
    flat_mean      = flat_output[:, :state_dim]
    flat_log_std   = flat_output[:, state_dim:]

    flat_log_probs = _gaussian_log_prob_sum(flat_delta_s, flat_mean, flat_log_std)  # (B*K,)
    all_log_probs  = jnp.reshape(flat_log_probs, (B, num_skills))  # (B, K)

    # log[ (1/K) Σ_{z'} q(s'|s,z') ]  =  logsumexp(all_log_probs) - log(K)
    log_marginal = jax.scipy.special.logsumexp(all_log_probs, axis=-1) - jnp.log(num_skills)

    return log_q_z - log_marginal  # (B,)


def make_skill_dynamics_update_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dynamics_optimizer: optax.GradientTransformation,
):
    """Create the skill dynamics update function.

    Args:
        dads_network:       DADS networks container.
        state_dim:          Dimension of the raw state (no skill appended).
        dynamics_optimizer: Optimizer for the skill dynamics network.

    Returns:
        Update function ``(dyn_params, transitions, optimizer_state, key)``
        → ``(loss, new_params, new_optimizer_state)``.
    """
    def loss_fn(dyn_params, transitions):
        return skill_dynamics_loss_fn(dads_network, state_dim, dyn_params, transitions)

    return gradients.gradient_update_fn(
        loss_fn,
        dynamics_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
