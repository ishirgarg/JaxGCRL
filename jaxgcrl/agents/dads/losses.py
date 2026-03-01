"""DADS losses and reward computation.

Dynamics-Aware Discovery of Skills (DADS): https://arxiv.org/abs/1907.01657

Contains:
- Skill dynamics loss (negative log-likelihood of a diagonal Gaussian)
- DADS intrinsic reward: r(s, z, s') = log q(s'|s,z) - log[(1/K) Σ_z' q(s'|s,z')]
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients
from brax.training.types import Params

from .networks import DADSNetworks

_PMAP_AXIS_NAME = "i"

# Type alias for transitions (to avoid circular import)
Transition = Any  # Will be the Transition NamedTuple from dads.py


def _prepare_dynamics_inputs(
    transitions: Transition,
    state_dim: int,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare inputs for dynamics network computation.
    
    Extracts states, computes delta_s, and prepares dynamics input.
    Handles use_xy_prior case by excluding goal_indices from input.
    
    Returns:
        delta_s: State differences, shape (B, output_dim)
        dynamics_input: Input to dynamics network, shape (B, input_dim)
        states: Raw states, shape (B, state_dim)
        next_states: Raw next states, shape (B, state_dim)
    """
    states = transitions.observation[:, :state_dim]  # (B, state_dim)
    next_states = transitions.next_observation[:, :state_dim]  # (B, state_dim)
    skills = transitions.extras["state_extras"]["skill"]  # (B, num_skills)
    
    if use_xy_prior:
        # Extract state without goal_indices for input
        state_without_xy = jnp.take(states, non_goal_indices, axis=1)  # (B, state_dim - len(goal_indices))
        # Only compute delta_s for goal_indices
        delta_s = next_states[:, goal_indices] - states[:, goal_indices]  # (B, len(goal_indices))
        # Input: state without goal_indices + skill
        dynamics_input = jnp.concatenate([state_without_xy, skills], axis=-1)  # (B, state_dim - len(goal_indices) + num_skills)
    else:
        delta_s = next_states - states  # (B, state_dim)
        dynamics_input = jnp.concatenate([states, skills], axis=-1)  # (B, state_dim+num_skills)
    
    return delta_s, dynamics_input, states, next_states


def _gaussian_log_prob_sum_identity_cov(
    x: jnp.ndarray,
    mean: jnp.ndarray,
) -> jnp.ndarray:
    """Log probability of ``x`` under N(mean, I), summed over the last dim.

    Uses identity covariance matrix (as per DADS paper).

    Args:
        x:    Target values, shape (..., D).
        mean: Predicted mean,  shape (..., D).

    Returns:
        Scalar-per-sample log probability, shape (...,).
    """
    # With identity covariance: log p(x|mean) = -0.5 * ||x - mean||^2 - 0.5*D*log(2*pi)
    return -0.5 * jnp.sum((x - mean) ** 2 + jnp.log(2.0 * jnp.pi), axis=-1)


def skill_dynamics_loss_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dyn_params: Params,
    transitions: Transition,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Negative log-likelihood loss for the skill dynamics model.

    Trains q_phi(s' | s, z) = N(mean_phi(s,z), I) with identity covariance.

    As per paper: "We normalize the output targets using their batch-average and 
    batch-standard deviation, similar to batch-normalization."

    Args:
        dads_network: DADS networks container.
        state_dim:    Dimension of the raw state (no skill appended).
        dyn_params:   Skill dynamics network parameters.
        transitions:  Batch of transitions with skill labels in extras.
        use_xy_prior: If True, only predict goal_indices differences and exclude goal_indices from input.
        goal_indices: Indices of x-y coordinates. Required if use_xy_prior=True.

    Returns:
        Scalar mean NLL loss.
    """
    delta_s, dynamics_input, _, _ = _prepare_dynamics_inputs(
        transitions, state_dim, use_xy_prior, goal_indices, non_goal_indices
    )

    # Normalize inputs using batch statistics (batch normalization on input)
    input_mean = jnp.mean(dynamics_input, axis=0, keepdims=True)  # (1, input_size)
    input_std = jnp.std(dynamics_input, axis=0, keepdims=True) + 1e-8  # (1, input_size)
    dynamics_input_norm = (dynamics_input - input_mean) / input_std

    # Network outputs mean (covariance is identity)
    mean = dads_network.skill_dynamics_network.apply(None, dyn_params, dynamics_input_norm)  # (B, output_size)

    # Normalize targets using batch statistics (as per paper)
    target_mean = jnp.mean(delta_s, axis=0, keepdims=True)  # (1, output_size)
    target_std = jnp.std(delta_s, axis=0, keepdims=True) + 1e-8  # (1, output_size)
    delta_s_norm = (delta_s - target_mean) / target_std
    mean_norm = (mean - target_mean) / target_std

    log_prob = _gaussian_log_prob_sum_identity_cov(delta_s_norm, mean_norm)  # (B,)
    return -jnp.mean(log_prob)


def compute_dads_reward(
    dads_network: DADSNetworks,
    state_dim: int,
    num_skills: int,
    dyn_params: Params,
    transitions: Transition,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute DADS intrinsic reward.

    r(s, z, s') = log q(s'|s,z) - log[ (1/K) Σ_{z'} q(s'|s,z') ]

    Equivalently:
        = log q(s'|s,z) - logsumexp_{z'}[log q(s'|s,z')] + log(K)

    This maximises the mutual information I(s'; z | s).

    Uses identity covariance and batch normalization as per paper.

    Args:
        dads_network: DADS networks container.
        state_dim:    Dimension of the raw state (no skill appended).
        num_skills:   Number of discrete skills K.
        dyn_params:   Skill dynamics network parameters.
        transitions:  Batch of transitions with skill labels in extras.
        use_xy_prior: If True, only predict goal_indices differences and exclude goal_indices from input.
        goal_indices: Indices of x-y coordinates. Required if use_xy_prior=True.

    Returns:
        Array of intrinsic rewards, shape (batch_size,).
    """
    delta_s, dynamics_input, states, _ = _prepare_dynamics_inputs(
        transitions, state_dim, use_xy_prior, goal_indices, non_goal_indices
    )
    
    B = states.shape[0]

    # Normalize targets using batch statistics (as per paper)
    target_mean = jnp.mean(delta_s, axis=0, keepdims=True)  # (1, output_size)
    target_std = jnp.std(delta_s, axis=0, keepdims=True) + 1e-8  # (1, output_size)
    delta_s_norm = (delta_s - target_mean) / target_std

    # ---- log q(s'|s,z) for the *actual* skill --------------------------------
    # Normalize input using batch statistics
    input_mean = jnp.mean(dynamics_input, axis=0, keepdims=True)  # (1, input_size)
    input_std = jnp.std(dynamics_input, axis=0, keepdims=True) + 1e-8  # (1, input_size)
    dynamics_input_norm = (dynamics_input - input_mean) / input_std
    
    mean = dads_network.skill_dynamics_network.apply(None, dyn_params, dynamics_input_norm)  # (B, output_size)
    mean_norm = (mean - target_mean) / target_std
    log_q_z = _gaussian_log_prob_sum_identity_cov(delta_s_norm, mean_norm)  # (B,)

    # ---- log q(s'|s,z') for *all* skills z' -----------------------------------
    # Build (B, K) grid: each sample paired with every skill.
    all_skills = jnp.eye(num_skills, dtype=jnp.float32)  # (K, num_skills)

    if use_xy_prior:
        # Broadcast state without xy and skills
        state_without_xy_bk = jnp.broadcast_to(state_without_xy[:, None, :], (B, num_skills, state_dim - len(goal_indices)))
        skills_bk = jnp.broadcast_to(all_skills[None, :, :], (B, num_skills, num_skills))
        delta_s_bk_norm = jnp.broadcast_to(delta_s_norm[:, None, :], (B, num_skills, len(goal_indices)))
        
        # Flatten to (B*K, ...)
        flat_input = jnp.reshape(jnp.concatenate([state_without_xy_bk, skills_bk], axis=-1),
                                 (B * num_skills, state_dim - len(goal_indices) + num_skills))
    else:
        # Broadcast to (B, K, state_dim) and (B, K, num_skills)
        states_bk = jnp.broadcast_to(states[:, None, :], (B, num_skills, state_dim))
        skills_bk = jnp.broadcast_to(all_skills[None, :, :], (B, num_skills, num_skills))
        delta_s_bk_norm = jnp.broadcast_to(delta_s_norm[:, None, :], (B, num_skills, state_dim))
        
        # Flatten to (B*K, ...)
        flat_input = jnp.reshape(jnp.concatenate([states_bk, skills_bk], axis=-1),
                                 (B * num_skills, state_dim + num_skills))
    
    # Normalize flat input using batch statistics
    flat_input_mean = jnp.mean(flat_input, axis=0, keepdims=True)  # (1, input_size)
    flat_input_std = jnp.std(flat_input, axis=0, keepdims=True) + 1e-8  # (1, input_size)
    flat_input_norm = (flat_input - flat_input_mean) / flat_input_std

    flat_mean = dads_network.skill_dynamics_network.apply(None, dyn_params, flat_input_norm)  # (B*K, output_size)
    flat_mean_norm = (flat_mean - target_mean) / target_std

    output_size = len(goal_indices) if use_xy_prior else state_dim
    flat_log_probs = _gaussian_log_prob_sum_identity_cov(
        jnp.reshape(delta_s_bk_norm, (B * num_skills, output_size)),
        flat_mean_norm
    )  # (B*K,)
    all_log_probs = jnp.reshape(flat_log_probs, (B, num_skills))  # (B, K)

    # log[ (1/K) Σ_{z'} q(s'|s,z') ]  =  logsumexp(all_log_probs) - log(K)
    log_marginal = jax.scipy.special.logsumexp(all_log_probs, axis=-1) - jnp.log(num_skills)

    return log_q_z - log_marginal  # (B,)


def make_skill_dynamics_update_fn(
    dads_network: DADSNetworks,
    state_dim: int,
    dynamics_optimizer: optax.GradientTransformation,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
    non_goal_indices: Optional[jnp.ndarray] = None,
):
    """Create the skill dynamics update function.

    Args:
        dads_network:       DADS networks container.
        state_dim:          Dimension of the raw state (no skill appended).
        dynamics_optimizer: Optimizer for the skill dynamics network.
        use_xy_prior:       If True, only predict goal_indices differences and exclude goal_indices from input.
        goal_indices:       Indices of x-y coordinates. Required if use_xy_prior=True.

    Returns:
        Update function ``(dyn_params, transitions, optimizer_state, key)``
        → ``(loss, new_params, new_optimizer_state)``.
    """
    def loss_fn(dyn_params, transitions):
        return skill_dynamics_loss_fn(
            dads_network, state_dim, dyn_params, transitions,
            use_xy_prior=use_xy_prior, goal_indices=goal_indices,
            non_goal_indices=non_goal_indices
        )

    return gradients.gradient_update_fn(
        loss_fn,
        dynamics_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
