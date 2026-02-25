"""DIAYN losses and reward computation.

Contains:
- Discriminator loss function (cross-entropy for skill prediction)
- DIAYN intrinsic reward computation
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients
from brax.training.types import Params

from .networks import DIAYNNetworks

_PMAP_AXIS_NAME = "i"

# Type alias for transitions (to avoid circular import)
Transition = Any  # Will be the Transition NamedTuple from diayn.py


def discriminator_loss_fn(
    diayn_network: DIAYNNetworks,
    state_dim: int,
    disc_params: Params,
    transitions: Transition,
) -> jnp.ndarray:
    """Cross-entropy loss for the discriminator q_phi(z | s').

    Uses the *next* state (state part only, no skill appended) to
    predict which skill produced this transition.

    Args:
        diayn_network: DIAYN networks container.
        state_dim: Dimension of the raw state (without skill).
        disc_params: Discriminator network parameters.
        transitions: Batch of transitions with skill labels in extras.

    Returns:
        Scalar cross-entropy loss.
    """
    next_states = transitions.next_observation[:, :state_dim]  # (B, state_dim)
    skills = transitions.extras["state_extras"]["skill"]  # (B, num_skills)
    skill_indices = jnp.argmax(skills, axis=-1)  # (B,)
    logits = diayn_network.discriminator_network.apply(
        None, disc_params, next_states
    )  # (B, num_skills)
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, num_skills)
    # Gather the log-probability of the correct skill
    loss = -jnp.mean(log_probs[jnp.arange(logits.shape[0]), skill_indices])
    return loss


def compute_diayn_reward(
    diayn_network: DIAYNNetworks,
    state_dim: int,
    num_skills: int,
    disc_params: Params,
    transitions: Transition,
) -> jnp.ndarray:
    """Compute DIAYN intrinsic reward: r(s', z) = log q_φ(z | s') + log(num_skills).

    Args:
        diayn_network: DIAYN networks container.
        state_dim: Dimension of the raw state (without skill).
        num_skills: Number of skills.
        disc_params: Discriminator network parameters (current, not target).
        transitions: Batch of transitions with skill labels in extras.

    Returns:
        Array of intrinsic rewards, shape (batch_size,).
    """
    next_states = transitions.next_observation[:, :state_dim]  # (B, state_dim)
    skills = transitions.extras["state_extras"]["skill"]  # (B, num_skills)
    skill_indices = jnp.argmax(skills, axis=-1)  # (B,)

    disc_logits = diayn_network.discriminator_network.apply(
        None, disc_params, next_states
    )  # (B, num_skills)
    log_q = jax.nn.log_softmax(disc_logits, axis=-1)  # (B, num_skills)
    log_q_z = log_q[jnp.arange(log_q.shape[0]), skill_indices]  # (B,)
    diayn_reward = log_q_z + jnp.log(num_skills)  # (B,)
    return diayn_reward


def make_discriminator_update_fn(
    diayn_network: DIAYNNetworks,
    state_dim: int,
    discriminator_optimizer: optax.GradientTransformation,
):
    """Create discriminator update function.

    Args:
        diayn_network: DIAYN networks container.
        state_dim: Dimension of the raw state (without skill).
        discriminator_optimizer: Optimizer for the discriminator.

    Returns:
        Update function that takes (disc_params, transitions, optimizer_state, key)
        and returns (loss, new_params, new_optimizer_state).
    """
    def loss_fn(disc_params, transitions):
        return discriminator_loss_fn(diayn_network, state_dim, disc_params, transitions)

    return gradients.gradient_update_fn(
        loss_fn,
        discriminator_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
