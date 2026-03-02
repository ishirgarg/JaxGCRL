"""Losses for the Distilled SAC agent.

Contains:
  - distillation_nll_loss : Phase 2a  – NLL of flow policy on replay-buffer actions
  - critic_loss           : Phase 2b/3 – SAC critic (Bellman) loss with flow actor
  - actor_loss            : Phase 3    – SAC actor loss (maximize Q - alpha * log_prob)
  - alpha_loss            : Phase 3    – automatic entropy tuning
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients, types
from brax.training.acme import running_statistics
from brax.training.types import Params, PRNGKey

from .networks import FlowPolicyNetworks, log_prob_of_action, sample_and_log_prob

_PMAP_AXIS_NAME = "i"

# Type alias – the actual NamedTuple lives in distilled_sac.py
Transition = Any


# ---------------------------------------------------------------------------
# Phase 2a – Distillation NLL loss
# ---------------------------------------------------------------------------

def distillation_nll_loss(
    vel_params: Params,
    flow_networks: FlowPolicyNetworks,
    transitions: Transition,
    goal_set: jnp.ndarray,          # (G, goal_dim)  – fixed set of goals
    state_dim: int,
    goal_key: PRNGKey,
    ode_key: PRNGKey,
    n_ode_steps: int,
    n_hutchinson_samples: int,
) -> jnp.ndarray:
    """Negative log-likelihood loss for flow policy distillation.

    Minimises:
        E_{(s,a) ~ replay}  E_{g ~ goal_set}  [ -log π_flow(a | [s, g]) ]

    A fresh random goal is sampled per gradient step so the flow policy
    learns to be goal-invariant over the provided goal set.

    Args:
        vel_params:           Flow velocity network parameters.
        flow_networks:        FlowPolicyNetworks container.
        transitions:          Sampled replay-buffer batch.
        goal_set:             Array of candidate goals, shape (G, goal_dim).
        state_dim:            Dimension of the raw state (first slice of obs).
        goal_key:             PRNGKey for goal sampling.
        ode_key:              PRNGKey for ODE Hutchinson estimator.
        n_ode_steps:          Backward-Euler steps.
        n_hutchinson_samples: Rademacher samples per step.
    Returns:
        Scalar NLL loss.
    """
    B = transitions.observation.shape[0]

    # ---- sample a random goal for every transition in the batch -------------
    # We sample ONE goal per gradient step (as specified) and broadcast to batch.
    goal_idx = jax.random.randint(goal_key, (), 0, goal_set.shape[0])
    g = goal_set[goal_idx]                  # (goal_dim,)
    g_batch = jnp.broadcast_to(g, (B, g.shape[0]))  # (B, goal_dim)

    # ---- build goal-conditioned observations --------------------------------
    state = transitions.observation[:, :state_dim]   # (B, state_dim)
    obs = jnp.concatenate([state, g_batch], axis=-1)  # (B, state_dim + goal_dim)

    # ---- compute log-prob via backward ODE ----------------------------------
    log_probs = log_prob_of_action(
        velocity_network=flow_networks.velocity_network,
        vel_params=vel_params,
        obs=obs,
        action_squashed=transitions.action,
        key=ode_key,
        n_ode_steps=n_ode_steps,
        n_hutchinson_samples=n_hutchinson_samples,
    )  # (B,)

    return -jnp.mean(log_probs)


def make_distillation_update_fn(
    flow_networks: FlowPolicyNetworks,
    optimizer: optax.GradientTransformation,
    state_dim: int,
    n_ode_steps: int,
    n_hutchinson_samples: int,
):
    """Create a pmappable distillation gradient-update function.

    Returns a function with signature:
        update(vel_params, transitions, goal_set, goal_key, ode_key,
               optimizer_state=...)
        -> (loss, new_vel_params, new_optimizer_state)
    """
    def loss_fn(vel_params, transitions, goal_set, goal_key, ode_key):
        return distillation_nll_loss(
            vel_params=vel_params,
            flow_networks=flow_networks,
            transitions=transitions,
            goal_set=goal_set,
            state_dim=state_dim,
            goal_key=goal_key,
            ode_key=ode_key,
            n_ode_steps=n_ode_steps,
            n_hutchinson_samples=n_hutchinson_samples,
        )

    return gradients.gradient_update_fn(loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME)


# ---------------------------------------------------------------------------
# Phase 2b / 3 – SAC critic loss (works with any actor)
# ---------------------------------------------------------------------------

def critic_loss(
    q_params: Params,
    vel_params: Params,
    normalizer_params,
    target_q_params: Params,
    alpha: float,
    transitions: Transition,
    flow_networks: FlowPolicyNetworks,
    discounting: float,
    reward_scaling: float,
    n_ode_steps: int,
    n_hutchinson_samples: int,
    key: PRNGKey,
) -> jnp.ndarray:
    """SAC critic (Bellman) loss using the flow policy for next-action sampling.

    Target:
        y = r + γ * (min_Q(s', a') - α * log π(a' | s'))
    where a' ~ π_flow(· | s').

    Args:
        q_params:       Current Q-network parameters.
        vel_params:     Flow velocity network parameters.
        normalizer_params: Running statistics for observation normalisation.
        target_q_params: Target Q-network parameters (lagged copy of q_params).
        alpha:          Entropy temperature coefficient.
        transitions:    Batch of (s, a, r, s', discount) transitions.
        flow_networks:  FlowPolicyNetworks container.
        discounting:    Discount factor γ.
        reward_scaling: Reward scaling factor.
        n_ode_steps:    ODE steps for next-action sampling.
        n_hutchinson_samples: Hutchinson samples for log-prob estimation.
        key:            PRNGKey.
    Returns:
        Scalar critic loss (mean of squared Bellman errors).
    """
    obs = transitions.observation
    next_obs = transitions.next_observation
    actions = transitions.action
    rewards = transitions.reward * reward_scaling
    discounts = transitions.discount * discounting

    # ---- sample next action + log-prob with target Q -----------------------
    next_action, next_log_prob = sample_and_log_prob(
        velocity_network=flow_networks.velocity_network,
        vel_params=vel_params,
        obs=next_obs,
        key=key,
        n_ode_steps=n_ode_steps,
        n_hutchinson_samples=n_hutchinson_samples,
        action_size=flow_networks.action_size,
    )

    # ---- target Q-value ----------------------------------------------------
    target_q = flow_networks.q_network.apply(
        normalizer_params, target_q_params, next_obs, next_action
    )  # (B, n_critics)
    min_target_q = jnp.min(target_q, axis=-1)          # (B,)
    target_v = min_target_q - alpha * next_log_prob     # (B,)
    q_target = jax.lax.stop_gradient(rewards + discounts * target_v)

    # ---- current Q-values --------------------------------------------------
    q_vals = flow_networks.q_network.apply(normalizer_params, q_params, obs, actions)  # (B, n)
    # Bellman error for each critic
    q_error = q_vals - jnp.expand_dims(q_target, axis=-1)  # (B, n)
    return 0.5 * jnp.mean(q_error ** 2)


def make_critic_update_fn(
    flow_networks: FlowPolicyNetworks,
    q_optimizer: optax.GradientTransformation,
    discounting: float,
    reward_scaling: float,
    n_ode_steps: int,
    n_hutchinson_samples: int,
):
    """Create a pmappable critic gradient-update function.

    Returns a function with signature:
        update(q_params, vel_params, normalizer_params, target_q_params,
               alpha, transitions, key, optimizer_state=...)
        -> (loss, new_q_params, new_optimizer_state)
    """
    def loss_fn(q_params, vel_params, normalizer_params, target_q_params, alpha, transitions, key):
        return critic_loss(
            q_params=q_params,
            vel_params=vel_params,
            normalizer_params=normalizer_params,
            target_q_params=target_q_params,
            alpha=alpha,
            transitions=transitions,
            flow_networks=flow_networks,
            discounting=discounting,
            reward_scaling=reward_scaling,
            n_ode_steps=n_ode_steps,
            n_hutchinson_samples=n_hutchinson_samples,
            key=key,
        )

    return gradients.gradient_update_fn(loss_fn, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)


# ---------------------------------------------------------------------------
# Phase 3 – SAC actor loss
# ---------------------------------------------------------------------------

def actor_loss(
    vel_params: Params,
    normalizer_params,
    q_params: Params,
    alpha: float,
    transitions: Transition,
    flow_networks: FlowPolicyNetworks,
    n_ode_steps: int,
    n_hutchinson_samples: int,
    key: PRNGKey,
) -> jnp.ndarray:
    """SAC actor loss for the flow matching policy.

    Maximises Q(s, a) - α * log π(a | s), i.e. minimises the negation.

    Args:
        vel_params:           Flow velocity network parameters.
        normalizer_params:    Running statistics.
        q_params:             Q-network parameters.
        alpha:                Entropy temperature.
        transitions:          Batch of replay-buffer transitions.
        flow_networks:        Networks container.
        n_ode_steps:          ODE steps.
        n_hutchinson_samples: Hutchinson samples.
        key:                  PRNGKey.
    Returns:
        Scalar actor loss.
    """
    obs = transitions.observation

    action, log_prob = sample_and_log_prob(
        velocity_network=flow_networks.velocity_network,
        vel_params=vel_params,
        obs=obs,
        key=key,
        n_ode_steps=n_ode_steps,
        n_hutchinson_samples=n_hutchinson_samples,
        action_size=flow_networks.action_size,
    )

    q_vals = flow_networks.q_network.apply(normalizer_params, q_params, obs, action)
    min_q = jnp.min(q_vals, axis=-1)   # (B,)

    return jnp.mean(alpha * log_prob - min_q)


def make_actor_update_fn(
    flow_networks: FlowPolicyNetworks,
    policy_optimizer: optax.GradientTransformation,
    n_ode_steps: int,
    n_hutchinson_samples: int,
):
    """Create a pmappable actor gradient-update function."""
    def loss_fn(vel_params, normalizer_params, q_params, alpha, transitions, key):
        return actor_loss(
            vel_params=vel_params,
            normalizer_params=normalizer_params,
            q_params=q_params,
            alpha=alpha,
            transitions=transitions,
            flow_networks=flow_networks,
            n_ode_steps=n_ode_steps,
            n_hutchinson_samples=n_hutchinson_samples,
            key=key,
        )

    return gradients.gradient_update_fn(loss_fn, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)


# ---------------------------------------------------------------------------
# Phase 3 – alpha (temperature) loss
# ---------------------------------------------------------------------------

def alpha_loss(
    log_alpha: jnp.ndarray,
    vel_params: Params,
    normalizer_params,
    transitions: Transition,
    flow_networks: FlowPolicyNetworks,
    n_ode_steps: int,
    n_hutchinson_samples: int,
    target_entropy: float,
    key: PRNGKey,
) -> jnp.ndarray:
    """Automatic entropy tuning loss.

    Minimises:  E[ -log_alpha * (log π(a|s) + target_entropy) ]
    """
    obs = transitions.observation
    _, log_prob = sample_and_log_prob(
        velocity_network=flow_networks.velocity_network,
        vel_params=vel_params,
        obs=obs,
        key=key,
        n_ode_steps=n_ode_steps,
        n_hutchinson_samples=n_hutchinson_samples,
        action_size=flow_networks.action_size,
    )
    return jnp.mean(-log_alpha * jax.lax.stop_gradient(log_prob + target_entropy))


def make_alpha_update_fn(
    flow_networks: FlowPolicyNetworks,
    alpha_optimizer: optax.GradientTransformation,
    n_ode_steps: int,
    n_hutchinson_samples: int,
    target_entropy: float,
):
    """Create a pmappable alpha gradient-update function."""
    def loss_fn(log_alpha, vel_params, normalizer_params, transitions, key):
        return alpha_loss(
            log_alpha=log_alpha,
            vel_params=vel_params,
            normalizer_params=normalizer_params,
            transitions=transitions,
            flow_networks=flow_networks,
            n_ode_steps=n_ode_steps,
            n_hutchinson_samples=n_hutchinson_samples,
            target_entropy=target_entropy,
            key=key,
        )

    return gradients.gradient_update_fn(loss_fn, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
