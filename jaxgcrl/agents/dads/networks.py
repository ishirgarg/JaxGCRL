"""DADS networks.

Dynamics-Aware Discovery of Skills (DADS): https://arxiv.org/abs/1907.01657

Architecture:
  - policy_network:         MLP taking [state | one-hot skill z]  -> action distribution params
  - q_network:              MLP taking [state | one-hot skill z, action] -> Q values
  - skill_dynamics_network: MLP taking [state | one-hot skill z]  -> predicted (normalized)
                            delta to a future state s+, i.e. q_phi(s+ | s, z).

This repo's variant of DADS models a *gamma-discounted future state* s+ rather
than the one-step next state s'.  The skill dynamics network therefore predicts
the (running-normalized) displacement delta = s+ - s.  Normalization of both the
network input and the delta target is handled in ``losses.py`` using persisted
running statistics (so the same function is used for the numerator and the
all-skills marginal, and so the post-training empowerment map is well-defined).
"""

from typing import Optional, Sequence

import flax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from jaxgcrl.agents.common_networks import (
    MLP,
    ActivationFn,
    make_inference_fn,
    make_policy_network,
    make_q_network,
)


@flax.struct.dataclass
class DADSNetworks:
    """Networks for the DADS agent."""
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    # Predicts the (normalized) delta = s+ - s given (s, z); diagonal Gaussian
    # with identity covariance.
    skill_dynamics_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_skill_dynamics_network(
    state_size: int,
    num_skills: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
) -> networks.FeedForwardNetwork:
    """Creates the skill dynamics network q_phi(s+ | s, z).

    The network takes [s | z] as input and outputs the mean for a diagonal
    Gaussian over the (normalized) displacement delta = s+ - s.  Covariance is
    fixed to identity; input and target normalization are applied in the loss
    using persisted running statistics.

    When ``use_xy_prior`` is True the network input excludes ``goal_indices``
    (e.g. x-y), and the output is restricted to those indices' deltas.

    Args:
        state_size: Dimension of the raw environment state (no skill appended).
        num_skills: Number of discrete skills.
        hidden_layer_sizes: Hidden layer sizes for the MLP.
        activation: Activation function.
        layer_norm: Whether to use layer normalisation.
        use_xy_prior: If True, only predict goal_indices deltas and exclude
            goal_indices from the input.
        goal_indices: Indices of the x-y coordinates. Required if
            ``use_xy_prior`` is True.
    """
    if use_xy_prior:
        if goal_indices is None:
            raise ValueError("goal_indices must be provided when use_xy_prior=True")
        input_size = (state_size - len(goal_indices)) + num_skills
        output_size = len(goal_indices)
    else:
        input_size = state_size + num_skills
        output_size = state_size  # Only mean; covariance is fixed to identity.

    dyn_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [output_size],
        activation=activation,
        layer_norm=layer_norm,
    )

    def apply(processor_params, dyn_params, state_skill_input):
        # processor_params intentionally unused: normalization happens in the
        # loss using running statistics, so the network sees pre-normalized
        # [s|z] input and predicts the normalized delta.
        del processor_params
        return dyn_module.apply(dyn_params, state_skill_input)

    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(
        init=lambda key: dyn_module.init(key, dummy_input), apply=apply
    )


def make_dads_networks(
    observation_size: int,
    action_size: int,
    state_size: int,
    num_skills: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
    use_xy_prior: bool = False,
    goal_indices: Optional[jnp.ndarray] = None,
) -> DADSNetworks:
    """Build all DADS networks.

    Args:
        observation_size: Dimension of the *skill-augmented* observation
            (= state_size + num_skills).  Used for policy and Q-networks.
        action_size: Action dimension.
        state_size: Dimension of the raw environment state (no skill appended).
            Used for the skill dynamics network.
        num_skills: Number of discrete skills.
        preprocess_observations_fn: Observation normaliser (applied to the
            skill-augmented observation for policy and Q-networks).
        hidden_layer_sizes: Hidden layer sizes shared by all MLPs.
        activation: Activation function.
        layer_norm: Whether to use layer normalisation.
        use_xy_prior: Restrict the skill dynamics network to goal_indices deltas.
        goal_indices: Indices of the x-y coordinates (needed for use_xy_prior).
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    skill_dynamics_network = make_skill_dynamics_network(
        state_size=state_size,
        num_skills=num_skills,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
        use_xy_prior=use_xy_prior,
        goal_indices=goal_indices,
    )
    return DADSNetworks(
        policy_network=policy_network,
        q_network=q_network,
        skill_dynamics_network=skill_dynamics_network,
        parametric_action_distribution=parametric_action_distribution,
    )
