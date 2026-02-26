"""DADS networks.

Dynamics-Aware Discovery of Skills (DADS): https://arxiv.org/abs/1907.01657

Architecture:
  - policy_network:         MLP taking [state | one-hot skill z]  -> action distribution params
  - q_network:              MLP taking [state | one-hot skill z, action] -> Q values
  - skill_dynamics_network: MLP taking [state | one-hot skill z]  -> (mean, log_std) of delta_s
                            where delta_s = s' - s  (size 2 * state_dim)
"""

from typing import Any, Callable, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class DADSNetworks:
    """Networks for the DADS agent."""
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    # Predicts a diagonal Gaussian over delta_s = s' - s given (s, z).
    skill_dynamics_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a Q-network taking (obs=[state|skill], action) as input."""

    class QModule(linen.Module):
        """Q Module."""
        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    layer_norm=layer_norm,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a policy network taking obs=[state|skill] as input."""

    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_skill_dynamics_network(
    state_size: int,
    num_skills: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates the skill dynamics network q_phi(s' | s, z).

    The network takes [s | z] as input and outputs (mean, log_std) for the
    diagonal Gaussian over delta_s = s' - s.  Output size is 2 * state_size.

    Args:
        state_size: Dimension of the raw environment state (no skill appended).
        num_skills: Number of discrete skills.
        hidden_layer_sizes: Hidden layer sizes for the MLP.
        activation: Activation function.
        layer_norm: Whether to use layer normalisation.
    """
    input_size = state_size + num_skills
    output_size = 2 * state_size  # (mean, log_std) each of size state_size

    dyn_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [output_size],
        activation=activation,
        layer_norm=layer_norm,
    )

    def apply(processor_params, dyn_params, state_skill_input):
        # processor_params intentionally unused: dynamics network sees raw [s|z]
        return dyn_module.apply(dyn_params, state_skill_input)

    dummy_input = jnp.zeros((1, input_size))
    return networks.FeedForwardNetwork(
        init=lambda key: dyn_module.init(key, dummy_input), apply=apply
    )


def make_inference_fn(dads_networks: DADSNetworks):
    """Creates an inference function for the DADS policy.

    The returned ``make_policy`` accepts ``(normalizer_params, policy_params)``
    and returns a callable ``policy(obs, key) -> (action, extras)`` where
    ``obs`` is the *skill-augmented* observation [state | skill_z].
    """

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = dads_networks.policy_network.apply(*params, observations)
            if deterministic:
                return dads_networks.parametric_action_distribution.mode(logits), {}
            return (
                dads_networks.parametric_action_distribution.sample(logits, key_sample),
                {},
            )

        return policy

    return make_policy


def make_dads_networks(
    observation_size: int,
    action_size: int,
    state_size: int,
    num_skills: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
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
    )
    return DADSNetworks(
        policy_network=policy_network,
        q_network=q_network,
        skill_dynamics_network=skill_dynamics_network,
        parametric_action_distribution=parametric_action_distribution,
    )
