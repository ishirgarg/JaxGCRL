"""Networks for Distilled SAC with Flow Matching Policy.

Architecture:
  - velocity_network : MLP taking [action | time | obs] -> velocity
      (a continuous normalizing flow / CNF velocity field)
  - q_network       : standard twin-critic MLP taking (obs, action) -> Q-values

The flow policy maps base-distribution noise (N(0,I)) to actions via an ODE:
    da/dt = v(a_t, t, obs)    t: 0 -> 1
Sampling uses forward Euler; log-prob uses backward Euler + Hutchinson trace.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import networks as brax_networks, types
from brax.training.types import PRNGKey
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


# ---------------------------------------------------------------------------
# Shared MLP building block
# ---------------------------------------------------------------------------

class MLP(linen.Module):
    """Feed-forward MLP with optional layer-norm."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        hidden = data
        for i, size in enumerate(self.layer_sizes):
            hidden = linen.Dense(size, name=f"hidden_{i}", kernel_init=self.kernel_init)(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


# ---------------------------------------------------------------------------
# Flow velocity network
# ---------------------------------------------------------------------------

class VelocityModule(linen.Module):
    """CNF velocity field: v([action | time | obs]) -> velocity.

    Input is the concatenation of the current action a_t (action_size),
    the scalar time t broadcast to (1,), and the goal-conditioned observation
    (obs_size).
    """

    hidden_sizes: Sequence[int]
    action_size: int
    layer_norm: bool = False
    activation: ActivationFn = linen.relu

    @linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            layer_sizes=list(self.hidden_sizes) + [self.action_size],
            activation=self.activation,
            layer_norm=self.layer_norm,
        )(x)


# ---------------------------------------------------------------------------
# Networks container
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class FlowPolicyNetworks:
    """Container for the flow policy + Q-network used by DistilledSAC."""

    velocity_network: brax_networks.FeedForwardNetwork   # v(a, t, obs) -> velocity
    q_network: brax_networks.FeedForwardNetwork          # Q(obs, action)  -> (B, n_critics)
    action_size: int
    obs_size: int


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_velocity_network(
    action_size: int,
    obs_size: int,
    hidden_sizes: Sequence[int] = (256, 256, 256, 256),
    layer_norm: bool = False,
) -> Tuple[brax_networks.FeedForwardNetwork, VelocityModule]:
    """Create the CNF velocity-field FeedForwardNetwork.

    The apply signature is:
        apply(None, vel_params, concat_input: (B, action_size+1+obs_size)) -> (B, action_size)
    """
    vel_module = VelocityModule(
        hidden_sizes=hidden_sizes,
        action_size=action_size,
        layer_norm=layer_norm,
    )
    input_size = action_size + 1 + obs_size
    dummy_input = jnp.zeros((1, input_size))

    def apply(_, vel_params, x):
        return vel_module.apply(vel_params, x)

    return (
        brax_networks.FeedForwardNetwork(
            init=lambda key: vel_module.init(key, dummy_input),
            apply=apply,
        ),
        vel_module,
    )


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256),
    n_critics: int = 2,
    layer_norm: bool = False,
) -> brax_networks.FeedForwardNetwork:
    """Twin-critic Q-network (same as SAC)."""

    class QModule(linen.Module):
        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            hidden = jnp.concatenate([obs, actions], axis=-1)
            qs = [
                MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    layer_norm=layer_norm,
                )(hidden)
                for _ in range(self.n_critics)
            ]
            return jnp.concatenate(qs, axis=-1)

    q_module = QModule(n_critics=n_critics)
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    return brax_networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action),
        apply=apply,
    )


def make_flow_policy_networks(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256),
    layer_norm: bool = False,
) -> Tuple[FlowPolicyNetworks, VelocityModule]:
    """Build the full FlowPolicyNetworks (velocity + Q) and return the VelocityModule."""
    velocity_network, vel_module = make_velocity_network(
        action_size=action_size,
        obs_size=obs_size,
        hidden_sizes=hidden_layer_sizes,
        layer_norm=layer_norm,
    )
    q_network = make_q_network(
        obs_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        layer_norm=layer_norm,
    )
    return (
        FlowPolicyNetworks(
            velocity_network=velocity_network,
            q_network=q_network,
            action_size=action_size,
            obs_size=obs_size,
        ),
        vel_module,
    )


# ---------------------------------------------------------------------------
# ODE integration helpers
# ---------------------------------------------------------------------------

def _build_x(a_t: jnp.ndarray, t: float, obs: jnp.ndarray) -> jnp.ndarray:
    """Concatenate [action | time | obs] for velocity network input.

    Args:
        a_t:  (B, act_size)   action at time t
        t:    scalar           current time
        obs:  (B, obs_size)   goal-conditioned observations
    Returns:
        x: (B, act_size + 1 + obs_size)
    """
    B = a_t.shape[0]
    t_emb = jnp.full((B, 1), t)
    return jnp.concatenate([a_t, t_emb, obs], axis=-1)


def sample_action(
    velocity_network: brax_networks.FeedForwardNetwork,
    vel_params,
    obs: jnp.ndarray,
    key: PRNGKey,
    n_ode_steps: int,
    action_size: int,
) -> jnp.ndarray:
    """Sample an action from the flow policy via forward Euler ODE integration.

    z ~ N(0,I)  ->  [ODE forward]  ->  a_raw  ->  tanh  ->  action ∈ (-1,1)

    Args:
        velocity_network: FeedForwardNetwork for the velocity field.
        vel_params:       Parameters of the velocity network.
        obs:              Goal-conditioned observations, shape (B, obs_size).
        key:              JAX PRNGKey.
        n_ode_steps:      Number of Euler integration steps.
        action_size:      Dimension of the action space.
    Returns:
        action: (B, action_size) tanh-squashed actions.
    """
    B = obs.shape[0]
    key, z_key = jax.random.split(key)
    a_t = jax.random.normal(z_key, (B, action_size))
    dt = 1.0 / n_ode_steps

    for step in range(n_ode_steps):
        t = step * dt
        x = _build_x(a_t, t, obs)
        v_t = velocity_network.apply(None, vel_params, x)
        a_t = a_t + dt * v_t

    return jnp.tanh(a_t)


def sample_and_log_prob(
    velocity_network: brax_networks.FeedForwardNetwork,
    vel_params,
    obs: jnp.ndarray,
    key: PRNGKey,
    n_ode_steps: int,
    n_hutchinson_samples: int,
    action_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample action AND estimate log-prob via forward ODE + Hutchinson trace.

    Used during SAC fine-tuning for the actor/alpha updates.

    Returns:
        action:   (B, action_size) tanh-squashed actions
        log_prob: (B,)  log π_flow(action | obs)
    """
    B = obs.shape[0]
    key, z_key = jax.random.split(key)
    z = jax.random.normal(z_key, (B, action_size))

    a_t = z
    logp_delta = jnp.zeros(B)
    dt = 1.0 / n_ode_steps

    for step in range(n_ode_steps):
        t = step * dt
        key, h_key = jax.random.split(key)

        # ---- velocity -------------------------------------------------------
        x = _build_x(a_t, t, obs)
        v_t = velocity_network.apply(None, vel_params, x)  # (B, A)

        # ---- Hutchinson trace estimate of div(v) ----------------------------
        eps_all = (
            2.0
            * jax.random.bernoulli(h_key, 0.5, (n_hutchinson_samples, B, action_size)).astype(
                jnp.float32
            )
            - 1.0
        )  # Rademacher samples  (n_h, B, A)

        def trace_for_one_eps(eps: jnp.ndarray) -> jnp.ndarray:
            """eps: (B, A).  Returns per-sample Hutchinson trace estimate (B,)."""
            a_t_sg = jax.lax.stop_gradient(a_t)  # trajectory treated as fixed

            def hutchinson_single(a_i, obs_i, eps_i):
                """Single-sample Hutchinson estimate for one env."""
                def v_fn(a):
                    t_emb_i = jnp.array([t])
                    xi = jnp.concatenate([a, t_emb_i, obs_i])[None]  # (1, A+1+O)
                    v_out = velocity_network.apply(None, vel_params, xi)  # (1, A)
                    return v_out[0]  # (A,)

                _, jvp_val = jax.jvp(v_fn, (a_i,), (eps_i,))
                return jnp.dot(eps_i, jvp_val)

            return jax.vmap(hutchinson_single)(a_t_sg, obs, eps)

        traces = jax.vmap(trace_for_one_eps)(eps_all)  # (n_h, B)
        div_estimate = traces.mean(axis=0)              # (B,)

        # ---- Euler step -----------------------------------------------------
        a_t = a_t + dt * v_t
        logp_delta = logp_delta - dt * div_estimate

    # base-distribution log-prob
    log_p0 = (
        -0.5 * jnp.sum(z ** 2, axis=-1)
        - 0.5 * action_size * jnp.log(2.0 * jnp.pi)
    )
    log_prob_raw = log_p0 + logp_delta

    # tanh squashing Jacobian correction
    action = jnp.tanh(a_t)
    log_prob = log_prob_raw - jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)

    return action, log_prob


def log_prob_of_action(
    velocity_network: brax_networks.FeedForwardNetwork,
    vel_params,
    obs: jnp.ndarray,
    action_squashed: jnp.ndarray,
    key: PRNGKey,
    n_ode_steps: int,
    n_hutchinson_samples: int,
) -> jnp.ndarray:
    """Compute log π_flow(action | obs) via backward ODE integration.

    Used for the distillation NLL loss.

    Args:
        velocity_network:   FeedForwardNetwork for the velocity field.
        vel_params:         Velocity network parameters.
        obs:                Goal-conditioned observations, shape (B, obs_size).
        action_squashed:    Tanh-squashed actions from replay buffer, shape (B, A).
        key:                JAX PRNGKey.
        n_ode_steps:        Number of backward Euler steps.
        n_hutchinson_samples: Rademacher samples for trace estimation.
    Returns:
        log_prob: (B,)
    """
    EPS = 1e-5
    B, A = action_squashed.shape
    dt = 1.0 / n_ode_steps

    # inverse-tanh to get pre-squash action (the point at t=1 in the ODE)
    a_t = jnp.arctanh(jnp.clip(action_squashed, -1.0 + EPS, 1.0 - EPS))

    logp_delta = jnp.zeros(B)

    for step in range(n_ode_steps):
        t = 1.0 - step * dt     # time goes from 1 → dt (backward)
        key, h_key = jax.random.split(key)

        # ---- velocity -------------------------------------------------------
        x = _build_x(a_t, t, obs)
        v_t = velocity_network.apply(None, vel_params, x)  # (B, A)

        # ---- Hutchinson trace estimate  -------------------------------------
        eps_all = (
            2.0
            * jax.random.bernoulli(h_key, 0.5, (n_hutchinson_samples, B, A)).astype(jnp.float32)
            - 1.0
        )

        def trace_for_one_eps(eps: jnp.ndarray) -> jnp.ndarray:
            """eps: (B, A).  Returns per-sample Hutchinson trace estimate (B,)."""
            a_t_sg = jax.lax.stop_gradient(a_t)  # trajectory treated as fixed

            def hutchinson_single(a_i, obs_i, eps_i):
                def v_fn(a):
                    t_emb_i = jnp.array([t])
                    xi = jnp.concatenate([a, t_emb_i, obs_i])[None]  # (1, A+1+O)
                    v_out = velocity_network.apply(None, vel_params, xi)  # (1, A)
                    return v_out[0]  # (A,)

                _, jvp_val = jax.jvp(v_fn, (a_i,), (eps_i,))
                return jnp.dot(eps_i, jvp_val)

            return jax.vmap(hutchinson_single)(a_t_sg, obs, eps)

        traces = jax.vmap(trace_for_one_eps)(eps_all)  # (n_h, B)
        div_estimate = traces.mean(axis=0)              # (B,)

        # ---- backward Euler step -------------------------------------------
        a_t = a_t - dt * v_t
        logp_delta = logp_delta + dt * div_estimate   # opposite sign from forward integration

    # base distribution: z = a_t at end of backward integration
    z = a_t
    log_p0 = (
        -0.5 * jnp.sum(z ** 2, axis=-1)
        - 0.5 * A * jnp.log(2.0 * jnp.pi)
    )
    log_prob_raw = log_p0 + logp_delta

    # squashing correction: log π(a|s,g) = log π_raw(arctanh(a)|s,g) - Σ log(1-a²)
    log_prob = log_prob_raw - jnp.sum(jnp.log(1.0 - action_squashed ** 2 + 1e-6), axis=-1)

    return log_prob


def make_inference_fn(
    flow_networks: FlowPolicyNetworks,
    n_ode_steps: int = 10,
):
    """Create a make_policy function compatible with the Brax Evaluator."""

    def make_policy(params, deterministic: bool = False):
        normalizer_params, vel_params = params

        def policy(observations: jnp.ndarray, key: PRNGKey):
            # observations: (B, obs_size) – already goal-conditioned
            action = sample_action(
                flow_networks.velocity_network,
                vel_params,
                observations,
                key,
                n_ode_steps=n_ode_steps,
                action_size=flow_networks.action_size,
            )
            return action, {}

        return policy

    return make_policy
