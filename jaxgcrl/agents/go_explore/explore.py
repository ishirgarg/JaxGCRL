"""EXPLORE: optimistic UCB reward labeling for offline data.

Per Liu et al., when prior offline data is available without reward labels for
the task at hand, we can:

  1. Train a reward predictor ``r_θ(s,a)`` and a termination predictor
     ``T̂_θ(s,a)`` from the agent's online interactions (which DO carry true
     sparse rewards and termination flags).
  2. Label every sampled offline transition with the optimistic UCB reward

         UCBR(s,a) = r_θ(s,a) + ucb_coeff * (1/L) * ||f_φ(s,a) - f̄(s,a)||²

     and discount ``1 - sigmoid(T̂_θ(s,a))``.
  3. Mix the relabeled offline batch 50/50 with the online batch (RLPD-style)
     and feed both into a standard off-policy update.

The novelty bonus is sourced from the existing RND pipeline in
``exploration.py`` (see ``ExplorationBonuses.compute_first_rnd_novelty``) so
this module owns ONLY the reward and termination predictors. The two
predictors are MLPs over ``[state || action]``; we reuse the project's
``Encoder`` for the architecture so EXPLORE inherits the same network shape
as everything else in this codebase (matching the user's ask: don't bake in
new architecture details).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass as fdataclass

from .networks import Encoder


@fdataclass
class ExploreUCBState:
    """Trainable state for the EXPLORE reward + termination predictors."""

    reward_params: Any
    reward_opt_state: Any
    term_params: Any
    term_opt_state: Any


def create_explore_ucb_models(
    *,
    state_size: int,
    action_size: int,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    learning_rate: float = 3e-4,
    key: jax.Array,
) -> Tuple[ExploreUCBState, Callable, Callable]:
    """Build the reward predictor, termination predictor, and their training/relabel helpers.

    Returns
    -------
    initial_state : ExploreUCBState
        Initialized predictor params + Adam states.
    train_fn : Callable[[ExploreUCBState, Transition, int], (ExploreUCBState, dict)]
        Trains both predictors on a batch of online transitions for
        ``num_grad_steps`` Adam steps.
    relabel_fn : Callable[[ExploreUCBState, Transition, jnp.ndarray, float], Transition]
        Returns ``transitions`` with ``reward`` replaced by UCBR(s,a) and
        ``discount`` replaced by ``1 - sigmoid(T̂(s,a))``. ``novelty`` is the
        per-transition (1/L)||f_φ - f̄||² value (any positive scaling fine —
        ``ucb_coeff`` is the user-facing knob).
    """
    reward_net = Encoder(
        repr_dim=1,
        network_width=hidden_dim,
        network_depth=num_hidden,
        skip_connections=0,
        use_relu=False,
        use_ln=True,
    )
    term_net = Encoder(
        repr_dim=1,
        network_width=hidden_dim,
        network_depth=num_hidden,
        skip_connections=0,
        use_relu=False,
        use_ln=True,
    )

    sa_dim = state_size + action_size
    dummy = jnp.zeros((1, sa_dim), dtype=jnp.float32)

    rkey, tkey = jax.random.split(key)
    reward_params = reward_net.init(rkey, dummy)
    term_params = term_net.init(tkey, dummy)

    optimizer = optax.adam(learning_rate)
    reward_opt = optimizer.init(reward_params)
    term_opt = optimizer.init(term_params)

    initial_state = ExploreUCBState(
        reward_params=reward_params,
        reward_opt_state=reward_opt,
        term_params=term_params,
        term_opt_state=term_opt,
    )

    def _reward_apply(params, sa):
        return reward_net.apply(params, sa).squeeze(-1)

    def _term_logit_apply(params, sa):
        return term_net.apply(params, sa).squeeze(-1)

    def relabel_fn(
        state: ExploreUCBState,
        transitions,
        novelty: jnp.ndarray,
        ucb_coeff: float,
    ):
        """Relabel ``transitions`` with UCB reward and predicted-termination discount.

        ``novelty`` must broadcast to ``transitions.reward.shape``.
        """
        s = transitions.observation[..., :state_size]
        a = transitions.action
        sa = jnp.concatenate([s, a], axis=-1)
        r_pred = _reward_apply(state.reward_params, sa)
        t_logit = _term_logit_apply(state.term_params, sa)
        t_prob = jax.nn.sigmoid(t_logit)
        ucb_r = r_pred + jnp.asarray(ucb_coeff, dtype=r_pred.dtype) * novelty
        new_discount = 1.0 - t_prob
        return transitions._replace(reward=ucb_r, discount=new_discount)

    def train_fn(
        state: ExploreUCBState,
        transitions,
        num_grad_steps: int = 1,
    ):
        """Train reward (MSE) + termination (BCE) predictors on online transitions.

        Termination target is derived from ``discount``: a transition with
        ``discount == 0`` is treated as terminated (T=1). Any leading batch
        dimensions (e.g. ``(num_envs, episode_length)``) are flattened into a
        single batch axis before the gradient step.
        """
        s = transitions.observation[..., :state_size]
        a = transitions.action
        sa = jnp.concatenate([s, a], axis=-1).reshape(-1, sa_dim)
        r_target = transitions.reward.reshape(-1)
        t_target = (1.0 - transitions.discount).reshape(-1)

        def reward_loss_fn(params):
            r_pred = _reward_apply(params, sa)
            return jnp.mean((r_pred - r_target) ** 2)

        def term_loss_fn(params):
            t_logit = _term_logit_apply(params, sa)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(t_logit, t_target))

        def step(carry, _):
            st = carry
            rloss, rg = jax.value_and_grad(reward_loss_fn)(st.reward_params)
            r_updates, new_r_opt = optimizer.update(rg, st.reward_opt_state, st.reward_params)
            new_r_params = optax.apply_updates(st.reward_params, r_updates)

            tloss, tg = jax.value_and_grad(term_loss_fn)(st.term_params)
            t_updates, new_t_opt = optimizer.update(tg, st.term_opt_state, st.term_params)
            new_t_params = optax.apply_updates(st.term_params, t_updates)

            new_st = ExploreUCBState(
                reward_params=new_r_params,
                reward_opt_state=new_r_opt,
                term_params=new_t_params,
                term_opt_state=new_t_opt,
            )
            return new_st, (rloss, tloss)

        new_state, (rlosses, tlosses) = jax.lax.scan(
            step, state, (), length=num_grad_steps,
        )
        with_pred = _reward_apply(new_state.reward_params, sa)
        with_term = jax.nn.sigmoid(_term_logit_apply(new_state.term_params, sa))
        metrics = {
            "explore_reward_loss": jnp.mean(rlosses),
            "explore_term_loss": jnp.mean(tlosses),
            "explore_reward_pred_mean": jnp.mean(with_pred),
            "explore_term_prob_mean": jnp.mean(with_term),
            "explore_reward_target_mean": jnp.mean(r_target),
            "explore_term_target_mean": jnp.mean(t_target),
        }
        return new_state, metrics

    return initial_state, train_fn, relabel_fn
