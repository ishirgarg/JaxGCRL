"""Exploration bonus functions for reward shaping.

Agents use the high-level :func:`create_exploration_bonuses` entry point,
which returns an :class:`ExplorationBonuses` bundle wrapping one or more
bonuses. The bundle exposes a tiny functional API::

    bonuses = create_exploration_bonuses(types, weights, env=..., ...)
    state = bonuses.initial_state                                # tuple pytree
    state = bonuses.init_from_states(state, state_batch)         # post-prefill seed
    total, state, metrics = bonuses.compute(state, transitions, key)

``total`` has shape ``transitions.reward.shape``, is the weighted sum across
all configured bonuses, and is what the agent adds to the reward. The agent
only needs to thread ``state`` through its training step.

Internally each bonus is a triple ``(bonus_fn, initial_state, init_fn)`` where

    bonus_fn(state, transitions, key) -> (bonus, new_state, metrics)
    init_fn(state, states_batch)      -> new_state   (optional; may be None)

``init_fn`` is called once after prefill with a batch of observed states so
bonuses like RND can seed their observation-normalization statistics from the
initial random-policy rollouts (per the RND paper, "initialize the normalization
parameters by stepping a random agent in the environment for a small number of
steps"). Bonuses that don't need post-prefill init return ``None`` instead.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass as fdataclass


@fdataclass
class StatelessBonusState:
    """Placeholder state for bonuses with no trainable parameters."""
    pass


@fdataclass
class RNDBonusState:
    """Mutable state for Random Network Distillation.

    Carries both the predictor (trainable) and two running statistics: an
    observation normalizer (mean/std over states) and an intrinsic-reward
    normalizer (mean/M2/count used to derive a running std of the raw
    intrinsic reward, approximating the "running std of intrinsic returns"
    scaling from the RND paper).
    """
    predictor_params: Any
    opt_state: Any
    obs_mean: jnp.ndarray  # (state_size,)
    obs_std: jnp.ndarray   # (state_size,)
    reward_mean: jnp.ndarray   # scalar
    reward_m2: jnp.ndarray     # scalar (Welford sum of squared deviations)
    reward_count: jnp.ndarray  # scalar


BonusFn = Callable[[Any, Any, jax.Array], Tuple[jnp.ndarray, Any, dict]]
InitFromStatesFn = Optional[Callable[[Any, jnp.ndarray], Any]]


def create_exploration_bonus(
    bonus_type: str,
    *,
    env,
    state_size: int,
    key: jax.Array,
    # empowerment-specific
    empowerment_run_dir: Optional[str] = None,
    empowerment_epoch: Optional[int] = None,
    empowerment_num_splus_samples: int = 128,
    empowerment_score_chunk_size: int = 32,
    empowerment_mean: float = 0.0,
    empowerment_scale: float = 1.0,
    empowerment_precomputed_scorer: Optional[Callable] = None,
    empowerment_use_full_obs: bool = False,
    # RND-specific
    rnd_feature_dim: int = 64,
    rnd_hidden_dim: int = 256,
    rnd_num_hidden: int = 2,
    rnd_learning_rate: float = 1e-4,
    rnd_obs_clip: float = 5.0,
    rnd_use_goal: bool = False,
    goal_indices: Optional[Sequence[int]] = None,
) -> Tuple[BonusFn, Any, InitFromStatesFn]:
    """Factory: returns ``(bonus_fn, initial_state, init_from_states_fn)``.

    Supported ``bonus_type`` values:
      - ``"empowerment"``: offline empowerment scorer. Emits
        ``(raw_emp - mean) / scale`` so the raw empowerment distribution can
        be globally recentered and rescaled before the per-bonus weight.
        If ``empowerment_precomputed_scorer`` is provided (e.g. one already
        built for a goal proposer), it is reused instead of loading a second
        copy of the agent.
      - ``"rnd"``: Random Network Distillation — prediction-error of a
        trainable predictor against a fixed random target network, both
        operating on observation-normalized, ``[-rnd_obs_clip, rnd_obs_clip]``
        clipped states. Normalization stats are seeded post-prefill via
        ``init_from_states_fn``. When ``rnd_use_goal`` is true, the state
        passed to RND is sliced to ``goal_indices`` so novelty is measured
        only over the goal-relevant dimensions.
    """
    if bonus_type == "empowerment":
        return _create_empowerment_bonus(
            env=env,
            state_size=state_size,
            key=key,
            run_dir=empowerment_run_dir,
            epoch=empowerment_epoch,
            num_splus_samples=empowerment_num_splus_samples,
            chunk_size=empowerment_score_chunk_size,
            mean=empowerment_mean,
            scale=empowerment_scale,
            precomputed_scorer=empowerment_precomputed_scorer,
            use_full_obs=empowerment_use_full_obs,
        )
    if bonus_type == "rnd":
        return _create_rnd_bonus(
            state_size=state_size,
            key=key,
            feature_dim=rnd_feature_dim,
            hidden_dim=rnd_hidden_dim,
            num_hidden=rnd_num_hidden,
            learning_rate=rnd_learning_rate,
            obs_clip=rnd_obs_clip,
            use_goal=rnd_use_goal,
            goal_indices=goal_indices,
        )
    raise ValueError(f"Unknown exploration bonus type: {bonus_type!r}")


def _create_empowerment_bonus(
    *,
    env,
    state_size: int,
    key: jax.Array,
    run_dir: Optional[str],
    epoch: Optional[int],
    num_splus_samples: int,
    chunk_size: int,
    mean: float,
    scale: float,
    precomputed_scorer: Optional[Callable] = None,
    use_full_obs: bool = False,
) -> Tuple[BonusFn, StatelessBonusState, None]:
    """Offline empowerment scorer wrapped as a bonus fn.

    Emits ``(raw_emp - mean) / scale``; ``mean`` recenters the raw empowerment
    and ``scale`` standard-deviation-style rescales, both as fixed globals.
    The per-bonus weight applied by the agent multiplies this value. When
    ``precomputed_scorer`` is supplied, the agent/obs-builder/scorer load is
    skipped (used by callers that already loaded one for the goal proposer).
    When ``use_full_obs`` is true, the state vector (obs with the goal sliced
    off) is fed directly to the empowerment network instead of overwriting
    specific indices of a cached OGBench template.
    """
    if precomputed_scorer is not None:
        scorer = precomputed_scorer
    else:
        if run_dir is None:
            raise ValueError(
                "empowerment_run_dir must be set when exploration_bonus_type='empowerment'."
            )
        from .empowerment import (
            infer_empowerment_override_indices_from_env,
            load_offline_empowerment_agent,
            make_empowerment_full_obs_builder,
            make_empowerment_obs_builder,
            make_offline_empowerment_scorer,
        )

        emp_agent, _ex_obs_dim, base_obs = load_offline_empowerment_agent(
            run_dir=run_dir,
            jax_env=env,
            template_rng=key,
            epoch=epoch,
            num_splus_samples=num_splus_samples,
        )
        if use_full_obs:
            if state_size != int(_ex_obs_dim):
                raise ValueError(
                    "empowerment_use_full_obs=True requires state_size == "
                    f"ex_obs_dim; got state_size={state_size}, ex_obs_dim={_ex_obs_dim}."
                )
            obs_builder = make_empowerment_full_obs_builder()
        else:
            ogbench_idx, jax_idx = infer_empowerment_override_indices_from_env(env)
            obs_builder = make_empowerment_obs_builder(
                jnp.asarray(base_obs),
                ogbench_idx,
                jax_idx,
                state_size=state_size,
            )
        scorer = make_offline_empowerment_scorer(
            emp_agent, obs_builder, chunk_size=chunk_size
        )

    mean_f = jnp.asarray(mean, dtype=jnp.float32)
    scale_f = jnp.asarray(scale, dtype=jnp.float32)

    def empowerment_bonus(state, transitions, bonus_key):
        # Score s_{t+1} to match EmpowermentSAC: reward(s,a,s') = emp(s').
        states = transitions.next_observation[..., :state_size]
        shape = states.shape[:2]
        flat_states = states.reshape(-1, state_size)
        raw = scorer(flat_states, bonus_key).reshape(shape)
        bonus = (raw - mean_f) / scale_f
        metrics = {
            "empowerment_raw_mean": jnp.mean(raw),
            "empowerment_shifted_mean": jnp.mean(bonus),
        }
        return bonus, state, metrics

    return empowerment_bonus, StatelessBonusState(), None


def _create_rnd_bonus(
    *,
    state_size: int,
    key: jax.Array,
    feature_dim: int,
    hidden_dim: int,
    num_hidden: int,
    learning_rate: float,
    obs_clip: float,
    use_goal: bool = False,
    goal_indices: Optional[Sequence[int]] = None,
) -> Tuple[BonusFn, RNDBonusState, InitFromStatesFn]:
    """Random Network Distillation with observation normalization.

    Target is a frozen randomly initialized MLP; predictor is a same-shape
    MLP trained each call on the batch's states via a single Adam step.
    Both nets operate on ``clip((s - mean) / std, -obs_clip, obs_clip)``,
    with ``mean``/``std`` seeded from the post-prefill state distribution
    via the returned ``init_from_states_fn``.

    When ``use_goal`` is true, states are first sliced to ``goal_indices`` so
    RND's prediction error (and normalization stats) are over only the
    goal-relevant state dimensions.
    """
    from .networks import Encoder

    if use_goal:
        if goal_indices is None or len(goal_indices) == 0:
            raise ValueError(
                "rnd_use_goal=True requires non-empty goal_indices."
            )
        goal_idx_arr = jnp.asarray(tuple(int(i) for i in goal_indices), dtype=jnp.int32)
        input_dim = int(goal_idx_arr.shape[0])
    else:
        goal_idx_arr = None
        input_dim = state_size

    target_net = Encoder(
        repr_dim=feature_dim,
        network_width=hidden_dim,
        network_depth=num_hidden,
        skip_connections=0,
        use_relu=False,
        use_ln=True,
    )
    predictor_net = Encoder(
        repr_dim=feature_dim,
        network_width=hidden_dim,
        network_depth=num_hidden,
        skip_connections=0,
        use_relu=False,
        use_ln=True,
    )

    tkey, pkey = jax.random.split(key)
    dummy = jnp.zeros((1, input_dim), dtype=jnp.float32)
    target_params = target_net.init(tkey, dummy)
    predictor_params0 = predictor_net.init(pkey, dummy)

    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(predictor_params0)

    clip_val = float(obs_clip)

    def _normalize(flat_states, mean, std):
        return jnp.clip(
            (flat_states - mean) / jnp.maximum(std, 1e-8),
            -clip_val,
            clip_val,
        )

    def _per_state_error(predictor_params, normalized):
        target_feat = jax.lax.stop_gradient(target_net.apply(target_params, normalized))
        pred_feat = predictor_net.apply(predictor_params, normalized)
        return jnp.sum((target_feat - pred_feat) ** 2, axis=-1)

    def _loss_fn(predictor_params, normalized):
        return jnp.mean(_per_state_error(predictor_params, normalized))

    # Pre-init (mean=0, std=1) is used until init_from_states runs post-prefill.
    initial_state = RNDBonusState(
        predictor_params=predictor_params0,
        opt_state=opt_state0,
        obs_mean=jnp.zeros((input_dim,), dtype=jnp.float32),
        obs_std=jnp.ones((input_dim,), dtype=jnp.float32),
        reward_mean=jnp.asarray(0.0, dtype=jnp.float32),
        reward_m2=jnp.asarray(0.0, dtype=jnp.float32),
        reward_count=jnp.asarray(0.0, dtype=jnp.float32),
    )

    def _welford_update(mean, m2, count, batch_flat):
        """Parallel Welford update for running mean/M2 over a flat batch."""
        n = jnp.asarray(batch_flat.shape[0], dtype=jnp.float32)
        batch_mean = jnp.mean(batch_flat)
        batch_m2 = jnp.sum((batch_flat - batch_mean) ** 2)
        new_count = count + n
        delta = batch_mean - mean
        new_mean = mean + delta * n / new_count
        new_m2 = m2 + batch_m2 + (delta ** 2) * count * n / new_count
        return new_mean, new_m2, new_count

    def rnd_bonus(state: RNDBonusState, transitions, bonus_key):
        del bonus_key
        states = transitions.observation[..., :state_size]
        shape = states.shape[:2]
        flat = states.reshape(-1, state_size)
        if goal_idx_arr is not None:
            flat = flat[:, goal_idx_arr]
        normalized = _normalize(flat, state.obs_mean, state.obs_std)

        per_err = _per_state_error(state.predictor_params, normalized)
        raw_bonus_flat = per_err

        # Update running reward stats *before* using std so the first batch
        # produces a sensible scale (std=1 fallback via reward_count<=1).
        new_reward_mean, new_reward_m2, new_reward_count = _welford_update(
            state.reward_mean, state.reward_m2, state.reward_count, raw_bonus_flat
        )
        reward_std = jnp.sqrt(
            jnp.where(new_reward_count > 1.0, new_reward_m2 / new_reward_count, 1.0)
        )
        bonus_flat = raw_bonus_flat / jnp.maximum(reward_std, 1e-8)
        bonus = bonus_flat.reshape(shape)

        loss_value, grads = jax.value_and_grad(_loss_fn)(state.predictor_params, normalized)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.predictor_params)
        new_predictor_params = optax.apply_updates(state.predictor_params, updates)
        new_state = state.replace(
            predictor_params=new_predictor_params,
            opt_state=new_opt_state,
            reward_mean=new_reward_mean,
            reward_m2=new_reward_m2,
            reward_count=new_reward_count,
        )
        metrics = {
            "rnd_loss": loss_value,
            "rnd_raw_bonus_mean": jnp.mean(raw_bonus_flat),
            "rnd_bonus_mean": jnp.mean(bonus),
            "rnd_reward_std": reward_std,
        }
        return bonus, new_state, metrics

    def init_from_states(state: RNDBonusState, states: jnp.ndarray) -> RNDBonusState:
        """Seed ``obs_mean``/``obs_std`` from a batch of observed states.

        Per the RND paper, normalization parameters are initialized by stepping
        a random agent in the environment for a small number of steps. Here we
        use the prefill (uniform-random buffer warmup) rollouts for that.
        """
        flat = states.reshape(-1, state_size)
        if goal_idx_arr is not None:
            flat = flat[:, goal_idx_arr]
        new_mean = jnp.mean(flat, axis=0)
        new_std = jnp.std(flat, axis=0)
        return state.replace(obs_mean=new_mean, obs_std=new_std)

    return rnd_bonus, initial_state, init_from_states


class ExplorationBonuses:
    """A static bundle of exploration bonuses with shared state management.

    State is a plain tuple of per-bonus states (a heterogeneous pytree); agents
    thread this tuple through their training step. An empty bundle (no bonus
    types configured) is still safe to call — ``compute`` returns zeros and
    the original state unchanged.
    """

    def __init__(
        self,
        bonus_types: Tuple[str, ...],
        weights: Tuple[float, ...],
        fns: Tuple[BonusFn, ...],
        initial_state: Tuple,
        init_from_states_fns: Tuple[InitFromStatesFn, ...],
    ):
        self.bonus_types = bonus_types
        self.weights = weights
        self.fns = fns
        self.initial_state = initial_state
        self._init_from_states_fns = init_from_states_fns
        self.is_empty = len(fns) == 0

    def compute(
        self,
        state: Tuple,
        transitions,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple, dict]:
        """Return ``(weighted_total, raw_total, new_state, metrics)``.

        ``weighted_total`` = ``sum_i weight_i * bonus_i`` is what gets added to
        the reward (matching existing SAC reward shaping). ``raw_total`` =
        ``sum_i bonus_i`` is the un-scaled sum of the raw bonuses; callers
        that want to train a separate Q_exp *and* apply the weight at the
        actor loss (rather than at the reward) should use this instead.
        """
        if self.is_empty:
            zeros = jnp.zeros_like(transitions.reward)
            return zeros, zeros, state, {}

        total = jnp.zeros_like(transitions.reward)
        raw_total = jnp.zeros_like(transitions.reward)
        new_states = []
        metrics: dict = {}
        keys = jax.random.split(key, len(self.fns))
        for i, (fn, weight, bt, sub_key) in enumerate(
            zip(self.fns, self.weights, self.bonus_types, keys)
        ):
            bonus, new_state_i, m = fn(state[i], transitions, sub_key)
            total = total + jnp.asarray(weight, dtype=total.dtype) * bonus
            raw_total = raw_total + bonus
            new_states.append(new_state_i)
            for mk, mv in m.items():
                metrics[f"bonus_{i}_{bt}_{mk}"] = mv
        metrics["bonus_total_mean"] = jnp.mean(total)
        metrics["bonus_raw_total_mean"] = jnp.mean(raw_total)
        return total, raw_total, tuple(new_states), metrics

    def init_from_states(self, state: Tuple, states_batch: jnp.ndarray) -> Tuple:
        """Seed per-bonus state from a batch of observed states (post-prefill)."""
        if self.is_empty:
            return state
        new_states = []
        for init_fn, s in zip(self._init_from_states_fns, state):
            if init_fn is not None:
                s = init_fn(s, states_batch)
            new_states.append(s)
        return tuple(new_states)


def create_exploration_bonuses(
    bonus_types: Union[None, str, Sequence[str]],
    bonus_weights: Union[float, Sequence[float]],
    *,
    env,
    state_size: int,
    key: jax.Array,
    # empowerment-specific
    empowerment_run_dir: Optional[str] = None,
    empowerment_epoch: Optional[int] = None,
    empowerment_num_splus_samples: int = 128,
    empowerment_score_chunk_size: int = 32,
    empowerment_mean: float = 0.0,
    empowerment_scale: float = 1.0,
    empowerment_precomputed_scorer: Optional[Callable] = None,
    empowerment_use_full_obs: bool = False,
    # RND-specific
    rnd_feature_dim: int = 64,
    rnd_hidden_dim: int = 256,
    rnd_num_hidden: int = 2,
    rnd_learning_rate: float = 1e-4,
    rnd_obs_clip: float = 5.0,
    rnd_use_goal: bool = False,
    goal_indices: Optional[Sequence[int]] = None,
) -> ExplorationBonuses:
    """Build an :class:`ExplorationBonuses` bundle from user-facing config.

    Accepts ``bonus_types`` / ``bonus_weights`` as either a single value
    (``str``/``float``) or a matching-length sequence. A single ``bonus_weights``
    scalar is broadcast to all bonus types. When both are sequences their
    lengths must match. ``bonus_types=None`` yields an empty bundle.
    """
    if bonus_types is None:
        types_tuple: Tuple[str, ...] = ()
    elif isinstance(bonus_types, str):
        types_tuple = (bonus_types,)
    else:
        types_tuple = tuple(bonus_types)

    if isinstance(bonus_weights, (int, float)):
        weights_tuple: Tuple[float, ...] = (float(bonus_weights),) * len(types_tuple)
    else:
        weights_tuple = tuple(float(w) for w in bonus_weights)

    if types_tuple and len(weights_tuple) != len(types_tuple):
        raise ValueError(
            "exploration_bonus_type and exploration_bonus_weight must have "
            f"matching lengths; got {len(types_tuple)} types ({types_tuple!r}) "
            f"vs {len(weights_tuple)} weights ({weights_tuple!r})."
        )

    fns, states, init_fns = [], [], []
    for bt in types_tuple:
        key, sub_key = jax.random.split(key)
        fn, st, init_fn = create_exploration_bonus(
            bt,
            env=env,
            state_size=state_size,
            key=sub_key,
            empowerment_run_dir=empowerment_run_dir,
            empowerment_epoch=empowerment_epoch,
            empowerment_num_splus_samples=empowerment_num_splus_samples,
            empowerment_score_chunk_size=empowerment_score_chunk_size,
            empowerment_mean=empowerment_mean,
            empowerment_scale=empowerment_scale,
            empowerment_precomputed_scorer=empowerment_precomputed_scorer,
            empowerment_use_full_obs=empowerment_use_full_obs,
            rnd_feature_dim=rnd_feature_dim,
            rnd_hidden_dim=rnd_hidden_dim,
            rnd_num_hidden=rnd_num_hidden,
            rnd_learning_rate=rnd_learning_rate,
            rnd_obs_clip=rnd_obs_clip,
            rnd_use_goal=rnd_use_goal,
            goal_indices=goal_indices,
        )
        fns.append(fn)
        states.append(st)
        init_fns.append(init_fn)

    return ExplorationBonuses(
        bonus_types=types_tuple,
        weights=weights_tuple,
        fns=tuple(fns),
        initial_state=tuple(states),
        init_from_states_fns=tuple(init_fns),
    )
