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
class UCBBonusState:
    """Combined state for the UCB (EXPLORE) bonus.

    Wraps two trainable substates:
      * ``explore`` — reward and termination predictors (see explore.py).
      * ``rnd`` — UCB's INTERNAL RND target/predictor for the novelty term.
        Separate from any standalone ``"rnd"`` bonus the user might co-list,
        so listing ``"ucb"`` alone gives full EXPLORE without polluting
        online rewards with an additive RND shaping.
    """
    explore: Any
    rnd: Any


@fdataclass
class MISCBonusState:
    """Mutable state for the MISC mutual-information intrinsic reward.

    Holds the trainable parameters of the MI estimator T_phi (a small MLP),
    its Adam optimizer state, and a PRNG key advanced each train step (to
    produce a fresh trajectory-internal s_c permutation per update). T_phi
    is trained by gradient ascent on the MINE lower bound (Eq. 7 LHS); the
    per-transition reward (Eq. 6 surrogate) is computed read-only from the
    same params.
    """
    predictor_params: Any
    opt_state: Any
    key: jax.Array


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
# Per-bonus train hook (only set for bonuses with trainable params, e.g. RND):
# updates trainable params from a batch of *online* transitions. Returns
# ``(new_state, metrics)``. The bonus pipeline calls this via
# ``ExplorationBonuses.train(state, online_transitions)`` so that bonus-side
# learning is restricted to online data even though bonuses are *emitted* on
# the full (online + offline) batch.
TrainFn = Optional[Callable[[Any, Any, int], Tuple[Any, dict]]]
# Read-only RND novelty: (state, transitions) -> per-transition (1/L) * ||f_phi - f_bar||²,
# WITHOUT updating predictor params. Returned alongside RND bonuses so EXPLORE
# can label offline data with the bonus while keeping RND trained on online.
NoveltyFn = Optional[Callable[[Any, Any], jnp.ndarray]]
# UCB callbacks (only set on the "ucb" bonus). ``train_fn`` updates the reward
# and termination predictors on a batch of online transitions; ``relabel_fn``
# returns offline ``transitions`` with ``reward`` overwritten by UCBR(s,a) and
# ``discount`` overwritten by 1 - sigmoid(T̂(s,a)). The agent supplies the
# RND novelty term to ``relabel_fn`` (sourced via compute_first_rnd_novelty).
UCBTrainFn = Optional[Callable[[Any, Any, int], Tuple[Any, dict]]]
UCBRelabelFn = Optional[Callable[[Any, Any, jnp.ndarray], Any]]
# Per-bonus online-bonus hook (only set on the "ucb" bonus): returns
# ``ucb_coeff * RND_novelty`` for the given transitions, used by the agent to
# add UCB's RND novelty term to ONLINE rewards without overwriting them.
UCBOnlineBonusFn = Optional[Callable[[Any, Any], jnp.ndarray]]


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
    # UCB-specific (EXPLORE algorithm)
    ucb_action_size: Optional[int] = None,
    ucb_coeff: float = 1.0,
    ucb_hidden_dim: int = 256,
    ucb_num_hidden: int = 2,
    ucb_learning_rate: float = 3e-4,
    # MISC-specific
    controllable_indices: Optional[Sequence[int]] = None,
    misc_hidden_dim: int = 256,
    misc_num_hidden: int = 3,
    misc_learning_rate: float = 1e-3,
    misc_alpha: float = 5000.0,
    # Online-empowerment-specific
    online_empowerment_action_size: Optional[int] = None,
    online_empowerment_lr: float = 3e-4,
    online_empowerment_value_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_empowerment_actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_empowerment_value_latent_dim: int = 128,
    online_empowerment_num_skills: int = 5,
    online_empowerment_num_splus_samples: int = 32,
    online_empowerment_discount: float = 0.99,
    online_empowerment_tau: float = 0.005,
    online_empowerment_separate_qv: bool = False,
    online_empowerment_use_self_q_loss: bool = True,
    online_empowerment_layer_norm: bool = True,
    online_empowerment_bc_alpha: float = 0.01,
    online_empowerment_bonus_mean: float = 0.0,
    online_empowerment_bonus_scale: float = 1.0,
) -> Tuple[BonusFn, Any, InitFromStatesFn, NoveltyFn, TrainFn, UCBTrainFn, UCBRelabelFn, UCBOnlineBonusFn]:
    """Factory: returns ``(bonus_fn, initial_state, init_from_states_fn, novelty_fn, train_fn, ucb_train_fn, ucb_relabel_fn, ucb_online_bonus_fn)``.

    ``novelty_fn`` is non-None only for RND bonuses; it returns the per-transition
    ``(1/L) * ||f_phi - f_bar||²`` raw novelty without updating predictor params,
    enabling EXPLORE-style UCB labeling on offline data without polluting the
    predictor with offline samples.

    ``ucb_train_fn`` and ``ucb_relabel_fn`` are non-None only for the ``"ucb"``
    bonus (the EXPLORE algorithm). Its ``bonus_fn`` is a no-op (returns zeros)
    because UCB doesn't shape the reward additively — it relabels offline
    transitions in place via ``ucb_relabel_fn``, and learns the predictors via
    ``ucb_train_fn`` on online transitions. The agent invokes both explicitly.

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
        bonus_fn, st, init_fn = _create_empowerment_bonus(
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
        return bonus_fn, st, init_fn, None, None, None, None, None
    if bonus_type == "rnd":
        bonus_fn, st, init_fn, novelty_fn, train_fn = _create_rnd_bonus(
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
        return bonus_fn, st, init_fn, novelty_fn, train_fn, None, None, None
    if bonus_type == "misc":
        if goal_indices is None or len(goal_indices) == 0:
            raise ValueError(
                "'misc' bonus requires non-empty goal_indices (s_g)."
            )
        if controllable_indices is None or len(controllable_indices) == 0:
            raise ValueError(
                "'misc' bonus requires non-empty controllable_indices (s_c)."
            )
        bonus_fn, st, init_fn, train_fn = _create_misc_bonus(
            state_size=state_size,
            key=key,
            goal_indices=goal_indices,
            controllable_indices=controllable_indices,
            hidden_dim=misc_hidden_dim,
            num_hidden=misc_num_hidden,
            learning_rate=misc_learning_rate,
            alpha=misc_alpha,
        )
        return bonus_fn, st, init_fn, None, train_fn, None, None, None
    if bonus_type == "online_empowerment":
        if online_empowerment_action_size is None:
            raise ValueError(
                "'online_empowerment' bonus requires online_empowerment_action_size."
            )
        from .online_empowerment import _create_online_empowerment_bonus
        bonus_fn, st, init_fn, train_fn = _create_online_empowerment_bonus(
            state_size=state_size,
            action_size=online_empowerment_action_size,
            key=key,
            lr=online_empowerment_lr,
            value_hidden_dims=online_empowerment_value_hidden_dims,
            actor_hidden_dims=online_empowerment_actor_hidden_dims,
            value_latent_dim=online_empowerment_value_latent_dim,
            num_skills=online_empowerment_num_skills,
            num_splus_samples=online_empowerment_num_splus_samples,
            discount=online_empowerment_discount,
            tau=online_empowerment_tau,
            separate_qv=online_empowerment_separate_qv,
            use_self_q_loss=online_empowerment_use_self_q_loss,
            layer_norm=online_empowerment_layer_norm,
            bc_alpha=online_empowerment_bc_alpha,
            bonus_mean=online_empowerment_bonus_mean,
            bonus_scale=online_empowerment_bonus_scale,
        )
        return bonus_fn, st, init_fn, None, train_fn, None, None, None
    if bonus_type == "ucb":
        if ucb_action_size is None:
            raise ValueError("'ucb' bonus requires ucb_action_size to be set.")
        bonus_fn, st, init_fn, ucb_train_fn, ucb_relabel_fn, ucb_online_bonus_fn = _create_ucb_bonus(
            state_size=state_size,
            action_size=ucb_action_size,
            key=key,
            ucb_coeff=ucb_coeff,
            hidden_dim=ucb_hidden_dim,
            num_hidden=ucb_num_hidden,
            learning_rate=ucb_learning_rate,
            rnd_feature_dim=rnd_feature_dim,
            rnd_hidden_dim=rnd_hidden_dim,
            rnd_num_hidden=rnd_num_hidden,
            rnd_learning_rate=rnd_learning_rate,
            rnd_obs_clip=rnd_obs_clip,
            rnd_use_goal=rnd_use_goal,
            goal_indices=goal_indices,
        )
        return bonus_fn, st, init_fn, None, None, ucb_train_fn, ucb_relabel_fn, ucb_online_bonus_fn
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
            use_full_obs=use_full_obs,
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
) -> Tuple[BonusFn, RNDBonusState, InitFromStatesFn, NoveltyFn, TrainFn]:
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
        """Compute per-transition RND bonus.

        Read-only over predictor params (those are trained separately on online
        transitions via ``rnd_train``), but DOES update the running
        reward-stats normalizer on the full batch — this keeps the bonus-scale
        normalization consistent with what we actually emit (online + offline).
        """
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

        new_state = state.replace(
            reward_mean=new_reward_mean,
            reward_m2=new_reward_m2,
            reward_count=new_reward_count,
        )
        metrics = {
            "rnd_raw_bonus_mean": jnp.mean(raw_bonus_flat),
            "rnd_bonus_mean": jnp.mean(bonus),
            "rnd_reward_std": reward_std,
        }
        return bonus, new_state, metrics

    def rnd_train(state: RNDBonusState, online_transitions, num_grad_steps: int = 1):
        """Update RND predictor params on online states only.

        Per the EXPLORE setup: the RND prediction error must encode "novelty
        relative to what's been seen ONLINE" so that offline transitions can
        get a meaningful novelty bonus. Training on offline would collapse the
        signal we're trying to elicit.
        """
        states = online_transitions.observation[..., :state_size]
        flat = states.reshape(-1, state_size)
        if goal_idx_arr is not None:
            flat = flat[:, goal_idx_arr]
        normalized = _normalize(flat, state.obs_mean, state.obs_std)

        def step(carry, _):
            st = carry
            loss_value, grads = jax.value_and_grad(_loss_fn)(st.predictor_params, normalized)
            updates, new_opt_state = optimizer.update(grads, st.opt_state, st.predictor_params)
            new_predictor_params = optax.apply_updates(st.predictor_params, updates)
            new_st = st.replace(
                predictor_params=new_predictor_params,
                opt_state=new_opt_state,
            )
            return new_st, loss_value

        new_state, losses = jax.lax.scan(step, state, (), length=num_grad_steps)
        return new_state, {"rnd_loss": jnp.mean(losses)}

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

    def novelty_only(state: RNDBonusState, transitions) -> jnp.ndarray:
        """Per-transition ``(1/L) * ||f_phi - f_bar||²`` without updating predictor.

        Uses the predictor + observation-normalization stats currently held in
        ``state``; safe to call on offline transitions because no parameters or
        running stats are touched. Returned shape matches ``transitions.reward``.
        """
        states_flat = transitions.observation[..., :state_size]
        shape = states_flat.shape[:-1]
        flat = states_flat.reshape(-1, state_size)
        if goal_idx_arr is not None:
            flat = flat[:, goal_idx_arr]
        normalized = _normalize(flat, state.obs_mean, state.obs_std)
        per_err = _per_state_error(state.predictor_params, normalized)
        return (per_err / jnp.asarray(feature_dim, dtype=per_err.dtype)).reshape(shape)

    return rnd_bonus, initial_state, init_from_states, novelty_only, rnd_train


def _create_misc_bonus(
    *,
    state_size: int,
    key: jax.Array,
    goal_indices: Sequence[int],
    controllable_indices: Sequence[int],
    hidden_dim: int = 256,
    num_hidden: int = 3,
    learning_rate: float = 1e-3,
    alpha: float = 5000.0,
) -> Tuple[BonusFn, MISCBonusState, InitFromStatesFn, TrainFn]:
    """MISC: Mutual Information State-Controllable intrinsic reward.

    Trains an MI estimator ``T_phi(s_g, s_c) -> R`` by gradient ascent on the
    MINE lower bound (Eq. 7 LHS) with random temporal shuffles inside each
    sampled trajectory (the marginal sampler). Per-transition reward uses the
    Eq. 6 surrogate over the (s_t, s_{t+1}) pair, scaled by ``alpha`` and
    clipped to ``[0, 1]``.

    Both s_g and s_c are sliced from the state portion of the observation
    (``transitions.observation[..., :state_size]``); they may overlap in
    principle but the MISC formulation assumes a clean (goal, controllable)
    split. The estimator is *frozen* during ``bonus_fn`` (no grad, no param
    update) — its only update path is ``misc_train`` invoked by the agent on
    online trajectories.
    """
    from .networks import Encoder

    goal_idx_arr = jnp.asarray(
        tuple(int(i) for i in goal_indices), dtype=jnp.int32
    )
    ctrl_idx_arr = jnp.asarray(
        tuple(int(i) for i in controllable_indices), dtype=jnp.int32
    )
    dim_g = int(goal_idx_arr.shape[0])
    dim_c = int(ctrl_idx_arr.shape[0])
    input_dim = dim_g + dim_c

    # 3-hidden-layer ReLU MLP, no LayerNorm, scalar output (per spec §2).
    t_phi = Encoder(
        repr_dim=1,
        network_width=hidden_dim,
        network_depth=num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )

    init_key, train_key0 = jax.random.split(key)
    dummy = jnp.zeros((1, input_dim), dtype=jnp.float32)
    predictor_params0 = t_phi.init(init_key, dummy)
    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(predictor_params0)

    initial_state = MISCBonusState(
        predictor_params=predictor_params0,
        opt_state=opt_state0,
        key=train_key0,
    )

    alpha_f = jnp.asarray(alpha, dtype=jnp.float32)

    def _t_phi(params, s_g, s_c):
        """Apply T_phi(concat(s_g, s_c)). Returns scalar (last dim squeezed)."""
        x = jnp.concatenate([s_g, s_c], axis=-1)
        return t_phi.apply(params, x)[..., 0]

    def misc_bonus(state: MISCBonusState, transitions, bonus_key):
        """Per-transition MISC intrinsic reward (Eq. 6 surrogate).

        Read-only over T_phi: the per-transition signal uses the *current*
        frozen estimator. The estimator itself is updated by ``misc_train``
        on full trajectories, decoupled from this call.
        """
        del bonus_key
        states = transitions.observation[..., :state_size]
        next_states = transitions.next_observation[..., :state_size]
        s_g_t = states[..., goal_idx_arr]
        s_c_t = states[..., ctrl_idx_arr]
        s_g_tp1 = next_states[..., goal_idx_arr]
        s_c_tp1 = next_states[..., ctrl_idx_arr]

        params = jax.lax.stop_gradient(state.predictor_params)
        v_joint_0 = _t_phi(params, s_g_t, s_c_t)
        v_joint_1 = _t_phi(params, s_g_tp1, s_c_tp1)
        v_marg_0 = _t_phi(params, s_g_t, s_c_tp1)
        v_marg_1 = _t_phi(params, s_g_tp1, s_c_t)

        # log(0.5 * (exp(a) + exp(b))) = logsumexp([a, b]) - log(2).
        stacked = jnp.stack([v_marg_0, v_marg_1], axis=-1)
        log_marg_mean = jax.scipy.special.logsumexp(stacked, axis=-1) - jnp.log(2.0)
        r_raw = 0.5 * (v_joint_0 + v_joint_1) - log_marg_mean
        r_clipped = jnp.clip(alpha_f * r_raw, 0.0, 1.0)

        metrics = {
            "misc_raw_reward_mean": jnp.mean(r_raw),
            "misc_clipped_reward_mean": jnp.mean(r_clipped),
            "misc_v_joint_mean": 0.5 * (jnp.mean(v_joint_0) + jnp.mean(v_joint_1)),
            "misc_v_marginal_mean": 0.5 * (jnp.mean(v_marg_0) + jnp.mean(v_marg_1)),
        }
        return r_clipped, state, metrics

    def _mine_loss(params, s_g_seq, s_c_seq, s_c_shuffled_seq):
        """Negative MINE lower bound (Eq. 7 LHS), pooled across the batch.

        ``s_g_seq``, ``s_c_seq``, ``s_c_shuffled_seq`` have a common leading
        shape ``(...)`` of any rank; we flatten to ``(N, dim)`` before the
        forward pass so the lower-bound expectation is computed over all
        timesteps from all trajectories jointly.
        """
        joint_t = _t_phi(params, s_g_seq, s_c_seq)
        marg_t = _t_phi(params, s_g_seq, s_c_shuffled_seq)
        joint_flat = joint_t.reshape(-1)
        marg_flat = marg_t.reshape(-1)
        e_joint = jnp.mean(joint_flat)
        # log(mean(exp(T))) via log-sum-exp for numerical stability.
        n = jnp.asarray(marg_flat.shape[0], dtype=marg_flat.dtype)
        e_marginal = jax.scipy.special.logsumexp(marg_flat) - jnp.log(n)
        # Negate because we maximize the lower bound but optax does descent.
        return -(e_joint - e_marginal)

    def misc_train(state: MISCBonusState, online_transitions, num_grad_steps: int = 1):
        """Train T_phi on online trajectories via the MINE objective (Eq. 7).

        ``online_transitions.observation`` has shape ``(num_envs, T, obs_size)``
        — i.e. one full episode per env. For each env we permute its s_c
        sequence along the time axis (the trajectory-internal marginal
        sampler from §3.2 / Lemma 1), then pool all envs and timesteps for
        the lower-bound expectation.
        """
        states = online_transitions.observation[..., :state_size]
        if states.ndim < 2:
            raise ValueError(
                "MISC train_fn expected trajectory-shaped transitions "
                f"(.., T, obs); got observation shape {states.shape}."
            )
        s_g_seq = states[..., goal_idx_arr]   # (..., T, dim_g)
        s_c_seq = states[..., ctrl_idx_arr]   # (..., T, dim_c)

        # Per-trajectory random temporal permutation of s_c. ``states`` may
        # have any number of leading batch dims; we flatten them, permute
        # each row's time axis independently, then unflatten.
        leading_shape = s_c_seq.shape[:-2]
        T = s_c_seq.shape[-2]
        flat_s_c = s_c_seq.reshape((-1, T, dim_c))
        num_traj = flat_s_c.shape[0]

        def step(carry, _):
            params, opt_state, k = carry
            k, perm_key = jax.random.split(k)
            perm_keys = jax.random.split(perm_key, num_traj)
            permutations = jax.vmap(lambda kk: jax.random.permutation(kk, T))(perm_keys)
            shuffled_flat = jnp.take_along_axis(
                flat_s_c, permutations[:, :, None], axis=1
            )
            s_c_shuffled = shuffled_flat.reshape(s_c_seq.shape)

            loss_value, grads = jax.value_and_grad(_mine_loss)(
                params, s_g_seq, s_c_seq, s_c_shuffled
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state, k), loss_value

        (new_params, new_opt_state, new_key), losses = jax.lax.scan(
            step,
            (state.predictor_params, state.opt_state, state.key),
            (),
            length=num_grad_steps,
        )
        new_state = state.replace(
            predictor_params=new_params,
            opt_state=new_opt_state,
            key=new_key,
        )
        return new_state, {
            "misc_mine_loss": jnp.mean(losses),
            "misc_mine_lower_bound": -jnp.mean(losses),
        }

    return misc_bonus, initial_state, None, misc_train


def _create_ucb_bonus(
    *,
    state_size: int,
    action_size: int,
    key: jax.Array,
    ucb_coeff: float,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    learning_rate: float = 3e-4,
    # Internal-RND knobs (separate from any standalone "rnd" bonus). Defaults
    # mirror _create_rnd_bonus so the novelty source behaves identically.
    rnd_feature_dim: int = 64,
    rnd_hidden_dim: int = 256,
    rnd_num_hidden: int = 2,
    rnd_learning_rate: float = 1e-4,
    rnd_obs_clip: float = 5.0,
    rnd_use_goal: bool = False,
    goal_indices: Optional[Sequence[int]] = None,
) -> Tuple[BonusFn, UCBBonusState, InitFromStatesFn, UCBTrainFn, UCBRelabelFn]:
    """EXPLORE: optimistic UCB reward labeling for offline data.

    Owns its own RND target/predictor for the novelty term so listing ``"ucb"``
    alone gives full EXPLORE — no need to co-list a standalone ``"rnd"`` bonus
    (which would also additively shape *online* rewards, defeating the
    EXPLORE design where online uses true env rewards untouched).

    The reward predictor r_θ(s,a), termination predictor T̂_θ(s,a), and
    internal RND networks live in :class:`UCBBonusState`. UCB doesn't shape
    rewards additively, so its ``bonus_fn`` returns zeros — the actual work
    happens via separate ``relabel_fn`` and ``train_fn`` invoked by the agent
    through :class:`ExplorationBonuses`.
    """
    from .explore import create_explore_ucb_models

    explore_key, rnd_key = jax.random.split(key)

    explore_initial_state, explore_train_fn, explore_relabel_fn = create_explore_ucb_models(
        state_size=state_size,
        action_size=action_size,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        learning_rate=learning_rate,
        key=explore_key,
    )

    # Internal RND novelty source. We use the bonus_fn's read-only side
    # (novelty_fn) for relabeling and the train_fn for keeping the predictor
    # focused on online transitions only.
    _rnd_bonus_fn, rnd_initial_state, rnd_init_from_states_fn, rnd_novelty_fn, rnd_train_fn = (
        _create_rnd_bonus(
            state_size=state_size,
            key=rnd_key,
            feature_dim=rnd_feature_dim,
            hidden_dim=rnd_hidden_dim,
            num_hidden=rnd_num_hidden,
            learning_rate=rnd_learning_rate,
            obs_clip=rnd_obs_clip,
            use_goal=rnd_use_goal,
            goal_indices=goal_indices,
        )
    )

    initial_state = UCBBonusState(
        explore=explore_initial_state,
        rnd=rnd_initial_state,
    )

    def ucb_bonus(state, transitions, bonus_key):
        """No-op additive bonus; UCB acts via separate relabel / train methods."""
        del bonus_key
        zeros = jnp.zeros_like(transitions.reward)
        return zeros, state, {}

    def ucb_init_from_states(state: UCBBonusState, states: jnp.ndarray) -> UCBBonusState:
        """Seed UCB's internal RND obs-norm stats from a batch of observed states."""
        return state.replace(rnd=rnd_init_from_states_fn(state.rnd, states))

    coeff = float(ucb_coeff)

    def ucb_relabel(state: UCBBonusState, transitions):
        """Relabel ``transitions`` (intended: offline) with UCBR + predicted-discount.

        Reads the novelty term from UCB's internal RND in a read-only way —
        UCB's RND predictor is updated separately on online transitions via
        :meth:`ExplorationBonuses.train_ucb`, so evaluating it on offline (s, a)
        does not contaminate the predictor.
        """
        novelty = rnd_novelty_fn(state.rnd, transitions)
        relabeled = explore_relabel_fn(state.explore, transitions, novelty, coeff)
        return relabeled

    def ucb_online_bonus(state: UCBBonusState, transitions):
        """Per-transition ``ucb_coeff * RND_novelty`` for online transitions.

        Read-only over the RND state. Lets the agent add UCB's RND novelty to
        online rewards (without overwriting them with the reward predictor
        ``r_θ``, which is the offline-relabel path).
        """
        novelty = rnd_novelty_fn(state.rnd, transitions)
        return jnp.asarray(coeff, dtype=novelty.dtype) * novelty

    def ucb_train(state: UCBBonusState, online_transitions, num_grad_steps: int = 1):
        """Train all UCB-internal trainables on an online batch.

        Updates: reward predictor (MSE on r), termination predictor (BCE on
        T from discount), and the internal RND predictor (MSE on frozen
        target features).
        """
        new_explore, m_explore = explore_train_fn(state.explore, online_transitions, num_grad_steps)
        new_rnd, m_rnd = rnd_train_fn(state.rnd, online_transitions, num_grad_steps)
        new_state = state.replace(explore=new_explore, rnd=new_rnd)
        metrics = {**m_explore, **{f"ucb_internal_{k}": v for k, v in m_rnd.items()}}
        return new_state, metrics

    return ucb_bonus, initial_state, ucb_init_from_states, ucb_train, ucb_relabel, ucb_online_bonus


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
        novelty_fns: Tuple[NoveltyFn, ...] = (),
        train_fns: Tuple[TrainFn, ...] = (),
        ucb_train_fns: Tuple[UCBTrainFn, ...] = (),
        ucb_relabel_fns: Tuple[UCBRelabelFn, ...] = (),
        ucb_online_bonus_fns: Tuple[UCBOnlineBonusFn, ...] = (),
    ):
        self.bonus_types = bonus_types
        self.weights = weights
        self.fns = fns
        self.initial_state = initial_state
        self._init_from_states_fns = init_from_states_fns
        self._novelty_fns = novelty_fns or tuple(None for _ in fns)
        self._train_fns = train_fns or tuple(None for _ in fns)
        self._ucb_train_fns = ucb_train_fns or tuple(None for _ in fns)
        self._ucb_relabel_fns = ucb_relabel_fns or tuple(None for _ in fns)
        self._ucb_online_bonus_fns = ucb_online_bonus_fns or tuple(None for _ in fns)
        self.is_empty = len(fns) == 0
        self.has_trainables = any(fn is not None for fn in self._train_fns)
        # Cache the index of the first "ucb" bonus for fast access.
        self._ucb_index: Optional[int] = next(
            (i for i, bt in enumerate(bonus_types) if bt == "ucb"), None
        )

    # Bonus types whose additive contribution is restricted to ONLINE rows.
    # RND alone qualifies: it's a novelty signal trained on online states, so
    # firing it on offline rows would just reward "different from online" not
    # "novel to the agent". (Offline) empowerment is an offline scorer that
    # gives a meaningful reading on offline rows too. UCB's bonus_fn returns
    # zeros (it shapes rewards via separate offline-relabel and online-novelty
    # paths), so masking it would be a no-op either way. MISC's T_phi is
    # trained only on the agent's online trajectories, so labeling offline
    # transitions with the per-(s,s') MI surrogate would compare them against
    # an online-only estimator — leave offline rewards alone. Online
    # empowerment trains its skill-conditioned Q/V/policy from scratch on
    # online data, so its empowerment estimate on offline-only states is
    # untrustworthy and out-of-distribution — same argument as MISC.
    _ONLINE_ONLY_BONUS_TYPES = frozenset({"rnd", "misc", "online_empowerment"})

    def compute(
        self,
        state: Tuple,
        transitions,
        key: jax.Array,
        is_online: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple, dict]:
        """Return ``(weighted_total, raw_total, new_state, metrics)``.

        ``weighted_total`` = ``sum_i weight_i * bonus_i`` is what gets added to
        the reward (matching existing SAC reward shaping). ``raw_total`` =
        ``sum_i bonus_i`` is the un-scaled sum of the raw bonuses; callers
        that want to train a separate Q_exp *and* apply the weight at the
        actor loss (rather than at the reward) should use this instead.

        When ``is_online`` is provided (per-row 0/1 mask shaped like
        ``transitions.reward``), bonuses whose type is in
        ``_ONLINE_ONLY_BONUS_TYPES`` (currently just ``"rnd"``) are multiplied
        by it so offline rows get no contribution from those bonuses.
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
            if is_online is not None and bt in self._ONLINE_ONLY_BONUS_TYPES:
                bonus = bonus * is_online
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

    @property
    def has_rnd_novelty(self) -> bool:
        """True iff at least one configured bonus exposes a read-only RND novelty fn."""
        return any(fn is not None for fn in self._novelty_fns)

    def train(self, state: Tuple, online_transitions, num_grad_steps: int = 1):
        """Train every per-bonus trainable on a batch of online transitions.

        Bonuses without a train hook (e.g. empowerment, ucb — which trains
        through ``train_ucb`` instead) pass through unchanged. Returns
        ``(new_state, metrics)`` with each trainable slot updated and a
        ``bonus_{i}_{type}_*`` prefix on metrics so they don't collide.
        """
        if self.is_empty or not self.has_trainables:
            return state, {}
        new_states = list(state)
        all_metrics: dict = {}
        for i, (train_fn, bt) in enumerate(zip(self._train_fns, self.bonus_types)):
            if train_fn is None:
                continue
            new_s, m = train_fn(state[i], online_transitions, num_grad_steps)
            new_states[i] = new_s
            for mk, mv in m.items():
                all_metrics[f"bonus_{i}_{bt}_{mk}"] = mv
        return tuple(new_states), all_metrics

    def is_trainable(self, idx: int) -> bool:
        """True iff the bonus at ``idx`` exposes a per-bonus train hook."""
        return self._train_fns[idx] is not None

    def train_one(self, state: Tuple, idx: int, online_transitions, num_grad_steps: int = 1):
        """Train just the bonus at ``idx`` on an online batch.

        Lets the agent run different grad-step counts per bonus (with the
        outer resampling loop, when needed, lifted into the agent so each
        bonus can scan with its own length). Bonuses with no train hook are
        a no-op pass-through.
        """
        train_fn = self._train_fns[idx]
        if train_fn is None:
            return state, {}
        bt = self.bonus_types[idx]
        new_s, m = train_fn(state[idx], online_transitions, num_grad_steps)
        new_states = list(state)
        new_states[idx] = new_s
        prefixed = {f"bonus_{idx}_{bt}_{mk}": mv for mk, mv in m.items()}
        return tuple(new_states), prefixed

    def normalize_grad_steps(
        self, grad_steps: Union[int, Sequence[int]],
    ) -> Tuple[int, ...]:
        """Resolve a per-bonus grad-step spec to a per-bonus tuple.

        Accepts an int (broadcast to all bonuses), a length-1 sequence
        (also broadcast), or a sequence whose length matches the number of
        configured bonuses. Returns ``()`` for an empty bundle.
        """
        n = len(self.bonus_types)
        if n == 0:
            return ()
        if isinstance(grad_steps, int):
            return (int(grad_steps),) * n
        gs = tuple(int(x) for x in grad_steps)
        if len(gs) == 1:
            return gs * n
        if len(gs) != n:
            raise ValueError(
                "bonus_grad_steps_per_env_step length must be 1 or match "
                f"exploration_bonus_type ({n}); got {len(gs)} ({gs!r})."
            )
        return gs

    @property
    def has_ucb(self) -> bool:
        """True iff a ``"ucb"`` bonus is configured (EXPLORE algorithm enabled)."""
        return self._ucb_index is not None

    def compute_ucb_online_novelty_bonus(self, state: Tuple, transitions) -> jnp.ndarray:
        """Per-transition ``ucb_coeff * RND_novelty`` from UCB's internal RND.

        Returns zeros (shaped like ``transitions.reward``) when no UCB bonus is
        configured. Read-only over UCB's internal RND state. Intended for
        adding UCB's RND novelty term to ONLINE rewards (the offline path uses
        :meth:`relabel_offline_with_ucb` which overwrites the reward with
        ``r_θ + ucb_coeff * novelty`` instead).
        """
        if self._ucb_index is None:
            return jnp.zeros_like(transitions.reward)
        i = self._ucb_index
        return self._ucb_online_bonus_fns[i](state[i], transitions)

    def relabel_offline_with_ucb(self, state: Tuple, offline_transitions):
        """Relabel offline transitions in place: reward → UCBR, discount → 1 - sigmoid(T̂).

        UCB owns its internal RND novelty source — no external novelty needs
        to be supplied. Read-only over the UCB bonus state; predictor updates
        happen separately via :meth:`train_ucb`.
        """
        if self._ucb_index is None:
            raise RuntimeError(
                "relabel_offline_with_ucb called but no 'ucb' bonus configured."
            )
        i = self._ucb_index
        return self._ucb_relabel_fns[i](state[i], offline_transitions)

    def train_ucb(self, state: Tuple, online_transitions, num_grad_steps: int = 1):
        """Train reward + termination predictors on online transitions.

        Returns ``(new_state, metrics)`` with the UCB slot updated; all other
        per-bonus states pass through unchanged.
        """
        if self._ucb_index is None:
            raise RuntimeError("train_ucb called but no 'ucb' bonus configured.")
        i = self._ucb_index
        new_ucb_state, metrics = self._ucb_train_fns[i](
            state[i], online_transitions, num_grad_steps,
        )
        new_state = list(state)
        new_state[i] = new_ucb_state
        return tuple(new_state), metrics

    def compute_first_rnd_novelty(self, state: Tuple, transitions) -> jnp.ndarray:
        """Per-transition ``(1/L)||f_phi - f_bar||²`` from the first RND bonus.

        Read-only: predictor params and observation-normalization stats in
        ``state`` are NOT modified — used by EXPLORE to label offline data
        without contaminating RND with offline samples. Returns zeros (shaped
        like ``transitions.reward``) when no RND bonus is configured.
        """
        for i, fn in enumerate(self._novelty_fns):
            if fn is not None:
                return fn(state[i], transitions)
        return jnp.zeros_like(transitions.reward)


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
    # UCB-specific
    ucb_action_size: Optional[int] = None,
    ucb_coeff: float = 1.0,
    ucb_hidden_dim: int = 256,
    ucb_num_hidden: int = 2,
    ucb_learning_rate: float = 3e-4,
    # MISC-specific
    controllable_indices: Optional[Sequence[int]] = None,
    misc_hidden_dim: int = 256,
    misc_num_hidden: int = 3,
    misc_learning_rate: float = 1e-3,
    misc_alpha: float = 5000.0,
    # Online-empowerment-specific
    online_empowerment_action_size: Optional[int] = None,
    online_empowerment_lr: float = 3e-4,
    online_empowerment_value_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_empowerment_actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_empowerment_value_latent_dim: int = 128,
    online_empowerment_num_skills: int = 5,
    online_empowerment_num_splus_samples: int = 32,
    online_empowerment_discount: float = 0.99,
    online_empowerment_tau: float = 0.005,
    online_empowerment_separate_qv: bool = False,
    online_empowerment_use_self_q_loss: bool = True,
    online_empowerment_layer_norm: bool = True,
    online_empowerment_bc_alpha: float = 0.01,
    online_empowerment_bonus_mean: float = 0.0,
    online_empowerment_bonus_scale: float = 1.0,
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

    fns, states, init_fns, novelty_fns = [], [], [], []
    train_fns: list = []
    ucb_train_fns, ucb_relabel_fns, ucb_online_bonus_fns = [], [], []
    for bt in types_tuple:
        key, sub_key = jax.random.split(key)
        fn, st, init_fn, novelty_fn, train_fn, ucb_train_fn, ucb_relabel_fn, ucb_online_bonus_fn = create_exploration_bonus(
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
            ucb_action_size=ucb_action_size,
            ucb_coeff=ucb_coeff,
            ucb_hidden_dim=ucb_hidden_dim,
            ucb_num_hidden=ucb_num_hidden,
            ucb_learning_rate=ucb_learning_rate,
            controllable_indices=controllable_indices,
            misc_hidden_dim=misc_hidden_dim,
            misc_num_hidden=misc_num_hidden,
            misc_learning_rate=misc_learning_rate,
            misc_alpha=misc_alpha,
            online_empowerment_action_size=online_empowerment_action_size,
            online_empowerment_lr=online_empowerment_lr,
            online_empowerment_value_hidden_dims=online_empowerment_value_hidden_dims,
            online_empowerment_actor_hidden_dims=online_empowerment_actor_hidden_dims,
            online_empowerment_value_latent_dim=online_empowerment_value_latent_dim,
            online_empowerment_num_skills=online_empowerment_num_skills,
            online_empowerment_num_splus_samples=online_empowerment_num_splus_samples,
            online_empowerment_discount=online_empowerment_discount,
            online_empowerment_tau=online_empowerment_tau,
            online_empowerment_separate_qv=online_empowerment_separate_qv,
            online_empowerment_use_self_q_loss=online_empowerment_use_self_q_loss,
            online_empowerment_layer_norm=online_empowerment_layer_norm,
            online_empowerment_bc_alpha=online_empowerment_bc_alpha,
            online_empowerment_bonus_mean=online_empowerment_bonus_mean,
            online_empowerment_bonus_scale=online_empowerment_bonus_scale,
        )
        fns.append(fn)
        states.append(st)
        init_fns.append(init_fn)
        novelty_fns.append(novelty_fn)
        train_fns.append(train_fn)
        ucb_train_fns.append(ucb_train_fn)
        ucb_relabel_fns.append(ucb_relabel_fn)
        ucb_online_bonus_fns.append(ucb_online_bonus_fn)

    return ExplorationBonuses(
        bonus_types=types_tuple,
        weights=weights_tuple,
        fns=tuple(fns),
        initial_state=tuple(states),
        init_from_states_fns=tuple(init_fns),
        novelty_fns=tuple(novelty_fns),
        train_fns=tuple(train_fns),
        ucb_train_fns=tuple(ucb_train_fns),
        ucb_relabel_fns=tuple(ucb_relabel_fns),
        ucb_online_bonus_fns=tuple(ucb_online_bonus_fns),
    )
