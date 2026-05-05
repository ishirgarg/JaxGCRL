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
class EMEBonusState:
    """Mutable state for the EME (Effective Metric-based Exploration) intrinsic reward.

    Carries the trainable EME metric ``d_E^phi(s_i, s_j)`` (with its Adam
    state) and an ensemble of ``K`` reward predictors ``g(s, a; eta_k)`` whose
    output variance acts as the diversity-enhanced scaling factor (Eq. 8 of
    the paper). Per-transition reward is read-only over both, mirroring the
    other trainable bonuses; the joint update is driven by ``train_fn``.

    The ensemble is stored as a single pytree with a leading ``K`` axis; we
    train it with ``jax.vmap`` so all members share one optimizer call but
    their parameters are kept distinct (paper §4.2: each model is initialized
    differently and trained on a bootstrap-sampled subset to maintain
    diversity in their predictions on unvisited states).
    """
    metric_params: Any
    metric_opt_state: Any
    ensemble_params: Any
    ensemble_opt_state: Any
    key: jax.Array


@fdataclass
class ICMBonusState:
    """Mutable state for the ICM (Intrinsic Curiosity Module) intrinsic reward.

    Holds the joint trainable parameters of the feature encoder ``phi``, the
    inverse model ``g(phi(s_t), phi(s_{t+1})) -> a_t``, and the forward model
    ``f(phi(s_t), a_t) -> phi_hat(s_{t+1})``, together with their shared Adam
    optimizer state. The per-transition reward is the read-only forward-model
    prediction error in the *current* feature space; all three networks are
    trained jointly via gradient descent on ``(1-beta) * L_I + beta * L_F``.
    """
    predictor_params: Any
    opt_state: Any


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
    # Shared knob used by ICM and EME (the agent's RL discount). The action
    # dimension is sourced directly from ``env.action_size`` inside each
    # branch — it's a property of the env, not a config parameter.
    discount: float = 0.99,
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
    # ICM-specific (Pathak et al., 2017). Defaults follow the paper:
    # learning rate 1e-3, beta=0.2, feature dim 288, hidden 256. The action
    # dimension is sourced from the top-level ``action_size`` argument.
    icm_feature_dim: int = 288,
    icm_encoder_hidden_dim: int = 256,
    icm_encoder_num_hidden: int = 2,
    icm_inverse_hidden_dim: int = 256,
    icm_inverse_num_hidden: int = 1,
    icm_forward_hidden_dim: int = 256,
    icm_forward_num_hidden: int = 1,
    icm_learning_rate: float = 1e-3,
    icm_beta: float = 0.2,
    icm_eta: float = 1.0,
    # EME-specific (Wang et al., 2024). Defaults follow the paper:
    # ensemble size 6, max reward scaling M=10. The discount is sourced
    # from the top-level ``discount`` argument (= the agent's discount).
    eme_actor_dist_fn: Optional[
        Callable[[Any, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    ] = None,
    eme_metric_hidden_dim: int = 256,
    eme_metric_num_hidden: int = 2,
    eme_reward_hidden_dim: int = 256,
    eme_reward_num_hidden: int = 2,
    eme_metric_learning_rate: float = 1e-3,
    eme_reward_learning_rate: float = 1e-3,
    eme_ensemble_size: int = 6,
    eme_max_reward_scaling: float = 10.0,
    eme_bootstrap_keep_prob: float = 0.8,
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
    # Online-MINE-empowerment-specific
    online_mine_empowerment_action_size: Optional[int] = None,
    online_mine_empowerment_actor_sample_fn: Optional[
        Callable[[Any, jnp.ndarray, jax.Array], jnp.ndarray]
    ] = None,
    online_mine_empowerment_lr_dyn: float = 1e-3,
    online_mine_empowerment_lr_t: float = 3e-4,
    online_mine_empowerment_dyn_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_mine_empowerment_t_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_mine_empowerment_layer_norm: bool = True,
    online_mine_empowerment_bonus_mean: float = 0.0,
    online_mine_empowerment_bonus_scale: float = 1.0,
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
    if bonus_type == "icm":
        bonus_fn, st, init_fn, train_fn = _create_icm_bonus(
            state_size=state_size,
            action_size=int(env.action_size),
            key=key,
            feature_dim=icm_feature_dim,
            encoder_hidden_dim=icm_encoder_hidden_dim,
            encoder_num_hidden=icm_encoder_num_hidden,
            inverse_hidden_dim=icm_inverse_hidden_dim,
            inverse_num_hidden=icm_inverse_num_hidden,
            forward_hidden_dim=icm_forward_hidden_dim,
            forward_num_hidden=icm_forward_num_hidden,
            learning_rate=icm_learning_rate,
            beta=icm_beta,
            eta=icm_eta,
        )
        return bonus_fn, st, init_fn, None, train_fn, None, None, None
    if bonus_type == "eme":
        if eme_actor_dist_fn is None:
            raise ValueError(
                "'eme' bonus requires eme_actor_dist_fn (callable "
                "(actor_params, obs) -> (mean, log_std) for the agent's actor) — "
                "needed for the KL term in the EME metric loss (Eq. 9)."
            )
        bonus_fn, st, init_fn, train_fn = _create_eme_bonus(
            state_size=state_size,
            action_size=int(env.action_size),
            actor_dist_fn=eme_actor_dist_fn,
            key=key,
            metric_hidden_dim=eme_metric_hidden_dim,
            metric_num_hidden=eme_metric_num_hidden,
            reward_hidden_dim=eme_reward_hidden_dim,
            reward_num_hidden=eme_reward_num_hidden,
            metric_learning_rate=eme_metric_learning_rate,
            reward_learning_rate=eme_reward_learning_rate,
            ensemble_size=eme_ensemble_size,
            max_reward_scaling=eme_max_reward_scaling,
            gamma=discount,
            bootstrap_keep_prob=eme_bootstrap_keep_prob,
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
    if bonus_type == "online_mine_empowerment":
        if online_mine_empowerment_action_size is None:
            raise ValueError(
                "'online_mine_empowerment' bonus requires online_mine_empowerment_action_size."
            )
        if online_mine_empowerment_actor_sample_fn is None:
            raise ValueError(
                "'online_mine_empowerment' bonus requires online_mine_empowerment_actor_sample_fn "
                "(callable (actor_params, obs, key) -> actions sourced from the agent's main actor)."
            )
        from .online_mine_empowerment import _create_online_mine_empowerment_bonus
        bonus_fn, st, init_fn, train_fn = _create_online_mine_empowerment_bonus(
            state_size=state_size,
            action_size=online_mine_empowerment_action_size,
            actor_sample_fn=online_mine_empowerment_actor_sample_fn,
            key=key,
            lr_dyn=online_mine_empowerment_lr_dyn,
            lr_t=online_mine_empowerment_lr_t,
            dyn_hidden_dims=online_mine_empowerment_dyn_hidden_dims,
            t_hidden_dims=online_mine_empowerment_t_hidden_dims,
            layer_norm=online_mine_empowerment_layer_norm,
            bonus_mean=online_mine_empowerment_bonus_mean,
            bonus_scale=online_mine_empowerment_bonus_scale,
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


def _create_icm_bonus(
    *,
    state_size: int,
    action_size: int,
    key: jax.Array,
    feature_dim: int = 288,
    encoder_hidden_dim: int = 256,
    encoder_num_hidden: int = 2,
    inverse_hidden_dim: int = 256,
    inverse_num_hidden: int = 1,
    forward_hidden_dim: int = 256,
    forward_num_hidden: int = 1,
    learning_rate: float = 1e-3,
    beta: float = 0.2,
    eta: float = 1.0,
) -> Tuple[BonusFn, ICMBonusState, InitFromStatesFn, TrainFn]:
    """ICM: Intrinsic Curiosity Module (Pathak et al., 2017).

    Three jointly trained networks share one Adam optimizer:
      * ``phi(s)``: feature encoder, ``state_size -> feature_dim``.
      * ``g(phi(s_t), phi(s_{t+1}))``: inverse dynamics, predicts ``a_t``.
      * ``f(phi(s_t), a_t)``: forward dynamics, predicts ``phi(s_{t+1})``.

    The intrinsic reward is the forward-model error in the *current* feature
    space, ``(eta/2) * ||phi_hat(s_{t+1}) - phi(s_{t+1})||^2`` (Eq. 6); the
    estimator is updated by descending ``(1-beta) * L_I + beta * L_F`` on the
    same online minibatch (Eq. 7 with the policy term excluded — that lives in
    the agent's RL loss). For continuous actions we use MSE for ``L_I``; the
    paper uses cross-entropy for discrete action spaces.
    """
    from .networks import Encoder

    encoder = Encoder(
        repr_dim=feature_dim,
        network_width=encoder_hidden_dim,
        network_depth=encoder_num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )
    inverse_net = Encoder(
        repr_dim=action_size,
        network_width=inverse_hidden_dim,
        network_depth=inverse_num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )
    forward_net = Encoder(
        repr_dim=feature_dim,
        network_width=forward_hidden_dim,
        network_depth=forward_num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )

    enc_key, inv_key, fwd_key = jax.random.split(key, 3)
    dummy_state = jnp.zeros((1, state_size), dtype=jnp.float32)
    dummy_phi_pair = jnp.zeros((1, 2 * feature_dim), dtype=jnp.float32)
    dummy_phi_act = jnp.zeros((1, feature_dim + action_size), dtype=jnp.float32)
    encoder_params = encoder.init(enc_key, dummy_state)
    inverse_params = inverse_net.init(inv_key, dummy_phi_pair)
    forward_params = forward_net.init(fwd_key, dummy_phi_act)

    predictor_params0 = {
        "encoder": encoder_params,
        "inverse": inverse_params,
        "forward": forward_params,
    }
    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(predictor_params0)

    initial_state = ICMBonusState(
        predictor_params=predictor_params0,
        opt_state=opt_state0,
    )

    beta_f = jnp.asarray(beta, dtype=jnp.float32)
    eta_f = jnp.asarray(eta, dtype=jnp.float32)

    def _encode(params, states):
        return encoder.apply(params["encoder"], states)

    def _predict_action(params, phi_t, phi_tp1):
        x = jnp.concatenate([phi_t, phi_tp1], axis=-1)
        return inverse_net.apply(params["inverse"], x)

    def _predict_phi_next(params, phi_t, actions):
        x = jnp.concatenate([phi_t, actions], axis=-1)
        return forward_net.apply(params["forward"], x)

    def icm_bonus(state: ICMBonusState, transitions, bonus_key):
        """Per-transition ICM intrinsic reward (Eq. 6).

        Read-only over all three networks; the joint estimator is updated only
        by ``icm_train`` on online transitions.
        """
        del bonus_key
        states = transitions.observation[..., :state_size]
        next_states = transitions.next_observation[..., :state_size]
        actions = transitions.action

        params = jax.lax.stop_gradient(state.predictor_params)
        phi_t = _encode(params, states)
        phi_tp1 = _encode(params, next_states)
        phi_hat_tp1 = _predict_phi_next(params, phi_t, actions)
        sq_err = jnp.sum((phi_hat_tp1 - phi_tp1) ** 2, axis=-1)
        bonus = 0.5 * eta_f * sq_err

        metrics = {
            "icm_forward_sq_err_mean": jnp.mean(sq_err),
            "icm_bonus_mean": jnp.mean(bonus),
        }
        return bonus, state, metrics

    def _icm_loss(params, states, next_states, actions):
        """Joint inverse + forward loss, ``(1-beta) * L_I + beta * L_F``.

        ``L_F`` uses ``stop_gradient(phi(s_{t+1}))`` as the target so the
        encoder is shaped only by the inverse-model objective (the standard
        ICM training trick: otherwise the encoder can collapse phi to a
        constant to drive the forward loss to zero).
        """
        phi_t = _encode(params, states)
        phi_tp1 = _encode(params, next_states)
        phi_hat_tp1 = _predict_phi_next(params, phi_t, actions)
        a_hat = _predict_action(params, phi_t, phi_tp1)

        l_inverse = jnp.mean(jnp.sum((a_hat - actions) ** 2, axis=-1))
        l_forward = 0.5 * jnp.mean(
            jnp.sum((phi_hat_tp1 - jax.lax.stop_gradient(phi_tp1)) ** 2, axis=-1)
        )
        loss = (1.0 - beta_f) * l_inverse + beta_f * l_forward
        return loss, (l_inverse, l_forward)

    def icm_train(state: ICMBonusState, online_transitions, num_grad_steps: int = 1):
        """Joint Adam update for encoder + inverse + forward on online data."""
        states = online_transitions.observation[..., :state_size]
        next_states = online_transitions.next_observation[..., :state_size]
        actions = online_transitions.action
        states = states.reshape(-1, state_size)
        next_states = next_states.reshape(-1, state_size)
        actions = actions.reshape(-1, action_size)

        def step(carry, _):
            params, opt_state = carry
            (loss_value, (l_inv, l_fwd)), grads = jax.value_and_grad(
                _icm_loss, has_aux=True
            )(params, states, next_states, actions)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state), (loss_value, l_inv, l_fwd)

        (new_params, new_opt_state), (losses, l_inv, l_fwd) = jax.lax.scan(
            step,
            (state.predictor_params, state.opt_state),
            (),
            length=num_grad_steps,
        )
        new_state = state.replace(
            predictor_params=new_params,
            opt_state=new_opt_state,
        )
        return new_state, {
            "icm_loss": jnp.mean(losses),
            "icm_inverse_loss": jnp.mean(l_inv),
            "icm_forward_loss": jnp.mean(l_fwd),
        }

    return icm_bonus, initial_state, None, icm_train


def _create_eme_bonus(
    *,
    state_size: int,
    action_size: int,
    actor_dist_fn: Callable[[Any, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    key: jax.Array,
    metric_hidden_dim: int = 256,
    metric_num_hidden: int = 2,
    reward_hidden_dim: int = 256,
    reward_num_hidden: int = 2,
    metric_learning_rate: float = 1e-3,
    reward_learning_rate: float = 1e-3,
    ensemble_size: int = 6,
    max_reward_scaling: float = 10.0,
    gamma: float = 0.99,
    bootstrap_keep_prob: float = 0.8,
) -> Tuple[BonusFn, EMEBonusState, InitFromStatesFn, TrainFn]:
    """EME: Effective Metric-based Exploration bonus (Wang et al., 2024).

    Two trainable subsystems:
      * EME metric ``d_E^phi(s_i, s_j) -> R``: a small MLP over the
        concatenated state pair. Trained on Eq. 9 of the paper, the tractable
        regression target for the unique fixed-point of the EME distance
        function (Theorem 1). The target uses a sample-based reward gap with
        ensemble-variance correction (Eq. 7), a bootstrapped next-pair metric
        with ``stop_gradient``, and a closed-form Gaussian KL between the
        actor's pre-tanh action distributions at the two states.
      * Reward ensemble ``{g(eta_1), ..., g(eta_K)}``: K MLPs each predicting
        ``r_{t+1}`` from ``(s_t, a_t)``. Trained jointly via ``jax.vmap`` with
        member-specific Bernoulli bootstrap masks (sample-with-replacement
        approximation), so each member's predictions diverge most on
        rarely-visited (s, a) — high cross-member variance there is what the
        scaling factor ``zeta`` keys on (paper §4.2).

    Per-transition reward (Eq. 10):
      ``b = d_E(s_t, s_{t+1}) * clip(max(zeta(s_t, a_t), 1), 1, M)``

    where ``zeta(s_t, a_t) = mean_d Var_eta[g(s_t, a_t; eta)_d]`` (squared L2
    radius of the ensemble's outputs around their mean). Both ``d_E`` and the
    ensemble are read-only inside ``bonus_fn``; learning happens in
    ``train_fn`` and consumes the agent's actor params for the KL term.
    """
    from .networks import Encoder

    metric_net = Encoder(
        repr_dim=1,
        network_width=metric_hidden_dim,
        network_depth=metric_num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )
    reward_net = Encoder(
        repr_dim=1,
        network_width=reward_hidden_dim,
        network_depth=reward_num_hidden,
        skip_connections=0,
        use_relu=True,
        use_ln=False,
    )

    metric_key, ensemble_key, train_key0 = jax.random.split(key, 3)
    dummy_pair = jnp.zeros((1, 2 * state_size), dtype=jnp.float32)
    dummy_sa = jnp.zeros((1, state_size + action_size), dtype=jnp.float32)
    metric_params0 = metric_net.init(metric_key, dummy_pair)

    ensemble_keys = jax.random.split(ensemble_key, ensemble_size)
    ensemble_params0 = jax.vmap(lambda k: reward_net.init(k, dummy_sa))(ensemble_keys)

    metric_optimizer = optax.adam(metric_learning_rate)
    metric_opt_state0 = metric_optimizer.init(metric_params0)
    # One Adam over the whole stacked ensemble pytree. Each leaf has a leading
    # K axis, but optax treats this as ordinary pytree shape and applies
    # per-element scaling — no per-member optimizer state needed (and
    # vmapping ``init`` is wrong here because Adam's bias-correction count
    # would become per-member, then broadcast incorrectly across the K axis).
    reward_optimizer = optax.adam(reward_learning_rate)
    reward_opt_state0 = reward_optimizer.init(ensemble_params0)

    initial_state = EMEBonusState(
        metric_params=metric_params0,
        metric_opt_state=metric_opt_state0,
        ensemble_params=ensemble_params0,
        ensemble_opt_state=reward_opt_state0,
        key=train_key0,
    )

    M = jnp.asarray(max_reward_scaling, dtype=jnp.float32)
    gamma_f = jnp.asarray(gamma, dtype=jnp.float32)
    keep_p = jnp.asarray(bootstrap_keep_prob, dtype=jnp.float32)

    def _metric(metric_params, s_i, s_j):
        x = jnp.concatenate([s_i, s_j], axis=-1)
        return metric_net.apply(metric_params, x)[..., 0]

    def _reward_one(reward_params_one, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        return reward_net.apply(reward_params_one, x)[..., 0]

    def _ensemble_predictions(ensemble_params, s, a):
        """Stacked predictions across ensemble members; shape (K, ...)."""
        return jax.vmap(_reward_one, in_axes=(0, None, None))(
            ensemble_params, s, a
        )

    def _ensemble_variance(ensemble_params, s, a):
        """Cross-member variance ``Var_eta[g(s, a; eta)]``, scalar per (s, a)."""
        preds = _ensemble_predictions(ensemble_params, s, a)
        return jnp.var(preds, axis=0)

    def eme_bonus(state: EMEBonusState, transitions, bonus_key, actor_params=None):
        """Per-transition EME bonus (Eq. 10). ``actor_params`` is unused here."""
        del bonus_key, actor_params
        states = transitions.observation[..., :state_size]
        next_states = transitions.next_observation[..., :state_size]
        actions = transitions.action

        params_metric = jax.lax.stop_gradient(state.metric_params)
        params_ensemble = jax.lax.stop_gradient(state.ensemble_params)

        d = _metric(params_metric, states, next_states)
        zeta = _ensemble_variance(params_ensemble, states, actions)
        scale = jnp.clip(jnp.maximum(zeta, 1.0), 1.0, M)
        bonus = d * scale

        metrics = {
            "eme_metric_mean": jnp.mean(d),
            "eme_zeta_mean": jnp.mean(zeta),
            "eme_scale_mean": jnp.mean(scale),
            "eme_bonus_mean": jnp.mean(bonus),
        }
        return bonus, state, metrics

    def _gaussian_kl(mean_p, log_std_p, mean_q, log_std_q):
        """Closed-form KL between two diagonal Gaussians, summed over action dims.

        Operates on the actor's pre-tanh distribution: SAC's actor returns
        ``(mean, log_std)`` of a diagonal Gaussian *before* the tanh squash.
        The tanh adds a constant log-determinant term that cancels exactly
        between ``pi(.|s_i)`` and ``pi(.|s_j)`` under KL, so the closed-form
        Gaussian KL on the pre-tanh parameters is exact.
        """
        var_p = jnp.exp(2.0 * log_std_p)
        var_q = jnp.exp(2.0 * log_std_q)
        return jnp.sum(
            log_std_q - log_std_p
            + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q + 1e-8)
            - 0.5,
            axis=-1,
        )

    def _metric_loss(
        metric_params,
        ensemble_params_sg,
        actor_params,
        s_i, a_i, r_i, sp_i,
        s_j, a_j, r_j, sp_j,
    ):
        """Eq. 9 regression loss, evaluated on a permuted-pair batch.

        Reward gap correction: ``sqrt(max(|r_i - r_j|^2 - var_i - var_j, 0))``.
        Bootstrap target: ``stop_gradient(d_E(sp_i, sp_j))``.
        Policy term: closed-form pre-tanh Gaussian KL between ``pi(.|s_i)`` and
        ``pi(.|s_j)``.
        """
        d_pred = _metric(metric_params, s_i, s_j)

        var_i = _ensemble_variance(ensemble_params_sg, s_i, a_i)
        var_j = _ensemble_variance(ensemble_params_sg, s_j, a_j)
        reward_gap_sq = (r_i - r_j) ** 2 - var_i - var_j
        reward_term = jnp.sqrt(jnp.maximum(reward_gap_sq, 0.0))

        d_next = jax.lax.stop_gradient(_metric(metric_params, sp_i, sp_j))

        mean_i, log_std_i = actor_dist_fn(actor_params, s_i)
        mean_j, log_std_j = actor_dist_fn(actor_params, s_j)
        kl_term = _gaussian_kl(mean_i, log_std_i, mean_j, log_std_j)

        target = jax.lax.stop_gradient(
            reward_term + gamma_f * d_next + gamma_f * kl_term
        )
        loss = jnp.mean((d_pred - target) ** 2)
        return loss, {
            "eme_target_mean": jnp.mean(target),
            "eme_pred_mean": jnp.mean(d_pred),
            "eme_reward_term_mean": jnp.mean(reward_term),
            "eme_bootstrap_mean": jnp.mean(d_next),
            "eme_kl_term_mean": jnp.mean(kl_term),
        }

    def _ensemble_loss(ensemble_params, states, actions, rewards, masks):
        """Per-member MSE on ``r_{t+1}``, weighted by member-specific masks.

        ``masks`` has shape ``(K, N)``; element ``(k, i) ∈ {0, 1}`` decides
        whether transition ``i`` is in member ``k``'s bootstrap subset. We
        sum over weighted squared errors and divide by the per-member kept
        count, recovering an unbiased per-member MSE on its own subset.
        """
        def _per_member(params_k, mask_k):
            preds = _reward_one(params_k, states, actions)
            sq = (preds - rewards) ** 2
            kept = jnp.maximum(jnp.sum(mask_k), 1.0)
            return jnp.sum(sq * mask_k) / kept

        per_member = jax.vmap(_per_member)(ensemble_params, masks)
        return jnp.mean(per_member), per_member

    def eme_train(
        state: EMEBonusState,
        online_transitions,
        num_grad_steps: int = 1,
        actor_params: Any = None,
    ):
        """Joint update for ``d_E^phi`` (Eq. 9) and the reward ensemble (MSE).

        Per inner step:
          1. Resample a permutation of the batch to form pairs ``(s_i, s_j)``,
             plus per-member Bernoulli bootstrap masks (independent of the
             permutation, drawn fresh each step from ``state.key``).
          2. Update the reward ensemble first so the metric loss in this same
             step sees the updated (still ``stop_gradient``'d) ensemble.
          3. Update ``d_E^phi`` against the Eq. 9 target.
        """
        if actor_params is None:
            raise ValueError("eme_train requires actor_params for the KL term.")

        states = online_transitions.observation[..., :state_size].reshape(-1, state_size)
        next_states = online_transitions.next_observation[..., :state_size].reshape(-1, state_size)
        actions = online_transitions.action.reshape(-1, action_size)
        rewards = online_transitions.reward.reshape(-1)
        N = states.shape[0]

        def step(carry, _):
            metric_p, metric_opt, ensemble_p, ensemble_opt, k = carry
            k, perm_key, mask_key = jax.random.split(k, 3)

            # 1. Bootstrap masks (independent across ensemble members).
            mask_keys = jax.random.split(mask_key, ensemble_size)
            masks = jax.vmap(
                lambda kk: (jax.random.uniform(kk, (N,)) < keep_p).astype(jnp.float32)
            )(mask_keys)

            # 2. Reward-ensemble Adam step.
            (e_loss, per_member), e_grads = jax.value_and_grad(
                _ensemble_loss, has_aux=True
            )(ensemble_p, states, actions, rewards, masks)
            e_updates, new_ensemble_opt = reward_optimizer.update(
                e_grads, ensemble_opt, ensemble_p
            )
            new_ensemble_p = optax.apply_updates(ensemble_p, e_updates)

            # 3. Metric step on a fresh permutation. ``stop_gradient`` on the
            # ensemble and on the bootstrap d_E target keeps the metric loss
            # from leaking gradients into the ensemble or the next-pair
            # prediction.
            perm = jax.random.permutation(perm_key, N)
            s_i = states
            a_i = actions
            r_i = rewards
            sp_i = next_states
            s_j = states[perm]
            a_j = actions[perm]
            r_j = rewards[perm]
            sp_j = next_states[perm]
            ensemble_sg = jax.lax.stop_gradient(new_ensemble_p)

            (m_loss, m_aux), m_grads = jax.value_and_grad(_metric_loss, has_aux=True)(
                metric_p,
                ensemble_sg,
                actor_params,
                s_i, a_i, r_i, sp_i,
                s_j, a_j, r_j, sp_j,
            )
            m_updates, new_metric_opt = metric_optimizer.update(
                m_grads, metric_opt, metric_p
            )
            new_metric_p = optax.apply_updates(metric_p, m_updates)

            return (new_metric_p, new_metric_opt, new_ensemble_p, new_ensemble_opt, k), {
                "eme_metric_loss": m_loss,
                "eme_ensemble_loss": e_loss,
                "eme_ensemble_member_loss_mean": jnp.mean(per_member),
                "eme_ensemble_member_loss_std": jnp.std(per_member),
                **m_aux,
            }

        (new_metric_p, new_metric_opt, new_ensemble_p, new_ensemble_opt, new_key), step_metrics = jax.lax.scan(
            step,
            (
                state.metric_params,
                state.metric_opt_state,
                state.ensemble_params,
                state.ensemble_opt_state,
                state.key,
            ),
            (),
            length=num_grad_steps,
        )
        new_state = state.replace(
            metric_params=new_metric_p,
            metric_opt_state=new_metric_opt,
            ensemble_params=new_ensemble_p,
            ensemble_opt_state=new_ensemble_opt,
            key=new_key,
        )
        return new_state, jax.tree_util.tree_map(jnp.mean, step_metrics)

    return eme_bonus, initial_state, None, eme_train


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
        self.requires_actor_params: bool = any(
            bt in self._ACTOR_PARAM_BONUS_TYPES for bt in bonus_types
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
    _ONLINE_ONLY_BONUS_TYPES = frozenset(
        {"rnd", "misc", "online_empowerment", "online_mine_empowerment", "icm", "eme"}
    )

    # Bonus types whose bonus_fn / train_fn require the agent's main actor
    # params at runtime (for marginal-action sampling, or for the policy KL
    # term in EME's metric loss). The dispatcher forwards ``actor_params``
    # to these and to no others.
    _ACTOR_PARAM_BONUS_TYPES = frozenset({"online_mine_empowerment", "eme"})

    def compute(
        self,
        state: Tuple,
        transitions,
        key: jax.Array,
        is_online: Optional[jnp.ndarray] = None,
        actor_params: Any = None,
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
            if bt in self._ACTOR_PARAM_BONUS_TYPES:
                if actor_params is None:
                    raise ValueError(
                        f"bonus '{bt}' requires actor_params to be passed to compute()."
                    )
                bonus, new_state_i, m = fn(
                    state[i], transitions, sub_key, actor_params=actor_params,
                )
            else:
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

    def train(self, state: Tuple, online_transitions, num_grad_steps: int = 1,
              actor_params: Any = None):
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
            if bt in self._ACTOR_PARAM_BONUS_TYPES:
                if actor_params is None:
                    raise ValueError(
                        f"bonus '{bt}' requires actor_params to be passed to train()."
                    )
                new_s, m = train_fn(
                    state[i], online_transitions, num_grad_steps,
                    actor_params=actor_params,
                )
            else:
                new_s, m = train_fn(state[i], online_transitions, num_grad_steps)
            new_states[i] = new_s
            for mk, mv in m.items():
                all_metrics[f"bonus_{i}_{bt}_{mk}"] = mv
        return tuple(new_states), all_metrics

    def is_trainable(self, idx: int) -> bool:
        """True iff the bonus at ``idx`` exposes a per-bonus train hook."""
        return self._train_fns[idx] is not None

    def requires_actor_params_at(self, idx: int) -> bool:
        """True iff bonus ``idx`` needs the main actor's params at train time."""
        return self.bonus_types[idx] in self._ACTOR_PARAM_BONUS_TYPES

    def train_one(self, state: Tuple, idx: int, online_transitions, num_grad_steps: int = 1,
                  actor_params: Any = None):
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
        if bt in self._ACTOR_PARAM_BONUS_TYPES:
            if actor_params is None:
                raise ValueError(
                    f"bonus '{bt}' requires actor_params to be passed to train_one()."
                )
            new_s, m = train_fn(
                state[idx], online_transitions, num_grad_steps,
                actor_params=actor_params,
            )
        else:
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
    # Shared knob used by ICM and EME (the agent's RL discount). The action
    # dimension is sourced directly from ``env.action_size`` inside each
    # branch — it's a property of the env, not a config parameter.
    discount: float = 0.99,
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
    # ICM-specific (Pathak et al., 2017). Defaults follow the paper:
    # learning rate 1e-3, beta=0.2, feature dim 288, hidden 256. The action
    # dimension is sourced from the top-level ``action_size`` argument.
    icm_feature_dim: int = 288,
    icm_encoder_hidden_dim: int = 256,
    icm_encoder_num_hidden: int = 2,
    icm_inverse_hidden_dim: int = 256,
    icm_inverse_num_hidden: int = 1,
    icm_forward_hidden_dim: int = 256,
    icm_forward_num_hidden: int = 1,
    icm_learning_rate: float = 1e-3,
    icm_beta: float = 0.2,
    icm_eta: float = 1.0,
    # EME-specific (Wang et al., 2024). Defaults follow the paper:
    # ensemble size 6, max reward scaling M=10. The discount is sourced
    # from the top-level ``discount`` argument (= the agent's discount).
    eme_actor_dist_fn: Optional[
        Callable[[Any, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    ] = None,
    eme_metric_hidden_dim: int = 256,
    eme_metric_num_hidden: int = 2,
    eme_reward_hidden_dim: int = 256,
    eme_reward_num_hidden: int = 2,
    eme_metric_learning_rate: float = 1e-3,
    eme_reward_learning_rate: float = 1e-3,
    eme_ensemble_size: int = 6,
    eme_max_reward_scaling: float = 10.0,
    eme_bootstrap_keep_prob: float = 0.8,
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
    # Online-MINE-empowerment-specific
    online_mine_empowerment_action_size: Optional[int] = None,
    online_mine_empowerment_actor_sample_fn: Optional[
        Callable[[Any, jnp.ndarray, jax.Array], jnp.ndarray]
    ] = None,
    online_mine_empowerment_lr_dyn: float = 1e-3,
    online_mine_empowerment_lr_t: float = 3e-4,
    online_mine_empowerment_dyn_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_mine_empowerment_t_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    online_mine_empowerment_layer_norm: bool = True,
    online_mine_empowerment_bonus_mean: float = 0.0,
    online_mine_empowerment_bonus_scale: float = 1.0,
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
            discount=discount,
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
            icm_feature_dim=icm_feature_dim,
            icm_encoder_hidden_dim=icm_encoder_hidden_dim,
            icm_encoder_num_hidden=icm_encoder_num_hidden,
            icm_inverse_hidden_dim=icm_inverse_hidden_dim,
            icm_inverse_num_hidden=icm_inverse_num_hidden,
            icm_forward_hidden_dim=icm_forward_hidden_dim,
            icm_forward_num_hidden=icm_forward_num_hidden,
            icm_learning_rate=icm_learning_rate,
            icm_beta=icm_beta,
            icm_eta=icm_eta,
            eme_actor_dist_fn=eme_actor_dist_fn,
            eme_metric_hidden_dim=eme_metric_hidden_dim,
            eme_metric_num_hidden=eme_metric_num_hidden,
            eme_reward_hidden_dim=eme_reward_hidden_dim,
            eme_reward_num_hidden=eme_reward_num_hidden,
            eme_metric_learning_rate=eme_metric_learning_rate,
            eme_reward_learning_rate=eme_reward_learning_rate,
            eme_ensemble_size=eme_ensemble_size,
            eme_max_reward_scaling=eme_max_reward_scaling,
            eme_bootstrap_keep_prob=eme_bootstrap_keep_prob,
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
            online_mine_empowerment_action_size=online_mine_empowerment_action_size,
            online_mine_empowerment_actor_sample_fn=online_mine_empowerment_actor_sample_fn,
            online_mine_empowerment_lr_dyn=online_mine_empowerment_lr_dyn,
            online_mine_empowerment_lr_t=online_mine_empowerment_lr_t,
            online_mine_empowerment_dyn_hidden_dims=online_mine_empowerment_dyn_hidden_dims,
            online_mine_empowerment_t_hidden_dims=online_mine_empowerment_t_hidden_dims,
            online_mine_empowerment_layer_norm=online_mine_empowerment_layer_norm,
            online_mine_empowerment_bonus_mean=online_mine_empowerment_bonus_mean,
            online_mine_empowerment_bonus_scale=online_mine_empowerment_bonus_scale,
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
