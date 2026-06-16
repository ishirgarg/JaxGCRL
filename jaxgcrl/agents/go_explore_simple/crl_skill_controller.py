"""Hierarchical *contrastive* (CRL) skill controller training routine.

Implements ``GoExploreSimple._train_crl_skill_controller``
(``agent_type="crl_skill"``): freeze a pretrained OGBench skill-conditioned
policy ``π(a | s, z)`` and train an online **CRL (contrastive RL)** *high-level
controller* that selects discrete skills ``z`` over a Semi-MDP (SMDP) with fixed
``k``-step temporal commitment.

This is the contrastive sibling of ``skill_controller.train_skill_controller``
(``agent_type="sac_discrete"``). Everything about the SMDP plumbing — the frozen
skill adapter, the ``k``-step macro-step rollout, HER relabeling over
macro-steps, the deterministic eval rollout, the skill-usage histogram, and the
skill-colored trajectory + brax HTML render — is identical and reused from
``skill_controller`` / ``sac_discrete``. The ONLY thing that changes is the
high-level *learner*: instead of SAC-discrete we train a CRL contrastive critic
plus a categorical actor with an auto-tuned entropy temperature ``α``, mirroring
``crl/losses.py`` as closely as the discrete-skill setting allows.

CRL high-level learner (mirrors ``crl/losses.py`` ``update_critic`` /
``update_actor_and_alpha``):

  - **Contrastive critic** ``φ(s, z), ψ(g)``: ``sa_encoder`` consumes
    ``[state, one_hot(z)]`` (state = controller_obs state slice; ``one_hot(z)``
    is the discrete skill — exactly how the frozen low-level policy consumes
    skills); ``g_encoder`` consumes the goal (controller_obs goal slice). InfoNCE
    over the in-batch goals via ``energy_fn`` + ``contrastive_loss_fn`` with the
    same ``logsumexp_penalty_coeff``. HER (uniform-future relabel) supplies the
    future-goal positives — the SMDP analogue of CRL's ``flatten_batch``.

  - **Categorical actor** ``π(z | s, g)``: an MLP -> logits over ``num_skills``.
    Actor loss is the exact-soft discrete analogue of CRL's actor loss
    ``E[α·log π − Q]``:
        ``J = E_s Σ_z π(z|s,g)·(α·log π(z|s,g) − Q(s,z,g))``,
    with ``Q(s,z,g) = energy_fn(φ(state, one_hot(z)), ψ(goal))`` evaluated for
    every skill ``z`` (batched over the ``num_skills`` one-hots).

  - **Entropy / α term**: auto-tuned exactly as in CRL's ``alpha_loss``,
    ``α·mean(stop_gradient(−log_prob − H̄))`` with ``−log_prob`` replaced by the
    categorical entropy ``H(π) = −Σ π log π`` and
    ``H̄ = controller_target_entropy_scale·log(num_skills)`` (the SAC-discrete
    controller's convention).

The critic is purely contrastive (goal-reaching InfoNCE over (s,z)->future-goal,
no reward bootstrap / Bellman target) — faithful to ``crl/losses.py``.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EpisodeWrapper,
    EvalAutoResetWrapper,
    TrainAutoResetWrapper,
    TrajectoryIdWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import EvalWrapper
from jaxgcrl.agents.go_explore.utils import save_params
from jaxgcrl.agents.go_explore.goal_proposers import create_random_env_goals_proposer
from jaxgcrl.agents.go_explore.networks import Encoder

# Frozen-skill adapters + the skill-colored trajectory plot are shared verbatim
# with the SAC-discrete controller; only the high-level learner differs.
from jaxgcrl.agents.go_explore_simple.skill_controller import (
    _plot_skill_colored_trajectory,
    load_frozen_skill_policy,
    make_skill_action_fn,
)
from jaxgcrl.agents.go_explore_simple.sac_discrete import (
    DiscreteActorNet,
    her_relabel_sequence,
    init_controller_replay,
    insert_controller_replay,
    sample_controller_replay,
)
from jaxgcrl.agents.crl.losses import contrastive_loss_fn, energy_fn


# ── CRL controller training state ───────────────────────────────────────────────


@dataclass
class CRLControllerState:
    """Training state for the CRL high-level controller (no target critic)."""

    actor_state: TrainState     # categorical actor π(z | s, g)
    critic_state: TrainState    # {"sa_encoder": ..., "g_encoder": ...}
    alpha_state: TrainState     # {"log_alpha": ...}
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray


def create_crl_controller_state(
    actor_net: DiscreteActorNet,
    sa_encoder: Encoder,
    g_encoder: Encoder,
    *,
    state_size: int,
    goal_size: int,
    num_skills: int,
    policy_lr: float,
    critic_lr: float,
    alpha_lr: float,
    actor_key: jax.Array,
    sa_key: jax.Array,
    g_key: jax.Array,
) -> CRLControllerState:
    """Initialize the categorical actor, contrastive critic, and α states.

    Mirrors CRL's network/state setup: the critic is a single ``TrainState`` whose
    params hold both encoders (``sa_encoder`` over ``[state, one_hot(z)]`` and
    ``g_encoder`` over the goal); ``log_alpha`` starts at 0.
    """
    obs_size = state_size + goal_size
    actor_params = actor_net.init(actor_key, jnp.ones((1, obs_size), dtype=jnp.float32))
    sa_params = sa_encoder.init(sa_key, jnp.ones((1, state_size + num_skills), dtype=jnp.float32))
    g_params = g_encoder.init(g_key, jnp.ones((1, goal_size), dtype=jnp.float32))

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=policy_lr),
    )
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_params, "g_encoder": g_params},
        tx=optax.adam(learning_rate=critic_lr),
    )
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
        tx=optax.adam(learning_rate=alpha_lr),
    )
    return CRLControllerState(
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
    )


# ── CRL contrastive update (mirrors crl/losses.py, discrete-skill adapted) ───────


def crl_controller_update(
    controller_state: CRLControllerState,
    batch: Dict[str, jnp.ndarray],
    *,
    actor_net: DiscreteActorNet,
    sa_encoder: Encoder,
    g_encoder: Encoder,
    num_skills: int,
    state_size: int,
    energy_fn_name: str,
    contrastive_loss_name: str,
    logsumexp_penalty_coeff: float,
    target_entropy: float,
):
    """One CRL gradient step for the high-level controller.

    Order follows ``crl.py``'s ``update_networks``: actor + α first, then the
    contrastive critic. ``α`` uses the OLD (pre-update) value for the actor loss,
    matching CRL. There is no Bellman target / Polyak averaging — the critic is
    purely contrastive (InfoNCE over (state, one_hot(z)) -> future-goal positives,
    with in-batch goals as negatives), exactly like ``update_critic``.
    """
    obs = batch["obs"]                       # (B, obs_size) = [state, goal]
    skill = batch["skill"].astype(jnp.int32)
    reward = batch["reward"]
    done = batch["done"]
    B = obs.shape[0]
    state = obs[:, :state_size]              # (B, state_size)
    goal = obs[:, state_size:]               # (B, goal_size)

    actor_params = controller_state.actor_state.params
    critic_params = controller_state.critic_state.params
    old_alpha = jnp.exp(controller_state.alpha_state.params["log_alpha"])

    # ── Actor + α update (mirror crl ``update_actor_and_alpha``, categorical) ──
    def actor_loss_fn(a_params):
        logits = actor_net.apply(a_params, obs)            # (B, K)
        log_pi = jax.nn.log_softmax(logits, axis=-1)       # (B, K)
        pi = jnp.exp(log_pi)                               # (B, K)
        entropy = -jnp.sum(pi * log_pi, axis=-1)           # (B,)  H(π)

        # Per-skill contrastive Q(s, z, g) for *every* z (batch the K one-hots).
        sa_params = critic_params["sa_encoder"]
        g_params = critic_params["g_encoder"]
        onehots = jnp.eye(num_skills, dtype=state.dtype)               # (K, K)
        state_b = jnp.broadcast_to(state[:, None, :], (B, num_skills, state_size))
        onehot_b = jnp.broadcast_to(onehots[None, :, :], (B, num_skills, num_skills))
        sa_in = jnp.concatenate([state_b, onehot_b], axis=-1)         # (B, K, state+K)
        sa_repr = sa_encoder.apply(sa_params, sa_in)                  # (B, K, repr)
        g_repr = g_encoder.apply(g_params, goal)                     # (B, repr)
        q = energy_fn(energy_fn_name, sa_repr, g_repr[:, None, :])    # (B, K)
        q = jax.lax.stop_gradient(q)

        # J = E_s Σ_z π(z|s,g)·(α·log π(z|s,g) − Q(s,z,g))
        per_state = jnp.sum(pi * (old_alpha * log_pi - q), axis=-1)   # (B,)
        return jnp.mean(per_state), entropy

    (actor_loss_val, entropy), actor_grad = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_params)
    new_actor_state = controller_state.actor_state.apply_gradients(grads=actor_grad)

    def alpha_loss_fn(alpha_params):
        alpha = jnp.exp(alpha_params["log_alpha"])
        # crl alpha_loss: α·mean(stop_gradient(−log_prob − H̄)); discrete
        # substitution −log_prob -> entropy H(π)  =>  α·mean(sg(H − H̄)).
        return alpha * jnp.mean(jax.lax.stop_gradient(entropy - target_entropy))

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        controller_state.alpha_state.params
    )
    new_alpha_state = controller_state.alpha_state.apply_gradients(grads=alpha_grad)

    # ── Critic update (mirror crl ``update_critic`` exactly) ─────────────────
    z_onehot = jax.nn.one_hot(skill, num_skills)           # (B, K)

    def critic_loss_fn(c_params):
        sa_params = c_params["sa_encoder"]
        g_params = c_params["g_encoder"]
        sa_repr = sa_encoder.apply(sa_params, jnp.concatenate([state, z_onehot], axis=-1))
        g_repr = g_encoder.apply(g_params, goal)

        # InfoNCE over the in-batch goals.
        logits = energy_fn(energy_fn_name, sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(contrastive_loss_name, logits)

        # logsumexp regularisation (identical to update_critic).
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        return loss, (logsumexp, correct, logits_pos, logits_neg)

    (critic_loss_val, (logsumexp, correct, logits_pos, logits_neg)), critic_grad = (
        jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params)
    )
    new_critic_state = controller_state.critic_state.apply_gradients(grads=critic_grad)

    new_controller_state = controller_state.replace(
        actor_state=new_actor_state,
        critic_state=new_critic_state,
        alpha_state=new_alpha_state,
        gradient_steps=controller_state.gradient_steps + 1,
    )

    metrics = {
        # Reuse the existing controller_* metric keys where they map cleanly.
        "controller_actor_loss": actor_loss_val,
        "controller_alpha": old_alpha,
        "controller_alpha_loss": alpha_loss_val,
        "controller_entropy": jnp.mean(entropy),
        "controller_target_entropy": jnp.asarray(target_entropy, dtype=jnp.float32),
        "controller_critic_loss": critic_loss_val,
        # Contrastive-critic-specific metrics (mirror update_critic).
        "controller_categorical_accuracy": jnp.mean(correct),
        "controller_logits_pos": logits_pos,
        "controller_logits_neg": logits_neg,
        "controller_logsumexp": logsumexp.mean(),
        "controller_reward_mean": jnp.mean(reward),
        "controller_done_mean": jnp.mean(done),
    }
    return new_controller_state, metrics


# ── Training routine ────────────────────────────────────────────────────────────


def train_crl_skill_controller(
    self,
    config,
    train_env,
    eval_env,
    progress_fn: Callable = lambda *a, **k: None,
):
    """CRL contrastive high-level controller over a frozen skill set.

    ``self`` is the ``GoExploreSimple`` dataclass instance (config holder). The
    SMDP plumbing mirrors ``train_skill_controller``; the learner is CRL.
    """
    # CRL fundamentally relies on future-goal positives (HER), so the contrastive
    # controller requires the HER data path. Non-HER has no contrastive positives.
    assert bool(self.use_her), (
        "agent_type='crl_skill' requires use_her=True: the contrastive critic "
        "is trained on HER future-goal positives (the SMDP analogue of CRL's "
        "flatten_batch)."
    )

    unwrapped_env = train_env
    eval_env_base = eval_env if eval_env is not None else train_env

    state_size = int(unwrapped_env.state_dim)
    goal_size = len(unwrapped_env.goal_indices)
    obs_size = state_size + goal_size
    num_envs = config.num_envs
    k = self.skill_commitment_k
    gamma_low = self.gamma_low

    rng = jax.random.PRNGKey(config.seed)
    rng, skill_key, actor_key, sa_key, g_key, env_key = jax.random.split(rng, 6)
    np.random.seed(config.seed)

    # ── Frozen skill policy + adapters ───────────────────────────────────────
    emp_agent, num_skills, skill_obs_builder = load_frozen_skill_policy(
        self, unwrapped_env, skill_key
    )
    logging.info(
        "crl skill controller: num_skills=%d, k=%d, use_her=%s, p_future_her_goal=%.3f",
        num_skills, k, bool(self.use_her), float(self.p_future_her_goal),
    )
    skill_action_train = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills,
        deterministic=self.deterministic_skill_actions,
    )
    skill_action_eval = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills, deterministic=True,
    )

    target_entropy = float(self.controller_target_entropy_scale) * float(np.log(num_skills))

    # ── Controller networks + state (categorical actor + contrastive critic) ──
    actor_net = DiscreteActorNet(
        num_skills=num_skills, h_dim=self.h_dim, n_hidden=self.n_hidden,
        skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
    )
    sa_encoder = Encoder(
        repr_dim=self.repr_dim, network_width=self.h_dim, network_depth=self.n_hidden,
        skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
    )
    g_encoder = Encoder(
        repr_dim=self.repr_dim, network_width=self.h_dim, network_depth=self.n_hidden,
        skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
    )
    controller_state = create_crl_controller_state(
        actor_net, sa_encoder, g_encoder,
        state_size=state_size, goal_size=goal_size, num_skills=num_skills,
        policy_lr=self.policy_lr, critic_lr=self.critic_lr, alpha_lr=self.alpha_lr,
        actor_key=actor_key, sa_key=sa_key, g_key=g_key,
    )

    replay_state = init_controller_replay(self.controller_replay_size, obs_size)

    # Bound the per-skill kwargs so the update is a clean JIT-friendly closure.
    _update = functools.partial(
        crl_controller_update,
        actor_net=actor_net, sa_encoder=sa_encoder, g_encoder=g_encoder,
        num_skills=num_skills, state_size=state_size,
        energy_fn_name=self.energy_fn, contrastive_loss_name=self.contrastive_loss_fn,
        logsumexp_penalty_coeff=self.logsumexp_penalty_coeff, target_entropy=target_entropy,
    )

    # ── Env stacks (identical to the SAC-discrete controller) ────────────────
    t_env = TrajectoryIdWrapper(unwrapped_env)
    t_env = VmapWrapper(t_env)
    t_env = EpisodeWrapper(t_env, config.episode_length, config.action_repeat)
    t_env = TrainAutoResetWrapper(t_env)

    e_env = TrajectoryIdWrapper(eval_env_base)
    e_env = VmapWrapper(e_env)
    e_env = EpisodeWrapper(e_env, config.episode_length, config.action_repeat)
    e_env = EvalAutoResetWrapper(e_env)
    e_env = EvalWrapper(e_env)

    random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, num_envs)

    # ── Initial train reset ──────────────────────────────────────────────────
    env_keys = jax.random.split(env_key, num_envs)
    initial_goals = jax.vmap(random_goals_proposer)(env_keys)
    env_state = t_env.reset(env_keys, goal=initial_goals)

    # ── Macro-step rollout (one SMDP transition per env) ─────────────────────
    def rollout_macro_step(env_state, z, key, skill_action_fn):
        """Execute skill ``z`` for ``k`` env steps; return SMDP transition fields.

        Identical to the SAC-discrete controller's macro-step: ``R`` accumulates
        ``γ_low``-discounted reward up to the first in-window termination,
        ``done`` is any termination in the window, ``next_obs`` is frozen at the
        first termination. (The contrastive critic ignores ``R`` — it learns from
        HER future-goal positives — but we keep the same transition fields so the
        replay buffer + relabeling code is shared verbatim.)
        """
        R0 = jnp.zeros((num_envs,), dtype=jnp.float32)
        disc0 = jnp.ones((num_envs,), dtype=jnp.float32)
        alive0 = jnp.ones((num_envs,), dtype=jnp.float32)
        done0 = jnp.zeros((num_envs,), dtype=jnp.float32)
        final_obs0 = env_state.obs

        def body(carry, _):
            es, R, disc, alive, done_any, final_obs, key = carry
            key, akey, skey = jax.random.split(key, 3)
            state = es.obs[:, :state_size]
            a = skill_action_fn(state, z, akey)
            nstate = t_env.step(es, a, skey)
            r = nstate.reward
            alive_b = alive > 0.5
            R = R + alive * disc * r
            final_obs = jnp.where(alive_b[:, None], nstate.obs, final_obs)
            step_done = nstate.done.astype(jnp.float32)
            done_any = jnp.where(alive_b, jnp.maximum(done_any, step_done), done_any)
            alive = alive * (1.0 - step_done)
            disc = disc * gamma_low
            return (nstate, R, disc, alive, done_any, final_obs, key), None

        (env_state, R, _, _, done_any, final_obs, _), _ = jax.lax.scan(
            body, (env_state, R0, disc0, alive0, done0, final_obs0, key), (), length=k
        )
        return env_state, R, final_obs, done_any

    # ── Collect one macro-step of experience for all envs ────────────────────
    def step_macro(controller_state, env_state, key):
        """One macro-step for all envs; returns the (un-inserted) SMDP transition."""
        goal_key, z_key, roll_key = jax.random.split(key, 3)

        # Refresh task goals so any auto-reset this macro-step draws a fresh goal.
        goal_keys = jax.random.split(goal_key, num_envs)
        fresh_goals = jax.vmap(random_goals_proposer)(goal_keys)
        info = dict(env_state.info)
        info["proposed_goals"] = fresh_goals
        env_state = env_state.replace(info=info)

        controller_obs = env_state.obs  # [state, goal]
        traj_id = env_state.info["traj_id"]
        logits = actor_net.apply(controller_state.actor_state.params, controller_obs)
        z = jax.random.categorical(z_key, logits)  # stochastic during training

        env_state, R, next_obs, done = rollout_macro_step(
            env_state, z, roll_key, skill_action_train
        )
        step = {
            "obs": controller_obs,
            "skill": z,
            "reward": R,
            "next_obs": next_obs,
            "done": done,
            "traj_id": traj_id,
        }
        return env_state, step, R, done

    def collect_macro(controller_state, env_state, replay_state, key):
        env_state, step, R, done = step_macro(controller_state, env_state, key)
        replay_state = insert_controller_replay(replay_state, step)
        return env_state, replay_state, R, done

    n_grad_steps = max(1, self.train_step_multiplier)

    # ── Step bookkeeping (identical accounting to the SAC-discrete path) ──────
    env_steps_per_macro = num_envs * k
    num_prefill_macro_steps = int(np.ceil(self.min_replay_size / num_envs))
    num_prefill_env_steps = num_prefill_macro_steps * env_steps_per_macro
    available_env_steps = max(env_steps_per_macro, config.total_env_steps - num_prefill_env_steps)
    env_steps_per_epoch = available_env_steps // config.num_evals
    macro_steps_per_epoch = max(1, env_steps_per_epoch // env_steps_per_macro)

    logging.info("crl controller num_prefill_macro_steps: %d", num_prefill_macro_steps)
    logging.info("crl controller macro_steps_per_epoch:   %d", macro_steps_per_epoch)

    @jax.jit
    def prefill(controller_state, env_state, replay_state, key):
        def f(carry, _):
            cs, es, rs, k_ = carry
            k_, ck = jax.random.split(k_)
            es, rs, _, _ = collect_macro(cs, es, rs, ck)
            return (cs, es, rs, k_), None
        (controller_state, env_state, replay_state, _), _ = jax.lax.scan(
            f, (controller_state, env_state, replay_state, key), (), length=num_prefill_macro_steps
        )
        return controller_state, env_state, replay_state

    # ── HER (hindsight relabel) — the contrastive data path ──────────────────
    # Collect a block of macro-steps, relabel each transition's goal with a
    # uniformly-sampled *future* achieved state from the SAME episode (matched by
    # traj_id), store in the flat replay, sample i.i.d. batches, and run the CRL
    # contrastive update with in-batch goals as negatives.
    HER_MAX_WINDOW = 512
    her_window = min(config.episode_length // k, HER_MAX_WINDOW)  # macro-steps / block
    blocks_per_epoch = max(1, int(round(macro_steps_per_epoch / her_window)))
    _goal_indices = jnp.asarray(
        [int(i) for i in np.asarray(unwrapped_env.goal_indices)]
    )
    _goal_reach_thresh = float(unwrapped_env.goal_reach_thresh)
    _relabel_seq = functools.partial(
        her_relabel_sequence,
        state_size=state_size,
        goal_indices=_goal_indices,
        goal_reach_thresh=_goal_reach_thresh,
        p_future_her_goal=float(self.p_future_her_goal),
    )

    def collect_block(controller_state, env_state, key):
        def f(carry, _):
            es_, k_ = carry
            k_, ck = jax.random.split(k_)
            es_, step, _, _ = step_macro(controller_state, es_, ck)
            return (es_, k_), step
        (env_state, _), block = jax.lax.scan(
            f, (env_state, key), (), length=her_window
        )
        return env_state, block  # block leaves: (her_window, num_envs, ...)

    @jax.jit
    def training_epoch_her(controller_state, env_state, replay_state, key):
        def block_step(carry, _):
            cs, es, rs, k_ = carry
            k_, collect_key, relabel_key, train_key = jax.random.split(k_, 4)

            es, block = collect_block(cs, es, collect_key)

            # Relabel per env over its macro-step sequence, then flatten to i.i.d.
            block_env_major = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), block
            )  # (num_envs, her_window, ...)
            rkeys = jax.random.split(relabel_key, num_envs)
            relabeled = jax.vmap(_relabel_seq)(block_env_major, rkeys)
            flat = jax.tree_util.tree_map(
                lambda x: x.reshape((num_envs * her_window,) + x.shape[2:]), relabeled
            )
            rs = insert_controller_replay(rs, flat)

            def upd(carry2, _):
                cs2, k2 = carry2
                k2, sk = jax.random.split(k2)
                batch = sample_controller_replay(rs, sk, self.batch_size)
                cs2, m = _update(cs2, batch)
                return (cs2, k2), m

            # Keep updates-per-collected-macro-step equal to the non-HER budget.
            (cs, _), m = jax.lax.scan(
                upd, (cs, train_key), (), length=her_window * n_grad_steps
            )
            cs = cs.replace(env_steps=cs.env_steps + num_envs * k * her_window)
            m = jax.tree_util.tree_map(jnp.mean, m)
            m["macro_reward_mean"] = jnp.mean(block["reward"])
            m["macro_done_mean"] = jnp.mean(block["done"])
            return (cs, es, rs, k_), m

        (controller_state, env_state, replay_state, _), metrics = jax.lax.scan(
            block_step,
            (controller_state, env_state, replay_state, key),
            (), length=blocks_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return controller_state, env_state, replay_state, metrics

    # ── Dedicated eval: argmax controller + deterministic skill policy ───────
    num_eval_envs = config.num_eval_envs
    episode_length = config.episode_length

    @jax.jit
    def evaluate(controller_state, key):
        reset_keys = jax.random.split(key, num_eval_envs)
        state = e_env.reset(reset_keys)  # goal=None -> env's eval task goals

        skill0 = jnp.zeros((num_eval_envs,), dtype=jnp.int32)

        def body(carry, i):
            state, skill = carry
            controller_obs = state.obs
            logits = actor_net.apply(controller_state.actor_state.params, controller_obs)
            new_skill = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            reselect = (i % k) == 0
            skill = jnp.where(reselect, new_skill, skill)
            a = skill_action_eval(state.obs[:, :state_size], skill, key)
            nstate = e_env.step(state, a)
            return (nstate, skill), skill

        (state, _), skills = jax.lax.scan(
            body, (state, skill0), jnp.arange(episode_length)
        )
        eval_metrics = state.info["eval_metrics"]
        return eval_metrics, skills  # skills: (episode_length, num_eval_envs)

    def run_eval(controller_state, base_metrics, key, num_steps):
        eval_metrics, skills = evaluate(controller_state, key)
        em = eval_metrics.episode_metrics
        out = dict(base_metrics)
        for name in ("reward", "success", "success_easy", "dist"):
            if name in em:
                out[f"eval/episode_{name}"] = float(np.mean(np.asarray(em[name])))
        if "success" in em:
            out["eval/episode_success_any"] = float(np.mean(np.asarray(em["success"]) > 0.0))
        out["eval/avg_episode_length"] = float(np.mean(np.asarray(eval_metrics.episode_steps)))

        # ── Skill-usage distribution (choices made at the reselection steps) ──
        skills_np = np.asarray(skills)                 # (episode_length, num_eval_envs)
        chosen = skills_np[0::k].reshape(-1).astype(int)  # i % k == 0 -> a skill choice
        counts = np.bincount(chosen, minlength=num_skills).astype(np.float64)
        fracs = counts / max(counts.sum(), 1.0)
        nz = fracs > 0
        out["eval/skill_entropy"] = float(-(fracs[nz] * np.log(fracs[nz])).sum())
        out["eval/skill_max_frac"] = float(fracs.max())
        out["eval/skill_active_count"] = float((counts > 0).sum())
        # Route the histogram through the single per-step log_wandb call (via the
        # reserved "_wandb_media" key) instead of a separate same-step wandb.log,
        # which the live server would treat as a step collision and drop the
        # scalar metrics logged afterward (zeros in wandb). See utils/env.py.
        media = out.setdefault("_wandb_media", {})
        media["eval/skill_usage_hist"] = wandb.Histogram(
            np_histogram=(counts, np.arange(num_skills + 1) - 0.5)
        )
        return out

    # ── Render: faithful hierarchical rollout (controller + frozen skill) ────
    prim_idx = np.asarray(unwrapped_env.goal_indices)[:2]  # tracked entity xy

    def render_skill_controller(controller_state, key, num_steps):
        from brax.io import html

        media = {}  # accumulate renders; merged into the single per-step log_wandb
        actor_params = controller_state.actor_state.params

        @jax.jit
        def pick(obs_b):
            logits = actor_net.apply(actor_params, obs_b)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        @jax.jit
        def act(state_b, skill_b, akey):
            return skill_action_eval(state_b, skill_b, akey)

        jit_reset = jax.jit(eval_env_base.reset)
        jit_step = jax.jit(eval_env_base.step)

        key, rk = jax.random.split(key)
        state = jit_reset(rk)
        rollout, xy_list, skill_list = [], [], []
        skill = jnp.zeros((1,), dtype=jnp.int32)
        for i in range(episode_length):
            rollout.append(state.pipeline_state)
            obs_b = state.obs[None]
            if i % k == 0:
                skill = pick(obs_b)
            key, ak = jax.random.split(key)
            a = act(obs_b[:, :state_size], skill, ak)
            obs_np = np.asarray(state.obs)
            xy_list.append(obs_np[prim_idx])
            skill_list.append(int(skill[0]))
            state = jit_step(state, a[0])

        xy = np.stack(xy_list)
        skills = np.asarray(skill_list)
        goal_xy = np.asarray(state.obs)[state_size:][:2]

        # (1) brax 3D HTML
        try:
            sys = eval_env_base.sys.tree_replace({"opt.timestep": eval_env_base.dt})
            url = html.render(sys, rollout, height=1024)
            media["render/skill_html"] = wandb.Html(url)
        except Exception as e:  # rendering must never crash training
            logging.warning("skill HTML render failed: %s", e)

        # (2) skill-colored 2D trajectory
        try:
            fig = _plot_skill_colored_trajectory(
                xy, skills, num_skills,
                unwrapped_env.x_bounds, unwrapped_env.y_bounds,
                goal_xy=goal_xy, title=f"skills over trajectory @ step {num_steps}",
            )
            media["render/skill_trajectory"] = wandb.Image(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            logging.warning("skill trajectory plot failed: %s", e)
        return media

    # ── Prefill ──────────────────────────────────────────────────────────────
    rng, prefill_key = jax.random.split(rng)
    controller_state, env_state, replay_state = prefill(
        controller_state, env_state, replay_state, prefill_key
    )

    # ── Main loop ────────────────────────────────────────────────────────────
    def make_policy(params, deterministic: bool = True):
        # Thin per-step hierarchical policy (used only for optional rendering).
        actor_params = params[0]

        def policy(obs, rng_key):
            obs_b = obs[None] if obs.ndim == 1 else obs
            logits = actor_net.apply(actor_params, obs_b)
            z = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            a = skill_action_eval(obs_b[:, :state_size], z, rng_key)
            a = a[0] if obs.ndim == 1 else a
            return a, {}
        return policy

    training_walltime = 0.0
    metrics = {}
    current_step = 0
    logging.info("starting crl skill-controller training....")

    for ne in range(config.num_evals):
        t = time.time()
        rng, epoch_key, eval_rng = jax.random.split(rng, 3)
        controller_state, env_state, replay_state, raw_metrics = training_epoch_her(
            controller_state, env_state, replay_state, epoch_key
        )
        raw_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), raw_metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time

        current_step = int(controller_state.env_steps) + num_prefill_env_steps
        sps = (env_steps_per_macro * macro_steps_per_epoch) / max(epoch_time, 1e-6)

        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": current_step,
        }
        for name, value in raw_metrics.items():
            metrics[f"training/{name}"] = float(value)

        metrics = run_eval(controller_state, metrics, eval_rng, current_step)
        logging.info("step: %d", current_step)

        params = (
            controller_state.actor_state.params,
            controller_state.critic_state.params,
            controller_state.alpha_state.params,
        )
        if config.checkpoint_logdir:
            save_params(f"{config.checkpoint_logdir}/step_{current_step}.pkl", params)

        vis_interval = getattr(config, "visualization_interval", 0)
        if vis_interval and (ne % vis_interval == 0):
            rng, render_key = jax.random.split(rng)
            try:
                render_media = render_skill_controller(controller_state, render_key, current_step)
                if render_media:
                    metrics.setdefault("_wandb_media", {}).update(render_media)
            except Exception as e:  # never let rendering crash training
                logging.warning("render_skill_controller failed: %s", e)
        progress_fn(current_step, metrics, make_policy, (controller_state.actor_state.params,),
                    unwrapped_env, do_render=False)

    logging.info("total steps: %s", current_step)
    final_params = (
        controller_state.actor_state.params,
        controller_state.critic_state.params,
        controller_state.alpha_state.params,
    )
    return make_policy, final_params, metrics
