"""Hierarchical *contrastive* (CRL) skill controller training routine.

Implements ``GoExploreSimple._train_crl_skill_controller``
(``agent_type="crl_skill"``): freeze a pretrained OGBench skill-conditioned
policy ``π(a | s, z)`` and train an online **CRL (contrastive RL)** *high-level
controller* that selects discrete skills ``z`` over a Semi-MDP (SMDP) with fixed
``k``-step temporal commitment.

This is the contrastive sibling of ``skill_controller.train_skill_controller``
(``agent_type="sac_discrete"``). Everything about the SMDP plumbing — the frozen
skill adapter, the ``k``-step macro-step rollout, the deterministic eval
rollout, the skill-usage histogram, and the skill-colored trajectory + brax
HTML render — is identical and reused from ``skill_controller`` /
``sac_discrete``. What changes is the high-level *learner* AND the data path:
instead of SAC-discrete + HER we train a CRL contrastive critic plus a
categorical actor with an auto-tuned entropy temperature ``α``, with InfoNCE
positives sampled EXACTLY the way the flat ``agent_type="crl"`` path does it —
raw macro-transitions (with ``traj_id``) are stored in a trajectory-preserving
queue, and at batch time each row's goal is drawn from a discount-weighted
*future* macro-step of the same episode (``go_explore/utils.flatten_batch``,
applied by the flat path via ``CRLActor.process_transitions``).

CRL high-level learner (mirrors ``crl/losses.py`` ``update_critic`` /
``update_actor_and_alpha``):

  - **Contrastive critic** ``φ(s, z), ψ(g)``: ``sa_encoder`` consumes
    ``[state, one_hot(z)]`` (state = controller_obs state slice; ``one_hot(z)``
    is the discrete skill — exactly how the frozen low-level policy consumes
    skills); ``g_encoder`` consumes the goal (controller_obs goal slice). InfoNCE
    over the in-batch goals via ``energy_fn`` + ``contrastive_loss_fn`` with the
    same ``logsumexp_penalty_coeff``. Positives are ALWAYS sampled future
    states: each batch row's goal is the achieved goal of a future macro-step
    from the same episode, sampled with probability ∝ γ^Δt (the SMDP port of
    CRL's ``flatten_batch``; γ measured in env steps, so γ_macro = γ^k per row).

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
no reward bootstrap / Bellman target) — faithful to ``crl/losses.py``. There is
NO HER path here: unlike the SAC-discrete controller, goals are never relabeled
at insert time and the stored reward is never recomputed; relabeling happens
purely at batch-construction time, exactly like the flat CRL agent.

When ``self.continuous_skill`` is True, the controller's action space is a
continuous skill vector instead of a discrete index: the categorical actor is
swapped for the same ``Actor`` network used for continuous action spaces
elsewhere in this repo (a learned state-dependent Gaussian mean + log-std), but
UNSQUASHED — the skill latent lives in all of R^d (whatever the frozen
low-level policy was trained to condition on), not tanh-bounded like an env
action, so sampling/log-prob use ``_gaussian_skill_and_log_prob`` (no tanh
correction) rather than ``go_explore/losses.py``'s squashed variant. Skills are
concatenated directly into the contrastive critic's ``sa_encoder`` input (no
one-hot), and the actor/alpha losses are the reparameterized-sample SAC updates
(mirroring ``update_actor_and_alpha``, minus the squashing) instead of the
exact enumeration over one-hot skills. Everything else — the SMDP macro-step
plumbing, the trajectory replay + future-goal positive sampling, the eval loop
— is unchanged; ``self.skill_dim`` (analogous to ``num_skills``) sets the
continuous skill dimensionality.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Dict

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
from jaxgcrl.agents.go_explore.networks import Actor, Encoder

# Frozen-skill adapters + the skill-colored trajectory plot are shared verbatim
# with the SAC-discrete controller; only the high-level learner differs.
from jaxgcrl.agents.go_explore_simple.skill_controller import (
    _plot_skill_colored_trajectory,
    load_frozen_skill_policy,
    make_skill_action_fn,
)
from jaxgcrl.agents.go_explore_simple.sac_discrete import DiscreteActorNet
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.losses import contrastive_loss_fn, energy_fn


def _gaussian_skill_and_log_prob(means, log_stds, key, keepdims: bool = True):
    """Reparameterized Gaussian sample, UNSQUASHED (skill latent lives in R^d).

    Unlike ``go_explore/losses.py::_squashed_gaussian_action_and_log_prob``
    (used for bounded env actions), a continuous skill has no valid range to
    tanh-squash into — it's whatever the frozen low-level policy was trained
    to condition on — so this omits the tanh + log-det-Jacobian correction.
    """
    stds = jnp.exp(log_stds)
    key, noise_key = jax.random.split(key)
    skill = means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
    log_prob = jax.scipy.stats.norm.logpdf(skill, loc=means, scale=stds)
    log_prob = log_prob.sum(-1, keepdims=keepdims)
    return skill, log_prob


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
    actor_net,
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
    """Initialize the actor, contrastive critic, and α states.

    Mirrors CRL's network/state setup: the critic is a single ``TrainState`` whose
    params hold both encoders (``sa_encoder`` over ``[state, skill]`` — skill is
    one-hot for a discrete controller (``actor_net`` a ``DiscreteActorNet``) or a
    raw continuous vector for a continuous controller (``actor_net`` an
    ``Actor``) — and ``g_encoder`` over the goal); ``log_alpha`` starts at 0.
    ``num_skills`` doubles as the continuous skill dimensionality in that case.
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
    key: jax.Array,
    *,
    actor_net,
    sa_encoder: Encoder,
    g_encoder: Encoder,
    num_skills: int,
    state_size: int,
    energy_fn_name: str,
    contrastive_loss_name: str,
    logsumexp_penalty_coeff: float,
    target_entropy: float,
    continuous: bool = False,
):
    """One CRL gradient step for the high-level controller.

    Order follows ``crl.py``'s ``update_networks``: actor + α first, then the
    contrastive critic. ``α`` uses the OLD (pre-update) value for the actor loss,
    matching CRL. There is no Bellman target / Polyak averaging — the critic is
    purely contrastive (InfoNCE over (state, skill) -> future-goal positives,
    with in-batch goals as negatives), exactly like ``update_critic``.

    When ``continuous=False`` the actor is categorical (``actor_net`` a
    ``DiscreteActorNet``) and the actor loss exactly enumerates every one-hot
    skill. When ``continuous=True`` the actor is a tanh-squashed Gaussian
    (``actor_net`` an ``Actor``, same as the flat CRL agent's continuous action
    space) and the actor/alpha losses are the reparameterized-sample versions
    from ``go_explore/losses.py::update_actor_and_alpha`` — ``num_skills``
    doubles as the continuous skill dimensionality, and no one-hot is used
    anywhere (the raw skill vector is concatenated with the state instead).
    """
    obs = batch["obs"]                       # (B, obs_size) = [state, goal]
    reward = batch["reward"]
    done = batch["done"]
    B = obs.shape[0]
    state = obs[:, :state_size]              # (B, state_size)
    goal = obs[:, state_size:]               # (B, goal_size)

    actor_params = controller_state.actor_state.params
    critic_params = controller_state.critic_state.params
    old_alpha = jnp.exp(controller_state.alpha_state.params["log_alpha"])

    # ── Actor + α update (mirror crl ``update_actor_and_alpha``) ─────────────
    if continuous:
        def actor_loss_fn(a_params, key):
            means, log_stds = actor_net.apply(a_params, obs)   # (B, K), (B, K)
            skill, log_prob = _gaussian_skill_and_log_prob(
                means, log_stds, key, keepdims=False
            )  # skill: (B, K), log_prob: (B,)

            sa_params = critic_params["sa_encoder"]
            g_params = critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_params, jnp.concatenate([state, skill], axis=-1))
            g_repr = g_encoder.apply(g_params, goal)
            q = energy_fn(energy_fn_name, sa_repr, g_repr)     # (B,); NOT stop-gradiented
            # -> gradient flows to the actor through the reparameterized skill.

            per_state = old_alpha * log_prob - q               # (B,)
            return jnp.mean(per_state), log_prob

        (actor_loss_val, entropy_term), actor_grad = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_params, key)
        # ``entropy_term`` here is log_prob (B,); reused below as ``entropy`` for
        # the alpha loss and the shared "controller_entropy" metric (-log_prob).
        entropy = -entropy_term
    else:
        def actor_loss_fn(a_params, key):
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
        )(actor_params, key)
    new_actor_state = controller_state.actor_state.apply_gradients(grads=actor_grad)

    def alpha_loss_fn(alpha_params):
        alpha = jnp.exp(alpha_params["log_alpha"])
        # crl alpha_loss: α·mean(stop_gradient(−log_prob − H̄)); discrete
        # substitution −log_prob -> entropy H(π)  =>  α·mean(sg(H − H̄)); the
        # continuous branch uses −log_prob directly (== ``entropy`` above).
        return alpha * jnp.mean(jax.lax.stop_gradient(entropy - target_entropy))

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        controller_state.alpha_state.params
    )
    new_alpha_state = controller_state.alpha_state.apply_gradients(grads=alpha_grad)

    # ── Critic update (mirror crl ``update_critic`` exactly) ─────────────────
    skill = batch["skill"] if continuous else jax.nn.one_hot(
        batch["skill"].astype(jnp.int32), num_skills
    )  # (B, K)

    def critic_loss_fn(c_params):
        sa_params = c_params["sa_encoder"]
        g_params = c_params["g_encoder"]
        sa_repr = sa_encoder.apply(sa_params, jnp.concatenate([state, skill], axis=-1))
        g_repr = g_encoder.apply(g_params, goal)

        # InfoNCE over the in-batch goals.
        logits = energy_fn(energy_fn_name, sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(contrastive_loss_name, logits)

        # logsumexp regularisation (identical to update_critic).
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

        eye = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(eye, axis=1)
        logits_pos = jnp.sum(logits * eye) / jnp.sum(eye)
        logits_neg = jnp.sum(logits * (1 - eye)) / jnp.sum(1 - eye)
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
    unwrapped_env = train_env
    eval_env_base = eval_env if eval_env is not None else train_env

    state_size = int(unwrapped_env.state_dim)
    goal_size = len(unwrapped_env.goal_indices)
    obs_size = state_size + goal_size
    num_envs = config.num_envs
    k = self.skill_commitment_k
    gamma_low = self.gamma_low
    continuous_skill = bool(getattr(self, "continuous_skill", False))

    rng = jax.random.PRNGKey(config.seed)
    rng, skill_key, actor_key, sa_key, g_key, env_key = jax.random.split(rng, 6)
    np.random.seed(config.seed)

    # ── Frozen skill policy + adapters ───────────────────────────────────────
    emp_agent, num_skills, skill_obs_builder = load_frozen_skill_policy(
        self, unwrapped_env, skill_key
    )
    logging.info(
        "crl skill controller: num_skills=%d, k=%d, continuous_skill=%s",
        num_skills, k, continuous_skill,
    )
    skill_action_train = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills,
        deterministic=self.deterministic_skill_actions, continuous=continuous_skill,
    )
    skill_action_eval = make_skill_action_fn(
        emp_agent, skill_obs_builder, num_skills, deterministic=True,
        continuous=continuous_skill,
    )

    # Continuous: standard SAC heuristic (-0.5 * skill_dim), matching the flat
    # agent's continuous-action target entropy. Discrete: scaled max entropy.
    if continuous_skill:
        target_entropy = -0.5 * float(num_skills)
    else:
        target_entropy = float(self.controller_target_entropy_scale) * float(np.log(num_skills))

    # ── Controller networks + state (actor + contrastive critic) ─────────────
    if continuous_skill:
        actor_net = Actor(
            action_size=num_skills, network_width=self.h_dim, network_depth=self.n_hidden,
            skip_connections=self.skip_connections, use_relu=self.use_relu, use_ln=self.use_ln,
        )
    else:
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

    # ── Skill selection (stochastic-train / deterministic-mode) ──────────────
    # Shared by ``step_macro``, ``evaluate``, and ``render_skill_controller`` so
    # the discrete/continuous branch lives in exactly one place.
    if continuous_skill:
        def sample_skill(actor_params, obs, key):
            means, log_stds = actor_net.apply(actor_params, obs)
            z, _ = _gaussian_skill_and_log_prob(means, log_stds, key, keepdims=False)
            return z

        def skill_mode(actor_params, obs):
            means, _ = actor_net.apply(actor_params, obs)
            return means
    else:
        def sample_skill(actor_params, obs, key):
            logits = actor_net.apply(actor_params, obs)
            return jax.random.categorical(key, logits)

        def skill_mode(actor_params, obs):
            logits = actor_net.apply(actor_params, obs)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    # ── Trajectory-preserving macro-step replay (mirrors the flat CRL buffer) ──
    # Raw macro-transitions (goals NOT relabeled, ``traj_id`` kept) are stored
    # per-env in time order; sampling returns contiguous per-env windows of
    # ``macro_window`` macro-steps, from which ``flatten_macro_batch`` draws the
    # future-goal positives — identical structure to the flat agent's
    # ``TrajectoryUniformSamplingQueue`` + ``flatten_batch`` pipeline.
    macro_window = config.episode_length // k  # macro-steps per episode
    assert macro_window >= 2, (
        "crl_skill needs episode_length/k >= 2 macro-steps so every window has "
        "at least one future state to sample as an InfoNCE positive."
    )
    # Capacity is in per-env rows; keep the same total macro-transition budget
    # as the flat controller replay (controller_replay_size) with a floor that
    # guarantees the sampler always has a full window plus history.
    max_replay_rows = max(2 * macro_window + 1, self.controller_replay_size // num_envs)
    dummy_skill = (
        jnp.zeros((num_skills,), dtype=jnp.float32)
        if continuous_skill
        else jnp.zeros((), dtype=jnp.int32)
    )
    dummy_macro_step = {
        "obs": jnp.zeros((obs_size,), dtype=jnp.float32),
        "skill": dummy_skill,
        "reward": jnp.zeros((), dtype=jnp.float32),
        "next_obs": jnp.zeros((obs_size,), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.float32),
        "traj_id": jnp.zeros((), dtype=jnp.float32),
    }
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_replay_rows,
        dummy_data_sample=dummy_macro_step,
        sample_batch_size=self.batch_size,
        num_envs=num_envs,
        episode_length=macro_window,
    )
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    rng, buffer_key = jax.random.split(rng)
    replay_state = replay_buffer.init(buffer_key)

    # Bound the per-skill kwargs so the update is a clean JIT-friendly closure.
    _update = functools.partial(
        crl_controller_update,
        actor_net=actor_net, sa_encoder=sa_encoder, g_encoder=g_encoder,
        num_skills=num_skills, state_size=state_size,
        energy_fn_name=self.energy_fn, contrastive_loss_name=self.contrastive_loss_fn,
        logsumexp_penalty_coeff=self.logsumexp_penalty_coeff, target_entropy=target_entropy,
        continuous=continuous_skill,
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
        sampled future-goal positives — but we keep the same transition fields as
        the SAC-discrete controller so the SMDP plumbing stays identical.)
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
        z = sample_skill(controller_state.actor_state.params, controller_obs, z_key)  # stochastic during training

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

    def collect_block(controller_state, env_state, key):
        """Collect one episode-length block of raw macro-steps for all envs."""
        def f(carry, _):
            es_, k_ = carry
            k_, ck = jax.random.split(k_)
            es_, step, _, _ = step_macro(controller_state, es_, ck)
            return (es_, k_), step
        (env_state, _), block = jax.lax.scan(
            f, (env_state, key), (), length=macro_window
        )
        return env_state, block  # block leaves: (macro_window, num_envs, ...)

    # ── Step bookkeeping (identical accounting to the SAC-discrete path) ──────
    env_steps_per_macro = num_envs * k
    # Prefill in whole blocks; at least 2 so the queue's sampler always has a
    # valid start range (insert_position > macro_window).
    num_prefill_blocks = max(
        2, int(np.ceil(self.min_replay_size / (macro_window * num_envs)))
    )
    num_prefill_macro_steps = num_prefill_blocks * macro_window
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
            es, block = collect_block(cs, es, ck)
            rs = replay_buffer.insert(rs, block)
            return (cs, es, rs, k_), None
        (controller_state, env_state, replay_state, _), _ = jax.lax.scan(
            f, (controller_state, env_state, replay_state, key), (), length=num_prefill_blocks
        )
        return controller_state, env_state, replay_state

    # ── Contrastive data path (exact mimic of the flat CRL agent) ────────────
    # Per block: collect raw macro-steps, insert into the trajectory queue,
    # sample per-env macro-step windows from the WHOLE buffer, draw each row's
    # goal from a γ-discounted future macro-step of the same episode
    # (``flatten_macro_batch`` below == ``go_explore/utils.flatten_batch`` on
    # macro rows), then permute, split into batches, and do one full pass of
    # contrastive updates — the structure of the flat path's ``training_step``
    # (get_experience -> buffer.sample -> process_transitions -> scan(update)).
    blocks_per_epoch = max(1, int(round(macro_steps_per_epoch / macro_window)))
    _goal_indices = jnp.asarray(
        [int(i) for i in np.asarray(unwrapped_env.goal_indices)]
    )
    # Rows are k env steps apart, so the flat agent's per-env-step ``discounting``
    # becomes ``discounting**k`` per macro-row: γ^Δt measured in env steps.
    gamma_macro = float(self.discounting) ** k

    def flatten_macro_batch(seq, sample_key):
        """``go_explore/utils.flatten_batch`` ported to one env's macro window.

        ``seq`` fields are (T, ...) raw macro-transitions. Each row's goal is the
        achieved goal of a future row of the same episode, sampled with
        probability ∝ γ_macro^Δrows (traj_id-masked, ``eye*1e-5`` self-fallback
        for rows with no future — identical to the flat implementation). The
        last row is dropped (it has no future). Rewards are NOT recomputed; the
        contrastive critic never reads them.
        """
        T = seq["obs"].shape[0]
        arrangement = jnp.arange(T)
        is_future = (arrangement[:, None] < arrangement[None]).astype(jnp.float32)
        discount = gamma_macro ** jnp.array(
            arrangement[None] - arrangement[:, None], dtype=jnp.float32
        )
        tid = seq["traj_id"]
        same_traj = (tid[None, :] == tid[:, None]).astype(jnp.float32)
        probs = is_future * discount * same_traj + jnp.eye(T) * 1e-5

        goal_index = jax.random.categorical(sample_key, jnp.log(probs))
        future_state = seq["obs"][goal_index[:-1], :state_size]
        goal = future_state[:, _goal_indices]

        state = seq["obs"][:-1, :state_size]
        next_state = seq["next_obs"][:-1, :state_size]
        return {
            "obs": jnp.concatenate([state, goal], axis=1),
            "skill": seq["skill"][:-1],
            "reward": seq["reward"][:-1],
            "next_obs": jnp.concatenate([next_state, goal], axis=1),
            "done": seq["done"][:-1],
        }

    flat_rows = num_envs * (macro_window - 1)
    num_batches = max(1, flat_rows // self.batch_size)

    @jax.jit
    def training_epoch(controller_state, env_state, replay_state, key):
        def block_step(carry, _):
            cs, es, rs, k_ = carry
            k_, collect_key, flatten_key, perm_key, update_key = jax.random.split(k_, 5)

            es, block = collect_block(cs, es, collect_key)
            rs = replay_buffer.insert(rs, block)

            # Sample windows -> future-goal positives -> permute -> batches.
            rs, seqs = replay_buffer.sample(rs)  # leaves: (num_envs, macro_window, ...)
            fkeys = jax.random.split(flatten_key, num_envs)
            flat = jax.vmap(flatten_macro_batch)(seqs, fkeys)
            flat = jax.tree_util.tree_map(
                lambda x: x.reshape((flat_rows,) + x.shape[2:]), flat
            )
            perm = jax.random.permutation(perm_key, flat_rows)
            batched = jax.tree_util.tree_map(
                lambda x: x[perm][: num_batches * self.batch_size].reshape(
                    (num_batches, self.batch_size) + x.shape[1:]
                ),
                flat,
            )

            update_keys = jax.random.split(update_key, num_batches)

            def scan_update(cs2, xs):
                batch, uk = xs
                return _update(cs2, batch, uk)

            cs, m = jax.lax.scan(scan_update, cs, (batched, update_keys))
            cs = cs.replace(env_steps=cs.env_steps + num_envs * k * macro_window)
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

        skill0 = (
            jnp.zeros((num_eval_envs, num_skills), dtype=jnp.float32)
            if continuous_skill
            else jnp.zeros((num_eval_envs,), dtype=jnp.int32)
        )

        def body(carry, i):
            state, skill = carry
            controller_obs = state.obs
            new_skill = skill_mode(controller_state.actor_state.params, controller_obs)
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

        if continuous_skill:
            # ── Continuous skill usage: just log the per-dim std of choices made
            # at the reselection steps (no discrete histogram/entropy applies).
            skills_np = np.asarray(skills)              # (episode_length, num_eval_envs, K)
            chosen = skills_np[0::k].reshape(-1, num_skills)
            out["eval/skill_std_mean"] = float(np.mean(np.std(chosen, axis=0)))
        else:
            # ── Skill-usage distribution (choices made at the reselection steps) ──
            skills_np = np.asarray(skills)                 # (episode_length, num_eval_envs)
            chosen = skills_np[0::k].reshape(-1).astype(int)  # i % k == 0 -> a skill choice
            counts = np.bincount(chosen, minlength=num_skills).astype(np.float64)
            fracs = counts / max(counts.sum(), 1.0)
            nz = fracs > 0
            out["eval/skill_entropy"] = float(-(fracs[nz] * np.log(fracs[nz])).sum())
            out["eval/skill_max_frac"] = float(fracs.max())
            out["eval/skill_active_count"] = float((counts > 0).sum())
            # Route the histogram through the single per-step log_wandb call (via
            # the reserved "_wandb_media" key) instead of a separate same-step
            # wandb.log, which the live server would treat as a step collision and
            # drop the scalar metrics logged afterward (zeros in wandb). See
            # utils/env.py.
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
            return skill_mode(actor_params, obs_b)

        @jax.jit
        def act(state_b, skill_b, akey):
            return skill_action_eval(state_b, skill_b, akey)

        jit_reset = jax.jit(eval_env_base.reset)
        jit_step = jax.jit(eval_env_base.step)

        key, rk = jax.random.split(key)
        state = jit_reset(rk)
        rollout, xy_list, skill_list = [], [], []
        skill = (
            jnp.zeros((1, num_skills), dtype=jnp.float32)
            if continuous_skill
            else jnp.zeros((1,), dtype=jnp.int32)
        )
        for i in range(episode_length):
            rollout.append(state.pipeline_state)
            obs_b = state.obs[None]
            if i % k == 0:
                skill = pick(obs_b)
            key, ak = jax.random.split(key)
            a = act(obs_b[:, :state_size], skill, ak)
            obs_np = np.asarray(state.obs)
            xy_list.append(obs_np[prim_idx])
            skill_list.append(int(skill[0]) if not continuous_skill else np.asarray(skill[0]))
            state = jit_step(state, a[0])

        xy = np.stack(xy_list)
        goal_xy = np.asarray(state.obs)[state_size:][:2]

        # (1) brax 3D HTML
        try:
            sys = eval_env_base.sys.tree_replace({"opt.timestep": eval_env_base.dt})
            url = html.render(sys, rollout, height=1024)
            media["render/skill_html"] = wandb.Html(url)
        except Exception as e:  # rendering must never crash training
            logging.warning("skill HTML render failed: %s", e)

        # (2) skill-colored 2D trajectory (discrete skills only — the plot colors
        # segments by a discrete skill index, which has no continuous analogue).
        if not continuous_skill:
            try:
                skills = np.asarray(skill_list)
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
            z = skill_mode(actor_params, obs_b)
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
        controller_state, env_state, replay_state, raw_metrics = training_epoch(
            controller_state, env_state, replay_state, epoch_key
        )
        raw_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), raw_metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time

        current_step = int(controller_state.env_steps) + num_prefill_env_steps
        sps = (env_steps_per_macro * macro_window * blocks_per_epoch) / max(epoch_time, 1e-6)

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
