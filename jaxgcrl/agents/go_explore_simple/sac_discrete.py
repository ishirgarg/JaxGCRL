"""SAC-discrete machinery for the hierarchical skill controller.

This module implements the *high-level* controller used by
``GoExploreSimple(agent_type="sac_discrete")``: a discrete-action Soft
Actor-Critic (Christodoulou 2019, arXiv:1910.07207) that selects discrete
skills ``z`` to maximize task reward over a Semi-MDP (SMDP) whose actions are
fixed-``k`` skill commitments.

Design (see ``SKILL_CONTROLLER_DESIGN.md``):
  - Discrete actor: ``MLP(obs) -> logits[num_skills] -> distrax.Categorical``.
  - Twin Q critics: ``MLP(obs) -> vector[num_skills]`` (no action input);
    elementwise ``min`` over the two critics for clipped double-Q.
  - Exact soft value (no sampling):
        ``V(s) = Σ_z π(z|s)·(Q_min(s)[z] − α·log π(z|s))``.
  - Actor loss: ``J = E_s Σ_z π(z|s)·(α·log π(z|s) − Q_min(s)[z])``.
  - Critic target: ``y = R + γ·(1−done)·V_target(s')``.
  - Auto-tuned temperature α with target entropy ``H̄ = scale·log(num_skills)``.

The networks reuse the codebase ``Encoder`` MLP (``networks.py``) for layout
consistency. Everything here is pure / JIT-friendly.
"""

from __future__ import annotations

from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.agents.go_explore.networks import Encoder


# ── Networks ───────────────────────────────────────────────────────────────────


class DiscreteActorNet(nn.Module):
    """MLP mapping controller obs -> logits over ``num_skills``."""

    num_skills: int
    h_dim: int
    n_hidden: int
    skip_connections: int
    use_relu: bool
    use_ln: bool

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        return Encoder(
            repr_dim=self.num_skills,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(obs)


class DiscreteQNet(nn.Module):
    """Twin (or n-critic) vector-output Q network.

    Returns Q-values of shape ``(..., n_critics, num_skills)`` — one full vector
    of per-skill Q-values for each critic, with **no action input** (the discrete
    action indexes the output vector).
    """

    num_skills: int
    n_critics: int
    h_dim: int
    n_hidden: int
    skip_connections: int
    use_relu: bool
    use_ln: bool

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        qs = []
        for _ in range(self.n_critics):
            q = Encoder(
                repr_dim=self.num_skills,
                network_width=self.h_dim,
                network_depth=self.n_hidden,
                skip_connections=self.skip_connections,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
            )(obs)
            qs.append(q[..., None, :])  # (..., 1, num_skills)
        return jnp.concatenate(qs, axis=-2)  # (..., n_critics, num_skills)


# ── Controller training state ──────────────────────────────────────────────────


@dataclass
class ControllerState:
    """Training state for the SAC-discrete high-level controller."""

    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    target_critic_params: Any
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray


def create_controller_state(
    actor_net: DiscreteActorNet,
    critic_net: DiscreteQNet,
    obs_size: int,
    *,
    policy_lr: float,
    critic_lr: float,
    alpha_lr: float,
    actor_key: jax.Array,
    critic_key: jax.Array,
) -> ControllerState:
    dummy_obs = jnp.ones((1, obs_size), dtype=jnp.float32)
    actor_params = actor_net.init(actor_key, dummy_obs)
    critic_params = critic_net.init(critic_key, dummy_obs)

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=policy_lr),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=critic_lr),
    )
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
        tx=optax.adam(learning_rate=alpha_lr),
    )
    return ControllerState(
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        target_critic_params=critic_params,
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
    )


# ── Flat uniform replay buffer (i.i.d. SMDP transitions) ────────────────────────


@dataclass
class ControllerReplay:
    """Flat circular buffer of i.i.d. SMDP macro-transitions.

    A dedicated flat uniform buffer (rather than the trajectory-structured
    ``TrajectoryUniformSamplingQueue``) because controller transitions are
    i.i.d. ``(s, z, R, s', done)`` macro-steps with no future-relabel path.
    """

    obs: jnp.ndarray       # (capacity, obs_size)
    skill: jnp.ndarray     # (capacity,) int32
    reward: jnp.ndarray    # (capacity,)
    next_obs: jnp.ndarray  # (capacity, obs_size)
    done: jnp.ndarray      # (capacity,)
    ptr: jnp.ndarray       # () int32 — next write position
    size: jnp.ndarray      # () int32 — number of valid entries


def init_controller_replay(capacity: int, obs_size: int) -> ControllerReplay:
    return ControllerReplay(
        obs=jnp.zeros((capacity, obs_size), dtype=jnp.float32),
        skill=jnp.zeros((capacity,), dtype=jnp.int32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, obs_size), dtype=jnp.float32),
        done=jnp.zeros((capacity,), dtype=jnp.float32),
        ptr=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def insert_controller_replay(state: ControllerReplay, batch: Dict[str, jnp.ndarray]) -> ControllerReplay:
    """Insert a batch of ``B`` macro-transitions (circular overwrite)."""
    capacity = state.obs.shape[0]
    b = batch["obs"].shape[0]
    idx = (state.ptr + jnp.arange(b, dtype=jnp.int32)) % capacity
    return state.replace(
        obs=state.obs.at[idx].set(batch["obs"]),
        skill=state.skill.at[idx].set(batch["skill"].astype(jnp.int32)),
        reward=state.reward.at[idx].set(batch["reward"]),
        next_obs=state.next_obs.at[idx].set(batch["next_obs"]),
        done=state.done.at[idx].set(batch["done"]),
        ptr=(state.ptr + b) % capacity,
        size=jnp.minimum(state.size + b, capacity).astype(jnp.int32),
    )


def sample_controller_replay(state: ControllerReplay, key: jax.Array, batch_size: int) -> Dict[str, jnp.ndarray]:
    """Uniformly sample ``batch_size`` transitions from the valid region."""
    # Clamp high to >=1 so randint is valid before the first insert; callers
    # only sample once size >= min_replay_size, so this never biases training.
    high = jnp.maximum(state.size, 1)
    idx = jax.random.randint(key, (batch_size,), 0, high)
    return {
        "obs": state.obs[idx],
        "skill": state.skill[idx],
        "reward": state.reward[idx],
        "next_obs": state.next_obs[idx],
        "done": state.done[idx],
    }


# ── HER (hindsight) relabeling for controller macro-transitions ─────────────────


def her_relabel_sequence(
    seq: Dict[str, jnp.ndarray],
    key: jax.Array,
    *,
    state_size: int,
    goal_indices: jnp.ndarray,
    goal_reach_thresh: float,
    p_future_her_goal: float,
):
    """Uniform-future HER relabel of one env's macro-step sequence.

    ``seq`` holds the macro-transitions of a single env over one collected block,
    each field shaped ``(T, ...)``: ``obs=[state, goal]``, ``skill``, ``reward``,
    ``next_obs=[next_state, goal]``, ``done``, ``traj_id``. For each macro-step,
    with probability ``p_future_her_goal`` the goal is replaced by the achieved
    goal of a **uniformly** sampled *future* macro-step in the **same episode**
    (matched by ``traj_id``), and the reward is recomputed as the window-end
    goal-reach bit ``1[‖achieved − new_goal‖ < thresh]`` against the new goal.
    Mirrors the flat agent's HER (``go_explore.algorithms``), at the SMDP level.

    Returns the relabeled transition dict (``traj_id`` dropped) ready for the
    flat controller replay.
    """
    T = seq["obs"].shape[0]
    arrangement = jnp.arange(T)
    is_future = (arrangement[:, None] < arrangement[None]).astype(jnp.float32)
    tid = seq["traj_id"]
    same_traj = (tid[None, :] == tid[:, None]).astype(jnp.float32)
    # Uniform over same-episode future macro-steps; eye*1e-5 keeps positive mass
    # for rows with no valid future (last step of an episode -> keeps own goal).
    probs = is_future * same_traj + jnp.eye(T) * 1e-5

    sample_key, bern_key = jax.random.split(key)
    future_idx = jax.random.categorical(sample_key, jnp.log(probs))  # (T,)

    gi = jnp.asarray(goal_indices)
    future_next_state = seq["next_obs"][future_idx][:, :state_size]
    new_goal = future_next_state[:, gi]                      # (T, goal_size)

    state = seq["obs"][:, :state_size]
    next_state = seq["next_obs"][:, :state_size]
    relabeled_obs = jnp.concatenate([state, new_goal], axis=1)
    relabeled_next_obs = jnp.concatenate([next_state, new_goal], axis=1)

    achieved = next_state[:, gi]
    dist = jnp.linalg.norm(achieved - new_goal, axis=1)
    relabeled_reward = (dist < goal_reach_thresh).astype(jnp.float32)

    use_future = jax.random.bernoulli(bern_key, p=p_future_her_goal, shape=(T,))
    m = use_future[:, None]
    return {
        "obs": jnp.where(m, relabeled_obs, seq["obs"]),
        "skill": seq["skill"],
        "reward": jnp.where(use_future, relabeled_reward, seq["reward"]),
        "next_obs": jnp.where(m, relabeled_next_obs, seq["next_obs"]),
        "done": seq["done"],
    }


# ── SAC-discrete update ─────────────────────────────────────────────────────────


def _soft_update(target_params, online_params, tau: float):
    return jax.tree_util.tree_map(
        lambda t, o: t * (1 - tau) + o * tau, target_params, online_params
    )


def sac_discrete_update(
    controller_state: ControllerState,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    *,
    actor_net: DiscreteActorNet,
    critic_net: DiscreteQNet,
    gamma: float,
    tau: float,
    target_entropy: float,
):
    """One SAC-discrete gradient step.

    Order matches the standard discrete-SAC / existing SAC convention:
      1. update α (auto-temperature)
      2. update twin-Q toward the exact soft target
      3. update the discrete actor
      4. Polyak-update the target Q.
    α uses the OLD value (pre-update) for the critic and actor losses.
    """
    del key  # exact (no sampling): the discrete SAC update is deterministic.

    obs = batch["obs"]
    next_obs = batch["next_obs"]
    skill = batch["skill"].astype(jnp.int32)
    reward = batch["reward"]
    done = batch["done"]
    batch_idx = jnp.arange(obs.shape[0])

    actor_params = controller_state.actor_state.params
    critic_params = controller_state.critic_state.params
    target_critic_params = controller_state.target_critic_params
    old_log_alpha = controller_state.alpha_state.params["log_alpha"]
    old_alpha = jnp.exp(old_log_alpha)

    # ── Current policy distribution (used for α + actor losses) ──────────────
    logits = actor_net.apply(actor_params, obs)          # (B, K)
    log_pi = jax.nn.log_softmax(logits, axis=-1)         # (B, K)
    pi = jnp.exp(log_pi)                                  # (B, K)
    entropy = -jnp.sum(pi * log_pi, axis=-1)             # (B,)

    # ── 1. α update (auto-temperature). H̄ = scale·log(K) ────────────────────
    def alpha_loss_fn(alpha_params):
        alpha = jnp.exp(alpha_params["log_alpha"])
        # Matches the continuous-SAC sign convention in losses.py:
        #   continuous: -alpha * (log_prob + H̄);  discrete: substitute
        #   log_prob -> -entropy  =>  alpha * (entropy - H̄).
        return alpha * jnp.mean(jax.lax.stop_gradient(entropy - target_entropy))

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        controller_state.alpha_state.params
    )
    new_alpha_state = controller_state.alpha_state.apply_gradients(grads=alpha_grad)

    # ── 2. Critic update toward the exact soft target ────────────────────────
    next_logits = actor_net.apply(actor_params, next_obs)     # (B, K)
    next_log_pi = jax.nn.log_softmax(next_logits, axis=-1)
    next_pi = jnp.exp(next_log_pi)
    target_q = critic_net.apply(target_critic_params, next_obs)  # (B, n_critics, K)
    target_q_min = jnp.min(target_q, axis=-2)                    # (B, K)
    next_v = jnp.sum(
        next_pi * (target_q_min - old_alpha * next_log_pi), axis=-1
    )  # (B,)
    target = jax.lax.stop_gradient(reward + gamma * (1.0 - done) * next_v)  # (B,)

    def critic_loss_fn(c_params):
        q = critic_net.apply(c_params, obs)                  # (B, n_critics, K)
        # Gather Q for the taken skill across all critics -> (B, n_critics)
        q_taken = q[batch_idx, :, skill]
        sq_err = (q_taken - target[:, None]) ** 2
        return jnp.mean(sq_err)

    critic_loss_val, critic_grad = jax.value_and_grad(critic_loss_fn)(critic_params)
    new_critic_state = controller_state.critic_state.apply_gradients(grads=critic_grad)

    # ── 3. Actor update (exact, no sampling) ─────────────────────────────────
    def actor_loss_fn(a_params):
        logits_a = actor_net.apply(a_params, obs)
        log_pi_a = jax.nn.log_softmax(logits_a, axis=-1)
        pi_a = jnp.exp(log_pi_a)
        q = critic_net.apply(critic_params, obs)             # OLD critic params
        q_min = jnp.min(q, axis=-2)                          # (B, K)
        # J = E_s Σ_z π(z|s)·(α·log π(z|s) − Q_min(s)[z])
        per_state = jnp.sum(
            pi_a * (old_alpha * log_pi_a - jax.lax.stop_gradient(q_min)), axis=-1
        )
        return jnp.mean(per_state)

    actor_loss_val, actor_grad = jax.value_and_grad(actor_loss_fn)(actor_params)
    new_actor_state = controller_state.actor_state.apply_gradients(grads=actor_grad)

    # ── 4. Polyak target update ──────────────────────────────────────────────
    new_target_critic_params = _soft_update(
        target_critic_params, new_critic_state.params, tau
    )

    new_controller_state = controller_state.replace(
        actor_state=new_actor_state,
        critic_state=new_critic_state,
        alpha_state=new_alpha_state,
        target_critic_params=new_target_critic_params,
        gradient_steps=controller_state.gradient_steps + 1,
    )

    metrics = {
        "controller_alpha": old_alpha,
        "controller_alpha_loss": alpha_loss_val,
        "controller_critic_loss": critic_loss_val,
        "controller_actor_loss": actor_loss_val,
        "controller_entropy": jnp.mean(entropy),
        "controller_target_entropy": jnp.asarray(target_entropy, dtype=jnp.float32),
        "controller_reward_mean": jnp.mean(reward),
        "controller_done_mean": jnp.mean(done),
    }
    return new_controller_state, metrics
