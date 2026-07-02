"""Dynamics-Aware Discovery of Skills (DADS) training — future-state variant.

Reference: https://arxiv.org/abs/1907.01657

Key ideas
---------
* ``num_skills`` discrete skills.  A skill ``z`` is a **one-hot vector** of
  length ``num_skills``, sampled uniformly at the start of every episode and
  held fixed for the whole episode.
* The environment's *goal* part of the observation is **replaced** by ``z``,
  giving the policy/Q an augmented observation ``[state | z]``.
* A skill dynamics model ``q_phi(s+ | s, z)`` (diagonal Gaussian over the
  displacement ``delta = s+ - s``) models the **gamma-discounted future state**
  ``s+`` reached under skill ``z`` — NOT the one-step next state.  ``s+`` is
  sampled within the trajectory via a Geometric(1 - future_discount) offset
  (same machinery as CRL's HER future-state sampling).
* The intrinsic reward maximises the mutual information I(s+; z | s):
      r(s, z, s+) = log q(s+|s,z) - log[ (1/K) * Sum_{z'} q(s+|s,z') ]
  This is the per-state "empowerment" lower bound.
* A SAC agent maximises this intrinsic reward (the extrinsic reward is ignored);
  the critic still bootstraps on the one-step next state s'.
* The dynamics model is trained by maximum likelihood (NLL) on (s, z, s+).

After training, an empowerment heatmap over the visited x-y region is computed
(see ``eval.compute_and_log_empowerment_heatmap``) and logged to wandb, and the
full parameter set (policy + Q + discriminator + normalizers) is saved.
"""

import functools
import logging
import os
import pickle
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import losses as sac_losses
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.agents.common_agent_utils import _unpmap
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from . import networks as dads_networks
from . import eval as dads_eval
from . import losses as dads_losses

Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"

ReplayBufferState = Any


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single environment transition stored in the replay buffer.

    ``observation`` / ``next_observation`` are the *skill-augmented*
    observations ``[state | z]``.  The skill ``z`` is also stored in
    ``extras["state_extras"]["skill"]`` so the future state ``s+`` (sampled at
    train time) inherits the right skill and the dynamics model can be trained
    from the buffer.  ``future_state`` is added at sample time (see
    ``_dads_future_sample``); it is NOT stored in the buffer.
    """

    observation: Any
    next_observation: Any
    action: Any
    reward: Any
    discount: Any
    extras: Any = ()


# ---------------------------------------------------------------------------
# Geometric future-state sampling (per trajectory)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("buffer_config",))
def _dads_future_sample(buffer_config, transition, sample_key):
    """Attach a Geometric(1-gamma) future state s+ to each timestep.

    Operates on a single trajectory (vmapped over the env/trajectory axis),
    so ``transition.observation`` has shape ``(episode_len, obs_dim)``.  Mirrors
    CRL's ``flatten_batch`` future-state sampling: the future index is drawn from
    a categorical with probability proportional to ``gamma**(j-i)`` over strictly
    future timesteps that share the same ``traj_id`` (so s+ never crosses an
    episode boundary, keeping the skill consistent).  The last timestep is
    dropped (it has no future).
    """
    gamma, state_size = buffer_config
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    same_traj = jnp.equal(single_trajectories, single_trajectories.T)
    # A timestep is "valid" iff it has at least one strictly-future timestep in
    # the same episode. Terminal steps have none; the eye*1e-5 fallback then
    # makes the sampler pick the step itself (s+ == s, delta == 0). We flag
    # those so the loss/reward can mask them instead of training on delta=0.
    future_valid = (jnp.sum(is_future_mask * same_traj, axis=-1) > 0).astype(jnp.float32)
    probs = probs * same_traj + jnp.eye(seq_len) * 1e-5

    future_index = jax.random.categorical(sample_key, jnp.log(probs))  # (seq_len,)
    future_state = jnp.take(transition.observation, future_index[:-1], axis=0)[:, :state_size]

    state_extras_in = transition.extras["state_extras"]
    new_extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": state_extras_in["truncation"][:-1],
            "traj_id": state_extras_in["traj_id"][:-1],
            "skill": state_extras_in["skill"][:-1],
        },
        "future_state": future_state,
        "future_valid": future_valid[:-1],
    }
    return transition._replace(
        observation=transition.observation[:-1],
        next_observation=transition.next_observation[:-1],
        action=transition.action[:-1],
        reward=transition.reward[:-1],
        discount=transition.discount[:-1],
        extras=new_extras,
    )


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Full training state for the DADS learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    dynamics_optimizer_state: optax.OptState
    dynamics_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    # Running statistics for the skill-dynamics input [s|z] and the delta
    # target (s+ - s).  Persisted so numerator/denominator use the same
    # normalization and the post-training empowerment map is well-defined.
    dyn_input_normalizer_params: running_statistics.RunningStatisticsState
    dyn_delta_normalizer_params: running_statistics.RunningStatisticsState


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    dyn_input_size: int,
    dyn_delta_size: int,
    local_devices_to_use: int,
    dads_network: dads_networks.DADSNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    dynamics_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Initialise a replicated TrainingState."""
    key_policy, key_q, key_dyn = jax.random.split(key, 3)

    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = dads_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    q_params = dads_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    dynamics_params = dads_network.skill_dynamics_network.init(key_dyn)
    dynamics_optimizer_state = dynamics_optimizer.init(dynamics_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype("float32"))
    )
    dyn_input_normalizer_params = running_statistics.init_state(
        specs.Array((dyn_input_size,), jnp.dtype("float32"))
    )
    dyn_delta_normalizer_params = running_statistics.init_state(
        specs.Array((dyn_delta_size,), jnp.dtype("float32"))
    )

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        dynamics_optimizer_state=dynamics_optimizer_state,
        dynamics_params=dynamics_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        dyn_input_normalizer_params=dyn_input_normalizer_params,
        dyn_delta_normalizer_params=dyn_delta_normalizer_params,
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


# ---------------------------------------------------------------------------
# DADS agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class DADS:
    """Dynamics-Aware Discovery of Skills (DADS) agent (future-state variant).

    Args:
        num_skills:        Number of discrete skills |Z|.
        future_discount:   Geometric discount for sampling the future state s+.
        learning_rate:     LR for policy, Q-network, and dynamics model.
        discounting:       SAC discount factor gamma.
        batch_size:        Per-update minibatch size (across all devices).
        normalize_observations: Normalise skill-augmented observations.
        reward_scaling:    Scalar multiplier on the intrinsic reward.
        tau:               Soft target-network update rate.
        min_replay_size:   Min transitions before training starts.
        max_replay_size:   Max replay buffer capacity.
        deterministic_eval:Deterministic actions during evaluation.
        train_step_multiplier: Gradient-update passes per environment step.
        unroll_length:     Environment steps collected between updates.
        h_dim, n_hidden:   MLP width / depth for all networks.
        use_ln:            Use layer normalisation in MLPs.
        use_xy_prior:      Restrict the discriminator to x-y (goal_indices)
                           deltas.  Default False -> models the full state.
        emp_grid_spacing:  Heatmap cell size (world units) for the empowerment map.
        emp_num_future_samples: # of s+ samples per (cell, skill) at eval.
        emp_rollout_horizon: Rollout length per (cell, skill); 0 -> auto
                           = ceil(3 / (1 - future_discount)).
        emp_collect_envs:  # parallel envs used to collect visited states.
        emp_max_cells:     Cap on the number of occupied cells evaluated.
        compute_empowerment_map: Whether to compute+log the heatmap at the end.
    """

    num_skills: int = 15
    future_discount: float = 0.99
    learning_rate: float = 3e-4
    discounting: float = 0.99
    batch_size: int = 256
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10_000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    use_ln: bool = False
    use_xy_prior: bool = False

    # ── Empowerment-map (post-training) hyperparameters ────────────────────
    emp_grid_spacing: float = 0.25
    emp_num_future_samples: int = 16
    emp_rollout_horizon: int = 0
    emp_collect_envs: int = 256
    emp_max_cells: int = 4000
    compute_empowerment_map: bool = True

    # ------------------------------------------------------------------
    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        """Run DADS training."""
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use, device_count,
        )

        num_skills: int = self.num_skills
        if num_skills < 2:
            raise ValueError(
                f"num_skills must be >= 2 (got {num_skills}); with K=1 the skill "
                "marginal equals the numerator and the intrinsic reward is identically 0."
            )
        # The minibatch reshape in train_steps requires this to be exact.
        if (config.num_envs * (config.episode_length - 1)) % self.batch_size != 0:
            raise ValueError(
                f"num_envs * (episode_length - 1) = "
                f"{config.num_envs * (config.episode_length - 1)} must be divisible by "
                f"batch_size = {self.batch_size}."
            )
        unwrapped_env = train_env
        state_dim: int = unwrapped_env.state_dim
        goal_indices = getattr(unwrapped_env, "goal_indices", None)

        if self.use_xy_prior and goal_indices is None:
            raise ValueError("use_xy_prior=True requires the env to expose goal_indices.")

        if self.use_xy_prior:
            goal_indices_list = goal_indices.tolist() if hasattr(goal_indices, "tolist") else list(goal_indices)
            non_goal_indices = jnp.array([i for i in range(state_dim) if i not in goal_indices_list])
            dyn_input_size = (state_dim - len(goal_indices_list)) + num_skills
            dyn_delta_size = len(goal_indices_list)
        else:
            non_goal_indices = None
            dyn_input_size = state_dim + num_skills
            dyn_delta_size = state_dim

        # Augmented observation seen by policy and Q-network.
        dads_obs_size: int = state_dim + num_skills

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError("min_replay_size >= total_env_steps; no training would happen.")
        max_replay_size = (
            config.total_env_steps if self.max_replay_size is None else self.max_replay_size
        )

        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
        # The buffer samples episode_length-long windows per env, so it must hold
        # at least episode_length timesteps before the first sample; otherwise
        # early samples read uninitialized (zero, traj_id=0) regions. Ensure the
        # prefill covers that regardless of min_replay_size.
        num_prefill_actor_steps = max(
            self.min_replay_size // self.unroll_length + 1,
            -(-config.episode_length // self.unroll_length),
        )
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        assert config.num_envs % device_count == 0
        # Needed so the per-device minibatch reshape (batch_size // device_count)
        # combined with the global divisibility check above is exact per device.
        assert self.batch_size % device_count == 0
        # The per-device buffer samples episode_length-long windows, so it must be
        # able to hold at least one full episode window per device.
        assert (max_replay_size // device_count) >= config.episode_length + 1, (
            f"max_replay_size // device_count ({max_replay_size // device_count}) must "
            f">= episode_length + 1 ({config.episode_length + 1})."
        )
        num_envs_per_device = config.num_envs // device_count

        # ------------------------------------------------------------------
        # Environment setup
        # ------------------------------------------------------------------
        env = train_env
        if isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, config.num_envs // jax.process_count() // local_devices_to_use
                ),
            )

        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        action_size = env.action_size

        # ------------------------------------------------------------------
        # Networks
        # ------------------------------------------------------------------
        normalize_fn = (lambda x, y: x)
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        dads_network = dads_networks.make_dads_networks(
            observation_size=dads_obs_size,
            action_size=action_size,
            state_size=state_dim,
            num_skills=num_skills,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
            layer_norm=self.use_ln,
            use_xy_prior=self.use_xy_prior,
            goal_indices=goal_indices,
        )
        make_policy = dads_networks.make_inference_fn(dads_network)

        # ------------------------------------------------------------------
        # Optimisers
        # ------------------------------------------------------------------
        alpha_optimizer = optax.adam(learning_rate=3e-4)
        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)
        dynamics_optimizer = optax.adam(learning_rate=self.learning_rate)

        # ------------------------------------------------------------------
        # Replay buffer
        # ------------------------------------------------------------------
        dummy_obs = jnp.zeros((dads_obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            next_observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    "skill": jnp.zeros((num_skills,)),
                },
                "policy_extras": {},
            },
        )
        # The buffer is per-device (max_replay_size and sample_batch_size are
        # divided by device_count), and each device only collects/inserts
        # num_envs_per_device envs, so num_envs must be per-device too. Using
        # config.num_envs here would leave (device_count-1)/device_count of every
        # sampled trajectory zero-filled under multi-device.
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=num_envs_per_device,
                episode_length=config.episode_length,
            )
        )

        # ------------------------------------------------------------------
        # SAC losses (generic; we replace transitions.reward with DADS reward)
        # ------------------------------------------------------------------
        alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
            sac_network=dads_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
            action_size=action_size,
        )
        alpha_update = gradients.gradient_update_fn(
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        critic_update = gradients.gradient_update_fn(
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        actor_update = gradients.gradient_update_fn(
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        dynamics_update = dads_losses.make_skill_dynamics_update_fn(
            dads_network, state_dim, dynamics_optimizer,
            use_xy_prior=self.use_xy_prior, goal_indices=goal_indices,
            non_goal_indices=non_goal_indices,
        )

        # ------------------------------------------------------------------
        # Update step (one minibatch)
        # ------------------------------------------------------------------
        def update_step(
            carry: Tuple[TrainingState, PRNGKey],
            transitions: Transition,
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry
            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            # ---- 0. Update dynamics input/delta running normalizers --------
            states = transitions.observation[:, :state_dim]
            future_states = transitions.extras["future_state"]
            skills = transitions.extras["state_extras"]["skill"]
            if self.use_xy_prior:
                dyn_input = jnp.concatenate(
                    [jnp.take(states, non_goal_indices, axis=1), skills], axis=-1
                )
                delta = future_states[:, goal_indices] - states[:, goal_indices]
            else:
                dyn_input = jnp.concatenate([states, skills], axis=-1)
                delta = future_states - states

            # Terminal rows have delta == 0 (s+ degenerates to s). Replace them
            # with the batch mean of valid deltas before updating the delta
            # running stats, so the normalization isn't biased toward 0 by the
            # spurious zeros. (dyn_input is fine: terminal states are real.)
            future_valid = transitions.extras["future_valid"]              # (B,)
            valid_count = jnp.maximum(jnp.sum(future_valid), 1.0)
            valid_delta_mean = jnp.sum(delta * future_valid[:, None], axis=0) / valid_count
            delta_for_norm = jnp.where(future_valid[:, None] > 0, delta, valid_delta_mean[None, :])

            dyn_input_norm = running_statistics.update(
                training_state.dyn_input_normalizer_params, dyn_input,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            dyn_delta_norm = running_statistics.update(
                training_state.dyn_delta_normalizer_params, delta_for_norm,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            # ---- 1. DADS intrinsic reward ---------------------------------
            dads_reward = dads_losses.compute_dads_reward(
                dads_network, state_dim, num_skills, training_state.dynamics_params,
                transitions, dyn_input_norm, dyn_delta_norm,
                use_xy_prior=self.use_xy_prior, goal_indices=goal_indices,
                non_goal_indices=non_goal_indices,
            )
            per_sample_nll = dads_losses.compute_per_sample_dynamics_nll(
                dads_network, state_dim, training_state.dynamics_params, transitions,
                dyn_input_norm, dyn_delta_norm,
                use_xy_prior=self.use_xy_prior, goal_indices=goal_indices,
                non_goal_indices=non_goal_indices,
            )
            transitions = transitions._replace(reward=dads_reward)

            # ---- 2. SAC updates -------------------------------------------
            alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params, training_state.policy_params,
                training_state.normalizer_params, transitions, key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
            alpha = jnp.exp(training_state.alpha_params)

            critic_loss_val, q_params, q_optimizer_state = critic_update(
                training_state.q_params, training_state.policy_params,
                training_state.normalizer_params, training_state.target_q_params,
                alpha, transitions, key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )
            actor_loss_val, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params, training_state.normalizer_params,
                training_state.q_params, alpha, transitions, key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )
            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params, q_params,
            )

            # ---- 3. Dynamics model update ---------------------------------
            dyn_loss_val, dynamics_params, dynamics_optimizer_state = dynamics_update(
                training_state.dynamics_params, transitions,
                dyn_input_norm, dyn_delta_norm,
                optimizer_state=training_state.dynamics_optimizer_state,
            )

            # ---- 4. Per-skill metrics -------------------------------------
            # The DADS reward (and a high-likelihood NLL) are routinely negative,
            # so validity must be tracked by an explicit sample count, not by the
            # sign of the value. Metrics are computed over VALID rows only
            # (future_valid) so they match the masked trained loss/reward.
            skill_indices = jnp.argmax(skills, axis=-1)
            per_skill_metrics = {}
            per_skill_nll_list, per_skill_reward_list, per_skill_has = [], [], []
            for skill_idx in range(num_skills):
                w = (skill_indices == skill_idx).astype(jnp.float32) * future_valid
                wsum = jnp.sum(w)
                has = wsum > 0
                denom = jnp.maximum(wsum, 1.0)
                skill_nll = jnp.where(has, jnp.sum(per_sample_nll * w) / denom, 0.0)
                skill_reward = jnp.where(has, jnp.sum(dads_reward * w) / denom, 0.0)
                # Logged per-skill value is 0.0 for an empty skill (NaN-safe; the
                # MetricsRecorder NaN-guard would otherwise raise). Empty skills
                # are excluded from the cross-skill mean below via has_arr.
                per_skill_metrics[f"skill_{skill_idx}_training/dynamics_loss"] = skill_nll
                per_skill_metrics[f"skill_{skill_idx}_training/dads_reward"] = skill_reward
                per_skill_nll_list.append(skill_nll)
                per_skill_reward_list.append(skill_reward)
                per_skill_has.append(has.astype(jnp.float32))

            nll_arr = jnp.array(per_skill_nll_list)
            reward_arr = jnp.array(per_skill_reward_list)
            has_arr = jnp.array(per_skill_has)
            n_has = jnp.maximum(jnp.sum(has_arr), 1.0)
            mean_metrics = {
                "mean_training/dynamics_loss": jnp.sum(nll_arr * has_arr) / n_has,
                "mean_training/dads_reward": jnp.sum(reward_arr * has_arr) / n_has,
            }

            metrics = {
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "alpha": jnp.exp(alpha_params),
                "dynamics_loss": dyn_loss_val,
                "dads_reward": jnp.mean(dads_reward),
                **per_skill_metrics,
                **mean_metrics,
            }

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                dynamics_optimizer_state=dynamics_optimizer_state,
                dynamics_params=dynamics_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
                normalizer_params=training_state.normalizer_params,
                dyn_input_normalizer_params=dyn_input_norm,
                dyn_delta_normalizer_params=dyn_delta_norm,
            )
            return (new_training_state, key), metrics

        # ------------------------------------------------------------------
        # Experience collection
        # ------------------------------------------------------------------
        def get_experience(
            normalizer_params, policy_params, env_state, current_skills, buffer_state, key,
        ):
            policy = make_policy((normalizer_params, policy_params))

            @jax.jit
            def f(carry, unused_t):
                env_state, skills, current_key = carry
                current_key, next_key, skill_key = jax.random.split(current_key, 3)

                state_part = env_state.obs[:, :state_dim]
                aug_obs = jnp.concatenate([state_part, skills], axis=-1)
                actions, policy_extras = policy(aug_obs, current_key)
                nstate = env.step(env_state, actions)
                next_state_part = nstate.obs[:, :state_dim]
                aug_next_obs = jnp.concatenate([next_state_part, skills], axis=-1)

                state_extras = {
                    "truncation": nstate.info["truncation"],
                    "traj_id": nstate.info["traj_id"],
                    "skill": skills,
                }
                transition = Transition(
                    observation=aug_obs,
                    next_observation=aug_next_obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"policy_extras": policy_extras, "state_extras": state_extras},
                )
                # Resample skills for envs that terminated.
                new_skill_idx = jax.random.randint(skill_key, (skills.shape[0],), 0, num_skills)
                new_skills = jax.nn.one_hot(new_skill_idx, num_skills, dtype=jnp.float32)
                next_skills = jnp.where(nstate.done[:, None], new_skills, skills)
                return (nstate, next_skills, next_key), transition

            (env_state, current_skills, _), data = jax.lax.scan(
                f, (env_state, current_skills, key), (), length=self.unroll_length
            )
            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data).observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, current_skills, buffer_state

        # ------------------------------------------------------------------
        # Gradient updates from the buffer (with s+ future sampling)
        # ------------------------------------------------------------------
        def train_steps(training_state, buffer_state, key):
            sampling_key, perm_key, training_key = jax.random.split(key, 3)
            buffer_state, trajectories = replay_buffer.sample(buffer_state)
            # trajectories: (num_envs, episode_length, ...)

            batch_keys = jax.random.split(sampling_key, trajectories.observation.shape[0])
            transitions = jax.vmap(_dads_future_sample, in_axes=(None, 0, 0))(
                (self.future_discount, state_dim), trajectories, batch_keys,
            )
            # -> (num_envs, episode_length-1, ...) with extras.future_state
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            batch_size_per_device = self.batch_size // device_count
            permutation = jax.random.permutation(perm_key, transitions.observation.shape[0])
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, batch_size_per_device) + x.shape[1:]), transitions
            )
            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def training_step(training_state, env_state, current_skills, buffer_state, key):
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, current_skills, buffer_state = get_experience(
                training_state.normalizer_params, training_state.policy_params,
                env_state, current_skills, buffer_state, experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            training_state, buffer_state, metrics = train_steps(
                training_state, buffer_state, training_key
            )
            return training_state, env_state, current_skills, buffer_state, metrics

        def prefill_replay_buffer(training_state, env_state, current_skills, buffer_state, key):
            def f(carry, unused):
                del unused
                training_state, env_state, current_skills, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                new_normalizer_params, env_state, current_skills, buffer_state = get_experience(
                    training_state.normalizer_params, training_state.policy_params,
                    env_state, current_skills, buffer_state, key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, current_skills, buffer_state, new_key), ()

            return jax.lax.scan(
                f, (training_state, env_state, current_skills, buffer_state, key),
                (), length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, _unused):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics
            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        def training_epoch(training_state, env_state, current_skills, buffer_state, key):
            def f(carry, unused_t):
                ts, es, skills, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, skills, bs, metrics = training_step(ts, es, skills, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(
                    self.train_step_multiplier - 1, ts, bs, update_key
                )
                return (ts, es, skills, bs, new_key), metrics

            (training_state, env_state, current_skills, buffer_state, key), metrics = jax.lax.scan(
                f, (training_state, env_state, current_skills, buffer_state, key),
                (), length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, current_skills, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        def training_epoch_with_timing(training_state, env_state, current_skills, buffer_state, key):
            nonlocal training_walltime
            t = time.time()
            (training_state, env_state, current_skills, buffer_state, metrics) = training_epoch(
                training_state, env_state, current_skills, buffer_state, key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics_dict = dict(metrics)
            final_metrics = {"training/sps": sps, "training/walltime": training_walltime}
            for name, value in metrics_dict.items():
                final_metrics[name if "/" in name else f"training/{name}"] = value
            return training_state, env_state, current_skills, buffer_state, final_metrics

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        training_state = _init_training_state(
            key=global_key, obs_size=dads_obs_size,
            dyn_input_size=dyn_input_size, dyn_delta_size=dyn_delta_size,
            local_devices_to_use=local_devices_to_use, dads_network=dads_network,
            alpha_optimizer=alpha_optimizer, policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer, dynamics_optimizer=dynamics_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key, skill_key = jax.random.split(local_key, 5)
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        skill_keys = jax.random.split(skill_key, local_devices_to_use)

        def _init_skills(key):
            idx = jax.random.randint(key, (num_envs_per_device,), 0, num_skills)
            return jax.nn.one_hot(idx, num_skills, dtype=jnp.float32)

        current_skills = jax.pmap(_init_skills)(skill_keys)
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        # ------------------------------------------------------------------
        # Evaluation setup
        # ------------------------------------------------------------------
        if not eval_env:
            eval_env = train_env
        v_randomization_fn_eval = None
        if randomization_fn is not None:
            v_randomization_fn_eval = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
            )
        eval_env_wrapped = TrajectoryIdWrapper(eval_env)
        eval_env_wrapped = wrap_for_training(
            eval_env_wrapped, episode_length=config.episode_length,
            action_repeat=config.action_repeat, randomization_fn=v_randomization_fn_eval,
        )
        eval_make_policy = dads_eval.make_dads_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=0
        )

        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            params = _unpmap((training_state.normalizer_params, training_state.policy_params))
            metrics = dads_eval.run_multi_skill_evaluation(
                make_policy, state_dim, num_skills, eval_env_wrapped,
                self.deterministic_eval, config.num_eval_envs, config.episode_length,
                config.action_repeat, eval_key, params, training_metrics={},
            )
            progress_fn(0, metrics, eval_make_policy, params, unwrapped_env, False)

        # ------------------------------------------------------------------
        # Prefill replay buffer
        # ------------------------------------------------------------------
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, current_skills, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, current_skills, buffer_state, prefill_keys
        )
        replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        logging.info("replay size after prefill %s", replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        # ------------------------------------------------------------------
        # Main training loop
        # ------------------------------------------------------------------
        current_step = 0
        last_checkpoint_step = 0
        checkpoint_interval = 5_000_000

        for eval_epoch_num in range(num_evals_after_init):
            logging.info("step %s", current_step)
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, current_skills, buffer_state, training_metrics) = (
                training_epoch_with_timing(
                    training_state, env_state, current_skills, buffer_state, epoch_keys
                )
            )
            current_step = int(_unpmap(training_state.env_steps))

            if process_id == 0:
                params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                if config.checkpoint_logdir and (current_step - last_checkpoint_step >= checkpoint_interval):
                    path = f"{config.checkpoint_logdir}_dads_{current_step}.pkl"
                    ckpt_dir = os.path.dirname(path)
                    if ckpt_dir:
                        os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_params(path, params)
                    logging.info("Saved checkpoint at step %s", current_step)
                    last_checkpoint_step = current_step

                metrics = dads_eval.run_multi_skill_evaluation(
                    make_policy, state_dim, num_skills, eval_env_wrapped,
                    self.deterministic_eval, config.num_eval_envs, config.episode_length,
                    config.action_repeat, eval_key, params, training_metrics,
                )
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                if do_render:
                    render_dir = config.checkpoint_logdir if config.checkpoint_logdir else "./runs"
                    os.makedirs(render_dir, exist_ok=True)
                    try:
                        dads_eval.render_all_skills(
                            make_policy, state_dim, num_skills, params, unwrapped_env,
                            render_dir, config.exp_name, current_step,
                        )
                    except Exception:  # never let rendering kill training
                        logging.exception("render_all_skills failed; continuing")
                merged_metrics = {**training_metrics, **metrics}
                progress_fn(current_step, merged_metrics, eval_make_policy, params, unwrapped_env, False)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # ------------------------------------------------------------------
        # Save FULL parameter set (policy + Q + discriminator + normalizers)
        # for reproducibility and offline empowerment recomputation.
        # ------------------------------------------------------------------
        if process_id == 0:
            ts0 = _unpmap(training_state)
            full_state = {
                "normalizer_params": ts0.normalizer_params,
                "policy_params": ts0.policy_params,
                "q_params": ts0.q_params,
                "target_q_params": ts0.target_q_params,
                "dynamics_params": ts0.dynamics_params,
                "alpha_params": ts0.alpha_params,
                "dyn_input_normalizer_params": ts0.dyn_input_normalizer_params,
                "dyn_delta_normalizer_params": ts0.dyn_delta_normalizer_params,
            }
            ball_spawn_xy = None
            if hasattr(unwrapped_env, "possible_balls"):
                ball_spawn_xy = [float(v) for v in jax.device_get(unwrapped_env.possible_balls[0])]
            metadata = {
                "agent": "DADS",
                "env": config.env,
                "num_skills": num_skills,
                "future_discount": self.future_discount,
                "discounting": self.discounting,
                "state_dim": state_dim,
                "goal_indices": (goal_indices.tolist() if goal_indices is not None else None),
                "use_xy_prior": self.use_xy_prior,
                "h_dim": self.h_dim,
                "n_hidden": self.n_hidden,
                "use_ln": self.use_ln,
                "normalize_observations": self.normalize_observations,
                "ball_spawn_xy": ball_spawn_xy,
                "total_env_steps": total_steps,
                "seed": config.seed,
            }
            if config.checkpoint_logdir:
                full_path = f"{config.checkpoint_logdir}_dads_full_{total_steps}.pkl"
            else:
                full_path = f"./runs/run_{config.exp_name}_s_{config.seed}/dads_full_{total_steps}.pkl"
            _full_dir = os.path.dirname(full_path)
            if _full_dir:
                os.makedirs(_full_dir, exist_ok=True)
            with open(full_path, "wb") as fout:
                pickle.dump({"params": jax.device_get(full_state), "metadata": metadata}, fout)
            logging.info("Saved full DADS parameter set to %s", full_path)

            # ----------------------------------------------------------------
            # Empowerment heatmap over the visited x-y region.
            # ----------------------------------------------------------------
            if self.compute_empowerment_map:
                try:
                    if self.emp_rollout_horizon:
                        rollout_horizon = self.emp_rollout_horizon
                    elif self.future_discount < 1.0:
                        rollout_horizon = int(jnp.ceil(3.0 / (1.0 - self.future_discount)))
                    else:
                        rollout_horizon = config.episode_length
                    dads_eval.compute_and_log_empowerment_heatmap(
                        dads_network=dads_network,
                        make_policy=make_policy,
                        unwrapped_env=unwrapped_env,
                        params=full_state,
                        state_dim=state_dim,
                        num_skills=num_skills,
                        future_discount=self.future_discount,
                        goal_indices=goal_indices,
                        non_goal_indices=non_goal_indices,
                        use_xy_prior=self.use_xy_prior,
                        grid_spacing=self.emp_grid_spacing,
                        rollout_horizon=rollout_horizon,
                        num_future_samples=self.emp_num_future_samples,
                        collect_envs=self.emp_collect_envs,
                        collect_steps=config.episode_length,
                        max_cells=self.emp_max_cells,
                        ball_spawn_xy=ball_spawn_xy,
                        exp_name=config.exp_name,
                        step=total_steps,
                        seed=config.seed,
                        save_dir=os.path.dirname(full_path),
                    )
                except Exception as exc:  # never crash training over the heatmap
                    logging.exception("Empowerment heatmap computation failed: %s", exc)

        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return eval_make_policy, params, metrics
