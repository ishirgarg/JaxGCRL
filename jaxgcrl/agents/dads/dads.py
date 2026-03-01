"""Dynamics-Aware Discovery of Skills (DADS) training.

Reference: https://arxiv.org/abs/1907.01657

Key ideas
---------
* A set of ``num_skills`` discrete skills is defined.  A skill ``z`` is
  represented as a **one-hot vector** of length ``num_skills``.
* At the start of every episode a skill is sampled uniformly: ``z ~ p(z)``.
  The skill remains fixed for the entire episode.
* The environment's *goal* part of the observation is **replaced** by ``z``,
  giving the agent an augmented observation ``[state | z]`` of dimension
  ``state_dim + num_skills``.
* A skill dynamics model ``q_phi(s' | s, z)`` is a diagonal Gaussian that
  predicts ``delta_s = s' - s`` given the current state and skill.
* The intrinsic reward maximises the mutual information I(s'; z | s):
      r(s, z, s') = log q(s'|s,z) - log[ (1/K) * Σ_{z'} q(s'|s,z') ]
  where the denominator marginalises over all K skills.
* A SAC agent is trained to maximise this intrinsic reward.  The environment's
  extrinsic reward is entirely ignored.
* The dynamics model is trained by maximum likelihood (NLL loss).
"""

import functools
import logging
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.agents.sac import losses as sac_losses
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, Policy, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from . import networks as dads_networks
from . import eval as dads_eval
from . import losses as dads_losses

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

_PMAP_AXIS_NAME = "i"

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Container for a single environment transition stored in the replay buffer.

    ``observation`` and ``next_observation`` are the *skill-augmented*
    observations ``[state | z]`` of dimension ``state_dim + num_skills``.
    The skill ``z`` is also stored in ``extras["state_extras"]["skill"]``
    so that the dynamics model can be trained without re-augmenting.
    """

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
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
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


# ---------------------------------------------------------------------------
# DADS agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class DADS:
    """Dynamics-Aware Discovery of Skills (DADS) agent.

    Trains a SAC policy with an intrinsic reward derived from a skill dynamics
    model.  The model ``q_phi(s' | s, z)`` is a diagonal Gaussian predicting
    ``delta_s = s' - s`` given current state and skill.  The reward maximises
    the mutual information I(s'; z | s):

        r(s, z, s') = log q(s'|s,z) - log[ (1/K) * Σ_{z'} q(s'|s,z') ]

    The environment's extrinsic reward is ignored; the goal part of the
    observation is replaced by a one-hot skill vector, exactly as in DIAYN.

    Args:
        num_skills:        Number of discrete skills ``|Z|``.
        learning_rate:     Learning rate for policy, Q-network, and dynamics model.
        discounting:       Discount factor γ.
        batch_size:        Total training batch size (summed across all devices).
        normalize_observations: Whether to normalise skill-augmented observations.
        reward_scaling:    Scalar multiplier on the intrinsic reward.
        tau:               Soft target-network update rate.
        min_replay_size:   Minimum transitions in buffer before training starts.
        max_replay_size:   Maximum replay buffer capacity (per device).
        deterministic_eval: Use deterministic actions during evaluation.
        train_step_multiplier: Gradient updates per environment step.
        unroll_length:     Environment steps collected between updates.
        h_dim:             Hidden layer width for all MLPs.
        n_hidden:          Number of hidden layers for all MLPs.
        use_ln:            Use layer normalisation in MLPs.
        use_xy_prior:      If True, restrict dynamics network to only predict goal_indices
                           differences (e.g., x-y) and exclude goal_indices from input.
                           Requires environment to have goal_indices.
    """

    num_skills: int = 8
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
        """Run DADS training.

        Parameters
        ----------
        config : RunConfig
            Top-level run configuration (num_envs, total_env_steps, …).
        train_env : Env
            Training environment.  The goal part of its observations will be
            replaced by a one-hot skill vector.
        eval_env : Env, optional
            Evaluation environment.  Same replacement happens here.
        randomization_fn : callable, optional
            Domain-randomisation function (passed to brax wrappers).
        progress_fn : callable
            Called after each evaluation epoch with
            ``(step, metrics, make_policy, params, unwrapped_env, do_render)``.
        """
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use,
            device_count,
        )

        # ------------------------------------------------------------------
        # Derived constants
        # ------------------------------------------------------------------
        num_skills: int = self.num_skills

        # state_dim: dimensionality of the raw environment state (no goal/skill)
        unwrapped_env = train_env
        state_dim: int = unwrapped_env.state_dim
        goal_indices = getattr(unwrapped_env, "goal_indices", None)

        if self.use_xy_prior and goal_indices is None:
            raise ValueError(
                "use_xy_prior=True requires the environment to have goal_indices. "
                "The environment does not have this attribute."
            )

        # Precompute non-goal indices once (outside JIT) for use_xy_prior
        if self.use_xy_prior:
            goal_indices_list = goal_indices.tolist() if hasattr(goal_indices, 'tolist') else list(goal_indices)
            non_goal_indices = jnp.array([i for i in range(state_dim) if i not in goal_indices_list])
        else:
            non_goal_indices = None

        # Augmented observation size seen by the policy and Q-network
        dads_obs_size: int = state_dim + num_skills

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError(
                "No training will happen because min_replay_size >= total_env_steps"
            )
        max_replay_size = (
            config.total_env_steps if self.max_replay_size is None else self.max_replay_size
        )

        env_steps_per_actor_step = (
            config.action_repeat * config.num_envs * self.unroll_length
        )
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        logging.info("num_prefill_actor_steps: %s", num_prefill_actor_steps)
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        assert config.num_envs % device_count == 0
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
        def normalize_fn(x, y):
            return x

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
                    # One-hot skill vector stored alongside each transition so
                    # the dynamics model can be trained from the replay buffer.
                    "skill": jnp.zeros((num_skills,)),
                },
                "policy_extras": {},
            },
        )
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        # The brax SAC losses are generic: they use transitions.reward /
        # transitions.observation / etc.  We replace transitions.reward with
        # the DADS intrinsic reward inside update_step before calling them.
        alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
            sac_network=dads_network,  # duck-typed: has policy_network, q_network, parametric_action_distribution
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
            use_xy_prior=self.use_xy_prior,
            goal_indices=goal_indices,
            non_goal_indices=non_goal_indices,
        )

        # ------------------------------------------------------------------
        # Update step
        # ------------------------------------------------------------------
        def update_step(
            carry: Tuple[TrainingState, PRNGKey],
            transitions: Transition,
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry
            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            # ---- 1. Compute DADS intrinsic reward ------------------------
            # r(s, z, s') = log q(s'|s,z) - log[(1/K) Σ_{z'} q(s'|s,z')]
            dads_reward = dads_losses.compute_dads_reward(
                dads_network,
                state_dim,
                num_skills,
                training_state.dynamics_params,
                transitions,
                use_xy_prior=self.use_xy_prior,
                goal_indices=goal_indices,
                non_goal_indices=non_goal_indices,
            )

            # Also compute per-sample log q(s'|s,z) for the actual skill
            # (used for per-skill NLL metrics).
            # Reuse the same preprocessing logic as the loss function
            delta_s, dynamics_input, _, _ = dads_losses._prepare_dynamics_inputs(
                transitions, state_dim, self.use_xy_prior, goal_indices, non_goal_indices
            )
            
            # Normalize inputs and targets using batch statistics (as per paper)
            input_mean = jnp.mean(dynamics_input, axis=0, keepdims=True)
            input_std = jnp.std(dynamics_input, axis=0, keepdims=True) + 1e-8
            dynamics_input_norm = (dynamics_input - input_mean) / input_std
            
            target_mean = jnp.mean(delta_s, axis=0, keepdims=True)
            target_std = jnp.std(delta_s, axis=0, keepdims=True) + 1e-8
            delta_s_norm = (delta_s - target_mean) / target_std
            
            dyn_mean = dads_network.skill_dynamics_network.apply(
                None, training_state.dynamics_params, dynamics_input_norm
            )  # (B, output_size) - only mean, covariance is identity
            dyn_mean_norm = (dyn_mean - target_mean) / target_std
            
            # Per-sample NLL: -log q(s'|s,z_actual) with identity covariance
            per_sample_log_prob = dads_losses._gaussian_log_prob_sum_identity_cov(
                delta_s_norm, dyn_mean_norm
            )  # (B,)
            per_sample_nll = -per_sample_log_prob  # (B,)

            # Replace the stored (env) reward with the DADS intrinsic reward
            transitions = transitions._replace(reward=dads_reward)

            # ---- 2. SAC updates ------------------------------------------
            alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                training_state.policy_params,
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
            alpha = jnp.exp(training_state.alpha_params)

            critic_loss_val, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )

            actor_loss_val, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )

            # Soft target update
            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                q_params,
            )

            # ---- 3. Dynamics model update ---------------------------------
            dyn_loss_val, dynamics_params, dynamics_optimizer_state = dynamics_update(
                training_state.dynamics_params,
                transitions,
                optimizer_state=training_state.dynamics_optimizer_state,
            )

            # ---- 4. Per-skill metrics ------------------------------------
            skill_indices = jnp.argmax(skills, axis=-1)  # (B,)

            per_skill_metrics = {}
            per_skill_nll_list = []
            per_skill_reward_list = []

            for skill_idx in range(num_skills):
                mask = skill_indices == skill_idx  # (B,)
                num_samples = jnp.sum(mask.astype(jnp.int32))

                # Mean NLL for this skill (lower is better)
                skill_nll = jnp.where(
                    num_samples > 0,
                    jnp.sum(per_sample_nll * mask) / num_samples,
                    -1.0,
                )
                # Mean DADS reward for this skill
                skill_reward = jnp.where(
                    num_samples > 0,
                    jnp.sum(dads_reward * mask) / num_samples,
                    -1.0,
                )

                per_skill_metrics[f"skill_{skill_idx}_training/dynamics_loss"] = skill_nll
                per_skill_metrics[f"skill_{skill_idx}_training/dads_reward"] = skill_reward
                per_skill_nll_list.append(skill_nll)
                per_skill_reward_list.append(skill_reward)

            # Mean across skills (excluding -1 sentinel for missing skills)
            nll_arr    = jnp.array(per_skill_nll_list)
            reward_arr = jnp.array(per_skill_reward_list)
            valid_nll    = nll_arr    >= 0
            valid_reward = reward_arr >= 0
            n_valid_nll    = jnp.sum(valid_nll.astype(jnp.int32))
            n_valid_reward = jnp.sum(valid_reward.astype(jnp.int32))

            mean_metrics = {
                "mean_training/dynamics_loss": jnp.where(
                    n_valid_nll > 0,
                    jnp.sum(nll_arr * valid_nll) / n_valid_nll,
                    -1.0,
                ),
                "mean_training/dads_reward": jnp.where(
                    n_valid_reward > 0,
                    jnp.sum(reward_arr * valid_reward) / n_valid_reward,
                    -1.0,
                ),
            }

            metrics = {
                "critic_loss":   critic_loss_val,
                "actor_loss":    actor_loss_val,
                "alpha_loss":    alpha_loss_val,
                "alpha":         jnp.exp(alpha_params),
                "dynamics_loss": dyn_loss_val,
                "dads_reward":   jnp.mean(dads_reward),
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
            )
            return (new_training_state, key), metrics

        # ------------------------------------------------------------------
        # Experience collection
        # ------------------------------------------------------------------
        def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params: Params,
            env_state: State,
            current_skills: jnp.ndarray,    # (num_envs_per_device, num_skills)
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[
            running_statistics.RunningStatisticsState,
            State,
            jnp.ndarray,
            ReplayBufferState,
        ]:
            policy = make_policy((normalizer_params, policy_params))

            @jax.jit
            def f(carry, unused_t):
                env_state, skills, current_key = carry
                current_key, next_key, skill_key = jax.random.split(current_key, 3)

                # ---- Build skill-augmented observation --------------------
                state_part = env_state.obs[:, :state_dim]          # (N, state_dim)
                aug_obs = jnp.concatenate([state_part, skills], axis=-1)  # (N, dads_obs_size)

                # ---- Policy step -----------------------------------------
                actions, policy_extras = policy(aug_obs, current_key)

                # ---- Environment step ------------------------------------
                nstate = env.step(env_state, actions)

                # ---- Build next augmented observation --------------------
                next_state_part = nstate.obs[:, :state_dim]
                aug_next_obs = jnp.concatenate([next_state_part, skills], axis=-1)

                state_extras = {
                    "truncation": nstate.info["truncation"],
                    "traj_id":    nstate.info["traj_id"],
                    "skill":      skills,   # (N, num_skills) one-hot
                }

                transition = Transition(
                    observation=aug_obs,
                    next_observation=aug_next_obs,
                    action=actions,
                    reward=nstate.reward,   # replaced by DADS reward during gradient update
                    discount=1 - nstate.done,
                    extras={
                        "policy_extras": policy_extras,
                        "state_extras":  state_extras,
                    },
                )

                # ---- Skill update ----------------------------------------
                # Sample new skills for environments that just terminated.
                new_skill_idx = jax.random.randint(
                    skill_key, (skills.shape[0],), 0, num_skills
                )
                new_skills = jax.nn.one_hot(new_skill_idx, num_skills, dtype=jnp.float32)
                next_skills = jnp.where(
                    nstate.done[:, None],   # (N, 1) broadcast
                    new_skills,
                    skills,
                )

                return (nstate, next_skills, next_key), transition

            (env_state, current_skills, _), data = jax.lax.scan(
                f, (env_state, current_skills, key), (), length=self.unroll_length
            )

            # Update running normaliser with the augmented observations
            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, current_skills, buffer_state

        # ------------------------------------------------------------------
        # Training step (one unroll + gradient updates)
        # ------------------------------------------------------------------
        def training_step(
            training_state: TrainingState,
            env_state: State,
            current_skills: jnp.ndarray,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, jnp.ndarray, ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, current_skills, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                current_skills,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            training_state, buffer_state, metrics = train_steps(
                training_state, buffer_state, training_key
            )
            return training_state, env_state, current_skills, buffer_state, metrics

        # ------------------------------------------------------------------
        # Replay buffer prefill
        # ------------------------------------------------------------------
        def prefill_replay_buffer(
            training_state: TrainingState,
            env_state: State,
            current_skills: jnp.ndarray,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, jnp.ndarray, ReplayBufferState, PRNGKey]:
            def f(carry, unused):
                del unused
                training_state, env_state, current_skills, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                new_normalizer_params, env_state, current_skills, buffer_state = get_experience(
                    training_state.normalizer_params,
                    training_state.policy_params,
                    env_state,
                    current_skills,
                    buffer_state,
                    key,
                )
                new_training_state = training_state.replace(
                    normalizer_params=new_normalizer_params,
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (new_training_state, env_state, current_skills, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, current_skills, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        prefill_replay_buffer = jax.pmap(
            prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
        )

        # ------------------------------------------------------------------
        # Gradient updates (no environment interaction)
        # ------------------------------------------------------------------
        def train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            experience_key, training_key, sampling_key = jax.random.split(key, 3)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # Reshape: (sample_batch, episode_len, ...) -> (sample_batch*episode_len, ...)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )

            # Shuffle and split into mini-batches of size batch_size // device_count
            batch_size_per_device = self.batch_size // device_count
            permutation = jax.random.permutation(
                experience_key, transitions.observation.shape[0]
            )
            transitions = jax.tree_util.tree_map(
                lambda x: x[permutation], transitions
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, batch_size_per_device) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, _unused):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        # ------------------------------------------------------------------
        # Training epoch (many training steps)
        # ------------------------------------------------------------------
        def training_epoch(
            training_state: TrainingState,
            env_state: State,
            current_skills: jnp.ndarray,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, jnp.ndarray, ReplayBufferState, Metrics]:
            def f(carry, unused_t):
                ts, es, skills, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, skills, bs, metrics = training_step(ts, es, skills, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(
                    self.train_step_multiplier - 1, ts, bs, update_key
                )
                return (ts, es, skills, bs, new_key), metrics

            (training_state, env_state, current_skills, buffer_state, key), metrics = (
                jax.lax.scan(
                    f,
                    (training_state, env_state, current_skills, buffer_state, key),
                    (),
                    length=num_training_steps_per_epoch,
                )
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, current_skills, buffer_state, metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # ------------------------------------------------------------------
        # Training epoch with timing
        # ------------------------------------------------------------------
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: State,
            current_skills: jnp.ndarray,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, jnp.ndarray, ReplayBufferState, Metrics]:
            nonlocal training_walltime
            t = time.time()
            (
                training_state,
                env_state,
                current_skills,
                buffer_state,
                metrics,
            ) = training_epoch(
                training_state, env_state, current_skills, buffer_state, key
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time

            # Convert to a plain Python dict so iteration is reliable.
            metrics_dict = dict(metrics)

            # Build final metrics dict, preserving existing prefixes.
            final_metrics = {
                "training/sps":      sps,
                "training/walltime": training_walltime,
            }
            for name, value in metrics_dict.items():
                if "/" in name:
                    final_metrics[name] = value        # already has a prefix
                else:
                    final_metrics[f"training/{name}"] = value

            return training_state, env_state, current_skills, buffer_state, final_metrics

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        training_state = _init_training_state(
            key=global_key,
            obs_size=dads_obs_size,
            local_devices_to_use=local_devices_to_use,
            dads_network=dads_network,
            alpha_optimizer=alpha_optimizer,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
            dynamics_optimizer=dynamics_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key, skill_key = jax.random.split(local_key, 5)

        # Environment state
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(
            env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
        )
        env_state = jax.pmap(env.reset)(env_keys)

        # Initial skills (one per env, replicated across devices)
        skill_keys = jax.random.split(skill_key, local_devices_to_use)

        def _init_skills(key):
            idx = jax.random.randint(key, (num_envs_per_device,), 0, num_skills)
            return jax.nn.one_hot(idx, num_skills, dtype=jnp.float32)

        current_skills = jax.pmap(_init_skills)(skill_keys)

        # Replay buffer
        buffer_state = jax.pmap(replay_buffer.init)(
            jax.random.split(rb_key, local_devices_to_use)
        )

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
            eval_env_wrapped,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn_eval,
        )

        # Wrap make_policy so it accepts raw env observations.
        eval_make_policy = dads_eval.make_dads_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=0
        )

        evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(eval_make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # ------------------------------------------------------------------
        # Initial evaluation
        # ------------------------------------------------------------------
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            params = _unpmap(
                (training_state.normalizer_params, training_state.policy_params)
            )
            metrics = dads_eval.run_multi_skill_evaluation(
                make_policy,
                state_dim,
                num_skills,
                eval_env_wrapped,
                self.deterministic_eval,
                config.num_eval_envs,
                config.episode_length,
                config.action_repeat,
                eval_key,
                params,
                training_metrics={},
            )
            progress_fn(0, metrics, eval_make_policy, params, unwrapped_env, False)

        # ------------------------------------------------------------------
        # Prefill replay buffer
        # ------------------------------------------------------------------
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, current_skills, buffer_state, _ = (
            prefill_replay_buffer(
                training_state, env_state, current_skills, buffer_state, prefill_keys
            )
        )

        replay_size = (
            jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        )
        logging.info("replay size after prefill %s", replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        # ------------------------------------------------------------------
        # Main training loop
        # ------------------------------------------------------------------
        current_step = 0
        last_checkpoint_step = 0
        checkpoint_interval = 1_000_000  # Save policy every 1M steps

        for eval_epoch_num in range(num_evals_after_init):
            logging.info("step %s", current_step)

            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (
                training_state,
                env_state,
                current_skills,
                buffer_state,
                training_metrics,
            ) = training_epoch_with_timing(
                training_state, env_state, current_skills, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            if process_id == 0:
                params = _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                )
                # Save checkpoint every 1M steps
                if config.checkpoint_logdir and (
                    current_step - last_checkpoint_step >= checkpoint_interval
                ):
                    path = f"{config.checkpoint_logdir}_dads_{current_step}.pkl"
                    model.save_params(path, params)
                    logging.info(f"Saved checkpoint at step {current_step}")
                    last_checkpoint_step = current_step

                # Run multi-skill evaluation
                metrics = dads_eval.run_multi_skill_evaluation(
                    make_policy,
                    state_dim,
                    num_skills,
                    eval_env_wrapped,
                    self.deterministic_eval,
                    config.num_eval_envs,
                    config.episode_length,
                    config.action_repeat,
                    eval_key,
                    params,
                    training_metrics,
                )

                # Render videos for each skill if visualization is enabled
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                if do_render:
                    render_dir = config.checkpoint_logdir if config.checkpoint_logdir else "./runs"
                    dads_eval.render_all_skills(
                        make_policy,
                        state_dim,
                        num_skills,
                        params,
                        unwrapped_env,
                        render_dir,
                        config.exp_name,
                        current_step,
                    )

                # Merge training metrics so progress_fn receives everything.
                merged_metrics = {**training_metrics, **metrics}

                progress_fn(
                    current_step,
                    merged_metrics,
                    eval_make_policy,
                    params,
                    unwrapped_env,
                    False,  # rendering already done above
                )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        params = _unpmap(
            (training_state.normalizer_params, training_state.policy_params)
        )

        # Save final checkpoint
        if process_id == 0 and config.checkpoint_logdir:
            path = f"{config.checkpoint_logdir}_dads_{total_steps}_final.pkl"
            model.save_params(path, params)
            logging.info(f"Saved final checkpoint at step {total_steps}")

        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return eval_make_policy, params, metrics
