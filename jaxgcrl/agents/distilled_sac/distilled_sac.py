"""Distilled SAC with Flow Matching Policy.

Three-phase training algorithm:
  Phase 1 – Prefill
      Roll out a pretrained skill-conditioned policy (DIAYN/DADS) for each of
      the K skills, collecting task-reward transitions into a replay buffer.

  Phase 2a – Flow Policy Distillation
      Minimise forward KL between the skill mixture (replay buffer) and the
      flow matching policy via NLL: loss = -log π_flow(a | s, g) where g is
      sampled fresh each step from env.possible_goals.

  Phase 2b – Critic Warmup
      Run SAC critic updates on the prefill replay buffer using the distilled
      flow policy as the target actor. No actor updates.

  Phase 3 – SAC Fine-tuning
      Standard SAC training with the flow policy as the actor and the
      warm-started critic.  The replay buffer carries over from Phase 1.
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
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from . import networks as flow_nets
from . import losses as flow_losses

import numpy as np
from sklearn.manifold import TSNE
import wandb

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

_PMAP_AXIS_NAME = "i"
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any


# ---------------------------------------------------------------------------
# Transition (same format as SAC – stores full goal-conditioned observations)
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Replay buffer transition in SAC / goal-conditioned format."""

    observation: NestedArray       # (obs_size,)  = [state | goal]
    next_observation: NestedArray
    action: NestedArray            # (action_size,)  tanh-squashed
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Full training state for the DistilledSAC learner."""

    # Flow policy
    velocity_optimizer_state: optax.OptState
    velocity_params: Params

    # Q-network
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params

    # Alpha (entropy temperature)
    alpha_optimizer_state: optax.OptState
    alpha_params: Params            # log_alpha (scalar)

    # Observation normaliser
    normalizer_params: running_statistics.RunningStatisticsState

    # Counters
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    flow_networks: flow_nets.FlowPolicyNetworks,
    alpha_optimizer: optax.GradientTransformation,
    velocity_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    key_vel, key_q = jax.random.split(key)

    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    vel_params = flow_networks.velocity_network.init(key_vel)
    velocity_optimizer_state = velocity_optimizer.init(vel_params)

    q_params = flow_networks.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype("float32"))
    )

    state = TrainingState(
        velocity_optimizer_state=velocity_optimizer_state,
        velocity_params=vel_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
    )
    return jax.device_put_replicated(state, jax.local_devices()[:local_devices_to_use])


# ---------------------------------------------------------------------------
# DistilledSAC agent
# ---------------------------------------------------------------------------

@dataclass
class DistilledSAC:
    """Distilled SAC agent with a flow matching policy.

    Takes a pretrained skill-conditioned policy (DIAYN / DADS) and produces a
    goal-conditioned flow matching policy, warm-started via distillation and
    then fine-tuned with standard SAC.

    Phase 1  – prefill replay buffer with skill-policy rollouts
    Phase 2a – distil flow policy via NLL on the replay buffer
    Phase 2b – warm-start critic with SAC critic updates (no actor update)
    Phase 3  – SAC fine-tuning with flow policy + warm critic

    Args:
        skill_policy_path:      Path to the saved DIAYN/DADS model params
                                (output of brax.io.model.save_params).
        skill_policy_type:      "diayn" or "dads".
        num_skills:             Number of discrete skill latents K.
        prefill_steps_per_skill: Environment steps collected per skill during Phase 1.
        distill_steps:          Number of distillation gradient steps (Phase 2a).
        distill_viz_freq:       Visualise flow policy every N distillation steps.
        critic_warmup_steps:    Critic-only gradient steps before SAC (Phase 2b).
        hutchinson_samples:     Rademacher vectors for Hutchinson trace estimator.
        n_ode_steps:            Fixed Euler steps for CNF ODE integration.
        learning_rate:          Learning rate for all optimisers.
        discounting:            Discount factor γ.
        batch_size:             Training batch size (total across devices).
        normalize_observations: Whether to use running normaliser.
        reward_scaling:         Scalar multiplier on task reward.
        tau:                    Soft target-network update rate.
        min_replay_size:        Minimum replay size before Phase 3 starts.
        max_replay_size:        Maximum replay buffer capacity.
        deterministic_eval:     Deterministic actions during evaluation.
        train_step_multiplier:  SAC gradient steps per environment step (Phase 3).
        unroll_length:          Env steps collected per actor-step in Phase 3.
        h_dim:                  Hidden layer dimension for all MLPs.
        n_hidden:               Number of hidden layers.
        use_ln:                 Use layer normalisation.
        use_her:                Use hindsight experience replay in Phase 3.
    """

    skill_policy_path: str = ""
    skill_policy_type: str = "diayn"   # "diayn" or "dads"
    num_skills: int = 8
    prefill_steps_per_skill: int = 5_000
    distill_steps: int = 50_000
    distill_viz_freq: int = 5_000
    critic_warmup_steps: int = 10_000

    hutchinson_samples: int = 1
    n_ode_steps: int = 10

    skill_h_dim: Optional[int] = None     # if None, defaults to h_dim
    skill_n_hidden: Optional[int] = None  # if None, defaults to n_hidden

    learning_rate: float = 3e-4
    discounting: float = 0.99
    batch_size: int = 256
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 100_000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    use_ln: bool = True
    use_her: bool = True

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
        """Run three-phase DistilledSAC training.

        Parameters mirror the existing SAC / DIAYN train_fn signatures.
        """
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info("local_device_count: %s; total_device_count: %s", local_devices_to_use, device_count)

        # ------------------------------------------------------------------
        # Environment setup
        # ------------------------------------------------------------------
        unwrapped_env = train_env
        state_dim: int = unwrapped_env.state_dim
        goal_indices = getattr(unwrapped_env, "goal_indices", None)

        # Goal set for Phase 2a distillation
        goal_set = getattr(unwrapped_env, "possible_goals", None)
        if goal_set is None and goal_indices is not None:
            # Fall back: use unique goal observations from the environment
            logging.warning(
                "env.possible_goals not found; sampling goals from env.goal_indices slice."
            )
        if goal_set is None:
            raise ValueError(
                "DistilledSAC requires env.possible_goals (array of candidate goals) "
                "to be set on the environment. Found neither 'possible_goals' nor a fallback."
            )
        goal_dim = goal_set.shape[-1]

        if isinstance(train_env, envs.Env):
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

        env = TrajectoryIdWrapper(train_env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        obs_size = env.observation_size   # state_dim + goal_dim
        action_size = env.action_size
        assert obs_size == state_dim + goal_dim, (
            f"obs_size {obs_size} != state_dim {state_dim} + goal_dim {goal_dim}. "
            "Make sure the environment observation is [state | goal]."
        )

        # ------------------------------------------------------------------
        # Flow policy + Q-network
        # ------------------------------------------------------------------
        def normalize_fn(x, y):
            return x

        if self.normalize_observations:
            normalize_fn = running_statistics.normalize

        flow_networks, _ = flow_nets.make_flow_policy_networks(
            obs_size=obs_size,
            action_size=action_size,
            preprocess_observations_fn=normalize_fn,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
            layer_norm=self.use_ln,
        )
        make_policy = flow_nets.make_inference_fn(flow_networks, n_ode_steps=self.n_ode_steps)

        # ------------------------------------------------------------------
        # Replay buffer (SAC format: [state | goal] observations)
        # ------------------------------------------------------------------
        max_replay_size = self.max_replay_size if self.max_replay_size is not None else config.total_env_steps
        assert config.num_envs % device_count == 0
        num_envs_per_device = config.num_envs // device_count

        dummy_obs = jnp.zeros((obs_size,))
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
        # Optimisers
        # ------------------------------------------------------------------
        alpha_optimizer = optax.adam(learning_rate=3e-4)
        velocity_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        target_entropy = -action_size  # standard SAC default

        # ------------------------------------------------------------------
        # Loss / update functions
        # ------------------------------------------------------------------
        distillation_update = flow_losses.make_distillation_update_fn(
            flow_networks=flow_networks,
            optimizer=velocity_optimizer,
            state_dim=state_dim,
            n_ode_steps=self.n_ode_steps,
            n_hutchinson_samples=self.hutchinson_samples,
        )
        critic_update = flow_losses.make_critic_update_fn(
            flow_networks=flow_networks,
            q_optimizer=q_optimizer,
            discounting=self.discounting,
            reward_scaling=self.reward_scaling,
            n_ode_steps=self.n_ode_steps,
            n_hutchinson_samples=self.hutchinson_samples,
        )
        actor_update = flow_losses.make_actor_update_fn(
            flow_networks=flow_networks,
            policy_optimizer=velocity_optimizer,
            n_ode_steps=self.n_ode_steps,
            n_hutchinson_samples=self.hutchinson_samples,
        )
        alpha_update = flow_losses.make_alpha_update_fn(
            flow_networks=flow_networks,
            alpha_optimizer=alpha_optimizer,
            n_ode_steps=self.n_ode_steps,
            n_hutchinson_samples=self.hutchinson_samples,
            target_entropy=target_entropy,
        )

        # ------------------------------------------------------------------
        # Skill policy loading (Phase 1)
        # ------------------------------------------------------------------
        # Reconstruct the skill-conditioned policy network (DIAYN / DADS).
        # The policy takes [state | skill_one_hot] as input.
        skill_h_dim = self.skill_h_dim if self.skill_h_dim is not None else self.h_dim
        skill_n_hidden = self.skill_n_hidden if self.skill_n_hidden is not None else self.n_hidden
        skill_obs_size = state_dim + self.num_skills

        if self.skill_policy_type.lower() == "diayn":
            from jaxgcrl.agents.diayn import networks as diayn_networks
            skill_network = diayn_networks.make_diayn_networks(
                observation_size=skill_obs_size,
                action_size=action_size,
                state_size=state_dim,
                num_skills=self.num_skills,
                hidden_layer_sizes=[skill_h_dim] * skill_n_hidden,
            )
            skill_make_policy = diayn_networks.make_inference_fn(skill_network)
        elif self.skill_policy_type.lower() == "dads":
            from jaxgcrl.agents.dads import networks as dads_networks
            skill_network = dads_networks.make_dads_networks(
                observation_size=skill_obs_size,
                action_size=action_size,
                state_size=state_dim,
                num_skills=self.num_skills,
                hidden_layer_sizes=[skill_h_dim] * skill_n_hidden,
            )
            skill_make_policy = dads_networks.make_inference_fn(skill_network)
        else:
            raise ValueError(f"Unknown skill_policy_type: {self.skill_policy_type!r}")

        # Load pretrained params
        if self.skill_policy_path:
            logging.info("Loading skill policy from %s", self.skill_policy_path)
            skill_params = model.load_params(self.skill_policy_path)
            # skill_params = (normalizer_params, policy_params)
        else:
            raise ValueError("Skill policy path is required")

        # Build JIT-compiled skill policy inference function
        # (deterministic=False to keep stochasticity during data collection)
        _jit_skill_policy = jax.jit(skill_make_policy(skill_params, deterministic=False))

        # ------------------------------------------------------------------
        # Phase 1 helper: actor step with skill policy
        # ------------------------------------------------------------------
        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length

        def _skill_actor_step(
            env_state: State,
            skills: jnp.ndarray,    # (num_envs_per_device, num_skills)
            normalizer_params,
            key: PRNGKey,
        ) -> Tuple[State, Transition, jnp.ndarray]:
            """One step with skill-conditioned policy; stores SAC-format transition."""
            key, step_key, skill_key = jax.random.split(key, 3)

            # Build skill-augmented obs for the skill policy
            raw_obs = env_state.obs       # (N, obs_size)  = [state | goal]
            state_part = raw_obs[:, :state_dim]           # (N, state_dim)
            aug_obs = jnp.concatenate([state_part, skills], axis=-1)  # (N, skill_obs_size)

            # Skill policy action
            actions, _ = _jit_skill_policy(aug_obs, step_key)

            # Step env
            nstate = env.step(env_state, actions)

            # SAC-format transition (store full goal-conditioned obs)
            transition = Transition(
                observation=raw_obs,                     # [state | goal]
                next_observation=nstate.obs,
                action=actions,
                reward=nstate.reward,                    # task reward
                discount=1 - nstate.done,
                extras={
                    "state_extras": {
                        "truncation": nstate.info["truncation"],
                        "traj_id": nstate.info["traj_id"],
                    },
                    "policy_extras": {},
                },
            )

            # Update skills for environments that finished their episode
            new_skill_idx = jax.random.randint(skill_key, (skills.shape[0],), 0, self.num_skills)
            new_skills = jax.nn.one_hot(new_skill_idx, self.num_skills, dtype=jnp.float32)
            next_skills = jnp.where(nstate.done[:, None], new_skills, skills)

            return nstate, transition, next_skills

        # ------------------------------------------------------------------
        # Phase 1: Prefill replay buffer with skill-policy rollouts
        # ------------------------------------------------------------------
        def _prefill_with_skill_policy(
            env_state: State,
            buffer_state: ReplayBufferState,
            normalizer_params,
            key: PRNGKey,
            n_steps_per_skill: int,
        ) -> Tuple[State, ReplayBufferState, running_statistics.RunningStatisticsState]:
            """Collect n_steps_per_skill * K transitions across all K skills."""
            total_steps_per_device = (n_steps_per_skill * self.num_skills) // config.num_envs
            # Each device uses num_envs_per_device envs; each env carries a random skill.

            # initialise random skills for each env
            key, sk = jax.random.split(key)
            skill_idx = jax.random.randint(sk, (num_envs_per_device,), 0, self.num_skills)
            skills = jax.nn.one_hot(skill_idx, self.num_skills, dtype=jnp.float32)

            def f(carry, _):
                env_state, skills, buf_state, norm_params, key = carry
                key, step_key = jax.random.split(key)

                def inner_step(carry, _):
                    env_state, skills, key = carry
                    key, k = jax.random.split(key)
                    env_state, transition, skills = _skill_actor_step(env_state, skills, norm_params, k)
                    return (env_state, skills, key), transition

                (env_state, skills, key), data = jax.lax.scan(
                    inner_step, (env_state, skills, step_key), (), length=self.unroll_length
                )
                # Update normaliser (optional, improves Phase 3 if normalize_observations=True)
                norm_params = running_statistics.update(
                    norm_params,
                    jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                    ).observation,
                    pmap_axis_name=_PMAP_AXIS_NAME,
                )
                buf_state = replay_buffer.insert(buf_state, data)
                return (env_state, skills, buf_state, norm_params, key), ()

            (env_state, _, buffer_state, normalizer_params, _), _ = jax.lax.scan(
                f,
                (env_state, skills, buffer_state, normalizer_params, key),
                (),
                length=max(1, total_steps_per_device // self.unroll_length),
            )
            return env_state, buffer_state, normalizer_params

        _prefill_pmapped = jax.pmap(_prefill_with_skill_policy, axis_name=_PMAP_AXIS_NAME,
                                    static_broadcasted_argnums=(4,))

        # ------------------------------------------------------------------
        # Phase 2a: Distillation update step (offline, uses only replay buffer)
        # ------------------------------------------------------------------
        def _distillation_step(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            goal_set_device: jnp.ndarray,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # Flatten trajectory dimension
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            B_per_dev = self.batch_size // device_count
            # Take exactly one mini-batch
            transitions = jax.tree_util.tree_map(lambda x: x[:B_per_dev], transitions)

            key, goal_key, ode_key = jax.random.split(key, 3)

            loss, new_vel_params, new_vel_opt_state = distillation_update(
                training_state.velocity_params,
                transitions,
                goal_set_device,
                goal_key,
                ode_key,
                optimizer_state=training_state.velocity_optimizer_state,
            )

            new_state = training_state.replace(
                velocity_params=new_vel_params,
                velocity_optimizer_state=new_vel_opt_state,
            )
            return new_state, buffer_state, {"distillation/nll_loss": loss}

        _distillation_step_pmapped = jax.pmap(
            _distillation_step, axis_name=_PMAP_AXIS_NAME
        )

        # ------------------------------------------------------------------
        # Phase 2b: Critic warmup step (offline)
        # ------------------------------------------------------------------
        def _critic_warmup_step(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            B_per_dev = self.batch_size // device_count
            transitions = jax.tree_util.tree_map(lambda x: x[:B_per_dev], transitions)

            key, critic_key = jax.random.split(key)
            alpha = jnp.exp(training_state.alpha_params)

            loss, new_q_params, new_q_opt_state = critic_update(
                training_state.q_params,
                training_state.velocity_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                critic_key,
                optimizer_state=training_state.q_optimizer_state,
            )

            new_target_q = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                new_q_params,
            )

            new_state = training_state.replace(
                q_params=new_q_params,
                q_optimizer_state=new_q_opt_state,
                target_q_params=new_target_q,
            )
            return new_state, buffer_state, {"critic_warmup/critic_loss": loss}

        _critic_warmup_step_pmapped = jax.pmap(
            _critic_warmup_step, axis_name=_PMAP_AXIS_NAME
        )

        # ------------------------------------------------------------------
        # Phase 3: SAC fine-tuning update step
        # ------------------------------------------------------------------
        def sac_update_step(
            carry: Tuple[TrainingState, PRNGKey],
            transitions: Transition,
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry
            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            alpha = jnp.exp(training_state.alpha_params)

            # ---- Critic ---------------------------------------------------
            critic_loss_val, new_q_params, new_q_opt_state = critic_update(
                training_state.q_params,
                training_state.velocity_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )

            # ---- Actor ----------------------------------------------------
            actor_loss_val, new_vel_params, new_vel_opt_state = actor_update(
                training_state.velocity_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.velocity_optimizer_state,
            )

            # ---- Alpha ----------------------------------------------------
            alpha_loss_val, new_alpha_params, new_alpha_opt_state = alpha_update(
                training_state.alpha_params,
                new_vel_params,  # use updated actor params
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )

            # ---- Soft target update ----------------------------------------
            new_target_q = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                new_q_params,
            )

            metrics = {
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "alpha": jnp.exp(new_alpha_params),
            }

            new_state = TrainingState(
                velocity_optimizer_state=new_vel_opt_state,
                velocity_params=new_vel_params,
                q_optimizer_state=new_q_opt_state,
                q_params=new_q_params,
                target_q_params=new_target_q,
                alpha_optimizer_state=new_alpha_opt_state,
                alpha_params=new_alpha_params,
                normalizer_params=training_state.normalizer_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
            )
            return (new_state, key), metrics

        def get_experience(
            normalizer_params,
            velocity_params: Params,
            env_state: State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[running_statistics.RunningStatisticsState, State, ReplayBufferState]:
            """Collect env data with the flow policy and insert into replay buffer."""
            policy = make_policy((normalizer_params, velocity_params))

            def f(carry, _):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                actions, _ = policy(env_state.obs, current_key)
                nstate = env.step(env_state, actions)
                transition = Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=nstate.obs,
                    extras={
                        "state_extras": {
                            "truncation": nstate.info["truncation"],
                            "traj_id": nstate.info["traj_id"],
                        },
                        "policy_extras": {},
                    },
                )
                return (nstate, next_key), transition

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=self.unroll_length
            )
            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, buffer_state

        def sac_train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            exp_key, train_key, sample_key = jax.random.split(key, 3)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # Flatten trajectory dimension
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            B_per_dev = self.batch_size // device_count
            perm = jax.random.permutation(exp_key, transitions.observation.shape[0])
            transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, B_per_dev) + x.shape[1:]), transitions
            )

            (training_state, _), metrics = jax.lax.scan(
                sac_update_step, (training_state, train_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_sac_train_steps(n, ts, bs, key):
            def body(carry, _):
                ts, bs, key = carry
                key, new_key = jax.random.split(key)
                ts, bs, metrics = sac_train_steps(ts, bs, key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, key), (), length=n)

        def sac_training_step(
            training_state: TrainingState,
            env_state: State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, ReplayBufferState, Metrics]:
            exp_key, train_key = jax.random.split(key)
            norm_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.velocity_params,
                env_state,
                buffer_state,
                exp_key,
            )
            training_state = training_state.replace(
                normalizer_params=norm_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            training_state, buffer_state, metrics = sac_train_steps(
                training_state, buffer_state, train_key
            )
            return training_state, env_state, buffer_state, metrics

        # ------------------------------------------------------------------
        # SAC training epoch
        # ------------------------------------------------------------------
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        def sac_training_epoch(
            training_state: TrainingState,
            env_state: State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, State, ReplayBufferState, Metrics]:
            def f(carry, _):
                ts, es, bs, k = carry
                k, new_key, up_key = jax.random.split(k, 3)
                ts, es, bs, metrics = sac_training_step(ts, es, bs, k)
                (ts, bs, up_key), _ = scan_sac_train_steps(
                    self.train_step_multiplier - 1, ts, bs, up_key
                )
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, env_state, buffer_state, metrics

        sac_training_epoch = jax.pmap(sac_training_epoch, axis_name=_PMAP_AXIS_NAME)

        # ------------------------------------------------------------------
        # t-SNE visualisation helper (Phase 2a)
        # ------------------------------------------------------------------
        def _visualise_flow_policy(vel_params, obs_sample, step, process_id):
            """Log a t-SNE scatter of sampled actions coloured by goal index."""
            if process_id != 0:
                return
            try:
                n_goals = min(4, goal_set.shape[0])
                n_actions_per_goal = 64
                key_vis = jax.random.PRNGKey(step)
                all_actions, all_colours = [], []
                for gi in range(n_goals):
                    g = goal_set[gi]
                    s = obs_sample[:, :state_dim]
                    obs_g = jnp.concatenate([s, jnp.broadcast_to(g, s.shape[:1] + g.shape)], axis=-1)
                    obs_g_repeated = jnp.tile(obs_g[:1], (n_actions_per_goal, 1))
                    key_vis, k = jax.random.split(key_vis)
                    acts = flow_nets.sample_action(
                        flow_networks.velocity_network,
                        vel_params,
                        obs_g_repeated,
                        k,
                        n_ode_steps=self.n_ode_steps,
                        action_size=action_size,
                    )
                    all_actions.append(np.array(acts))
                    all_colours.extend([gi] * n_actions_per_goal)

                actions_np = np.concatenate(all_actions, axis=0)
                colours_np = np.array(all_colours)

                if actions_np.shape[1] > 2:
                    emb = TSNE(n_components=2, random_state=step).fit_transform(actions_np)
                else:
                    emb = actions_np

                fig, ax = plt.subplots(figsize=(6, 6))
                scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colours_np, cmap="tab10", alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, label="Goal index")
                ax.set_title(f"Flow policy actions (t-SNE) – distill step {step}")
                wandb.log({"distillation/tsne_actions": wandb.Image(fig)}, step=step)
                plt.close(fig)
            except Exception as exc:
                logging.warning("t-SNE visualisation failed: %s", exc)

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)

        training_state = _init_training_state(
            key=global_key,
            obs_size=obs_size,
            local_devices_to_use=local_devices_to_use,
            flow_networks=flow_networks,
            alpha_optimizer=alpha_optimizer,
            velocity_optimizer=velocity_optimizer,
            q_optimizer=q_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env state
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        # Replay buffer
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        # Eval environment
        if not eval_env:
            eval_env = train_env
        eval_env_wrapped = TrajectoryIdWrapper(eval_env)
        eval_env_wrapped = wrap_for_training(
            eval_env_wrapped,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # Initial eval (untrained policy)
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.velocity_params)),
                training_metrics={},
            )
            progress_fn(0, metrics, make_policy,
                        _unpmap((training_state.normalizer_params, training_state.velocity_params)),
                        unwrapped_env)

        # ======================================================================
        # PHASE 1: Prefill replay buffer with skill-policy rollouts
        # ======================================================================
        logging.info("=== Phase 1: Prefill replay buffer ===")
        t0 = time.time()

        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)

        # Replicate normaliser and env_state are already distributed via pmap init
        norm_params_rep = training_state.normalizer_params  # already replicated

        env_state, buffer_state, norm_params_rep = _prefill_pmapped(
            env_state,
            buffer_state,
            norm_params_rep,
            prefill_keys,
            self.prefill_steps_per_skill,
        )

        # Update the training state's normalizer_params with the updated stats
        training_state = training_state.replace(normalizer_params=norm_params_rep)

        replay_size_phase1 = (
            jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
        )
        logging.info(
            "Phase 1 done. Replay size: %s. Time: %.1fs",
            replay_size_phase1,
            time.time() - t0,
        )

        # Gather one obs-batch for Phase 2a t-SNE visualisation (on device 0)
        _obs_sample_for_vis = None
        if process_id == 0:
            buf_state_0, sample_transitions = replay_buffer.sample(
                jax.tree_util.tree_map(lambda x: x[0], buffer_state)
            )
            _obs_sample_for_vis = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:])[:64], sample_transitions
            ).observation

        # ======================================================================
        # PHASE 2a: Flow policy distillation (NLL)
        # ======================================================================
        logging.info("=== Phase 2a: Flow policy distillation (%d steps) ===", self.distill_steps)
        t0 = time.time()

        # Broadcast goal_set to all devices
        goal_set_rep = jnp.broadcast_to(goal_set, (local_devices_to_use,) + goal_set.shape)

        distill_key, local_key = jax.random.split(local_key)

        for distill_step in range(self.distill_steps):
            distill_key, step_key = jax.random.split(distill_key)
            step_keys = jax.random.split(step_key, local_devices_to_use)

            training_state, buffer_state, distill_metrics = _distillation_step_pmapped(
                training_state, buffer_state, goal_set_rep, step_keys
            )

            if distill_step % 500 == 0:
                loss_val = float(jax.tree_util.tree_map(jnp.mean, distill_metrics)["distillation/nll_loss"])
                logging.info("  distill step %d / %d  NLL=%.4f", distill_step, self.distill_steps, loss_val)

            if distill_step % self.distill_viz_freq == 0 and process_id == 0 and _obs_sample_for_vis is not None:
                vel_params_0 = _unpmap(training_state.velocity_params)
                _visualise_flow_policy(vel_params_0, _obs_sample_for_vis, distill_step, process_id)

        logging.info("Phase 2a done. Time: %.1fs", time.time() - t0)

        # ======================================================================
        # PHASE 2b: Critic warmup
        # ======================================================================
        logging.info("=== Phase 2b: Critic warmup (%d steps) ===", self.critic_warmup_steps)
        t0 = time.time()

        warmup_key, local_key = jax.random.split(local_key)

        for warmup_step in range(self.critic_warmup_steps):
            warmup_key, step_key = jax.random.split(warmup_key)
            step_keys = jax.random.split(step_key, local_devices_to_use)
            training_state, buffer_state, warmup_metrics = _critic_warmup_step_pmapped(
                training_state, buffer_state, step_keys
            )
            if warmup_step % 500 == 0:
                loss_val = float(jax.tree_util.tree_map(jnp.mean, warmup_metrics)["critic_warmup/critic_loss"])
                logging.info(
                    "  warmup step %d / %d  critic_loss=%.4f",
                    warmup_step, self.critic_warmup_steps, loss_val,
                )

        logging.info("Phase 2b done. Time: %.1fs", time.time() - t0)

        # ======================================================================
        # PHASE 3: SAC fine-tuning
        # ======================================================================
        logging.info("=== Phase 3: SAC fine-tuning ===")

        # (optional) SAC prefill to top up buffer
        def sac_prefill(training_state, env_state, buffer_state, key):
            def f(carry, _):
                ts, es, bs, key = carry
                key, new_key = jax.random.split(key)
                norm_p, es, bs = get_experience(
                    ts.normalizer_params, ts.velocity_params, es, bs, key
                )
                ts = ts.replace(normalizer_params=norm_p,
                                env_steps=ts.env_steps + env_steps_per_actor_step)
                return (ts, es, bs, new_key), ()

            return jax.lax.scan(
                f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps
            )[0]

        if self.min_replay_size > 0:
            sac_prefill_pmapped = jax.pmap(sac_prefill, axis_name=_PMAP_AXIS_NAME)
            prefill_key, local_key = jax.random.split(local_key)
            prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
            training_state, env_state, buffer_state, _ = sac_prefill_pmapped(
                training_state, env_state, buffer_state, prefill_keys
            )

        training_walltime = 0.0
        current_step = 0

        def sac_epoch_with_timing(training_state, env_state, buffer_state, key):
            nonlocal training_walltime
            t = time.time()
            training_state, env_state, buffer_state, epoch_metrics = sac_training_epoch(
                training_state, env_state, buffer_state, key
            )
            epoch_metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), epoch_metrics)

            elapsed = time.time() - t
            training_walltime += elapsed
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / elapsed
            final_metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
            }
            for name, value in dict(epoch_metrics).items():
                key_str = f"training/{name}" if "/" not in name else name
                final_metrics[key_str] = value
            return training_state, env_state, buffer_state, final_metrics

        for eval_epoch_num in range(num_evals_after_init):
            logging.info("SAC step %s", current_step)

            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

            training_state, env_state, buffer_state, training_metrics = sac_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            if process_id == 0:
                if config.checkpoint_logdir:
                    params = _unpmap(
                        (training_state.normalizer_params, training_state.velocity_params)
                    )
                    path = f"{config.checkpoint_logdir}_dsac_{current_step}.pkl"
                    model.save_params(path, params)

                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.velocity_params)),
                    training_metrics,
                )
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    _unpmap((training_state.normalizer_params, training_state.velocity_params)),
                    unwrapped_env,
                    do_render,
                )

        total_steps = current_step
        params = _unpmap((training_state.normalizer_params, training_state.velocity_params))

        pmap.assert_is_replicated(training_state)
        logging.info("DistilledSAC total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
