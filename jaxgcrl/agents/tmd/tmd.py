"""Go-Explore TMD agent.

Temporal Metric Distillation (TMD) adapted for the Go-Explore two-policy
framework used in GoExploreCRLSimple.  The structure (replay buffers, goal
proposers, environment interaction, training loop) is kept identical to
``GoExploreCRLSimple``; only the critic / actor networks and losses differ.

Key differences from GoExploreCRLSimple:
  * phi(s,a) + psi(g) encoders with ensemble of 2, using MRN or IQE distance.
  * Critic loss: contrastive InfoNCE + backup (LINEX) + action-invariance.
  * Actor loss: Q-maximisation + BC (no entropy / no alpha).
  * No alpha (entropy coefficient) – alpha in GoExploreTMD is the fixed BC coeff.
  * flatten_batch_tmd stores ``next_state`` (immediate next state) needed for
    the backup loss.
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.visualize import (
    visualize_dual_crl_trajectories_2d,
    visualize_scatter_sample,
)

from jaxgcrl.agents.crl.networks import Actor
from jaxgcrl.agents.crl.proposers import (
    FinalReplayBufferProposer,
    RandomEnvironmentGoalProposer,
    UCGRProposer,
    MaxWaypointRatioOneEnvProposer,
    QEpistemicProposer,
    MEGAProposer,
    OMEGAProposer,
    NearestEnvGoalProposer,
    NearestEnvGoalToGCPGoalProposer,
    EmpowermentDifferenceGoalProposer,
)

from .networks import TMDEncoder
from .losses import update_tmd_actor, update_tmd_critic

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


# ---------------------------------------------------------------------------
# Training state (no alpha – TMD uses fixed BC coefficient)
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Training state for GoExploreTMD."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    gcp_actor_state: TrainState
    gcp_critic_state: TrainState
    ep_actor_state: TrainState
    ep_critic_state: TrainState
    traj_counter: jnp.ndarray


# ---------------------------------------------------------------------------
# Transition container (identical to CRL)
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# flatten_batch_tmd – like the CRL version but adds ``next_state``
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_batch_tmd(buffer_config, transition, sample_key):
    """Process a trajectory from the replay buffer into training transitions.

    Identical to ``flatten_batch`` in GoExploreCRLSimple but also stores
    ``next_state`` (the immediate next state) in ``extras``, which is required
    by the TMD backup loss.
    """
    gamma, state_size, goal_indices = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )
    discount = gamma ** jnp.array(
        arrangement[None] - arrangement[:, None], dtype=jnp.float32
    )
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    probs = (
        probs * jnp.equal(single_trajectories, single_trajectories.T)
        + jnp.eye(seq_len) * 1e-5
    )
    proposed_goals = transition.observation[:, -len(goal_indices):]

    traj_ids = transition.extras["state_extras"]["traj_id"]

    def last_state_for_each_step(obs, traj_ids):
        seq_len = obs.shape[0]
        def last_state_for_t(i):
            mask = traj_ids == traj_ids[i]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs[last_idx]
        return jax.vmap(last_state_for_t)(jnp.arange(seq_len))

    def get_intermediate_trajectory_states(obs, traj_ids):
        seq_len = obs.shape[0]
        obs_dim = obs.shape[1]

        def intermediate_states_for_t(i, num_intermediate):
            same_traj_mask = traj_ids == traj_ids[i]
            future_mask = jnp.arange(seq_len) >= i
            mask = same_traj_mask & future_mask
            indices = jnp.where(mask, jnp.arange(seq_len), seq_len)
            sorted_indices = jnp.sort(indices)
            num_future = jnp.sum(mask)
            fractions = jnp.arange(1, num_intermediate + 1) / (num_intermediate + 1)
            idxs = jnp.floor(fractions * num_future).astype(jnp.int32)
            idxs = jnp.clip(idxs, 0, jnp.maximum(num_future - 1, 0))
            actual_idxs = sorted_indices[idxs]

            def get_state(idx):
                return jnp.where(num_future > 0, obs[idx], jnp.zeros(obs_dim))
            return jax.vmap(get_state)(actual_idxs)

        return jax.vmap(
            functools.partial(intermediate_states_for_t, num_intermediate=6)
        )(jnp.arange(seq_len))

    last_traj_state = last_state_for_each_step(transition.observation, traj_ids)
    intermediate_traj = get_intermediate_trajectory_states(
        transition.observation, traj_ids
    )

    def is_last_occurrence(i):
        traj_id = traj_ids[i]
        future_mask = jnp.arange(seq_len) > i
        has_future_same_id = jnp.any((traj_ids == traj_id) & future_mask)
        return ~has_future_same_id

    last_traj_state_mask = jax.vmap(is_last_occurrence)(jnp.arange(seq_len))

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]

    # Immediate next state (needed for TMD backup loss)
    next_state = transition.observation[1:, :state_size]

    new_obs = jnp.concatenate([state, goal], axis=1)

    original_state_extras = transition.extras["state_extras"]
    state_extras = jax.tree_util.tree_map(
        lambda x: x[:-1] if len(x.shape) > 0 else x,
        original_state_extras,
    )
    state_extras["truncation"] = jnp.squeeze(state_extras["truncation"])
    state_extras["traj_id"] = jnp.squeeze(state_extras["traj_id"])

    extras = {
        "policy_extras": {},
        "state_extras": state_extras,
        "state": state,
        "future_state": future_state,
        "next_state": next_state,          # <-- TMD addition
        "future_action": future_action,
        "proposed_goals": proposed_goals[:-1],
        "last_traj_state": last_traj_state[:-1],
        "intermediate_traj": intermediate_traj[:-1],
        "last_traj_state_mask": last_traj_state_mask[:-1],
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Utilities (identical to CRL)
# ---------------------------------------------------------------------------

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# GoExploreTMD agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class GoExploreTMDSimple:
    """Go-Explore Temporal Metric Distillation agent.

    Mirrors the structure of GoExploreCRLSimple but uses TMD (phi/psi
    encoders with MRN or IQE quasimetric) instead of CRL (InfoNCE + dot /
    norm energy function).
    """

    # ---- Optimiser --------------------------------------------------------
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    train_step_multiplier: int = 1  # kept for run.py compatibility

    # ---- RL ---------------------------------------------------------------
    discounting: float = 0.99

    # ---- Network architecture --------------------------------------------
    h_dim: int = 512
    n_hidden: int = 3
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = True       # layer norm (recommended for TMD)
    latent_dim: int = 512     # phi / psi output dimension

    # ---- TMD hyperparameters ---------------------------------------------
    alpha: float = 0.1        # BC coefficient (NOT entropy temperature)
    zeta: float = 0.05        # weight for backup + invariance losses
    t: float = 3.0            # LINEX clipping threshold
    diag_backup: float = 0.5  # mixing weight for diagonal backup term
    use_iqe: bool = False     # True → IQE distance, False → MRN distance
    tmd_components: int = 8   # number of MRN / IQE components K
    const_std: bool = True    # use mode (not sample) for Q-actions
    stopgrad_psi_backup: bool = False
    stopgrad_phi_invariance: bool = False

    # ---- Goal proposers --------------------------------------------------
    # energy_fn is used by goal proposers (not by TMD losses themselves)
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"
    gcp_goal_proposer_name: Literal[
        "gcp_final_rb", "ep_final_rb", "env_goals",
        "ucgr", "maxwaypointratio_one_env", "q_epistemic",
        "mega", "omega", "empowerment_diff",
    ] = "gcp_final_rb"
    ep_goal_proposer_name: Literal[
        "gcp_final_rb", "ep_final_rb", "env_goals",
        "nearest_env_goal", "nearest_env_goal_to_gcp_goal",
    ] = "ep_final_rb"
    goal_sampling_temperature: float = 1.0
    num_rb_goals: int = 256
    candidate_goals_type: Literal["final", "any"] = "final"
    filter_successful_waypoints: bool = False

    # ---- Replay buffer ---------------------------------------------------
    max_replay_size: int = 10000
    min_replay_size: int = 1000

    # ---- Policy sharing --------------------------------------------------
    use_same_policy: bool = True
    use_gcp_noise: bool = False
    use_gcp_critic_ensemble: bool = False
    gcp_num_critic_ensemble: int = 5   # kept for proposer compat

    # ---- Empowerment proposer (optional) ---------------------------------
    empowerment_num_outer_samples: int = 10
    empowerment_num_inner_actions: int = 10
    gcp_empowerment_penalty: float = 1.0

    # ---- Training loop ---------------------------------------------------
    unroll_length: int = 50

    # ---- Unused CRL params (kept for interface compatibility) ------------
    logsumexp_penalty_coeff: float = 0.1
    contrastive_loss_fn: str = "fwd_infonce"
    disable_entropy_actor: bool = False

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0
        assert config.num_goal_conditioned_steps % self.unroll_length == 0
        assert config.num_exploratory_steps % self.unroll_length == 0
        assert self.latent_dim % self.tmd_components == 0, (
            f"latent_dim ({self.latent_dim}) must be divisible by "
            f"tmd_components ({self.tmd_components})"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        logging.info("Num env: %d", config.num_envs)

        unroll_length = self.unroll_length
        total_steps_per_training_step = (
            config.num_goal_conditioned_steps + config.num_exploratory_steps
        )
        env_steps_per_actor_step = config.num_envs * unroll_length
        env_steps_per_training_step = config.num_envs * total_steps_per_training_step
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(
            self.min_replay_size
            / (config.num_goal_conditioned_steps + config.num_exploratory_steps)
        )
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (config.num_evals * env_steps_per_training_step)
        )
        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        # ---- Seeds & env reset -------------------------------------------
        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        (key, buffer_key, eval_env_key, env_key,
         gcp_actor_key, gcp_phi1_key, gcp_phi2_key, gcp_psi1_key, gcp_psi2_key,
         ep_actor_key, ep_phi1_key, ep_phi2_key, ep_psi1_key, ep_psi2_key) = (
            jax.random.split(key, 14)
        )

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
        initial_info = dict(env_state.info)
        initial_info["traj_id"] = env_indices
        initial_info["gc_proposed_goals"] = jnp.zeros(
            (config.num_envs, len(train_env.goal_indices))
        )
        initial_info["ep_proposed_goals"] = jnp.zeros(
            (config.num_envs, len(train_env.goal_indices))
        )
        env_state = env_state.replace(info=initial_info)

        # ---- Dimensions --------------------------------------------------
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size

        # ---- Networks ----------------------------------------------------
        # Actors (same architecture as CRL)
        gcp_actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor.init(gcp_actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        ep_actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        ep_actor_state = TrainState.create(
            apply_fn=ep_actor.apply,
            params=ep_actor.init(ep_actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # TMD phi (state+action → latent) and psi (goal → latent) encoders
        # Both GCP and EP share the same encoder *architecture* but have
        # independent parameters (two critics = two sets of phi/psi params).
        phi_encoder = TMDEncoder(
            repr_dim=self.latent_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            value_exp=True,
        )
        psi_encoder = TMDEncoder(
            repr_dim=self.latent_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            value_exp=True,
        )

        # ---- Critic params (ensemble of 2 per policy) --------------------
        def make_critic_params(phi1_key, phi2_key, psi1_key, psi2_key):
            params = {
                "phi_1": phi_encoder.init(phi1_key, np.ones([1, state_size + action_size])),
                "phi_2": phi_encoder.init(phi2_key, np.ones([1, state_size + action_size])),
                "psi_1": psi_encoder.init(psi1_key, np.ones([1, goal_size])),
                "psi_2": psi_encoder.init(psi2_key, np.ones([1, goal_size])),
            }
            if self.use_iqe:
                params["iqe_alpha_raw"] = jnp.zeros(())
            return params

        gcp_critic_state = TrainState.create(
            apply_fn=None,
            params=make_critic_params(gcp_phi1_key, gcp_phi2_key,
                                      gcp_psi1_key, gcp_psi2_key),
            tx=optax.adam(learning_rate=self.critic_lr),
        )
        ep_critic_state = TrainState.create(
            apply_fn=None,
            params=make_critic_params(ep_phi1_key, ep_phi2_key,
                                      ep_psi1_key, ep_psi2_key),
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # ---- Training state (no alpha) -----------------------------------
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            gcp_actor_state=gcp_actor_state,
            ep_actor_state=ep_actor_state,
            gcp_critic_state=gcp_critic_state,
            ep_critic_state=ep_critic_state,
            traj_counter=jnp.zeros((), dtype=jnp.int32),
        )

        # ---- Replay buffers (identical to CRL) ---------------------------
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_goal = jnp.zeros((goal_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    "in_gc_phase": 0.0,
                    "in_ep_phase": 0.0,
                    "gc_proposed_goals": dummy_goal,
                    "ep_proposed_goals": dummy_goal,
                    "terminated": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        main_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        main_buffer_state = jax.jit(main_replay_buffer.init)(buffer_key)

        gcp_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        gcp_buffer_state = jax.jit(gcp_replay_buffer.init)(buffer_key)

        ep_replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        ep_buffer_state = jax.jit(ep_replay_buffer.init)(buffer_key)

        # ---- Goal proposers (same as CRL) --------------------------------
        gcp_final_rb_proposer = FinalReplayBufferProposer(
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        ep_final_rb_proposer = FinalReplayBufferProposer(
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        env_goals_proposer = RandomEnvironmentGoalProposer()
        nearest_env_goal_proposer = NearestEnvGoalProposer(
            energy_fn_name=self.energy_fn
        )
        nearest_env_goal_to_gcp_goal_proposer = NearestEnvGoalToGCPGoalProposer(
            energy_fn_name=self.energy_fn
        )
        ucgr_proposer = UCGRProposer(
            energy_fn_name=self.energy_fn,
            num_rb_samples=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        maxwaypointratio_one_env_proposer = MaxWaypointRatioOneEnvProposer(
            energy_fn_name=self.energy_fn,
            goal_sampling_temperature=self.goal_sampling_temperature,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
            filter_successful_waypoints=self.filter_successful_waypoints,
        )
        q_epistemic_proposer = QEpistemicProposer(
            energy_fn_name=self.energy_fn,
            num_ensemble=self.gcp_num_critic_ensemble,
            use_env_goals=False,
            zero_center=False,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        mega_proposer = MEGAProposer(
            bandwidth=0.1,
            use_q_cutoff=True,
            cutoff_percentile=0.3,
            energy_fn_name=self.energy_fn,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        omega_proposer = OMEGAProposer(
            bandwidth=0.1,
            use_q_cutoff=True,
            cutoff_percentile=0.3,
            energy_fn_name=self.energy_fn,
            bias_param=-3.0,
            num_rb_goals=self.num_rb_goals,
            candidate_goals_type=self.candidate_goals_type,
        )
        empowerment_diff_proposer = EmpowermentDifferenceGoalProposer(
            energy_fn_name=self.energy_fn,
            empowerment_num_outer_samples=self.empowerment_num_outer_samples,
            empowerment_num_inner_actions=self.empowerment_num_inner_actions,
            gcp_empowerment_penalty=self.gcp_empowerment_penalty,
            num_rb_goals=self.num_rb_goals,
            discounting=self.discounting,
            goal_sampling_temperature=self.goal_sampling_temperature,
        )

        # ---- Compatibility helpers for proposers -------------------------
        # Goal proposers expect CRL-style {"sa_encoder": ..., "g_encoder": ...}
        # params.  We map phi_1 → sa_encoder and psi_1 → g_encoder.
        def gcp_compat_critic_params(training_state):
            cp = training_state.gcp_critic_state.params
            return {"sa_encoder": cp["phi_1"], "g_encoder": cp["psi_1"]}

        def ep_compat_critic_params(training_state):
            cp = training_state.ep_critic_state.params
            return {"sa_encoder": cp["phi_1"], "g_encoder": cp["psi_1"]}

        # ---- GCP goal proposer selection ---------------------------------
        if self.gcp_goal_proposer_name == "gcp_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(
                    gcp_replay_buffer, gcp_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, updated_gcp, ep_buffer_state

        elif self.gcp_goal_proposer_name == "ep_final_rb":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.gcp_goal_proposer_name == "env_goals":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = env_goals_proposer.propose_goals(
                    None, gcp_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state

        elif self.gcp_goal_proposer_name == "ucgr":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ucgr_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.gcp_goal_proposer_name == "maxwaypointratio_one_env":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = maxwaypointratio_one_env_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.gcp_goal_proposer_name == "q_epistemic":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_gcp = q_epistemic_proposer.propose_goals(
                    gcp_replay_buffer, gcp_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, updated_gcp, ep_buffer_state

        elif self.gcp_goal_proposer_name == "mega":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = mega_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.gcp_goal_proposer_name == "omega":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = omega_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.gcp_goal_proposer_name == "empowerment_diff":
            def gcp_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = empowerment_diff_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                    gcp_replay_buffer=gcp_replay_buffer,
                    gcp_buffer_state=gcp_buffer_state,
                    ep_replay_buffer=ep_replay_buffer,
                    ep_buffer_state=ep_buffer_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        else:
            raise ValueError(f"Unknown gcp_goal_proposer_name: {self.gcp_goal_proposer_name}")

        # ---- EP goal proposer selection ----------------------------------
        if self.ep_goal_proposer_name == "gcp_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_gcp = gcp_final_rb_proposer.propose_goals(
                    gcp_replay_buffer, gcp_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params,
                    ep_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, updated_gcp, ep_buffer_state

        elif self.ep_goal_proposer_name == "ep_final_rb":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, updated_ep = ep_final_rb_proposer.propose_goals(
                    ep_replay_buffer, ep_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params,
                    ep_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, updated_ep

        elif self.ep_goal_proposer_name == "env_goals":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = env_goals_proposer.propose_goals(
                    None, ep_buffer_state, env, env_state, key,
                    ep_actor, training_state.ep_actor_state.params,
                    ep_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state

        elif self.ep_goal_proposer_name == "nearest_env_goal":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = nearest_env_goal_proposer.propose_goals(
                    None, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state

        elif self.ep_goal_proposer_name == "nearest_env_goal_to_gcp_goal":
            def ep_propose_goals(gcp_buffer_state, ep_buffer_state, env, env_state, key, training_state):
                proposed_goals, _ = nearest_env_goal_to_gcp_goal_proposer.propose_goals(
                    None, ep_buffer_state, env, env_state, key,
                    gcp_actor, training_state.gcp_actor_state.params,
                    gcp_compat_critic_params(training_state),
                    phi_encoder, psi_encoder, training_state,
                )
                return proposed_goals, gcp_buffer_state, ep_buffer_state

        else:
            raise ValueError(f"Unknown ep_goal_proposer_name: {self.ep_goal_proposer_name}")

        # ---- Actor step helpers (identical to CRL) -----------------------
        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = gcp_actor.apply(training_state.gcp_actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def deterministic_ep_actor_step(training_state, env, env_state, extra_fields):
            means, _ = ep_actor.apply(training_state.ep_actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def deterministic_actor_step_with_proposals(
            actor_state, env, env_state, proposed_goals, extra_fields
        ):
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)
            means, _ = actor_state.apply_fn(actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, proposed_goals, key, extra_fields):
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)
            means, log_stds = actor_state.apply_fn(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(
                means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        # ---- Experience collection (identical to CRL) --------------------
        @jax.jit
        def get_experience_chunk(
            gcp_actor_state, ep_actor_state, env_state, training_state,
            main_buffer_state, gcp_buffer_state, ep_buffer_state,
            gc_steps_collected, ep_steps_collected, gc_goals_proposed, ep_goals_proposed,
            gc_proposed_goals_state, ep_proposed_goals_state, key,
        ):
            num_envs = config.num_envs
            num_goal_conditioned_steps = config.num_goal_conditioned_steps
            num_exploratory_steps = config.num_exploratory_steps

            in_gc_phase = gc_steps_collected < num_goal_conditioned_steps
            steps_to_collect = unroll_length

            key, gc_key, ep_key = jax.random.split(key, 3)

            def collect_gc_phase():
                def propose_gc_goals_fn():
                    proposed, updated_gcp, updated_ep = gcp_propose_goals(
                        gcp_buffer_state, ep_buffer_state, train_env, env_state,
                        gc_key, training_state,
                    )
                    return proposed, updated_gcp, updated_ep

                def use_existing_gc_goals():
                    return gc_proposed_goals_state, gcp_buffer_state, ep_buffer_state

                gc_proposed_goals_new, gcp_buffer_state_new, ep_buffer_state_new = (
                    jax.lax.cond(~gc_goals_proposed, propose_gc_goals_fn, use_existing_gc_goals)
                )
                gc_proposed_goals = jnp.where(
                    ~gc_goals_proposed, gc_proposed_goals_new, gc_proposed_goals_state
                )

                new_info = dict(env_state.info)
                new_info["gc_proposed_goals"] = gc_proposed_goals
                new_info["ep_proposed_goals"] = env_state.info.get(
                    "ep_proposed_goals", jnp.zeros_like(gc_proposed_goals)
                )

                def update_env_with_goals():
                    return env_state.replace(
                        obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(gc_proposed_goals),
                        info=new_info,
                    )

                def keep_env_state():
                    return env_state.replace(info=new_info)

                env_state_updated = jax.lax.cond(
                    ~gc_goals_proposed, update_env_with_goals, keep_env_state
                )

                def gc_rollout_step(carry, unused_t):
                    env_state_inner, gcp_actor_state_inner, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    gc_goals_inner = env_state_inner.info.get(
                        "gc_proposed_goals",
                        env_state_inner.obs[:, -len(train_env.goal_indices):]
                    )
                    if self.use_gcp_noise:
                        nstate, transition = actor_step(
                            gcp_actor_state_inner, train_env, env_state_inner,
                            gc_goals_inner, current_key, ("truncation", "traj_id"),
                        )
                    else:
                        nstate, transition = deterministic_actor_step_with_proposals(
                            gcp_actor_state_inner, train_env, env_state_inner,
                            gc_goals_inner, ("truncation", "traj_id"),
                        )
                    truncation = transition.extras["state_extras"]["truncation"]
                    terminated = (transition.discount < 1.0) | (truncation > 0.5)
                    new_info = dict(nstate.info)
                    new_info["gc_proposed_goals"] = gc_goals_inner
                    env_state_inner = nstate.replace(info=new_info)
                    transition_extras = {
                        **transition.extras["state_extras"],
                        "in_gc_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                        "in_ep_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                        "gc_proposed_goals": gc_goals_inner,
                        "ep_proposed_goals": gc_goals_inner,
                        "terminated": terminated.astype(jnp.float32),
                    }
                    transition = transition._replace(extras={"state_extras": transition_extras})
                    return (env_state_inner, gcp_actor_state_inner, next_key), transition

                (env_state_final, _, _), gc_transitions = jax.lax.scan(
                    gc_rollout_step,
                    (env_state_updated, gcp_actor_state, key),
                    (),
                    length=steps_to_collect,
                )

                main_buffer_state_new2 = main_replay_buffer.insert(main_buffer_state, gc_transitions)
                gcp_buffer_state_final = gcp_replay_buffer.insert(gcp_buffer_state_new, gc_transitions)

                return (
                    env_state_final, main_buffer_state_new2, gcp_buffer_state_final,
                    ep_buffer_state_new,
                    gc_steps_collected + steps_to_collect, ep_steps_collected,
                    True, ep_goals_proposed,
                    gc_proposed_goals, ep_proposed_goals_state, gc_transitions,
                )

            def collect_ep_phase():
                def propose_ep_goals_fn():
                    proposed, updated_gcp, updated_ep = ep_propose_goals(
                        gcp_buffer_state, ep_buffer_state, train_env, env_state,
                        ep_key, training_state,
                    )
                    return proposed, updated_gcp, updated_ep

                def use_existing_ep_goals():
                    return ep_proposed_goals_state, gcp_buffer_state, ep_buffer_state

                ep_proposed_goals_new, gcp_buffer_state_new, ep_buffer_state_new = (
                    jax.lax.cond(~ep_goals_proposed, propose_ep_goals_fn, use_existing_ep_goals)
                )
                ep_proposed_goals = jnp.where(
                    ~ep_goals_proposed, ep_proposed_goals_new, ep_proposed_goals_state
                )

                new_info = dict(env_state.info)
                new_info["ep_proposed_goals"] = ep_proposed_goals
                new_info["gc_proposed_goals"] = env_state.info.get(
                    "gc_proposed_goals", jnp.zeros_like(ep_proposed_goals)
                )

                def update_env_with_ep_goals():
                    return env_state.replace(
                        obs=env_state.obs.at[:, -len(train_env.goal_indices):].set(ep_proposed_goals),
                        info=new_info,
                    )

                def keep_env_state():
                    return env_state.replace(info=new_info)

                env_state_updated = jax.lax.cond(
                    ~ep_goals_proposed, update_env_with_ep_goals, keep_env_state
                )

                def ep_rollout_step(carry, unused_t):
                    env_state_inner, ep_actor_state_inner, current_key = carry
                    current_key, next_key = jax.random.split(current_key)
                    ep_goals_inner = env_state_inner.info.get(
                        "ep_proposed_goals",
                        env_state_inner.obs[:, -len(train_env.goal_indices):]
                    )
                    if self.use_same_policy:
                        nstate, transition = actor_step(
                            gcp_actor_state, train_env, env_state_inner,
                            ep_goals_inner, current_key, ("truncation", "traj_id"),
                        )
                    else:
                        nstate, transition = actor_step(
                            ep_actor_state_inner, train_env, env_state_inner,
                            ep_goals_inner, current_key, ("truncation", "traj_id"),
                        )
                    truncation = transition.extras["state_extras"]["truncation"]
                    terminated = (transition.discount < 1.0) | (truncation > 0.5)
                    new_info = dict(nstate.info)
                    new_info["ep_proposed_goals"] = ep_goals_inner
                    new_info["gc_proposed_goals"] = env_state_inner.info.get(
                        "gc_proposed_goals", ep_goals_inner
                    )
                    env_state_inner = nstate.replace(info=new_info)
                    gc_goals_for_transition = env_state_inner.info.get(
                        "gc_proposed_goals", ep_goals_inner
                    )
                    transition_extras = {
                        **transition.extras["state_extras"],
                        "in_gc_phase": jnp.zeros((num_envs,), dtype=jnp.float32),
                        "in_ep_phase": jnp.ones((num_envs,), dtype=jnp.float32),
                        "gc_proposed_goals": gc_goals_for_transition,
                        "ep_proposed_goals": ep_goals_inner,
                        "terminated": terminated.astype(jnp.float32),
                    }
                    transition = transition._replace(extras={"state_extras": transition_extras})
                    return (env_state_inner, ep_actor_state_inner, next_key), transition

                (env_state_final, _, _), ep_transitions = jax.lax.scan(
                    ep_rollout_step,
                    (env_state_updated, ep_actor_state, key),
                    (),
                    length=steps_to_collect,
                )

                main_buffer_state_new2 = main_replay_buffer.insert(main_buffer_state, ep_transitions)
                ep_buffer_state_final = ep_replay_buffer.insert(ep_buffer_state_new, ep_transitions)

                return (
                    env_state_final, main_buffer_state_new2, gcp_buffer_state_new,
                    ep_buffer_state_final,
                    gc_steps_collected, ep_steps_collected + steps_to_collect,
                    gc_goals_proposed, True,
                    gc_proposed_goals_state, ep_proposed_goals, ep_transitions,
                )

            return jax.lax.cond(in_gc_phase, collect_gc_phase, collect_ep_phase)

        # ---- Prefill (identical to CRL) ----------------------------------
        def prefill_replay_buffer(
            training_state, env_state, main_buffer_state, gcp_buffer_state,
            ep_buffer_state, key,
        ):
            initial_info = dict(env_state.info)
            initial_info["gc_proposed_goals"] = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )
            initial_info["ep_proposed_goals"] = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )
            env_state = env_state.replace(info=initial_info)

            @jax.jit
            def f(carry, unused):
                del unused
                (training_state, env_state, main_buffer_state,
                 gcp_buffer_state, ep_buffer_state, key) = carry
                key, reset_key, new_key = jax.random.split(key, 3)

                num_gc_chunks = config.num_goal_conditioned_steps // unroll_length
                num_ep_chunks = config.num_exploratory_steps // unroll_length
                total_chunks = num_gc_chunks + num_ep_chunks
                experience_keys = jax.random.split(key, total_chunks)

                gc_steps_collected = 0
                ep_steps_collected = 0
                gc_goals_proposed = False
                ep_goals_proposed = False
                gc_proposed_goals_state = jnp.zeros(
                    (config.num_envs, len(train_env.goal_indices))
                )
                ep_proposed_goals_state = jnp.zeros(
                    (config.num_envs, len(train_env.goal_indices))
                )

                def collect_chunk(carry_inner, chunk_idx):
                    (env_s, mbs, gcbs, ebs,
                     gc_steps, ep_steps, gc_prop, ep_prop,
                     gc_goals, ep_goals) = carry_inner

                    (env_s_new, mbs_new, gcbs_new, ebs_new,
                     gc_steps_new, ep_steps_new, gc_prop_new, ep_prop_new,
                     gc_goals_new, ep_goals_new, _) = get_experience_chunk(
                        training_state.gcp_actor_state,
                        training_state.ep_actor_state,
                        env_s, training_state,
                        mbs, gcbs, ebs,
                        gc_steps, ep_steps, gc_prop, ep_prop,
                        gc_goals, ep_goals,
                        experience_keys[chunk_idx],
                    )
                    return (env_s_new, mbs_new, gcbs_new, ebs_new,
                            gc_steps_new, ep_steps_new, gc_prop_new, ep_prop_new,
                            gc_goals_new, ep_goals_new), ()

                initial_carry = (
                    env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state,
                    gc_steps_collected, ep_steps_collected,
                    gc_goals_proposed, ep_goals_proposed,
                    gc_proposed_goals_state, ep_proposed_goals_state,
                )
                (env_state_final, mbs_final, gcbs_final, ebs_final,
                 _, _, _, _, _, _), _ = jax.lax.scan(
                    collect_chunk, initial_carry, jnp.arange(total_chunks)
                )

                reset_keys = jax.random.split(reset_key, config.num_envs)
                env_state_final = jax.jit(train_env.reset)(reset_keys)
                env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
                final_info = dict(env_state_final.info)
                final_info["traj_id"] = (
                    training_state.traj_counter * config.num_envs + env_indices
                )
                final_info["gc_proposed_goals"] = jnp.zeros(
                    (config.num_envs, len(train_env.goal_indices))
                )
                final_info["ep_proposed_goals"] = jnp.zeros(
                    (config.num_envs, len(train_env.goal_indices))
                )
                env_state_final = env_state_final.replace(info=final_info)

                training_state = training_state.replace(
                    env_steps=training_state.env_steps + (
                        config.num_goal_conditioned_steps + config.num_exploratory_steps
                    ) * config.num_envs,
                    traj_counter=training_state.traj_counter + 1,
                )
                return (training_state, env_state_final, mbs_final, gcbs_final, ebs_final, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state,
                 ep_buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ---- Network update (TMD-specific) --------------------------------
        @jax.jit
        def update_networks(carry, gcp_transitions, ep_transitions):
            training_state, key = carry
            key, gcp_critic_key, gcp_actor_key, ep_critic_key, ep_actor_key = (
                jax.random.split(key, 5)
            )

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
            )

            gcp_networks = dict(actor=gcp_actor, phi=phi_encoder, psi=psi_encoder)
            ep_networks = dict(actor=ep_actor, phi=phi_encoder, psi=psi_encoder)

            # GCP critic → GCP actor
            new_gcp_critic_state, gcp_critic_metrics = update_tmd_critic(
                context, gcp_networks, gcp_transitions,
                training_state.gcp_critic_state, gcp_critic_key,
            )
            new_gcp_actor_state, gcp_actor_metrics = update_tmd_actor(
                context, gcp_networks, gcp_transitions,
                training_state.gcp_actor_state, new_gcp_critic_state, gcp_actor_key,
            )

            # EP critic → EP actor
            new_ep_critic_state, ep_critic_metrics = update_tmd_critic(
                context, ep_networks, ep_transitions,
                training_state.ep_critic_state, ep_critic_key,
            )
            new_ep_actor_state, ep_actor_metrics = update_tmd_actor(
                context, ep_networks, ep_transitions,
                training_state.ep_actor_state, new_ep_critic_state, ep_actor_key,
            )

            training_state = training_state.replace(
                gcp_actor_state=new_gcp_actor_state,
                gcp_critic_state=new_gcp_critic_state,
                ep_actor_state=new_ep_actor_state,
                ep_critic_state=new_ep_critic_state,
                gradient_steps=training_state.gradient_steps + 1,
            )

            metrics = {
                # GCP
                "gcp_actor_loss": gcp_actor_metrics["actor_loss"],
                "gcp_q_mean": gcp_actor_metrics["q_mean"],
                "gcp_q_abs_mean": gcp_actor_metrics["q_abs_mean"],
                "gcp_bc_log_prob": gcp_actor_metrics["bc_log_prob"],
                "gcp_critic_loss": gcp_critic_metrics["critic_loss"],
                "gcp_contrastive_loss": gcp_critic_metrics["contrastive_loss"],
                "gcp_backup_loss": gcp_critic_metrics["backup_loss"],
                "gcp_invariance_loss": gcp_critic_metrics["invariance_loss"],
                "gcp_categorical_accuracy": gcp_critic_metrics["categorical_accuracy"],
                "gcp_logits_pos": gcp_critic_metrics["logits_pos"],
                "gcp_logits_neg": gcp_critic_metrics["logits_neg"],
                "gcp_dist_mean": gcp_critic_metrics["dist_mean"],
                # EP
                "ep_actor_loss": ep_actor_metrics["actor_loss"],
                "ep_q_mean": ep_actor_metrics["q_mean"],
                "ep_q_abs_mean": ep_actor_metrics["q_abs_mean"],
                "ep_bc_log_prob": ep_actor_metrics["bc_log_prob"],
                "ep_critic_loss": ep_critic_metrics["critic_loss"],
                "ep_contrastive_loss": ep_critic_metrics["contrastive_loss"],
                "ep_backup_loss": ep_critic_metrics["backup_loss"],
                "ep_invariance_loss": ep_critic_metrics["invariance_loss"],
                "ep_categorical_accuracy": ep_critic_metrics["categorical_accuracy"],
                "ep_logits_pos": ep_critic_metrics["logits_pos"],
                "ep_logits_neg": ep_critic_metrics["logits_neg"],
                "ep_dist_mean": ep_critic_metrics["dist_mean"],
            }

            return (training_state, key), metrics

        # ---- Training step (identical structure to CRL) ------------------
        @jax.jit
        def training_step(
            training_state, env_state, main_buffer_state,
            gcp_buffer_state, ep_buffer_state, key,
        ):
            reset_key, experience_keys, sampling_keys, permute_keys, training_keys = (
                jax.random.split(key, 5)
            )

            num_gc_chunks = config.num_goal_conditioned_steps // unroll_length
            num_ep_chunks = config.num_exploratory_steps // unroll_length
            total_chunks = num_gc_chunks + num_ep_chunks

            experience_keys = jax.random.split(experience_keys, total_chunks)
            sampling_keys = jax.random.split(sampling_keys, total_chunks)
            permute_keys = jax.random.split(permute_keys, total_chunks)
            training_keys = jax.random.split(training_keys, total_chunks)

            gc_steps_collected = 0
            ep_steps_collected = 0
            gc_goals_proposed = False
            ep_goals_proposed = False
            gc_proposed_goals_state = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )
            ep_proposed_goals_state = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )

            def collect_and_update_chunk(carry, chunk_idx):
                (ts_inner, env_s, mbs, gcbs, ebs,
                 gc_steps, ep_steps, gc_prop, ep_prop,
                 gc_goals, ep_goals) = carry

                (env_s_new, mbs_new, gcbs_new, ebs_new,
                 gc_steps_new, ep_steps_new, gc_prop_new, ep_prop_new,
                 gc_goals_new, ep_goals_new, collected_transitions) = get_experience_chunk(
                    ts_inner.gcp_actor_state, ts_inner.ep_actor_state,
                    env_s, ts_inner,
                    mbs, gcbs, ebs,
                    gc_steps, ep_steps, gc_prop, ep_prop,
                    gc_goals, ep_goals,
                    experience_keys[chunk_idx],
                )

                ts_inner = ts_inner.replace(
                    env_steps=ts_inner.env_steps + env_steps_per_actor_step
                )

                # Sample training batches
                main_buffer_state_sampled, gcp_batch = main_replay_buffer.sample(mbs_new)
                if self.train_ep_on_main_buffer if hasattr(self, "train_ep_on_main_buffer") else False:
                    ep_batch = gcp_batch
                else:
                    ebs_sampled, ep_batch = ep_replay_buffer.sample(ebs_new)
                    ebs_new = ebs_sampled

                # Process (flatten) transitions
                total_trajs = gcp_batch.observation.shape[0]
                batch_keys = jax.random.split(sampling_keys[chunk_idx], total_trajs * 2)
                gcp_keys = batch_keys[:total_trajs]
                ep_keys = batch_keys[total_trajs:]

                def process_transitions(transitions, batch_keys_inner, permute_key):
                    transitions = jax.vmap(flatten_batch_tmd, in_axes=(None, 0, 0))(
                        (self.discounting, state_size, tuple(train_env.goal_indices)),
                        transitions,
                        batch_keys_inner,
                    )
                    transitions = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                        transitions,
                    )
                    permutation = jax.random.permutation(
                        permute_key, len(transitions.observation)
                    )
                    transitions = jax.tree_util.tree_map(
                        lambda x: x[permutation], transitions
                    )
                    transitions = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                        transitions,
                    )
                    return transitions

                gcp_last_batch = process_transitions(gcp_batch, gcp_keys, permute_keys[chunk_idx])
                ep_last_batch = process_transitions(ep_batch, ep_keys, permute_keys[chunk_idx])

                # Update networks (scan over minibatches)
                ((ts_updated, _), metrics) = jax.lax.scan(
                    lambda carry, xs: update_networks(carry, xs[0], xs[1]),
                    (ts_inner, training_keys[chunk_idx]),
                    (gcp_last_batch, ep_last_batch),
                )

                carry_new = (
                    ts_updated, env_s_new, mbs_new, gcbs_new, ebs_new,
                    gc_steps_new, ep_steps_new, gc_prop_new, ep_prop_new,
                    gc_goals_new, ep_goals_new,
                )
                return carry_new, (metrics, collected_transitions)

            initial_carry = (
                training_state, env_state, main_buffer_state,
                gcp_buffer_state, ep_buffer_state,
                gc_steps_collected, ep_steps_collected,
                gc_goals_proposed, ep_goals_proposed,
                gc_proposed_goals_state, ep_proposed_goals_state,
            )

            (ts_final, env_s_final, mbs_final, gcbs_final, ebs_final,
             _, _, _, _, _, _), (all_metrics, all_collected) = jax.lax.scan(
                collect_and_update_chunk, initial_carry, jnp.arange(total_chunks)
            )

            # Reset environments
            reset_keys = jax.random.split(reset_key, config.num_envs)
            env_s_final = jax.jit(train_env.reset)(reset_keys)
            env_indices = jnp.arange(config.num_envs, dtype=jnp.int32)
            final_info = dict(env_s_final.info)
            final_info["traj_id"] = ts_final.traj_counter * config.num_envs + env_indices
            final_info["gc_proposed_goals"] = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )
            final_info["ep_proposed_goals"] = jnp.zeros(
                (config.num_envs, len(train_env.goal_indices))
            )
            env_s_final = env_s_final.replace(info=final_info)
            ts_final = ts_final.replace(traj_counter=ts_final.traj_counter + 1)

            collected_transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), all_collected
            )
            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), all_metrics)

            return (ts_final, env_s_final, mbs_final, gcbs_final, ebs_final,
                    collected_transitions), metrics

        # ---- Training epoch (identical to CRL) ---------------------------
        @jax.jit
        def training_epoch(
            training_state, env_state, main_buffer_state,
            gcp_buffer_state, ep_buffer_state, key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, mbs, gcbs, ebs, k, _ = carry
                k, train_key = jax.random.split(k, 2)
                (ts, es, mbs, gcbs, ebs, collected_transitions), metrics = training_step(
                    ts, es, mbs, gcbs, ebs, train_key
                )
                return (ts, es, mbs, gcbs, ebs, k, collected_transitions), metrics

            key, first_key = jax.random.split(key)
            ((training_state, env_state, main_buffer_state, gcp_buffer_state,
              ep_buffer_state, init_collected), first_metrics) = training_step(
                training_state, env_state, main_buffer_state,
                gcp_buffer_state, ep_buffer_state, first_key,
            )

            (training_state, env_state, main_buffer_state, gcp_buffer_state,
             ep_buffer_state, _, collected_transitions), rest_metrics = jax.lax.scan(
                f,
                (training_state, env_state, main_buffer_state, gcp_buffer_state,
                 ep_buffer_state, key, init_collected),
                (),
                length=num_training_steps_per_epoch - 1,
            )

            metrics = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a[None], b]),
                first_metrics, rest_metrics,
            )
            return (training_state, env_state, main_buffer_state, gcp_buffer_state,
                    ep_buffer_state, metrics, collected_transitions)

        # ---- Visualisation (identical to CRL) ----------------------------
        def visualize_goals(train_env, transitions, wandb_key):
            obs = np.array(transitions.observation)
            in_gc_phase = np.array(transitions.extras["state_extras"]["in_gc_phase"])
            in_ep_phase = np.array(transitions.extras["state_extras"]["in_ep_phase"])
            gc_proposed_goals = np.array(transitions.extras["state_extras"]["gc_proposed_goals"])
            ep_proposed_goals = np.array(transitions.extras["state_extras"]["ep_proposed_goals"])
            traj_ids = np.array(transitions.extras["state_extras"]["traj_id"])

            obs_flat = obs.reshape(-1, obs.shape[-1])
            in_gc_phase_flat = in_gc_phase.reshape(-1)
            in_ep_phase_flat = in_ep_phase.reshape(-1)
            gc_proposed_goals_flat = gc_proposed_goals.reshape(-1, len(train_env.goal_indices))
            ep_proposed_goals_flat = ep_proposed_goals.reshape(-1, len(train_env.goal_indices))
            traj_ids_flat = traj_ids.reshape(-1)

            unique_traj_ids = np.unique(traj_ids_flat)
            gc_final_states, ep_final_states = [], []
            start_states, gc_goals, ep_goals = [], [], []
            gc_intermediate_states_list, ep_intermediate_states_list = [], []

            for traj_id in unique_traj_ids:
                traj_mask = traj_ids_flat == traj_id
                traj_indices = np.sort(np.where(traj_mask)[0])
                start_states.append(obs_flat[traj_indices[0]][train_env.goal_indices])

                gc_mask = in_gc_phase_flat[traj_indices] > 0.5
                gc_indices = traj_indices[gc_mask]
                if len(gc_indices) > 0:
                    gc_final_states.append(obs_flat[gc_indices[-1]][train_env.goal_indices])
                    gc_goals.append(gc_proposed_goals_flat[gc_indices[-1]])
                    gc_states = obs_flat[gc_indices][:, train_env.goal_indices]
                    n = len(gc_states)
                    idxs = np.linspace(0, n - 1, 6).astype(int) if n > 0 else []
                    gc_intermediate_states_list.append(gc_states[idxs] if n > 0 else np.array([]).reshape(0, len(train_env.goal_indices)))

                ep_mask = in_ep_phase_flat[traj_indices] > 0.5
                ep_indices = traj_indices[ep_mask]
                if len(ep_indices) > 0:
                    ep_final_states.append(obs_flat[ep_indices[-1]][train_env.goal_indices])
                    ep_goals.append(ep_proposed_goals_flat[ep_indices[-1]])
                    ep_states = obs_flat[ep_indices][:, train_env.goal_indices]
                    n = len(ep_states)
                    idxs = np.linspace(0, n - 1, 6).astype(int) if n > 0 else []
                    ep_intermediate_states_list.append(ep_states[idxs] if n > 0 else np.array([]).reshape(0, len(train_env.goal_indices)))

            goal_dim = len(train_env.goal_indices)
            start_states = np.array(start_states).reshape(-1, goal_dim)
            gc_final_states = np.array(gc_final_states).reshape(-1, goal_dim)
            ep_final_states = np.array(ep_final_states).reshape(-1, goal_dim)
            gc_goals = np.array(gc_goals).reshape(-1, goal_dim)
            ep_goals = np.array(ep_goals).reshape(-1, goal_dim)

            num_trajs = start_states.shape[0]
            num_viz = min(4, num_trajs)
            sample_indices = np.random.choice(num_trajs, num_viz, replace=False)

            visualize_dual_crl_trajectories_2d(
                start_states[sample_indices],
                gc_final_states[sample_indices],
                ep_final_states[sample_indices],
                gc_goals[sample_indices],
                ep_goals[sample_indices],
                [gc_intermediate_states_list[i] for i in sample_indices],
                [ep_intermediate_states_list[i] for i in sample_indices],
                f"{wandb_key}/dual_tmd_trajectories",
                x_bounds=train_env.x_bounds,
                y_bounds=train_env.y_bounds,
            )
            if len(gc_final_states) > 0:
                visualize_scatter_sample(
                    gc_final_states, "GC Final States",
                    f"{wandb_key}/gc_final_states_scatter",
                    train_env.x_bounds, train_env.y_bounds,
                )
            if len(ep_final_states) > 0:
                visualize_scatter_sample(
                    ep_final_states, "EP Final States",
                    f"{wandb_key}/ep_final_states_scatter",
                    train_env.x_bounds, train_env.y_bounds,
                )
            if len(gc_goals) > 0:
                visualize_scatter_sample(
                    gc_goals, "GC Goal Proposals",
                    f"{wandb_key}/gc_goal_proposals_scatter",
                    train_env.x_bounds, train_env.y_bounds,
                )
            if len(ep_goals) > 0:
                visualize_scatter_sample(
                    ep_goals, "EP Goal Proposals",
                    f"{wandb_key}/ep_goal_proposals_scatter",
                    train_env.x_bounds, train_env.y_bounds,
                )
            logging.info(f"TMD: plotted visualizations at step {training_state.env_steps.item()}")

        # ---- Main training loop ------------------------------------------
        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, main_buffer_state, gcp_buffer_state, ep_buffer_state, _ = (
            prefill_replay_buffer(
                training_state, env_state, main_buffer_state,
                gcp_buffer_state, ep_buffer_state, prefill_key,
            )
        )

        key, eval_gcp_key, eval_ep_key = jax.random.split(key, 3)
        gcp_evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_gcp_key,
        )
        ep_evaluator = ActorEvaluator(
            deterministic_ep_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_ep_key,
        )

        training_walltime = 0
        logging.info("starting TMD training....")
        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            (training_state, env_state, main_buffer_state,
             gcp_buffer_state, ep_buffer_state, metrics,
             collected_transitions) = training_epoch(
                training_state, env_state, main_buffer_state,
                gcp_buffer_state, ep_buffer_state, epoch_key,
            )

            # Visualisation
            @jax.jit
            def process_for_viz(transitions, batch_keys):
                processed = jax.vmap(flatten_batch_tmd, in_axes=(None, 1, 0))(
                    (self.discounting, state_size, tuple(train_env.goal_indices)),
                    transitions,
                    batch_keys,
                )
                processed = jax.tree_util.tree_map(
                    lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, len(x.shape)))),
                    processed,
                )
                return processed

            viz_key = jax.random.PRNGKey(0)
            num_envs_viz = collected_transitions.observation.shape[1]
            viz_batch_keys = jax.random.split(viz_key, num_envs_viz)
            processed_transitions = process_for_viz(collected_transitions, viz_batch_keys)
            visualize_goals(train_env, processed_transitions, wandb_key="training")

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                env_steps_per_training_step * num_training_steps_per_epoch
            ) / epoch_training_time

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = gcp_evaluator.run_evaluation(training_state, metrics)
            ep_eval_metrics = ep_evaluator.run_evaluation(training_state, {})
            for metric_key, metric_value in ep_eval_metrics.items():
                if metric_key.startswith("eval/"):
                    metrics[metric_key.replace("eval/", "ep_eval/")] = metric_value

            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: gcp_actor_state.apply_fn(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.gcp_actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                params = (
                    training_state.gcp_actor_state.params,
                    training_state.gcp_critic_state.params,
                    training_state.ep_actor_state.params,
                    training_state.ep_critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
            else:
                params = None

        total_steps = current_step
        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
