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
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import (
    flatten_q_ensemble_params,
    soft_update_q_ensemble_target,
    update_actor_and_alpha,
    update_critic,
    update_exploration_q_critic,
)
from .networks import Actor, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    # Set when an exploration bonus is configured. ``exploration_q_critic_states``
    # is a tuple of per-critic TrainStates (SAC-style ensemble); the target
    # params hold the polyak-averaged copy used for the Bellman backup.
    exploration_q_critic_states: Optional[Tuple[TrainState, ...]] = None
    exploration_q_target_critic_params: Optional[Any] = None


class Transition(NamedTuple):
    """Container for a transition.

    ``next_observation`` is populated unconditionally so the SAC-style
    exploration Q-critic and bonus pipeline have access to s_{t+1}; CRL's
    own contrastive critic and actor never read it.
    """

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray = jnp.zeros(())
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    next_state = transition.observation[1:, :state_size]
    new_obs = jnp.concatenate([state, goal], axis=1)
    next_obs = jnp.concatenate([next_state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            # is_online is the per-row mask routing online-only bonuses to
            # online rows after the RLPD online/offline concat. Always present —
            # set to ones for non-RLPD runs in training_step.
            "is_online": jnp.squeeze(transition.extras["state_extras"]["is_online"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        # Pair next_observation with the same HER goal as observation so
        # the SAC-style exploration Q backup and any bonus that reads
        # s_{t+1} (ICM, EME, APT, MISC, empowerment, online_empowerment)
        # see a consistent (s, a, s') triple.
        next_observation=jnp.squeeze(next_obs),
        extras=extras,
    )


def _make_actor_sample_fn(actor):
    """Build a ``(params, obs, key) -> sampled_action`` callable for the
    online_mine_empowerment marginal-action sampler. The bonus calls this
    with the agent's current actor params at each train step.
    """
    def sample_fn(params, obs, key):
        means, log_stds = actor.apply(params, obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        return nn.tanh(x_ts)
    return sample_fn


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRL:
    """Contrastive Reinforcement Learning (CRL) agent."""

    policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 30000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 512
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # ── RLPD (offline data mixing) ─────────────────────────────────────────
    # When True, mix 50% offline OGBench data into each training batch. The
    # online half is tagged is_online=1 in extras and the offline half with 0;
    # the mask is forwarded to the exploration-bonus dispatcher so online-only
    # bonuses (RND/MISC/ICM/EME/APT/online_empowerment/online_mine_empowerment/
    # max_empowerment) zero out on offline rows. The offline buffer is loaded
    # via ``load_and_prepare_offline_buffer`` so the env must have an OGBench
    # mapping in ``JAXGCRL_TO_OGBENCH``.
    use_rlpd: bool = False

    # ── Exploration bonus (added via a separate Q-critic on the bonus reward).
    # When ``exploration_bonus_type`` is set, a SAC-style Q-critic is trained
    # on the per-transition weighted bonus (per-bonus weight baked into the
    # Bellman target) and ``-min_i Q_exp_i(obs, action)`` is added to the CRL
    # actor loss. Mirrors the GoExploreSimple wiring for CRL with bonus.
    exploration_bonus_type: Optional[Tuple[str, ...]] = None
    exploration_bonus_weight: Tuple[float, ...] = (0.1,)
    bonus_grad_steps_per_env_step: Tuple[int, ...] = (1,)

    # Exploration Q-critic: SAC-style ensemble; tau is the target soft-update
    # rate; n_exp_critics is the ensemble size.
    tau: float = 0.005
    n_exp_critics: int = 2

    # ── RND ────────────────────────────────────────────────────────────────
    rnd_feature_dim: int = 64
    rnd_hidden_dim: int = 256
    rnd_num_hidden: int = 3
    rnd_learning_rate: float = 1e-4
    rnd_obs_clip: float = 5.0
    rnd_train_batch_size: int = 512
    use_goal_for_rnd: bool = False

    # ── MISC ───────────────────────────────────────────────────────────────
    misc_hidden_dim: int = 256
    misc_num_hidden: int = 3
    misc_learning_rate: float = 1e-3
    misc_alpha: float = 5000.0

    # ── ICM (Pathak et al., 2017) ──────────────────────────────────────────
    icm_feature_dim: int = 64
    icm_encoder_hidden_dim: int = 128
    icm_encoder_num_hidden: int = 2
    icm_inverse_hidden_dim: int = 128
    icm_inverse_num_hidden: int = 1
    icm_forward_hidden_dim: int = 128
    icm_forward_num_hidden: int = 1
    icm_learning_rate: float = 1e-3
    icm_beta: float = 0.2
    icm_eta: float = 1.0
    icm_obs_clip: float = 5.0
    icm_train_batch_size: int = 512

    # ── EME (Wang et al., 2024) ────────────────────────────────────────────
    eme_metric_hidden_dim: int = 256
    eme_metric_num_hidden: int = 2
    eme_reward_hidden_dim: int = 256
    eme_reward_num_hidden: int = 2
    eme_metric_learning_rate: float = 1e-3
    eme_reward_learning_rate: float = 1e-3
    eme_ensemble_size: int = 6
    eme_max_reward_scaling: float = 10.0
    eme_bootstrap_keep_prob: float = 0.8

    # ── APT (Liu & Abbeel, 2021) ───────────────────────────────────────────
    apt_repr_dim: int = 16
    apt_encoder_hidden_dim: int = 256
    apt_encoder_num_hidden: int = 2
    apt_projection_hidden_dim: int = 256
    apt_projection_num_hidden: int = 1
    apt_learning_rate: float = 1e-3
    apt_temperature: float = 0.1
    apt_knn_k: int = 3
    apt_knn_c: float = 1.0
    apt_aug_noise_std: float = 0.1
    apt_train_batch_size: int = 256

    # ── Offline empowerment scorer (loaded from disk) ──────────────────────
    empowerment_run_dir: Optional[str] = None
    empowerment_epoch: Optional[int] = None
    empowerment_num_splus_samples: int = 12
    empowerment_score_chunk_size: int = 32
    empowerment_bonus_mean: float = 1.3
    empowerment_bonus_scale: float = 0.2
    use_full_empowerment: bool = True

    # ── Online empowerment ────────────────────────────────────────────────
    online_empowerment_lr: float = 3e-4
    online_empowerment_value_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_empowerment_actor_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_empowerment_value_latent_dim: int = 128
    online_empowerment_num_skills: int = 5
    online_empowerment_num_splus_samples: int = 32
    online_empowerment_discount: float = 0.99
    online_empowerment_tau: float = 0.005
    online_empowerment_separate_qv: bool = False
    online_empowerment_use_self_q_loss: bool = True
    online_empowerment_layer_norm: bool = True
    online_empowerment_bc_alpha: float = 0.01
    online_empowerment_bonus_mean: float = 0.0
    online_empowerment_bonus_scale: float = 1.0

    # ── Online MINE empowerment ───────────────────────────────────────────
    online_mine_empowerment_lr_dyn: float = 1e-3
    online_mine_empowerment_lr_t: float = 3e-4
    online_mine_empowerment_dyn_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_mine_empowerment_t_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_mine_empowerment_layer_norm: bool = True
    online_mine_empowerment_bonus_mean: float = 0.0
    online_mine_empowerment_bonus_scale: float = 1.0

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
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

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size]))
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # ── Exploration bonus + Q-critic ─────────────────────────────────────
        # When ``exploration_bonus_type`` is set, build:
        #   1. A SAC-style Q-critic ensemble that takes (obs, action) and is
        #      trained on the per-transition weighted bonus reward.
        #   2. The exploration-bonus bundle (RND, ICM, EME, APT, MISC,
        #      empowerment, online_empowerment, online_mine_empowerment, ...)
        #      that produces the per-transition bonus on each training step.
        # The bonus is *not* added to the env reward — instead the actor loss
        # subtracts ``min_i Q_exp_i(obs, action)``. This mirrors how
        # GoExploreSimple handles bonuses for CRL agents.
        exploration_q_critic = None
        exploration_bonuses = None
        exploration_bonus_state = None
        bonus_types_tuple = tuple(self.exploration_bonus_type or ())
        use_exploration_q = len(bonus_types_tuple) > 0

        if use_exploration_q:
            from jaxgcrl.agents.go_explore.algorithms import get_exploration_q_critic
            from jaxgcrl.agents.go_explore.exploration import create_exploration_bonuses

            key, exp_q_key, bonus_init_key = jax.random.split(key, 3)
            exploration_q_critic = get_exploration_q_critic(
                obs_size=obs_size,
                action_size=action_size,
                h_dim=self.h_dim,
                n_hidden=self.n_hidden,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
                n_critics=self.n_exp_critics,
            )
            exploration_q_critic_params = exploration_q_critic.init(
                exp_q_key, np.ones([1, obs_size])
            )
            exploration_q_critic_states = exploration_q_critic.create_critic_states(
                exploration_q_critic_params, self.critic_lr,
            )
            exploration_q_target_critic_params = exploration_q_critic_params

            goal_indices_tuple = tuple(int(i) for i in np.asarray(train_env.goal_indices))
            controllable_indices_tuple = (
                tuple(int(i) for i in np.asarray(unwrapped_env.controllable_indices))
                if hasattr(unwrapped_env, "controllable_indices") else None
            )
            exploration_bonuses = create_exploration_bonuses(
                self.exploration_bonus_type,
                self.exploration_bonus_weight,
                env=unwrapped_env,
                state_size=state_size,
                key=bonus_init_key,
                discount=self.discounting,
                empowerment_run_dir=self.empowerment_run_dir,
                empowerment_epoch=self.empowerment_epoch,
                empowerment_num_splus_samples=self.empowerment_num_splus_samples,
                empowerment_score_chunk_size=self.empowerment_score_chunk_size,
                empowerment_mean=self.empowerment_bonus_mean,
                empowerment_scale=self.empowerment_bonus_scale,
                empowerment_use_full_obs=self.use_full_empowerment,
                rnd_feature_dim=self.rnd_feature_dim,
                rnd_hidden_dim=self.rnd_hidden_dim,
                rnd_num_hidden=self.rnd_num_hidden,
                rnd_learning_rate=self.rnd_learning_rate,
                rnd_obs_clip=self.rnd_obs_clip,
                rnd_use_goal=self.use_goal_for_rnd,
                rnd_train_batch_size=self.rnd_train_batch_size,
                goal_indices=goal_indices_tuple,
                controllable_indices=controllable_indices_tuple,
                misc_hidden_dim=self.misc_hidden_dim,
                misc_num_hidden=self.misc_num_hidden,
                misc_learning_rate=self.misc_learning_rate,
                misc_alpha=self.misc_alpha,
                icm_feature_dim=self.icm_feature_dim,
                icm_encoder_hidden_dim=self.icm_encoder_hidden_dim,
                icm_encoder_num_hidden=self.icm_encoder_num_hidden,
                icm_inverse_hidden_dim=self.icm_inverse_hidden_dim,
                icm_inverse_num_hidden=self.icm_inverse_num_hidden,
                icm_forward_hidden_dim=self.icm_forward_hidden_dim,
                icm_forward_num_hidden=self.icm_forward_num_hidden,
                icm_learning_rate=self.icm_learning_rate,
                icm_beta=self.icm_beta,
                icm_eta=self.icm_eta,
                icm_obs_clip=self.icm_obs_clip,
                icm_train_batch_size=self.icm_train_batch_size,
                # EME's metric loss reads the actor's pre-tanh (mean, log_std)
                # at arbitrary states for the closed-form Gaussian KL term.
                eme_actor_dist_fn=(lambda params, obs: actor.apply(params, obs)),
                eme_metric_hidden_dim=self.eme_metric_hidden_dim,
                eme_metric_num_hidden=self.eme_metric_num_hidden,
                eme_reward_hidden_dim=self.eme_reward_hidden_dim,
                eme_reward_num_hidden=self.eme_reward_num_hidden,
                eme_metric_learning_rate=self.eme_metric_learning_rate,
                eme_reward_learning_rate=self.eme_reward_learning_rate,
                eme_ensemble_size=self.eme_ensemble_size,
                eme_max_reward_scaling=self.eme_max_reward_scaling,
                eme_bootstrap_keep_prob=self.eme_bootstrap_keep_prob,
                apt_repr_dim=self.apt_repr_dim,
                apt_encoder_hidden_dim=self.apt_encoder_hidden_dim,
                apt_encoder_num_hidden=self.apt_encoder_num_hidden,
                apt_projection_hidden_dim=self.apt_projection_hidden_dim,
                apt_projection_num_hidden=self.apt_projection_num_hidden,
                apt_learning_rate=self.apt_learning_rate,
                apt_temperature=self.apt_temperature,
                apt_knn_k=self.apt_knn_k,
                apt_knn_c=self.apt_knn_c,
                apt_aug_noise_std=self.apt_aug_noise_std,
                apt_train_batch_size=self.apt_train_batch_size,
                online_empowerment_action_size=action_size,
                online_empowerment_lr=self.online_empowerment_lr,
                online_empowerment_value_hidden_dims=self.online_empowerment_value_hidden_dims,
                online_empowerment_actor_hidden_dims=self.online_empowerment_actor_hidden_dims,
                online_empowerment_value_latent_dim=self.online_empowerment_value_latent_dim,
                online_empowerment_num_skills=self.online_empowerment_num_skills,
                online_empowerment_num_splus_samples=self.online_empowerment_num_splus_samples,
                online_empowerment_discount=self.online_empowerment_discount,
                online_empowerment_tau=self.online_empowerment_tau,
                online_empowerment_separate_qv=self.online_empowerment_separate_qv,
                online_empowerment_use_self_q_loss=self.online_empowerment_use_self_q_loss,
                online_empowerment_layer_norm=self.online_empowerment_layer_norm,
                online_empowerment_bc_alpha=self.online_empowerment_bc_alpha,
                online_empowerment_bonus_mean=self.online_empowerment_bonus_mean,
                online_empowerment_bonus_scale=self.online_empowerment_bonus_scale,
                online_mine_empowerment_action_size=action_size,
                online_mine_empowerment_actor_sample_fn=_make_actor_sample_fn(actor),
                online_mine_empowerment_lr_dyn=self.online_mine_empowerment_lr_dyn,
                online_mine_empowerment_lr_t=self.online_mine_empowerment_lr_t,
                online_mine_empowerment_dyn_hidden_dims=self.online_mine_empowerment_dyn_hidden_dims,
                online_mine_empowerment_t_hidden_dims=self.online_mine_empowerment_t_hidden_dims,
                online_mine_empowerment_layer_norm=self.online_mine_empowerment_layer_norm,
                online_mine_empowerment_bonus_mean=self.online_mine_empowerment_bonus_mean,
                online_mine_empowerment_bonus_scale=self.online_mine_empowerment_bonus_scale,
            )
            exploration_bonus_state = exploration_bonuses.initial_state
        else:
            exploration_q_critic_states = None
            exploration_q_target_critic_params = None

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
            exploration_q_critic_states=exploration_q_critic_states,
            exploration_q_target_critic_params=exploration_q_target_critic_params,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            next_observation=dummy_obs,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # ── RLPD: offline buffer ─────────────────────────────────────────────
        # Sampled per training step and concatenated with the online batch
        # along axis 0 so HER/flatten_batch processes both online and offline
        # trajectories. Each offline transition is tagged is_online=0 so
        # online-only bonuses are masked off on those rows.
        offline_buffer = None
        if self.use_rlpd:
            from jaxgcrl.utils.offline_buffer import load_and_prepare_offline_buffer
            offline_buffer = load_and_prepare_offline_buffer(
                env_name=config.env,
                episode_length=config.episode_length,
                num_slots=config.num_envs,
                obs_size=obs_size,
                action_size=action_size,
                state_size=state_size,
                agent_type="crl",
                include_phase=False,
                include_max_empowerment=False,
            )

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=nstate.obs,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, key, extra_fields):
            means, log_stds = actor.apply(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=nstate.obs,
                extras={"state_extras": state_extras},
            )

        @jax.jit
        def get_experience(actor_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    buffer_state,
                    key,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, xs):
            transitions, exploration_reward = xs
            training_state, key = carry
            key, exp_q_key, critic_key, actor_key = jax.random.split(key, 4)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )
            if use_exploration_q:
                networks["exploration_q_critic"] = exploration_q_critic

            metrics = {}

            # Train Q_exp first so the actor loss in the same step uses the
            # freshly-updated values; Q_exp's Bellman target uses the *target*
            # net so this is consistent.
            if use_exploration_q:
                training_state, exp_q_metrics = update_exploration_q_critic(
                    context, networks, transitions, exploration_reward,
                    training_state, exp_q_key,
                )
                metrics.update(exp_q_metrics)

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )

            if use_exploration_q:
                training_state = training_state.replace(
                    exploration_q_target_critic_params=soft_update_q_ensemble_target(
                        training_state.exploration_q_target_critic_params,
                        training_state.exploration_q_critic_states,
                        self.tau,
                    ),
                )

            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, exploration_bonus_state, key):
            experience_key1, experience_key2, sampling_key, bonus_key, training_key, offline_key = jax.random.split(key, 6)

            # update buffer
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                env_state,
                buffer_state,
                experience_key1,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, online_transitions = replay_buffer.sample(buffer_state)

            # Tag online rows so the exploration bonus can be masked to apply
            # only to online data after concat+permute. flatten_batch reads
            # this and carries it through to the post-HER batch.
            online_transitions = online_transitions._replace(
                extras={
                    **online_transitions.extras,
                    "state_extras": {
                        **online_transitions.extras["state_extras"],
                        "is_online": jnp.ones_like(online_transitions.reward),
                    },
                }
            )

            if self.use_rlpd:
                # Mix 50% offline data: concatenate along the num_envs axis so
                # flatten_batch sees one batched set of trajectories. The
                # offline buffer's Transition is the go_explore.types one;
                # rebuild it as the local NamedTuple for tree_map structural
                # compatibility.
                raw_offline = offline_buffer.sample(offline_key, config.num_envs)
                offline_transitions = Transition(
                    observation=raw_offline.observation,
                    action=raw_offline.action,
                    reward=raw_offline.reward,
                    discount=raw_offline.discount,
                    next_observation=raw_offline.next_observation,
                    extras={
                        **raw_offline.extras,
                        "state_extras": {
                            **raw_offline.extras["state_extras"],
                            "is_online": jnp.zeros_like(raw_offline.reward),
                        },
                    },
                )
                pre_her_transitions = jax.tree_util.tree_map(
                    lambda a, b: jnp.concatenate([a, b], axis=0),
                    online_transitions, offline_transitions,
                )
            else:
                pre_her_transitions = online_transitions

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, pre_her_transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                pre_her_transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            if self.use_rlpd:
                # Slice to the original num_batches so the gradient-step count
                # per env step stays the same as the online-only path. The
                # permutation above already mixed online+offline ~50/50 within
                # each batch, so the surviving slice still covers both.
                num_batches = config.num_envs * (config.episode_length - 1) // self.batch_size
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:num_batches], transitions
                )

            # Compute per-transition exploration bonus across all configured
            # bonuses; the weighted total is what Q_exp is trained on. The
            # is_online mask routes online-only bonuses (RND, MISC, ICM, EME,
            # APT, online_empowerment, ...) to online rows only inside compute().
            if use_exploration_q:
                is_online = transitions.extras["state_extras"]["is_online"]
                bonus_compute_kwargs = (
                    {"actor_params": training_state.actor_state.params}
                    if exploration_bonuses.requires_actor_params else {}
                )
                total_bonus, _, exploration_bonus_state, bonus_metrics = (
                    exploration_bonuses.compute(
                        exploration_bonus_state, transitions, bonus_key,
                        is_online=is_online,
                        **bonus_compute_kwargs,
                    )
                )
                exploration_reward_per_batch = total_bonus
            else:
                exploration_reward_per_batch = jnp.zeros_like(transitions.reward)
                bonus_metrics = {}

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(
                update_networks,
                (training_state, training_key),
                (transitions, exploration_reward_per_batch),
            )
            metrics.update(bonus_metrics)

            # Train bonus-side trainables (RND predictor, ICM/EME/APT
            # encoders, online_empowerment networks, ...) on the online
            # batch only — bonus *emission* above runs read-only on the full
            # post-HER batch, but bonus *learning* must stay scoped to the
            # agent's own visited states.
            if use_exploration_q and exploration_bonuses.has_trainables:
                # Per-bonus train_fns advance their own internal PRNG state,
                # so no key needs to be threaded in here.
                for i in range(len(exploration_bonuses.bonus_types)):
                    if not exploration_bonuses.is_trainable(i):
                        continue
                    bonus_train_kwargs = (
                        {"actor_params": training_state.actor_state.params}
                        if exploration_bonuses.requires_actor_params_at(i) else {}
                    )
                    exploration_bonus_state, m_i = exploration_bonuses.train_one(
                        exploration_bonus_state, i, online_transitions, 1,
                        **bonus_train_kwargs,
                    )
                    metrics.update(m_i)

            return (
                training_state,
                env_state,
                buffer_state,
                exploration_bonus_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            exploration_bonus_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, ebs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                        ebs,
                    ),
                    metrics,
                ) = training_step(ts, es, bs, ebs, train_key)
                return (ts, es, bs, ebs, k), metrics

            (training_state, env_state, buffer_state, exploration_bonus_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, exploration_bonus_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, exploration_bonus_state, metrics

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # Seed per-bonus observation-normalization statistics (e.g. RND's
        # obs_mean / obs_std) from the post-prefill rollouts, per the RND
        # paper's "step a random agent for a small number of steps" warm-up.
        if use_exploration_q and not exploration_bonuses.is_empty:
            buffer_state, seed_transitions = replay_buffer.sample(buffer_state)
            seed_states = seed_transitions.observation[..., :state_size]
            exploration_bonus_state = exploration_bonuses.init_from_states(
                exploration_bonus_state, seed_states,
            )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, exploration_bonus_state, metrics = training_epoch(
                training_state, env_state, buffer_state, exploration_bonus_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        # assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
