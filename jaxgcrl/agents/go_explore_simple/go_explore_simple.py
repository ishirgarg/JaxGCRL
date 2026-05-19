"""Go Explore agent.

Two-phase training loop:
  - Go phase   (phase == 0): GCP navigates to a proposed frontier goal.
  - Explore phase (phase == 1): continuation of go phase with eps-random actions
    and stochastic policy sampling.

Phase management is handled by ``GoExploreWrapper`` (see ``jaxgcrl/envs/wrappers.py``).
"""

import logging
import random
import time
from typing import Callable, List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    GoExploreWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization

from jaxgcrl.agents.go_explore.types import TrainingState, Transition, GoalProposerState
from jaxgcrl.agents.go_explore.algorithms import get_algorithm, get_exploration_q_critic
from jaxgcrl.agents.go_explore.utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
)
from jaxgcrl.agents.go_explore.losses import (
    flatten_crl_critic_params,
    flatten_sac_critic_params,
    soft_update_target_params,
    update_alpha_sac,
    update_exploration_q_critic,
)
from jaxgcrl.agents.go_explore.visualization import (
    all_visualizations,
    visualize_exploration_q_xy,
    visualize_go_explore_phases,
)
from jaxgcrl.agents.go_explore.goal_proposers import (
    create_goal_proposer,
    create_random_env_goals_proposer,
)
from jaxgcrl.agents.go_explore.empowerment import (
    infer_empowerment_override_indices_from_env,
    load_offline_empowerment_agent,
    make_empowerment_full_obs_builder,
    make_empowerment_obs_builder,
    make_offline_empowerment_scorer,
)
import io
import numpy as np
import os

try:
    import matplotlib.pyplot as _plt
    import wandb as _wandb
except Exception:  # pragma: no cover
    _plt = None
    _wandb = None


def _scatter_panel(ax, fig, xs, ys, values, title, xlabel, ylabel, cbar_label,
                   x_bounds=None, y_bounds=None):
    """One scatter + colorbar panel; shared body for the reward/bonus heatmaps."""
    sc = ax.scatter(xs, ys, c=values, cmap="viridis", s=6, linewidths=0)
    if x_bounds is not None:
        ax.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
    if y_bounds is not None:
        ax.set_ylim(float(y_bounds[0]), float(y_bounds[1]))
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _log_reward_heatmap(reward_viz, x_bounds, y_bounds, current_step: int):
    """Log scatters of raw (ant x, ant y) transition samples colored by their
    post-bonus reward and (if present) the scaled exploration bonus alone.
    ``reward_viz`` is the ``(xs, ys, rewards, scaled_bonus, ...)`` snapshot taken
    inside ``training_step`` right after the exploration bonus is added."""
    if _plt is None or _wandb is None:
        return

    xs = np.asarray(reward_viz[0]).reshape(-1)
    ys = np.asarray(reward_viz[1]).reshape(-1)
    rs = np.asarray(reward_viz[2]).reshape(-1)
    bonus = np.asarray(reward_viz[3]).reshape(-1) if len(reward_viz) > 3 else None
    has_bonus = bonus is not None and np.any(bonus != 0)

    ncols = 2 if has_bonus else 1
    fig, axes = _plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    _scatter_panel(axes[0, 0], fig, xs, ys, rs,
                   "reward (post-bonus)", "Ant x", "Ant y",
                   "reward (post-bonus)", x_bounds, y_bounds)
    if has_bonus:
        _scatter_panel(axes[0, 1], fig, xs, ys, bonus,
                       "scaled exploration bonus", "Ant x", "Ant y",
                       "scaled exploration bonus", x_bounds, y_bounds)

    fig.suptitle(f"step={current_step}")
    fig.tight_layout()
    _wandb.log({"viz/reward_heatmap": _wandb.Image(fig)}, step=current_step)
    _plt.close(fig)


def _log_trajectory_reward(reward_viz, current_step: int):
    """Plot the post-override reward for a single training batch.

    ``reward_viz[2]`` is ``transitions.reward`` after the ``reward=total_bonus``
    replace inside ``training_step`` — the exact values the critic trains on.
    Batch is post-HER and post-permute, so the x axis is sample index within
    the batch, not chronological time.
    """
    if _plt is None or _wandb is None:
        return

    single = np.asarray(reward_viz[2]).reshape(-1)

    fig, ax = _plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(single.shape[0]), single, lw=1.0)
    ax.set_xlabel("sample index (shuffled)")
    ax.set_ylabel("reward (post-bonus override)")
    ax.set_title(f"training-batch reward  step={current_step}")
    fig.tight_layout()
    _wandb.log({"viz/trajectory_reward": _wandb.Image(fig)}, step=current_step)
    _plt.close(fig)


def _log_online_empowerment_xy_map(
    score_fn,
    bonus_state,
    template_state,
    state_size: int,
    x_bounds,
    y_bounds,
    grid_res: int,
    rng,
    current_step: int,
):
    """2D (x, y) heatmap of per-state empowerment from the online-empowerment bonus.

    ``score_fn`` is the ``bonus_fn.score_states`` attribute exposed by the
    online-empowerment factory: ``(bonus_state, states, key) -> (B,)``.
    A representative state vector ``template_state`` (shape ``(state_size,)``)
    is broadcast over the grid and columns 0,1 are overwritten with the
    grid's x and y values, mirroring the prior art in
    ``empowerment_sac._log_empowerment_map``.
    """
    if _plt is None or _wandb is None:
        return

    xs = np.linspace(float(x_bounds[0]), float(x_bounds[1]), grid_res)
    ys = np.linspace(float(y_bounds[0]), float(y_bounds[1]), grid_res)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = jnp.asarray(xx.reshape(-1), dtype=jnp.float32)
    flat_y = jnp.asarray(yy.reshape(-1), dtype=jnp.float32)
    n = flat_x.shape[0]

    obs_batch = jnp.broadcast_to(
        jnp.asarray(template_state, dtype=jnp.float32)[None, :], (n, state_size),
    )
    obs_batch = obs_batch.at[:, 0].set(flat_x).at[:, 1].set(flat_y)

    emps = np.asarray(score_fn(bonus_state, obs_batch, rng))
    emp_map = emps.reshape(grid_res, grid_res)

    fig, ax = _plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        emp_map, origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis", aspect="equal",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Empowerment")
    ax.set_xlabel("obs[0] (x)")
    ax.set_ylabel("obs[1] (y)")
    ax.set_title(f"online empowerment  step={current_step}")
    fig.tight_layout()
    _wandb.log({"viz/online_empowerment_xy_map": _wandb.Image(fig)}, step=current_step)
    _plt.close(fig)


def _log_exploration_bonus_goal_heatmap(reward_viz, goal_indices, current_step: int):
    """Scatter the total exploration bonus (no env reward) over the first-two
    and last-two ``goal_indices`` columns of the sampled states. ``reward_viz``
    must carry ``(..., goal_feat)`` at index 4."""
    if _plt is None or _wandb is None:
        return
    if len(reward_viz) < 5 or len(goal_indices) < 2:
        return

    bonus = np.asarray(reward_viz[3]).reshape(-1)
    goal_feat = np.asarray(reward_viz[4]).reshape(-1, len(goal_indices))

    fig, axes = _plt.subplots(1, 2, figsize=(10, 5))
    _scatter_panel(
        axes[0], fig, goal_feat[:, 0], goal_feat[:, 1], bonus,
        "bonus vs first two goal idx",
        f"obs[{goal_indices[0]}]", f"obs[{goal_indices[1]}]",
        "exploration bonus",
    )
    _scatter_panel(
        axes[1], fig, goal_feat[:, -2], goal_feat[:, -1], bonus,
        "bonus vs last two goal idx",
        f"obs[{goal_indices[-2]}]", f"obs[{goal_indices[-1]}]",
        "exploration bonus",
    )

    fig.suptitle(f"step={current_step}")
    fig.tight_layout()
    _wandb.log(
        {"viz/exploration_bonus_goal_heatmap": _wandb.Image(fig)},
        step=current_step,
    )
    _plt.close(fig)


Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class GoExploreSimple:
    """Go Explore agent with a single goal-conditioned policy.

    The go phase navigates to a proposed frontier goal. The explore phase
    continues with the same policy but samples stochastically and injects
    uniform random actions with probability ``eps_random_action``.
    """

    # Algorithm type for the goal-conditioned policy
    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 60000
    min_replay_size: int = 1000
    unroll_length: int = 100
    h_dim: int = 256
    n_hidden: int = 3
    skip_connections: int = 3
    use_relu: bool = False

    repr_dim: int = 64
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    tau: float = 0.005
    n_critics: int = 2
    use_her: bool = True
    p_future_her_goal: float = 0.8
    use_sac_critic_mean: bool = False  # SAC: average critic ensemble instead of min

    goal_proposer_name: Literal["random_env_goals", "rb", "q_epistemic", "ucgr", "max_critic_to_env", "mega", "omega", "empowerment", "empowerment_density_ratio"] = "random_env_goals"
    num_candidates: int = 512
    goal_proposer_temperature: float = 0.0
    empowerment_alpha: float = 1.0
    empowerment_run_dir: Optional[str] = None
    empowerment_epoch: Optional[int] = None
    empowerment_num_splus_samples: int = 12
    empowerment_score_chunk_size: int = 32
    # If True, feed the full state (obs with goal sliced off) to the
    # empowerment network instead of overwriting a few indices of a cached
    # OGBench template. Requires state_size == checkpoint ex_obs_dim.
    use_full_empowerment: bool = True

    # ── RLPD (offline data mixing) ─────────────────────────────────────────
    use_rlpd: bool = True  # Mix 50% offline OGBench data into each training batch

    # ── Exploration bonus (added to reward after HER) ──────────────────────
    exploration_bonus_type: Optional[Tuple[str, ...]] = None
    exploration_bonus_weight: Tuple[float, ...] = (0.1,)

    # Number of bonus-train gradient steps per env step, per-bonus. Mirrors
    # the layout of `exploration_bonus_weight`: a length-1 tuple is broadcast
    # to all configured bonuses, otherwise the length must match
    # `exploration_bonus_type` and each entry applies to the bonus at that
    # index. For each bonus with n>1, successive steps draw fresh online
    # batches from the replay buffer (needed for online_empowerment so the
    # estimator and policy gradient see uncorrelated data); n=1 reuses the
    # main policy batch bit-for-bit. UCB has its own ucb_grad_steps_per_train_step.
    bonus_grad_steps_per_env_step: Tuple[int, ...] = (1,)

    # Empowerment global normalization: emitted bonus is (raw_emp - mean) / scale.
    empowerment_bonus_mean: float = 1.3
    empowerment_bonus_scale: float = 0.2

    # RND predictor/target network shape.
    rnd_feature_dim: int = 64
    rnd_hidden_dim: int = 256
    rnd_num_hidden: int = 3
    rnd_learning_rate: float = 1e-4
    rnd_obs_clip: float = 5.0
    rnd_train_batch_size: int = 512
    # If True, RND operates on state[goal_indices] only (novelty over goal dims).
    use_goal_for_rnd: bool = False

    # ── MISC (mutual-information state-controllable intrinsic reward) ──────
    # Enabled by listing "misc" in `exploration_bonus_type`. Trains a small
    # MI estimator T_phi on online trajectories (MINE lower bound with
    # random temporal s_c shuffling) and adds clip(α * MI_surrogate, 0, 1)
    # to ONLINE rewards only — offline (RLPD) rows are not augmented.
    # Requires the env to expose `controllable_indices` (s_c); s_g comes
    # from `goal_indices`.
    misc_hidden_dim: int = 256
    misc_num_hidden: int = 3
    misc_learning_rate: float = 1e-3
    misc_alpha: float = 5000.0

    # ── ICM (Pathak et al., 2017) ──────────────────────────────────────────
    # Enabled by listing "icm" in `exploration_bonus_type`. Trains a feature
    # encoder + inverse model + forward model and adds the forward-model
    # prediction error in feature space to ONLINE rewards only — offline
    # (RLPD) rows are not augmented. Defaults match the paper.
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
    # Enabled by listing "eme" in `exploration_bonus_type`. Trains an EME
    # metric d_E^phi(s_i, s_j) and a K-member reward ensemble; per-transition
    # bonus is d_E(s_t, s_{t+1}) * clip(max(zeta(s_t, a_t), 1), 1, M). Added
    # to ONLINE rewards only. Defaults match the paper (M=10, K=6, gamma=0.99).
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
    # Enabled by listing "apt" in `exploration_bonus_type`. Trains a state
    # encoder + projection head via a SimCLR contrastive loss (with
    # Gaussian-noise augmentations as the state-based analogue of the
    # paper's random-shift / color-jitter on images); per-transition
    # bonus is the particle-based entropy estimator
    # log(c + (1/k) * sum_{z^{(j)} in N_k(f(s'))} ||f(s') - z^{(j)}||),
    # normalized by a running mean. Added to ONLINE rewards only — APT
    # measures novelty relative to the agent's online visitation, so
    # offline (RLPD) rows are masked out.
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

    # ── EXPLORE (UCB labeling of offline data) ─────────────────────────────
    # Enabled by listing "ucb" alongside "rnd" in `exploration_bonus_type`.
    # When present, trains a reward predictor r_θ(s,a) and termination
    # predictor T̂_θ(s,a) on online transitions; relabels each sampled offline
    # transition with UCBR = r_θ + ucb_coeff * (1/L)||f_φ - f̄||² (RND novelty)
    # and discount = 1 - sigmoid(T̂). Requires use_rlpd=True, agent_type='sac',
    # and a co-listed "rnd" bonus to source novelty (its weight may be 0.0 if
    # you only want EXPLORE labeling without additive RND reward shaping).
    ucb_coeff: float = 1.0
    ucb_lr: float = 3e-4
    ucb_hidden_dim: int = 256
    ucb_num_hidden: int = 2
    ucb_grad_steps_per_train_step: int = 1
    ucb_warmup_env_steps: int = 10000  # state-based default from the paper

    # ── Online empowerment bonus ───────────────────────────────────────────
    # Enabled by listing "online_empowerment" in `exploration_bonus_type`.
    # Trains a skill-conditioned policy + Q (and optionally V) on samples
    # from the replay buffer; per-state empowerment I(Z;S+|s) is added to
    # the reward identical to the offline empowerment bonus path.
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
    # Emit (raw_emp - mean) / scale per the existing empowerment bonus path.
    online_empowerment_bonus_mean: float = 0.0
    online_empowerment_bonus_scale: float = 1.0

    # ── Online MINE empowerment bonus ──────────────────────────────────────
    # Enabled by listing "online_mine_empowerment" in `exploration_bonus_type`.
    # Trains a forward dynamics model f(s,a)->s' and a MINE statistics network
    # T(s,a,s') on samples from the replay buffer; per-transition
    # Donsker-Varadhan contribution T(s,a,s') - log E_marg[exp(T)] is added to
    # the reward (online rows only). Marginal actions are sampled from the
    # main GCP actor at the current params.
    online_mine_empowerment_lr_dyn: float = 1e-3
    online_mine_empowerment_lr_t: float = 3e-4
    online_mine_empowerment_dyn_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_mine_empowerment_t_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    online_mine_empowerment_layer_norm: bool = True
    online_mine_empowerment_bonus_mean: float = 0.0
    online_mine_empowerment_bonus_scale: float = 1.0

    # ── Go Explore specific parameters ──────────────────────────────────────
    num_gcp_steps: int = 250      # max steps in go phase before forcing explore
    num_ep_steps: int = 250        # steps in explore phase before reset to go
    deterministic_go_phase: bool = False  # if True, go phase uses policy mode
    eps_random_action: float = 0.1        # probability of uniform random action in explore phase
    reset_on_explore_goal_reached: bool = True  # if False, explore phase runs to completion regardless of goal reach

    def check_config(self, config):
        assert config.episode_length - 1 == self.num_gcp_steps + self.num_ep_steps, (
            "episode_length - 1 must be equal to num_gcp_steps + num_ep_steps"
        )
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

        action_size = train_env.action_size
        state_size  = train_env.state_dim
        goal_size   = len(train_env.goal_indices)
        obs_size    = state_size + goal_size
        goal_indices_tuple = tuple(int(i) for i in np.asarray(train_env.goal_indices))

        # ── Eval env ──────────────────────────────────────────────────────────
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # ── Train env: GoExploreWrapper manages phase transitions ─────────────
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        train_env = GoExploreWrapper(
            train_env,
            num_gcp_steps=self.num_gcp_steps,
            num_ep_steps=self.num_ep_steps,
            state_size=state_size,
            goal_size=goal_size,
            goal_indices=unwrapped_env.goal_indices,
            reset_on_explore_goal_reached=self.reset_on_explore_goal_reached,
        )

        # ── Step count bookkeeping ────────────────────────────────────────────
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps    = self.min_replay_size * config.num_envs
        num_prefill_actor_steps  = int(np.ceil(self.min_replay_size / self.unroll_length))

        available_env_steps          = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch          = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps:          %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps:         %d", num_prefill_actor_steps)
        logging.info("env_steps_per_epoch:             %d", env_steps_per_epoch)
        logging.info("num_training_steps_per_epoch:    %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        # ── GCP (goal-conditioned policy) — the only policy ───────────────────
        gcp_actor, gcp_critic = get_algorithm(
            agent_type=self.agent_type,
            action_size=action_size,
            obs_size=obs_size,
            state_size=state_size,
            goal_indices=train_env.goal_indices,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            repr_dim=self.repr_dim,
            discounting=self.discounting,
            energy_fn=self.energy_fn,
            n_critics=self.n_critics,
        )

        gcp_actor_params  = gcp_actor.init(actor_key,  np.ones([1, obs_size]))
        gcp_critic_params = gcp_critic.init(critic_key, np.ones([1, obs_size]))

        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        gcp_critic_states = gcp_critic.create_critic_states(gcp_critic_params, self.critic_lr)

        target_entropy = -0.5 * action_size
        log_alpha      = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state    = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        target_critic_params = None
        if self.agent_type == "sac":
            target_critic_params = gcp_critic_params

        # ── Exploration Q-critic (CRL only, when exploration bonuses are set).
        # Trained on the raw (unweighted, but globally shifted/rescaled) bonus
        # via SAC Bellman backups; its Q is added to the CRL actor loss scaled
        # by sum(exploration_bonus_weight).
        exploration_q_critic = None
        exploration_q_critic_states = None
        exploration_q_target_critic_params = None
        use_exploration_q_critic = (
            self.agent_type == "crl" and self.exploration_bonus_type is not None
        )
        if use_exploration_q_critic:
            key, exp_q_critic_key = jax.random.split(key)
            exploration_q_critic = get_exploration_q_critic(
                obs_size=obs_size,
                action_size=action_size,
                h_dim=self.h_dim,
                n_hidden=self.n_hidden,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
                n_critics=self.n_critics,
            )
            exploration_q_critic_params = exploration_q_critic.init(
                exp_q_critic_key, np.ones([1, obs_size])
            )
            exploration_q_critic_states = exploration_q_critic.create_critic_states(
                exploration_q_critic_params, self.critic_lr,
            )
            exploration_q_target_critic_params = exploration_q_critic_params

        # ── TrainingState ─────────────────────────────────────────────────────
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=gcp_actor_state,
            critic_states=gcp_critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
            exploration_q_critic_states=exploration_q_critic_states,
            exploration_q_target_critic_params=exploration_q_target_critic_params,
        )

        # ── Optional offline-empowerment scorer for goal proposer ───────────
        bonus_types_tuple = tuple(self.exploration_bonus_type or ())
        has_max_empowerment_bonus = "max_empowerment" in bonus_types_tuple
        needs_empowerment_scorer = (
            self.goal_proposer_name in ("empowerment", "empowerment_density_ratio")
            or "empowerment" in bonus_types_tuple
            or has_max_empowerment_bonus
        )
        offline_empowerment_scorer = None
        if needs_empowerment_scorer:
            if self.empowerment_run_dir is None:
                raise ValueError(
                    "empowerment_run_dir must be set when goal_proposer_name or "
                    f"exploration_bonus_type uses empowerment (got goal_proposer_name="
                    f"'{self.goal_proposer_name}', exploration_bonus_type={bonus_types_tuple})."
                )
            key, empowerment_template_key = jax.random.split(key)
            emp_agent, _ex_obs_dim, base_obs_template = load_offline_empowerment_agent(
                run_dir=self.empowerment_run_dir,
                jax_env=unwrapped_env,
                template_rng=empowerment_template_key,
                epoch=self.empowerment_epoch,
                num_splus_samples=self.empowerment_num_splus_samples,
                use_full_obs=self.use_full_empowerment,
            )
            if self.use_full_empowerment:
                if state_size != int(_ex_obs_dim):
                    raise ValueError(
                        "use_full_empowerment=True requires state_size == ex_obs_dim; "
                        f"got state_size={state_size}, ex_obs_dim={_ex_obs_dim}."
                    )
                obs_builder = make_empowerment_full_obs_builder()
            else:
                ogbench_obs_indices, jaxgcrl_state_indices = infer_empowerment_override_indices_from_env(
                    unwrapped_env
                )
                obs_builder = make_empowerment_obs_builder(
                    jnp.asarray(base_obs_template),
                    ogbench_obs_indices,
                    jaxgcrl_state_indices,
                    state_size=state_size,
                )
            offline_empowerment_scorer = make_offline_empowerment_scorer(
                emp_agent,
                obs_builder,
                chunk_size=self.empowerment_score_chunk_size,
                mean=self.empowerment_bonus_mean,
                scale=self.empowerment_bonus_scale,
            )

        # ── Goal proposer ────────────────────────────────────────────────────
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
            self.num_candidates,
            state_size=unwrapped_env.state_dim,
            goal_indices=unwrapped_env.goal_indices,
            actor=gcp_actor,
            critic=gcp_critic,
            discounting=self.discounting,
            offline_empowerment_scorer=offline_empowerment_scorer,
            goal_proposer_temperature=self.goal_proposer_temperature,
            empowerment_alpha=self.empowerment_alpha,
        )

        # ── Env reset ────────────────────────────────────────────────────────
        random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys      = jax.random.split(env_key, config.num_envs)
        initial_goals = jax.vmap(random_goals_proposer)(env_keys)

        env_state = train_env.reset(env_keys, goal=initial_goals)
        info = dict(env_state.info)
        info['proposed_goals'] = initial_goals
        env_state = env_state.replace(info=info)

        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # ── Replay buffer ────────────────────────────────────────────────────
        dummy_transition = create_single_dummy_transition(
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
            include_max_empowerment=has_max_empowerment_bonus,
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

        dummy_batch_transition = create_dummy_transition_for_buffer(
            unroll_length=self.unroll_length,
            num_envs=config.num_envs,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
            include_max_empowerment=has_max_empowerment_bonus,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)

        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
            include_phase=True,
            include_max_empowerment=has_max_empowerment_bonus,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=gcp_actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(gcp_critic_states)},
        )

        # ── RLPD: offline buffer ─────────────────────────────────────────────
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
                agent_type=self.agent_type,
                include_phase=True,
                include_max_empowerment=has_max_empowerment_bonus,
            )

        # ── Exploration bonus(es) ─────────────────────────────────────────────
        from jaxgcrl.agents.go_explore.exploration import create_exploration_bonuses
        key, bonus_init_key = jax.random.split(key)
        # Reuse the goal-proposer's empowerment scorer if it's already loaded —
        # config exposes only one set of empowerment_* knobs, so they're identical.
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
            # Normalization is baked into ``offline_empowerment_scorer`` above
            # via mean=self.empowerment_bonus_mean, scale=self.empowerment_bonus_scale,
            # so the bonus path emits the scorer output unchanged.
            empowerment_mean=0.0,
            empowerment_scale=1.0,
            empowerment_precomputed_scorer=offline_empowerment_scorer,
            empowerment_use_full_obs=self.use_full_empowerment,
            rnd_feature_dim=self.rnd_feature_dim,
            rnd_hidden_dim=self.rnd_hidden_dim,
            rnd_num_hidden=self.rnd_num_hidden,
            rnd_learning_rate=self.rnd_learning_rate,
            rnd_obs_clip=self.rnd_obs_clip,
            rnd_use_goal=self.use_goal_for_rnd,
            rnd_train_batch_size=self.rnd_train_batch_size,
            goal_indices=goal_indices_tuple,
            ucb_action_size=action_size,
            ucb_coeff=self.ucb_coeff,
            ucb_hidden_dim=self.ucb_hidden_dim,
            ucb_num_hidden=self.ucb_num_hidden,
            ucb_learning_rate=self.ucb_lr,
            controllable_indices=(
                tuple(int(i) for i in np.asarray(unwrapped_env.controllable_indices))
                if hasattr(unwrapped_env, "controllable_indices") else None
            ),
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
            # EME's metric loss needs the actor's pre-tanh (mean, log_std)
            # at arbitrary states for the closed-form Gaussian KL term.
            eme_actor_dist_fn=(lambda params, obs: gcp_actor.apply(params, obs)),
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
            online_mine_empowerment_actor_sample_fn=(
                lambda params, obs, k: gcp_actor.sample_actions(
                    params, obs, k, is_deterministic=False,
                )
            ),
            online_mine_empowerment_lr_dyn=self.online_mine_empowerment_lr_dyn,
            online_mine_empowerment_lr_t=self.online_mine_empowerment_lr_t,
            online_mine_empowerment_dyn_hidden_dims=self.online_mine_empowerment_dyn_hidden_dims,
            online_mine_empowerment_t_hidden_dims=self.online_mine_empowerment_t_hidden_dims,
            online_mine_empowerment_layer_norm=self.online_mine_empowerment_layer_norm,
            online_mine_empowerment_bonus_mean=self.online_mine_empowerment_bonus_mean,
            online_mine_empowerment_bonus_scale=self.online_mine_empowerment_bonus_scale,
        )
        exploration_bonus_state = exploration_bonuses.initial_state

        # ── EXPLORE preconditions ───────────────────────────────────────────
        # "ucb" in the bonus list switches on offline UCB labeling. Validate
        # the algorithm preconditions here so a misconfiguration fails fast.
        if exploration_bonuses.has_ucb:
            if not self.use_rlpd:
                raise ValueError("'ucb' bonus requires use_rlpd=True (no offline data otherwise).")
            # 'ucb' owns its internal RND novelty source, so no co-listed
            # 'rnd' bonus is required (and listing one would also additively
            # shape ONLINE rewards, defeating EXPLORE's "online uses true
            # reward only" design).
            #
            # CRL also supported: the exploration_q_critic (auto-enabled when
            # exploration_bonus_type is non-None) is reward-based, so UCB-
            # relabeled offline rewards flow into it via
            # exploration_reward_per_batch below.

        # Goal-index columns captured for the exploration-bonus heatmap viz.
        goal_indices_arr = jnp.asarray(
            tuple(unwrapped_env.goal_indices), dtype=jnp.int32
        )

        # ── actor_step ────────────────────────────────────────────────────────
        deterministic_go = self.deterministic_go_phase
        eps_random = self.eps_random_action

        def actor_step(training_state, env, env_state, key, extra_fields):
            """One env step using a single GCP policy for both phases."""
            key, action_key, random_key, eps_key, env_rng = jax.random.split(key, 5)

            phase     = env_state.info['phase']           # (num_envs,)
            go_goal   = env_state.info['go_goal']         # (num_envs, goal_size)
            raw_state = env_state.obs[:, :state_size]     # (num_envs, state_size)

            # GCP always sees [state, go_goal] in both phases
            gcp_obs = jnp.concatenate([raw_state, go_goal], axis=-1)

            # Stochastic everywhere by default; only branch when go phase is deterministic.
            stoch_actions = gcp_actor.sample_actions(
                training_state.actor_state.params, gcp_obs, action_key, is_deterministic=False
            )
            if deterministic_go:
                det_actions = gcp_actor.sample_actions(
                    training_state.actor_state.params, gcp_obs, action_key, is_deterministic=True
                )
                in_go = (phase == 0)
                policy_actions = jnp.where(in_go[:, None], det_actions, stoch_actions)
            else:
                policy_actions = stoch_actions

            # Explore phase: with probability eps_random_action, use uniform random action.
            if eps_random > 0.0:
                in_explore = (phase == 1)
                random_actions = jax.random.uniform(
                    random_key, shape=policy_actions.shape, minval=-1.0, maxval=1.0
                )
                use_random = jax.random.uniform(eps_key, shape=(policy_actions.shape[0],)) < eps_random
                use_random = jnp.logical_and(in_explore, use_random)
                actions = jnp.where(use_random[:, None], random_actions, policy_actions)
            else:
                actions = policy_actions

            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras['phase'] = phase

            next_obs = jnp.concatenate(
                [nstate.obs[:, :state_size], go_goal], axis=-1
            )

            return nstate, Transition(
                observation=gcp_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=next_obs,
                extras={"state_extras": state_extras},
            )

        # ── get_experience ────────────────────────────────────────────────────
        def get_experience(training_state, env_state, buffer_state, key,
                           experience_count, goal_proposer_state):
            num_envs_     = config.num_envs
            episode_length = config.episode_length
            info           = dict(env_state.info)

            reset_threshold    = jnp.array(episode_length // (self.unroll_length * 2), dtype=jnp.int32)
            new_experience_count = experience_count + 1

            def propose_new_goals(env_state, key, info, buffer_state, goal_proposer_state):
                # Sample + refresh proposer state only when we actually propose:
                # both buffer.sample and the params snapshot are unused otherwise.
                buffer_state, transitions_sample = replay_buffer.sample(buffer_state)
                goal_proposer_state = goal_proposer_state.replace(
                    transitions_sample=transitions_sample,
                    actor_params=training_state.actor_state.params,
                    critic_params={i: cs.params for i, cs in enumerate(training_state.critic_states)},
                )

                viz_key, goal_key = jax.random.split(key)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs_)
                goal_keys   = jax.random.split(goal_key, num_envs_)
                first_obs   = info['first_obs']

                def propose_single(rng_key, obs, state):
                    goal, updated_state, log_data = goal_proposer(rng_key, obs, state)
                    return goal, log_data

                new_goals, log_data_tree = jax.vmap(
                    propose_single, in_axes=(0, 0, None)
                )(goal_keys, first_obs, goal_proposer_state)
                info['proposed_goals'] = new_goals

                env_steps = training_state.env_steps

                def log_viz(log_data_tree_np, viz_idx, steps):
                    selected = {k: v[viz_idx] for k, v in log_data_tree_np.items()}
                    handle_goal_proposer_visualization(
                        selected, self.goal_proposer_name,
                        unwrapped_env.x_bounds, unwrapped_env.y_bounds, steps
                    )
                    return jnp.array(0, dtype=jnp.int32)

                jax.experimental.io_callback(
                    log_viz, jnp.array(0, dtype=jnp.int32),
                    log_data_tree, viz_env_idx, env_steps
                )
                return env_state, info, jnp.array(0, dtype=jnp.int32), buffer_state, goal_proposer_state

            def keep_existing_goals(env_state, key, info, buffer_state, goal_proposer_state):
                return env_state, info, new_experience_count, buffer_state, goal_proposer_state

            # Split key so proposal and rollout use independent randomness.
            _, propose_key, rollout_key = jax.random.split(key, 3)
            (
                env_state, info, updated_experience_count,
                buffer_state, updated_goal_proposer_state,
            ) = jax.lax.cond(
                new_experience_count >= reset_threshold,
                propose_new_goals,
                keep_existing_goals,
                env_state, propose_key, info, buffer_state, goal_proposer_state,
            )
            env_state = env_state.replace(info=info)

            @jax.jit
            def f(carry, _):
                env_state, buffer_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    k,
                    extra_fields=("truncation", "traj_id", "phase"),
                )
                return (env_state, buffer_state, next_k), transition

            rollout_key, scorer_key = jax.random.split(rollout_key)
            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, rollout_key), (), length=self.unroll_length
            )

            # When max_empowerment bonus is configured, score every collected
            # next_observation with the offline empowerment scorer and take a
            # cumulative max along the unroll-length axis (per env). The
            # resulting per-step value is stored on the transition so the
            # bonus_fn just reads it back at training time. Scoring uses
            # next_observation[..., :state_size] to match the existing offline
            # ``empowerment`` bonus convention (reward at s_{t+1}).
            if has_max_empowerment_bonus:
                batch_shape = data.observation.shape[:2]
                states = data.next_observation[..., :state_size]
                flat_states = states.reshape(-1, state_size)
                raw = offline_empowerment_scorer(flat_states, scorer_key)
                raw = raw.reshape(batch_shape).astype(jnp.float32)
                max_emp_field = jax.lax.cummax(raw, axis=0)
                new_state_extras = {
                    **data.extras["state_extras"],
                    "max_empowerment": max_emp_field,
                }
                data = data._replace(
                    extras={**data.extras, "state_extras": new_state_extras}
                )
            buffer_state = replay_buffer.insert(buffer_state, data)

            return env_state, buffer_state, updated_experience_count, updated_goal_proposer_state

        # ── prefill_replay_buffer ─────────────────────────────────────────────
        def prefill_replay_buffer(training_state, env_state, buffer_state, key, goal_proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, new_k = jax.random.split(k)
                es, bs, ec, gps = get_experience(ts, es, bs, k, ts.experience_count, gps)
                ts = ts.replace(
                    env_steps=ts.env_steps + config.num_envs * self.unroll_length,
                    experience_count=ec,
                )
                return (ts, es, bs, new_k, gps), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ── update_networks (GCP only) ────────────────────────────────────────
        # For CRL+exploration: also train an extra SAC-style Q critic on the
        # weighted exploration bonus (per-bonus weight baked into the Bellman
        # target), and add -Q_exp to the CRL actor loss.

        @jax.jit
        def update_networks(carry, xs):
            transitions, exploration_reward = xs
            training_state, key = carry
            needs_exp_q = use_exploration_q_critic
            n_splits = 3 + int(self.agent_type == "sac") + int(needs_exp_q)
            keys = jax.random.split(key, n_splits)
            key, sub_keys = keys[0], iter(keys[1:])

            context = dict(
                **vars(self), **vars(config),
                state_size=state_size, action_size=action_size, goal_size=goal_size,
                obs_size=obs_size, goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(actor=gcp_actor, critic=gcp_critic)
            if needs_exp_q:
                networks["exploration_q_critic"] = exploration_q_critic

            metrics = {}
            if self.agent_type == "crl":
                if needs_exp_q:
                    exp_transitions = transitions._replace(reward=exploration_reward)
                    training_state, m = update_exploration_q_critic(
                        context, networks, exp_transitions, training_state, next(sub_keys),
                    )
                    metrics.update(m)
                training_state, m = gcp_actor.update(context, networks, transitions, training_state, next(sub_keys))
                metrics.update(m)
                training_state, m = gcp_critic.update(context, networks, transitions, training_state, next(sub_keys))
                metrics.update(m)
            else:  # sac
                training_state, m = update_alpha_sac(context, networks, transitions, training_state, next(sub_keys))
                metrics.update(m)
                training_state, m = gcp_critic.update(context, networks, transitions, training_state, next(sub_keys))
                metrics.update(m)
                training_state, m = gcp_actor.update(context, networks, transitions, training_state, next(sub_keys))
                metrics.update(m)

            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                training_state = training_state.replace(
                    target_critic_params=soft_update_target_params(
                        training_state.target_critic_params,
                        training_state.critic_states,
                        self.tau,
                    ),
                )
            if needs_exp_q:
                training_state = training_state.replace(
                    exploration_q_target_critic_params=soft_update_target_params(
                        training_state.exploration_q_target_critic_params,
                        training_state.exploration_q_critic_states,
                        self.tau,
                    ),
                )

            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
            return (training_state, key), metrics

        # ── training_step ─────────────────────────────────────────────────────
        has_ucb = exploration_bonuses.has_ucb
        ucb_warmup = jnp.asarray(self.ucb_warmup_env_steps, dtype=jnp.float32)
        ucb_grad_steps = int(self.ucb_grad_steps_per_train_step)
        # Per-bonus grad-step counts. Validated/broadcast against the
        # configured bonus types so a length-1 tuple still works for any
        # number of bonuses (default behavior).
        bonus_grad_steps_per_bonus = exploration_bonuses.normalize_grad_steps(
            self.bonus_grad_steps_per_env_step
        )

        @jax.jit
        def training_step(
            training_state, env_state, buffer_state, key,
            goal_proposer_state, exploration_bonus_state,
        ):
            exp_key, process_key, train_key = jax.random.split(key, 3)

            env_state, buffer_state, updated_ec, updated_gps = get_experience(
                training_state, env_state, buffer_state, exp_key,
                training_state.experience_count, goal_proposer_state,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
                experience_count=updated_ec,
            )

            # GCP update on all transitions
            buffer_state, online_transitions = replay_buffer.sample(buffer_state)

            # Tag online rows so the exploration bonus can be masked to apply
            # only to online data after the concat+permute below. Also add
            # UCB's RND novelty to online rewards directly (the offline path
            # gets full UCBR via relabel_offline_with_ucb instead).
            online_is_online = jnp.ones_like(online_transitions.reward)
            online_extras = {
                **online_transitions.extras,
                "state_extras": {
                    **online_transitions.extras["state_extras"],
                    "is_online": online_is_online,
                },
            }
            if has_ucb:
                ucb_novelty_online = exploration_bonuses.compute_ucb_online_novelty_bonus(
                    exploration_bonus_state, online_transitions,
                )
                # Gate by UCB warmup the same way offline relabeling is.
                past_warmup = training_state.env_steps >= ucb_warmup
                ucb_novelty_online = jnp.where(
                    past_warmup, ucb_novelty_online, jnp.zeros_like(ucb_novelty_online),
                )
                online_transitions = online_transitions._replace(
                    reward=online_transitions.reward + ucb_novelty_online,
                    extras=online_extras,
                )
            else:
                online_transitions = online_transitions._replace(extras=online_extras)

            if self.use_rlpd:
                # Mix 50% offline data: concatenate along num_envs axis
                offline_key, process_key = jax.random.split(process_key)
                offline_transitions = offline_buffer.sample(offline_key, config.num_envs)

                # Tag offline rows with 0s so the bonus mask zeroes them out.
                offline_is_online = jnp.zeros_like(offline_transitions.reward)
                offline_transitions = offline_transitions._replace(
                    extras={
                        **offline_transitions.extras,
                        "state_extras": {
                            **offline_transitions.extras["state_extras"],
                            "is_online": offline_is_online,
                        },
                    }
                )

                if has_ucb:
                    # EXPLORE: relabel offline transitions with optimistic
                    # UCB. UCB owns its internal RND novelty source, so we
                    # just pass the offline batch — the relabel reads novelty
                    # from UCB's predictor (read-only) and adds r_θ + ucb_coeff
                    # * novelty for the reward, plus 1 - sigmoid(T̂) for the
                    # discount. Gated by warmup: the reward/term predictors
                    # are noise until enough online data has trained them.
                    relabeled_offline = exploration_bonuses.relabel_offline_with_ucb(
                        exploration_bonus_state, offline_transitions,
                    )
                    past_warmup = training_state.env_steps >= ucb_warmup
                    new_reward = jnp.where(
                        past_warmup, relabeled_offline.reward, offline_transitions.reward,
                    )
                    new_discount = jnp.where(
                        past_warmup, relabeled_offline.discount, offline_transitions.discount,
                    )
                    offline_transitions = offline_transitions._replace(
                        reward=new_reward, discount=new_discount,
                    )

                transitions = jax.tree_util.tree_map(
                    lambda a, b: jnp.concatenate([a, b], axis=0),
                    online_transitions, offline_transitions,
                )
            else:
                transitions = online_transitions

            transitions, _ = gcp_actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                state_size, goal_indices_tuple,
                train_env.goal_reach_thresh, self.use_her,
                p_future_her_goal=self.p_future_her_goal,
            )

            if self.use_rlpd:
                # Slice to original num_batches so gradient step count stays the same.
                # The random permutation in process_transitions already mixed
                # online+offline within each batch (~50/50), so the surviving
                # slice still covers both offline and online transitions.
                num_batches = config.num_envs * (config.episode_length - 1) // self.batch_size
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:num_batches], transitions
                )

            # Exploration bonus is summed across all configured bonuses. For
            # SAC we add the weighted total to the post-HER reward so it shapes
            # the critic/actor via Q; for CRL we instead train a separate Q_exp
            # on the weighted per-transition bonus and add -Q_exp to the CRL
            # actor loss (see update_networks). The is_online mask routes
            # online-only bonuses (RND) to online rows only inside compute().
            bonus_key, train_key = jax.random.split(train_key)
            is_online = transitions.extras["state_extras"]["is_online"]
            bonus_compute_kwargs = (
                {"actor_params": training_state.actor_state.params}
                if exploration_bonuses.requires_actor_params else {}
            )
            total_bonus, raw_total_bonus, exploration_bonus_state, bonus_metrics = (
                exploration_bonuses.compute(
                    exploration_bonus_state, transitions, bonus_key,
                    is_online=is_online,
                    **bonus_compute_kwargs,
                )
            )
            if self.agent_type == "sac":
                transitions = transitions._replace(
                    reward=transitions.reward + total_bonus
                )
                exploration_reward_per_batch = jnp.zeros_like(transitions.reward)
            else:
                # CRL: feed weighted bonus to Q_exp. With UCB enabled, also
                # add transitions.reward — which now carries true env reward
                # on online rows AND UCBR(s,a) on offline rows (set in
                # relabel_offline_with_ucb above) — so the exploration_q_critic
                # learns the EXPLORE-shaped signal in CRL mode the same way
                # the SAC critic does.
                if has_ucb:
                    exploration_reward_per_batch = total_bonus + transitions.reward
                else:
                    exploration_reward_per_batch = total_bonus

            reward_viz = (
                transitions.observation[..., 0],
                transitions.observation[..., 1],
                transitions.reward,
                total_bonus,
                transitions.observation[..., goal_indices_arr],
            )

            (training_state, _), metrics = jax.lax.scan(
                update_networks,
                (training_state, train_key),
                (transitions, exploration_reward_per_batch),
            )
            metrics.update(bonus_metrics)
            metrics["reward_mean"] = jnp.mean(transitions.reward)

            # Train all bonus-side trainables (RND predictor, online
            # empowerment networks, etc.) on the *online* batch only.
            # compute() above is read-only over predictor params, so RND
            # novelty stays meaningful for offline rows: it reflects "novel
            # relative to what's been seen ONLINE".
            #
            # Per-bonus grad-step counts (bonus_grad_steps_per_env_step) let
            # different bonuses train at different cadences in the same env
            # step. For each bonus i with n_i steps:
            #   n_i == 1: single train call on the same online_transitions
            #             batch the main policy used — preserves prior
            #             RND/MISC behavior bit-for-bit when no per-bonus
            #             override is set.
            #   n_i  > 1: jax.lax.scan that samples a fresh online batch
            #             from the replay buffer per grad step. Each step's
            #             samples are independent — needed for
            #             online_empowerment so the estimator and policy
            #             gradient see uncorrelated batches.
            if exploration_bonuses.has_trainables:
                for i, n_i in enumerate(bonus_grad_steps_per_bonus):
                    if not exploration_bonuses.is_trainable(i):
                        continue
                    bonus_train_kwargs = (
                        {"actor_params": training_state.actor_state.params}
                        if exploration_bonuses.requires_actor_params_at(i) else {}
                    )
                    if n_i == 1:
                        exploration_bonus_state, m_i = exploration_bonuses.train_one(
                            exploration_bonus_state, i, online_transitions, 1,
                            **bonus_train_kwargs,
                        )
                        metrics.update(m_i)
                    else:
                        bonus_loop_key, train_key = jax.random.split(train_key)

                        def _train_bonus_step(carry, _, _i=i):
                            bs, bonus_st, k = carry
                            sample_k, k_next = jax.random.split(k)
                            bs, bs_online = replay_buffer.sample(bs)
                            # Re-tag is_online so any train_fn that
                            # introspects extras sees the same shape as
                            # the main path.
                            bs_online_extras = {
                                **bs_online.extras,
                                "state_extras": {
                                    **bs_online.extras["state_extras"],
                                    "is_online": jnp.ones_like(bs_online.reward),
                                },
                            }
                            bs_online = bs_online._replace(extras=bs_online_extras)
                            bonus_st, m = exploration_bonuses.train_one(
                                bonus_st, _i, bs_online, 1,
                                **bonus_train_kwargs,
                            )
                            return (bs, bonus_st, k_next), m

                        (buffer_state, exploration_bonus_state, _), bonus_metrics_seq = jax.lax.scan(
                            _train_bonus_step,
                            (buffer_state, exploration_bonus_state, bonus_loop_key),
                            (),
                            length=int(n_i),
                        )
                        metrics.update(jax.tree_util.tree_map(
                            jnp.mean, bonus_metrics_seq,
                        ))

            if has_ucb:
                # Train reward + termination predictors on the *online* batch
                # (true r and termination flags). Always train; UCB labeling
                # is the part gated by warmup, so the predictors keep learning
                # from step 0 and are usable as soon as warmup ends.
                exploration_bonus_state, ucb_metrics = exploration_bonuses.train_ucb(
                    exploration_bonus_state, online_transitions, ucb_grad_steps,
                )
                metrics.update(ucb_metrics)

            # Snapshot Q_exp(obs, action) on the final batch of the scan so the
            # training loop can log an xy scatter of the exploration Q values.
            if use_exploration_q_critic:
                last_batch = jax.tree_util.tree_map(lambda x: x[-1], transitions)
                last_obs = last_batch.observation
                last_action = last_batch.action
                exp_q_vals = exploration_q_critic.apply(
                    flatten_sac_critic_params(training_state.exploration_q_critic_states),
                    last_obs, last_action,
                )
                exp_q_viz = (
                    last_obs[..., goal_indices_arr[0]],
                    last_obs[..., goal_indices_arr[1]],
                    jnp.min(exp_q_vals, axis=-1),
                )
            else:
                zero = jnp.zeros_like(transitions.reward[-1])
                exp_q_viz = (zero, zero, zero)

            return (
                training_state, env_state, buffer_state,
                updated_gps, exploration_bonus_state,
                reward_viz, exp_q_viz,
            ), metrics

        # ── training_epoch ────────────────────────────────────────────────────
        @jax.jit
        def training_epoch(
            training_state, env_state, buffer_state, key,
            goal_proposer_state, exploration_bonus_state,
        ):
            # Snapshot cumulative counters *before* the epoch so we can compute
            # epoch-level deltas (rather than a lifetime average).
            pre_completions   = jnp.sum(env_state.info['go_completions_total'])
            pre_successes     = jnp.sum(env_state.info['go_successes_total'])
            pre_success_steps = jnp.sum(env_state.info['go_success_steps_total'])

            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps, ebs = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, gps, ebs, reward_viz, exp_q_viz), metrics = training_step(
                    ts, es, bs, train_key, gps, ebs
                )
                return (ts, es, bs, k, gps, ebs), (metrics, reward_viz, exp_q_viz)

            (
                (training_state, env_state, buffer_state, key,
                 goal_proposer_state, exploration_bonus_state),
                (metrics, reward_viz, exp_q_viz),
            ) = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key,
                 goal_proposer_state, exploration_bonus_state),
                (),
                length=num_training_steps_per_epoch,
            )

            # Keep only the final step's snapshots for the once-per-epoch viz.
            last_reward_viz = jax.tree_util.tree_map(lambda x: x[-1], reward_viz)
            last_exp_q_viz  = jax.tree_util.tree_map(lambda x: x[-1], exp_q_viz)

            # Go Explore phase metrics — epoch-level (current policy performance)
            epoch_completions   = jnp.sum(env_state.info['go_completions_total']) - pre_completions
            epoch_successes     = jnp.sum(env_state.info['go_successes_total']) - pre_successes
            epoch_success_steps = jnp.sum(env_state.info['go_success_steps_total']) - pre_success_steps

            go_success_rate = jnp.where(epoch_completions > 0,
                                        epoch_successes / epoch_completions,
                                        0.0)
            avg_go_steps    = jnp.where(epoch_successes > 0,
                                        epoch_success_steps / epoch_successes,
                                        0.0)

            scan_shape = jax.tree_util.tree_leaves(metrics)[0].shape if metrics else (1,)
            metrics["go_phase_success_rate"] = jnp.broadcast_to(go_success_rate, scan_shape)
            metrics["avg_go_phase_steps"]    = jnp.broadcast_to(avg_go_steps, scan_shape)
            metrics["buffer_current_size"]   = jnp.broadcast_to(replay_buffer.size(buffer_state), scan_shape)

            return (
                training_state, env_state, buffer_state,
                goal_proposer_state, exploration_bonus_state,
                metrics, last_reward_viz, last_exp_q_viz,
            )

        # ── prefill ───────────────────────────────────────────────────────────
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, goal_proposer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state
        )

        # Seed per-bonus observation normalization (e.g. RND) from post-prefill
        # rollouts, per the RND paper's random-agent warmup.
        if not exploration_bonuses.is_empty:
            key, seed_key = jax.random.split(key)
            buffer_state, seed_transitions = replay_buffer.sample(buffer_state)
            seed_states = seed_transitions.observation[..., :state_size]
            exploration_bonus_state = exploration_bonuses.init_from_states(
                exploration_bonus_state, seed_states
            )

        # ── Evaluator ─────────────────────────────────────────────────────────
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            actions = gcp_actor.sample_actions(
                training_state.actor_state.params,
                env_state.obs,
                jax.random.PRNGKey(0),
                is_deterministic=True,
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=None,
                extras={"state_extras": state_extras},
            )

        evaluator = ActorEvaluator(
            eval_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ── Main training loop ────────────────────────────────────────────────
        training_walltime      = 0
        last_visualization_step = -1
        logging.info("starting training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            (
                training_state, env_state, buffer_state,
                goal_proposer_state, exploration_bonus_state,
                metrics, last_reward_viz, last_exp_q_viz,
            ) = training_epoch(
                training_state, env_state, buffer_state, epoch_key,
                goal_proposer_state, exploration_bonus_state,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time  = time.time() - t
            training_walltime   += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time

            metrics_dict = {}
            for name, value in metrics.items():
                if hasattr(value, 'item'):
                    metrics_dict[f"training/{name}"] = float(value.item())
                elif hasattr(value, '__float__'):
                    metrics_dict[f"training/{name}"] = float(value)
                else:
                    metrics_dict[f"training/{name}"] = value

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **metrics_dict,
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            # Visualize trajectories every 1M steps
            if current_step // 1_000_000 > last_visualization_step // 1_000_000:
                key, viz_key = jax.random.split(key)
                buffer_state = all_visualizations(
                    replay_buffer=replay_buffer,
                    buffer_state=buffer_state,
                    env=unwrapped_env,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                    rng_key=viz_key,
                    current_step=current_step,
                )
                _, phase_transitions = replay_buffer.sample(buffer_state)
                visualize_go_explore_phases(
                    phase_transitions,
                    unwrapped_env.x_bounds,
                    unwrapped_env.y_bounds,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                    current_step=current_step,
                )
                if not exploration_bonuses.is_empty:
                    _log_trajectory_reward(
                        reward_viz=last_reward_viz,
                        current_step=current_step,
                    )
                    _log_reward_heatmap(
                        reward_viz=last_reward_viz,
                        x_bounds=unwrapped_env.x_bounds,
                        y_bounds=unwrapped_env.y_bounds,
                        current_step=current_step,
                    )
                    _log_exploration_bonus_goal_heatmap(
                        reward_viz=last_reward_viz,
                        goal_indices=tuple(unwrapped_env.goal_indices),
                        current_step=current_step,
                    )
                    if (
                        self.exploration_bonus_type is not None
                        and "online_empowerment" in self.exploration_bonus_type
                    ):
                        oe_idx = self.exploration_bonus_type.index("online_empowerment")
                        oe_bonus_fn = exploration_bonuses.fns[oe_idx]
                        score_fn = getattr(oe_bonus_fn, "score_states", None)
                        if score_fn is not None:
                            buffer_state, viz_transitions = replay_buffer.sample(buffer_state)
                            template_state = viz_transitions.observation[0, 0, :state_size]
                            key, viz_emp_key = jax.random.split(key)
                            _log_online_empowerment_xy_map(
                                score_fn=score_fn,
                                bonus_state=exploration_bonus_state[oe_idx],
                                template_state=template_state,
                                state_size=state_size,
                                x_bounds=unwrapped_env.x_bounds,
                                y_bounds=unwrapped_env.y_bounds,
                                grid_res=64,
                                rng=viz_emp_key,
                                current_step=current_step,
                            )
                if use_exploration_q_critic:
                    visualize_exploration_q_xy(
                        xs=last_exp_q_viz[0],
                        ys=last_exp_q_viz[1],
                        q_values=last_exp_q_viz[2],
                        x_bounds=unwrapped_env.x_bounds,
                        y_bounds=unwrapped_env.y_bounds,
                        current_step=current_step,
                    )
                last_visualization_step = current_step

            do_render = ne % config.visualization_interval == 0
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: gcp_actor.apply(param, obs)
            else:
                make_policy = lambda param: lambda obs, rng: (
                    gcp_actor.sample_actions(param, obs, rng, is_deterministic=True), {}
                )

            # Build full GCP critic params for checkpointing
            if self.agent_type == "crl":
                full_critic_params = flatten_crl_critic_params(training_state.critic_states)
            else:
                full_critic_params = flatten_sac_critic_params(training_state.critic_states)

            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                full_critic_params,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)
        return make_policy, params, metrics
