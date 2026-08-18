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
from typing import Callable, Literal, Optional, Tuple, Union

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

from jaxgcrl.agents.go_explore.types import TrainingState, Transition, GoalProposerState
from jaxgcrl.agents.go_explore.algorithms import get_algorithm
from jaxgcrl.agents.go_explore.utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
    format_epoch_metrics,
)
from jaxgcrl.agents.go_explore.losses import (
    flatten_crl_critic_params,
    flatten_sac_critic_params,
    soft_update_target_params,
    update_alpha_sac,
)
from jaxgcrl.agents.go_explore.visualization import (
    all_visualizations,
    visualize_go_explore_phases,
    log_online_empowerment_heatmap,
    handle_goal_proposer_visualization,
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
from jaxgcrl.agents.go_explore.online_empowerment import (
    create_online_empowerment_agent,
    make_online_empowerment_scorer,
    make_online_empowerment_train_fn,
)


Metrics = types.Metrics


@dataclass
class GoExploreSimple:
    """Go Explore agent with a single goal-conditioned policy.

    The go phase navigates to a proposed frontier goal. The explore phase
    continues with the same policy but samples stochastically and injects
    uniform random actions with probability ``eps_random_action``.
    """

    # Algorithm type for the goal-conditioned policy.
    # "sac_discrete" branches into the hierarchical SAC-discrete skill controller;
    # "crl_skill" branches into the hierarchical *contrastive* (CRL) skill
    # controller (see skill_controller.py / crl_skill_controller.py).
    agent_type: Literal["sac", "crl", "sac_discrete", "crl_skill"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 50
    h_dim: int = 512
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    repr_dim: int = 64
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    tau: float = 0.005
    n_critics: int = 1
    use_her: bool = True
    p_future_her_goal: float = 0.8
    use_sac_critic_mean: bool = False  # SAC: average critic ensemble instead of min

    goal_proposer_name: Literal["random_env_goals", "rb", "q_epistemic", "ucgr", "max_critic_to_env", "mega", "omega", "empowerment", "empowerment_density_ratio", "empowerment_density_product"] = "random_env_goals"
    num_candidates: int = 512
    goal_proposer_temperature: float = 0.0
    empowerment_alpha: float = 1.0
    empowerment_run_dir: Optional[str] = None
    empowerment_epoch: Optional[int] = None
    empowerment_num_splus_samples: int = 16
    empowerment_score_chunk_size: int = 32
    # If True, feed the full state (obs with goal sliced off) to the
    # empowerment network instead of overwriting a few indices of a cached
    # OGBench template. Requires state_size == checkpoint ex_obs_dim.
    use_full_empowerment: bool = True

    # Empowerment scorer global normalization for the empowerment goal
    # proposers: the scorer emits (raw_emp - mean) / scale.
    empowerment_bonus_mean: float = 0
    empowerment_bonus_scale: float = 1

    # ── Online empowerment ──────────────────────────────────────────────────
    # When True, the empowerment goal proposer scores candidates with an
    # OGBench ``empowerment_skill`` agent trained online (in lockstep with the
    # main agent) instead of a pretrained checkpoint. The proposer logic is
    # otherwise identical. Requires ``ogbench_root`` for importing the agent.
    online_empowerment: bool = False
    # Gradient steps on the empowerment agent per training step (update loop).
    online_empowerment_num_grad_steps: int = 1
    # Hyperparameters — defaults transcribed from the reference flags.json.
    online_empowerment_num_skills: int = 15
    online_empowerment_num_splus_samples: int = 16
    online_empowerment_value_latent_dim: int = 256
    online_empowerment_separate_qv: bool = True
    online_empowerment_use_self_q_loss: bool = True
    online_empowerment_use_self_v_loss: bool = True
    online_empowerment_no_target_q_for_policy: bool = True
    online_empowerment_sample_z: bool = True
    online_empowerment_discount: float = 0.99
    online_empowerment_tau: float = 0.005
    online_empowerment_lr: float = 3e-4
    online_empowerment_batch_size: int = 1024
    online_empowerment_layer_norm: bool = True
    online_empowerment_const_std: bool = True
    online_empowerment_bc_alpha: float = 0.01
    online_empowerment_anneal_alpha: bool = False
    online_empowerment_log_interval: int = 5000
    # Heatmap: random replay-buffer states scored + scattered, logged each eval.
    online_empowerment_heatmap_num_states: int = 512

    # Explicit OGBench repo root (the dir containing impls/). If None it is
    # inferred from the checkpoint run_dir, which only works when the ckpt lives
    # under <root>/impls/...; set this when checkpoints are stored elsewhere
    # (e.g. a scratch dir) so OGBench's agents/utils imports resolve. Applies to
    # both empowerment_run_dir and skill_policy_run_dir loading.
    ogbench_root: Optional[str] = None

    # ── RLPD (offline data mixing) ─────────────────────────────────────────
    use_rlpd: bool = False  # Mix 50% offline OGBench data into each training batch

    # ── Go Explore specific parameters ──────────────────────────────────────
    num_gcp_steps: int = 250      # max steps in go phase before forcing explore
    num_ep_steps: int = 250        # steps in explore phase before reset to go
    deterministic_go_phase: bool = False  # if True, go phase uses policy mode
    eps_random_action: float = 0.2        # probability of uniform random action in explore phase
    reset_on_explore_goal_reached: bool = False  # if False, explore phase runs to completion regardless of goal reach

    # ── Hierarchical skill controller (agent_type="sac_discrete") ───────────────
    # Freeze a pretrained OGBench skill-conditioned policy π(a|s,z) and train an
    # online SAC-discrete high-level controller that picks discrete skills.
    skill_policy_run_dir: Optional[str] = None    # OGBench skill-agent checkpoint dir (flags.json + params_*.pkl)
    skill_policy_epoch: Optional[int] = None       # None -> latest
    num_skills: Optional[int] = None               # None -> infer from checkpoint config
    # If True, the high-level controller's action space is a *continuous* skill
    # vector (Gaussian actor, reparameterized SAC-style updates) instead of a
    # discrete skill index. Requires a frozen low-level policy checkpoint whose
    # ``policy(obs, z)`` submodule already accepts a continuous ``z`` directly
    # (no one-hot encoding); DDS checkpoints (VQ codebook) are inherently
    # discrete and are incompatible with continuous_skill=True.
    continuous_skill: bool = False
    skill_dim: Optional[int] = 8                    # continuous skill vector dim; None -> infer from checkpoint config
    # Skill-policy family, purely for wandb categorization. When set it is
    # ASSERTED against the checkpoint's flags.json agent_name (dds -> "dds",
    # empowerment -> "empowerment_skill") rather than derived from it; the config
    # value is what gets logged. None disables the check. Likewise num_skills
    # (when set) is asserted against the checkpoint and the config value is used.
    skill_policy_type: Optional[str] = None        # {"dds", "empowerment", "dads"}
    skill_commitment_k: int = 10                   # fixed open-loop temporal commitment
    use_full_skill_obs: bool = True                # full state row -> skill net (else override-index template)
    deterministic_skill_actions: bool = True       # frozen policy uses dist.mode() (vs sample)
    controller_target_entropy_scale: float = 0.98  # H̄ = scale * log(num_skills)
    gamma_low: float = 1.0                          # intra-macro-step reward discount (within macro-step)
    controller_replay_size: int = 50000

    def check_config(self, config):
        assert not self.continuous_skill or self.agent_type == "crl_skill", (
            "continuous_skill=True is only supported for agent_type='crl_skill' "
            "(the SAC-discrete controller is inherently discrete)."
        )
        if self.agent_type in ("sac_discrete", "crl_skill"):
            assert self.skill_policy_run_dir is not None, (
                f"agent_type='{self.agent_type}' requires skill_policy_run_dir (OGBench skill checkpoint)."
            )
            assert self.skill_commitment_k > 0, "skill_commitment_k must be > 0"
            assert config.episode_length % self.skill_commitment_k == 0, (
                "episode_length must be divisible by skill_commitment_k so macro-steps "
                "tile the episode exactly."
            )
            if self.continuous_skill:
                assert self.skill_dim is None or self.skill_dim > 0, (
                    "skill_dim (if set) must be > 0 for a continuous controller."
                )
                assert self.skill_policy_type not in ("dds", "dads"), (
                    "continuous_skill=True is incompatible with skill_policy_type='dds' "
                    "(DDS uses a discrete VQ codebook) or 'dads' (its skill prior/decoder "
                    "are trained only on one-hot discrete skills)."
                )
            else:
                assert self.num_skills is None or self.num_skills > 1, (
                    "num_skills (if set) must be > 1 for a discrete controller."
                )
            assert self.skill_policy_type in (None, "dds", "empowerment", "dads"), (
                f"skill_policy_type={self.skill_policy_type!r} must be one of "
                "None, 'dds', 'empowerment', 'dads'."
            )
            assert config.num_evals > 0, "num_evals must be > 0"
            if self.agent_type == "crl_skill":
                # The contrastive controller always samples γ-discounted future
                # states for its InfoNCE positives (flatten_batch, like the flat
                # CRL path); use_her / p_future_her_goal are ignored by it.
                assert config.episode_length // self.skill_commitment_k >= 2, (
                    "agent_type='crl_skill' needs episode_length/k >= 2 macro-steps "
                    "so each window has a future state to sample as a positive."
                )
            return

        assert config.episode_length - 1 == self.num_gcp_steps + self.num_ep_steps, (
            "episode_length - 1 must be equal to num_gcp_steps + num_ep_steps"
        )
        eff_len = config.episode_length if self.agent_type == "sac" else config.episode_length - 1
        assert config.num_envs * eff_len % self.batch_size == 0, (
            f"num_envs * effective_trajectory_length ({config.num_envs} * {eff_len}) "
            f"must be divisible by batch_size ({self.batch_size}); effective length is "
            "episode_length for SAC and episode_length-1 for CRL."
        )

    def _train_skill_controller(self, config, train_env, eval_env, progress_fn):
        """Hierarchical SAC-discrete controller over a frozen skill policy.

        Thin wrapper; all logic lives in ``skill_controller.train_skill_controller``.
        """
        from jaxgcrl.agents.go_explore_simple.skill_controller import (
            train_skill_controller,
        )
        return train_skill_controller(self, config, train_env, eval_env, progress_fn)

    def _train_crl_skill_controller(self, config, train_env, eval_env, progress_fn):
        """Hierarchical *contrastive* (CRL) controller over a frozen skill policy.

        Thin wrapper; all logic lives in
        ``crl_skill_controller.train_crl_skill_controller``.
        """
        from jaxgcrl.agents.go_explore_simple.crl_skill_controller import (
            train_crl_skill_controller,
        )
        return train_crl_skill_controller(self, config, train_env, eval_env, progress_fn)

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

        if self.agent_type == "sac_discrete":
            return self._train_skill_controller(
                config, train_env, eval_env, progress_fn
            )

        if self.agent_type == "crl_skill":
            return self._train_crl_skill_controller(
                config, train_env, eval_env, progress_fn
            )

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

        # ── TrainingState ─────────────────────────────────────────────────────
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=gcp_actor_state,
            critic_states=gcp_critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )

        # ── Empowerment scorer for goal proposer (offline ckpt or online) ────
        needs_empowerment_scorer = self.goal_proposer_name in (
            "empowerment", "empowerment_density_ratio", "empowerment_density_product"
        )
        offline_empowerment_scorer = None
        online_empowerment_score_fn = None
        online_empowerment_train_fn = None
        if needs_empowerment_scorer and self.online_empowerment:
            # ── Fully online empowerment: fresh OGBench empowerment_skill agent
            # trained in lockstep, scored live. No checkpoint / dataset load.
            if self.ogbench_root is None:
                raise ValueError(
                    "ogbench_root must be set when online_empowerment=True "
                    "(the OGBench repo root containing impls/)."
                )
            emp_online_agent = create_online_empowerment_agent(
                ogbench_root=self.ogbench_root,
                state_size=state_size,
                action_size=action_size,
                seed=config.seed,
                num_skills=self.online_empowerment_num_skills,
                num_splus_samples=self.online_empowerment_num_splus_samples,
                value_latent_dim=self.online_empowerment_value_latent_dim,
                separate_qv=self.online_empowerment_separate_qv,
                use_self_q_loss=self.online_empowerment_use_self_q_loss,
                use_self_v_loss=self.online_empowerment_use_self_v_loss,
                no_target_q_for_policy=self.online_empowerment_no_target_q_for_policy,
                sample_z=self.online_empowerment_sample_z,
                discount=self.online_empowerment_discount,
                tau=self.online_empowerment_tau,
                lr=self.online_empowerment_lr,
                batch_size=self.online_empowerment_batch_size,
                layer_norm=self.online_empowerment_layer_norm,
                const_std=self.online_empowerment_const_std,
                bc_alpha=self.online_empowerment_bc_alpha,
                anneal_alpha=self.online_empowerment_anneal_alpha,
                log_interval=self.online_empowerment_log_interval,
            )
            training_state = training_state.replace(
                online_empowerment_agent=emp_online_agent
            )
            online_empowerment_score_fn = make_online_empowerment_scorer(
                mean=self.empowerment_bonus_mean, scale=self.empowerment_bonus_scale,
                # Chunk candidates (as the offline scorer does) to bound memory
                # under the per-env vmap; variance comes from the agent's own
                # num_splus_samples.
                chunk_size=self.empowerment_score_chunk_size,
            )
            online_empowerment_train_fn = make_online_empowerment_train_fn(
                state_size=state_size,
                action_size=action_size,
                batch_size=self.online_empowerment_batch_size,
            )
        elif needs_empowerment_scorer:
            if self.empowerment_run_dir is None:
                raise ValueError(
                    "empowerment_run_dir must be set when goal_proposer_name uses "
                    f"empowerment (got goal_proposer_name='{self.goal_proposer_name}'); "
                    "or set online_empowerment=True."
                )
            key, empowerment_template_key = jax.random.split(key)
            emp_agent, ex_obs_dim, base_obs_template = load_offline_empowerment_agent(
                run_dir=self.empowerment_run_dir,
                jax_env=unwrapped_env,
                template_rng=empowerment_template_key,
                epoch=self.empowerment_epoch,
                num_splus_samples=self.empowerment_num_splus_samples,
                use_full_obs=self.use_full_empowerment,
                ogbench_root=self.ogbench_root,
            )
            if self.use_full_empowerment:
                if state_size != int(ex_obs_dim):
                    raise ValueError(
                        "use_full_empowerment=True requires state_size == ex_obs_dim; "
                        f"got state_size={state_size}, ex_obs_dim={ex_obs_dim}."
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
            online_empowerment_score_fn=online_empowerment_score_fn,
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
            include_phase=True,
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
            include_phase=True,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)

        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            include_phase=True,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=gcp_actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(gcp_critic_states)},
            empowerment_state=training_state.online_empowerment_agent,
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
                include_phase=True,
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
            num_envs     = config.num_envs
            episode_length = config.episode_length
            info           = dict(env_state.info)

            goal_reproposal_interval    = jnp.array(episode_length // (self.unroll_length * 2), dtype=jnp.int32)
            new_experience_count = experience_count + 1

            def propose_new_goals(env_state, key, info, buffer_state, goal_proposer_state):
                # Sample + refresh proposer state only when we actually propose:
                # both buffer.sample and the params snapshot are unused otherwise.
                buffer_state, transitions_sample = replay_buffer.sample(buffer_state)
                goal_proposer_state = goal_proposer_state.replace(
                    transitions_sample=transitions_sample,
                    actor_params=training_state.actor_state.params,
                    critic_params={i: cs.params for i, cs in enumerate(training_state.critic_states)},
                    # Refresh the empowerment snapshot to the latest online agent.
                    empowerment_state=training_state.online_empowerment_agent,
                )

                viz_key, goal_key = jax.random.split(key)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs)
                goal_keys   = jax.random.split(goal_key, num_envs)
                first_obs   = info['first_obs']

                def propose_single(rng_key, obs, state):
                    goal, _, log_data = goal_proposer(rng_key, obs, state)
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
                new_experience_count >= goal_reproposal_interval,
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

            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, rollout_key), (), length=self.unroll_length
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
        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            n_splits = 3 + int(self.agent_type == "sac")
            keys = jax.random.split(key, n_splits)
            key, sub_keys = keys[0], iter(keys[1:])

            context = dict(
                **vars(self), **vars(config),
                state_size=state_size, action_size=action_size, goal_size=goal_size,
                obs_size=obs_size, goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(actor=gcp_actor, critic=gcp_critic)

            metrics = {}
            if self.agent_type == "crl":
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

            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
            return (training_state, key), metrics

        # ── training_step ─────────────────────────────────────────────────────
        @jax.jit
        def training_step(
            training_state, env_state, buffer_state, key,
            goal_proposer_state,
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

            if self.use_rlpd:
                # Mix 50% offline data: concatenate along num_envs axis
                offline_key, process_key = jax.random.split(process_key)
                offline_transitions = offline_buffer.sample(offline_key, config.num_envs)
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
                eff_len = config.episode_length if self.agent_type == "sac" else config.episode_length - 1
                num_batches = config.num_envs * eff_len // self.batch_size
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:num_batches], transitions
                )

            (training_state, _), metrics = jax.lax.scan(
                update_networks,
                (training_state, train_key),
                transitions,
            )
            metrics["reward_mean"] = jnp.mean(transitions.reward)

            # ── Online empowerment update (when enabled) ──────────────────────
            # Train the OGBench empowerment_skill agent on the online replay
            # sample for ``online_empowerment_num_grad_steps`` gradient steps.
            if online_empowerment_train_fn is not None:
                emp_key = jax.random.fold_in(train_key, 1)
                new_emp_agent, emp_metrics = online_empowerment_train_fn(
                    training_state.online_empowerment_agent,
                    online_transitions,
                    emp_key,
                    self.online_empowerment_num_grad_steps,
                )
                training_state = training_state.replace(
                    online_empowerment_agent=new_emp_agent
                )
                metrics.update(
                    {f"online_empowerment/{k}": v for k, v in emp_metrics.items()}
                )

            return (
                training_state, env_state, buffer_state, updated_gps,
            ), metrics

        # ── training_epoch ────────────────────────────────────────────────────
        @jax.jit
        def training_epoch(
            training_state, env_state, buffer_state, key,
            goal_proposer_state,
        ):
            # Snapshot cumulative counters *before* the epoch so we can compute
            # epoch-level deltas (rather than a lifetime average).
            pre_completions   = jnp.sum(env_state.info['go_completions_total'])
            pre_successes     = jnp.sum(env_state.info['go_successes_total'])
            pre_success_steps = jnp.sum(env_state.info['go_success_steps_total'])

            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, gps), metrics = training_step(
                    ts, es, bs, train_key, gps
                )
                return (ts, es, bs, k, gps), metrics

            (
                (training_state, env_state, buffer_state, key,
                 goal_proposer_state),
                metrics,
            ) = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key,
                 goal_proposer_state),
                (),
                length=num_training_steps_per_epoch,
            )

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
                goal_proposer_state, metrics,
            )

        # ── prefill ───────────────────────────────────────────────────────────
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, goal_proposer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state
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
                goal_proposer_state, metrics,
            ) = training_epoch(
                training_state, env_state, buffer_state, epoch_key,
                goal_proposer_state,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time  = time.time() - t
            training_walltime   += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time

            # Online empowerment metrics keep their own top-level
            # "online_empowerment/" prefix (dedicated wandb tab); the rest go
            # under "training/".
            metrics_dict = format_epoch_metrics(metrics)

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
            if current_step // 2_000_000 > last_visualization_step // 2_000_000:
                key, _ = jax.random.split(key)
                buffer_state = all_visualizations(
                    replay_buffer=replay_buffer,
                    buffer_state=buffer_state,
                    env=unwrapped_env,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
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
                last_visualization_step = current_step

            # ── Online empowerment heatmap (periodic: every eval epoch) ───────
            if online_empowerment_score_fn is not None:
                key, emp_viz_key = jax.random.split(key)
                # Discard the post-sample buffer_state so visualization sampling
                # does not perturb the training trajectory (mirrors phase viz).
                _, emp_viz_trans = replay_buffer.sample(buffer_state)
                emp_states = jnp.reshape(
                    emp_viz_trans.observation,
                    (-1, emp_viz_trans.observation.shape[-1]),
                )[:, :state_size]
                log_online_empowerment_heatmap(
                    online_empowerment_score_fn,
                    training_state.online_empowerment_agent,
                    emp_states,
                    tuple(train_env.goal_indices),
                    getattr(unwrapped_env, "x_bounds", None),
                    getattr(unwrapped_env, "y_bounds", None),
                    emp_viz_key,
                    current_step,
                    max_points=self.online_empowerment_heatmap_num_states,
                )

            do_render = ne % config.visualization_interval == 0
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
