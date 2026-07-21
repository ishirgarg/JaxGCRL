import logging
import os
import pickle
import pprint

import tyro
from brax.io import model

import wandb
from jaxgcrl.utils.config import Config
from jaxgcrl.utils.env import MetricsRecorder, create_env


def main(config: Config):
    """Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Creates the following directory structure:
        ./runs/
            run_{name}_s_{seed}/  # Run-specific directory
                args.pkl          # Saved command-line arguments
                ckpt/            # Model checkpoints
    Initializes wandb logging if enabled. Runs training with profiling and
    saves profiling results.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    info = {**vars(config.run), **vars(config.agent)}

    utd_ratio = (
        config.run.num_envs
        * config.run.episode_length
        * config.agent.train_step_multiplier
        / config.agent.batch_size
    ) / (config.run.num_envs * config.agent.unroll_length)
    info["utd_ratio"] = utd_ratio
    info["agent"] = type(config.agent).__name__

    logging.info("Arguments:\n%s", pprint.pformat(info))

    wandb.init(
        project=config.run.wandb_project_name,
        group=config.run.wandb_group,
        name=config.run.exp_name,
        config=info,
        mode="online" if config.run.log_wandb else "disabled",
    )

    env = create_env(env_name=config.run.env, backend=config.run.backend)
    if config.run.eval_env:
        eval_env = create_env(env_name=config.run.eval_env, backend=config.run.backend)
    else:
        eval_env = env

    run_dir = f"./runs/run_{config.run.exp_name}_s_{config.run.seed}"
    ckpt_dir = run_dir + "/ckpt"
    if not config.run.no_save:
        os.makedirs("./runs", exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(run_dir + "/args.pkl", "wb") as f:
            pickle.dump(vars(config), f)

    metrics_to_collect = [
        "eval/episode_dist",
        "eval/episode_reward",
        "eval/episode_reward_ctrl",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_survive",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_easy",
        "eval/episode_success_hard",
        "eval/episode_distance_from_origin",
        "eval/avg_episode_length",
        "eval/epoch_eval_time",
        "eval/sps",
        "eval/walltime",
        "training/actor_loss",
        "training/log_alpha",
        "training/alpha",
        "training/alpha_loss",
        "training/critic_loss",
        "training/entropy",
        "training/reward_mean",
        "training/sps",
        "training/walltime",
        "training/envsteps",
        "training/go_phase_success_rate",
        "training/avg_go_phase_steps",
        "training/buffer_current_size",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
        "training/emp_abs_mean",
        "training/emp_abs_min",
        "training/emp_abs_max",
        "training/emp_abs_scaled_mean",
        "training/emp_delta_mean",
        "training/emp_delta_min",
        "training/emp_delta_max",
        "training/emp_delta_scaled_mean",
        # Hierarchical skill controller (agent_type="sac_discrete" / "crl_skill")
        "training/controller_actor_loss",
        "training/controller_critic_loss",
        "training/controller_alpha",
        "training/controller_alpha_loss",
        "training/controller_entropy",
        "training/controller_target_entropy",
        "training/controller_reward_mean",
        "training/controller_done_mean",
        "training/macro_reward_mean",
        "training/macro_done_mean",
        # CRL contrastive-critic controller (agent_type="crl_skill")
        "training/controller_categorical_accuracy",
        "training/controller_logits_pos",
        "training/controller_logits_neg",
        "training/controller_logsumexp",
        "eval/skill_entropy",
        "eval/skill_max_frac",
        "eval/skill_active_count",
        # Online empowerment (goal proposer trained in lockstep). Logged under a
        # dedicated "online_empowerment/" wandb tab; 0 for runs that don't use it.
        "online_empowerment/bc/alpha",
        "online_empowerment/bc/bc_log_prob_max",
        "online_empowerment/bc/bc_log_prob_mean",
        "online_empowerment/bc/bc_log_prob_min",
        "online_empowerment/bc/bc_loss",
        "online_empowerment/empowerment/max",
        "online_empowerment/empowerment/mean",
        "online_empowerment/empowerment/min",
        "online_empowerment/grad/max",
        "online_empowerment/grad/min",
        "online_empowerment/grad/norm",
        "online_empowerment/policy/e_delta_max",
        "online_empowerment/policy/e_delta_mean",
        "online_empowerment/policy/e_delta_min",
        "online_empowerment/policy/policy_loss",
        "online_empowerment/q/q_log_current_mean",
        "online_empowerment/q/q_log_mean",
        "online_empowerment/q/q_log_next_current_mean",
        "online_empowerment/q/q_log_next_future_mean",
        "online_empowerment/q/q_loss",
        "online_empowerment/q/q_loss_current",
        "online_empowerment/q/q_loss_future",
        "online_empowerment/q/v_next_current_log_mean",
        "online_empowerment/q/v_next_future_log_mean",
        "online_empowerment/total_loss",
        "online_empowerment/v/q_pi_log_mean",
        "online_empowerment/v/v_log_mean",
        "online_empowerment/v/v_loss",
        "online_empowerment/v/v_loss_current",
        "online_empowerment/v/v_loss_future",
        "online_empowerment/v/v_max",
        "online_empowerment/v/v_min",
    ]

    # DADS (skill-discovery) metrics. The empowerment heatmap + empowerment/*
    # stats are logged directly to wandb from the agent; here we register the
    # scalar training/eval metrics so MetricsRecorder forwards them.
    if type(config.agent).__name__ == "DADS":
        metrics_to_collect += [
            "training/dynamics_loss",
            "training/dads_reward",
            "mean_training/dynamics_loss",
            "mean_training/dads_reward",
        ]
        for i in range(getattr(config.agent, "num_skills", 0)):
            metrics_to_collect += [
                f"skill_{i}_training/dynamics_loss",
                f"skill_{i}_training/dads_reward",
                f"skill_{i}_eval/success",
                f"skill_{i}_eval/success_easy",
            ]

    metrics_recorder = MetricsRecorder(
        metrics_to_collect,
        None if config.run.no_save else run_dir,
        config.run.exp_name,
        mode=config.run.wandb_mode,
    )

    _, params, _ = config.agent.train_fn(
        train_env=env,
        eval_env=eval_env,
        config=config.run,
        progress_fn=metrics_recorder.progress,
    )
    if not config.run.no_save:
        model.save_params(ckpt_dir + "/final", params)


def cli():
    tyro.cli(
        main,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
        ),
    )


if __name__ == "__main__":
    cli()
