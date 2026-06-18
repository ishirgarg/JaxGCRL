#!/bin/bash
#SBATCH --job-name=umaze_mega
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0

# Basic MEGA goal proposer on ant u-maze via the go-explore-simple agent.
# No RLPD (offline data mixing disabled), 250 go steps + 250 explore steps.
# Array index selects the seed (0..4).

SEEDS=(0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

ENV=ant_u_maze

# Go-explore phase budget. check_config requires:
#   episode_length - 1 == num_gcp_steps + num_ep_steps
# so 250 + 250 -> episode_length must be 501.
NUM_GCP_STEPS=250
NUM_EP_STEPS=250
EPISODE_LENGTH=501

# num_envs * (episode_length - 1) must be divisible by batch_size (256).
# 256 * 500 % 256 == 0, so num_envs=256 satisfies the assertion (the default
# of 32 does not for episode_length=501).
NUM_ENVS=256

echo "ENV=$ENV  SEED=$SEED  proposer=mega  no_rlpd  gcp=$NUM_GCP_STEPS  ep=$NUM_EP_STEPS"

python run.py go-explore-simple \
        --env $ENV \
        --seed $SEED \
        --total_env_steps 80000000 \
        --episode_length $EPISODE_LENGTH \
        --num_envs $NUM_ENVS \
        --goal_proposer_name mega \
        --no_use_rlpd \
        --num_gcp_steps $NUM_GCP_STEPS \
        --num_ep_steps $NUM_EP_STEPS \
        --exp_name umaze_mega__s${SEED} \
        --wandb_project_name jaxgcrl_new \
        --wandb_group umaze_mega \
        --log_wandb
