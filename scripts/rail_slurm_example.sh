#!/bin/bash
#SBATCH --job-name=go_explore_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-4

EMP_DIR_BALL="--empowerment_run_dir /global/home/users/ishirgarg/ogbench/impls/ckpts/antsoccer/sd000_s_33708849.0.20260423_043239"
EMP_DIR_MAZE="--empowerment_run_dir /global/home/users/ishirgarg/ogbench/impls/ckpts/antmaze/sd000_s_33711690.0.20260423_095122"

IDX=${SLURM_ARRAY_TASK_ID}

# -----------------------------
# Sweep definitions
# -----------------------------
SEEDS=(0 1 2 3 4)

# Decode index
SEED_INDEX=$((IDX % 5))

SEED=${SEEDS[$SEED_INDEX]}


# -----------------------------
# Run variants
# -----------------------------
python run.py crl \
        --no-use_rlpd \
        --env ant_ball_4d_ogbench_small_easy_square_1g \
        --total_env_steps 80000000 \
        --episode_length 801 \
        --seed $SEED \