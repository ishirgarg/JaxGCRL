#!/bin/bash
#SBATCH --job-name=dads_asoc_scale
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-3

# ─────────────────────────────────────────────────────────────────────────────
# DADS (Dynamics-Aware Discovery of Skills, future-state variant) on the two
# antsoccer "scale" arenas, then log a post-training empowerment heatmap.
#
# The discriminator models a gamma-discounted future state s+ (future_discount
# = 0.99, ~100-step horizon) instead of the one-step next state.  The SAC policy
# is trained to maximise the per-state MI lower bound r(s,z,s+).  After 80M
# steps the full parameter set (policy + Q + discriminator + normalizers) is
# saved and an empowerment heatmap over the visited x-y region is logged to
# wandb (ball fixed at its spawn, 1/4-unit grid cells).
#
# Two envs (same [-6, 26] world frame; only the ant->ball gap differs):
#   asoc_ar1g_s2 : ant_ball_4d_ogbench_arena_1g_scale2   (ant/ball 2 units apart)
#   asoc_ar1g_s1 : ant_ball_4d_ogbench_arena_1g_scale1   (ant/ball 1 unit apart)
#
# 2 envs x 2 skill counts x 1 seed = 4 array tasks (--array=0-3).
#   ENV_IDX   = SLURM_ARRAY_TASK_ID / 2   (0..1)
#   SKILL_IDX = SLURM_ARRAY_TASK_ID % 2   (0..1)  -> num_skills in {15, 50}
# ─────────────────────────────────────────────────────────────────────────────

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl

ENVS=(
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale1"
)
ENV_TAGS=(asoc_ar1g_s2 asoc_ar1g_s1)
NUM_SKILLS=(15 50)

GLOBAL_IDX=$SLURM_ARRAY_TASK_ID
ENV_IDX=$((GLOBAL_IDX / 2))
SKILL_IDX=$((GLOBAL_IDX % 2))
SEED=0
ENV=${ENVS[$ENV_IDX]}
TAG=${ENV_TAGS[$ENV_IDX]}
K=${NUM_SKILLS[$SKILL_IDX]}
EXP_NAME="dads__${TAG}__k${K}__s${SEED}"

echo "GLOBAL_IDX=$GLOBAL_IDX  ENV=$ENV  K=$K  SEED=$SEED  EXP=$EXP_NAME"

# ── DADS hyperparameters ────────────────────────────────────────────────────
#   num_skills in {15, 50}, future_discount=0.99 (s+ ~ Geom, ~100-step horizon).
#   x-y prior discriminator (use_xy_prior): restricts skill dynamics to the
#   goal_indices (x-y) deltas.
#   Empowerment heatmap: 1/4-unit cells over the visited region, 16 s+ samples
#   per (cell, skill), rollout horizon auto = ceil(3 / (1 - 0.99)) = 300.
#
#   Buffer/UTD note: train_steps processes num_envs*(episode_length-1)/batch_size
#   minibatches per actor step. With these values that is 64*600/256 = 150.
python run.py dads \
  --env "$ENV" \
  --total_env_steps 100000000 \
  --episode_length 601 \
  --num_envs 64 \
  --num_eval_envs 64 \
  --num_evals 20 \
  --unroll_length 50 \
  --batch_size 256 \
  --min_replay_size 1000 \
  --max_replay_size 20000 \
  --num_skills "$K" \
  --use_xy_prior \
  --future_discount 0.99 \
  --discounting 0.99 \
  --learning_rate 3e-4 \
  --h_dim 256 --n_hidden 4 \
  --train_step_multiplier 1 \
  --visualization_interval 5 \
  --emp_grid_spacing 0.25 \
  --emp_num_future_samples 16 \
  --emp_rollout_horizon 0 \
  --emp_collect_envs 256 \
  --emp_max_cells 4000 \
  --compute_empowerment_map \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --wandb_project_name jaxgcrl_new \
  --wandb_group dads_antsoccer_scale_empowerment \
  --log_wandb
