#!/bin/bash
#SBATCH --job-name=asoc_s2_mega_emp_pipeline
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00

# ============================================================================
# Single-job, THREE-STAGE pipeline (no array sweep):
#
#   Stage 1  Train a MEGA goal-conditioned policy on the scale2 arena antsoccer
#            env, exactly like the MEGA branch of
#            rail_slurm_antsoccer_scale2_mega_emp_online_combined_sweep_lowest.sh
#            (--goal_proposer_name mega, --use_rlpd), and SAVE the checkpoint to
#            BRC scratch.
#   Stage 2  Use that frozen policy to COLLECT an OGBench-format offline dataset
#            (scripts/collect_ogbench_dataset.py) with the `line_to_goal`
#            proposer (goals sampled along the ant->goal line, +-noise), and
#            save it to scratch.
#   Stage 3  Run OGBench's `empowerment_skill` agent (../ogbench/impls) ON THAT
#            dataset to recover empowerment.
#
# The stages run sequentially in one job; if a stage fails the job stops.
#
# NOTE on walltime: Stage 1 is the reference MEGA run (200M env steps) and is by
# far the longest part. If 200M steps will not finish inside --time, lower
# TOTAL_ENV_STEPS below (the collection + empowerment stages are comparatively
# cheap). To sweep seeds, add `#SBATCH --array=0-3` and set `SEED=$SLURM_ARRAY_TASK_ID`.
# ============================================================================

set -euo pipefail

# ── Repo + scratch locations (edit to match your cluster) ───────────────────
JAXGCRL_ROOT=/global/home/users/ishirgarg/JaxGCRL         # dir containing run.py (submit dir)
OGBENCH_ROOT=/global/home/users/ishirgarg/ogbench   # dir containing impls/
SCRATCH_BASE=/global/scratch/users/ishirgarg/jaxgcrl

# Local wandb run data + mujoco rendering (home quota is small).
export WANDB_DIR=$SCRATCH_BASE
mkdir -p "$WANDB_DIR"
export MUJOCO_GL=egl

# Optional: activate a conda env per repo if JaxGCRL and ogbench/impls do not
# share one. Leave empty to use whatever environment the job was submitted with
# (matches the existing rail_slurm_* scripts, which do not activate).
JAXGCRL_CONDA_ENV=""
OGBENCH_CONDA_ENV=""
maybe_activate () {
  if [ -n "$1" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
  fi
}

# ── Experiment config ───────────────────────────────────────────────────────
SEED=0
ENV=ant_ball_4d_ogbench_arena_1g_scale2
ENV_TAG=asoc_ar1g_s2
# OGBench env/dataset name this JaxGCRL env maps to (see utils/offline_buffer.py
# JAXGCRL_TO_OGBENCH). Stage 3 builds the env from this name and loads OUR data.
OGBENCH_ENV=antsoccer-arena-navigate-v0

# Stage 1 — MEGA training (mirrors the reference MEGA branch).
TOTAL_ENV_STEPS=200000000
EP_LEN=601          # episode_length - 1 == num_gcp_steps + num_ep_steps
GCP_STEPS=300
EP_STEPS=300

# Stage 2 — offline collection (line_to_goal: goals between the ant and the goal).
COLLECT_EP_LEN=$GCP_STEPS   # one go-phase-length goal-reaching attempt per episode
NUM_EPISODES=1000           # train episodes (+ NUM_EPISODES//10 val); ~300k transitions
NUM_ENVS=256                # parallel rollout envs (lower if mjx OOMs)
ACTION_NOISE=0.1
NOISE_SCALE=1.0             # world-unit spread of the proposed waypoint

# Stage 3 — OGBench empowerment_skill (mirrors empowerment_skill_navigate_50.sh).
EMP_NUM_SKILLS=50
EMP_BC_ALPHA=0.03
EMP_TRAIN_STEPS=1500000

# ── Derived paths — everything lands under scratch ──────────────────────────
EXP_NAME="${ENV_TAG}__mega__rlpdon__s${SEED}"
PIPE_DIR=$SCRATCH_BASE/mega_emp_pipeline/${EXP_NAME}
CKPT_DIR=$PIPE_DIR/ckpt
DATASET_DIR=$PIPE_DIR/dataset
DATASET=$DATASET_DIR/${ENV_TAG}_mega_line.npz
EMP_SAVE_DIR=$PIPE_DIR/empowerment_skill
mkdir -p "$CKPT_DIR" "$DATASET_DIR" "$EMP_SAVE_DIR"

echo "=============================================================="
echo "Pipeline: $EXP_NAME"
echo "  env        : $ENV  ->  ogbench: $OGBENCH_ENV"
echo "  scratch    : $PIPE_DIR"
echo "=============================================================="

# ── Stage 1: train MEGA policy, checkpoint to scratch ───────────────────────
echo "[pipeline] STAGE 1: MEGA training ($TOTAL_ENV_STEPS env steps)"
maybe_activate "$JAXGCRL_CONDA_ENV"
cd "$JAXGCRL_ROOT"

# --no_no_save flips no_save=False so run.py writes runs/<exp>/{args.pkl,ckpt/final}.
# --checkpoint_logdir also dumps periodic step_*.pkl straight to scratch (a safety
# net for this long run).
python run.py go-explore-simple \
    --env $ENV \
    --total_env_steps $TOTAL_ENV_STEPS \
    --episode_length $EP_LEN \
    --num_gcp_steps $GCP_STEPS \
    --num_ep_steps $EP_STEPS \
    --n_critics 1 \
    --use_rlpd \
    --goal_proposer_name mega \
    --seed $SEED \
    --no_no_save \
    --checkpoint_logdir "$CKPT_DIR" \
    --exp_name $EXP_NAME \
    --wandb_project_name jaxgcrl_new \
    --wandb_group antsoccer_scale2_mega_emp_pipeline \
    --log_wandb

# Promote run.py's final policy + training config (args.pkl needed so the
# collector reconstructs the exact actor architecture) into the scratch ckpt dir.
RUN_DIR="$JAXGCRL_ROOT/runs/run_${EXP_NAME}_s_${SEED}"
[ -f "$RUN_DIR/ckpt/final" ] || { echo "ERROR: stage 1 produced no checkpoint at $RUN_DIR/ckpt/final"; exit 1; }
cp "$RUN_DIR/ckpt/final" "$CKPT_DIR/final"
cp "$RUN_DIR/args.pkl"   "$CKPT_DIR/args.pkl"
echo "[pipeline] checkpoint -> $CKPT_DIR/final"

# ── Stage 2: collect OGBench-format offline dataset from the trained policy ──
echo "[pipeline] STAGE 2: collecting offline dataset (line_to_goal)"
python scripts/collect_ogbench_dataset.py \
    --env $ENV \
    --checkpoint "$CKPT_DIR/final" \
    --train_args_pkl "$CKPT_DIR/args.pkl" \
    --save_path "$DATASET" \
    --goal_proposer line_to_goal \
    --noise_scale $NOISE_SCALE \
    --num_episodes $NUM_EPISODES \
    --episode_length $COLLECT_EP_LEN \
    --num_envs $NUM_ENVS \
    --action_noise $ACTION_NOISE \
    --seed $SEED

[ -f "$DATASET" ] && [ -f "${DATASET%.npz}-val.npz" ] || {
    echo "ERROR: stage 2 did not produce $DATASET (+ -val.npz)"; exit 1; }
echo "[pipeline] dataset -> $DATASET (+ -val.npz)"

# ── Stage 3: recover empowerment with OGBench empowerment_skill on our data ──
echo "[pipeline] STAGE 3: empowerment_skill on the collected dataset"
maybe_activate "$OGBENCH_CONDA_ENV"
cd "$OGBENCH_ROOT/impls"

python main.py \
    --env_name=$OGBENCH_ENV \
    --dataset_path="$DATASET" \
    --agent=agents/empowerment_skill.py \
    --agent.num_skills=$EMP_NUM_SKILLS \
    --agent.bc_alpha=$EMP_BC_ALPHA \
    --train_steps=$EMP_TRAIN_STEPS \
    --save_dir="$EMP_SAVE_DIR" \
    --run_group=asoc_s2_mega_emp_pipeline \
    --seed=$SEED

echo "[pipeline] DONE. empowerment_skill outputs -> $EMP_SAVE_DIR"
