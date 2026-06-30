#!/bin/bash
#SBATCH --job-name=asoc_s2_mega_emp_only_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-8

# ============================================================================
# Stage-3-ONLY bc_alpha SWEEP on HIGH priority (rail_gpu4_high).
#
# The full pipeline lives in
# rail_slurm_antsoccer_scale2_mega_collect_empowerment_pipeline.sh and runs
# three stages: (1) train MEGA policy, (2) collect an OGBench-format offline
# dataset, (3) run OGBench's empowerment_skill on that dataset. Stages 1+2
# already completed and the dataset is on scratch; only Stage 3 crashed.
#
# This script reuses the SAME path derivation so it picks up that exact dataset
# and runs ONLY Stage 3 (OGBench empowerment_skill). It does NOT retrain or
# recollect anything. Walltime is dropped to 48h since the long MEGA training
# stage is skipped.
#
# It sweeps the empowerment_skill bc_alpha over 9 values (one array task each):
#   BC_ALPHAS = (0 0.003 0.01 0.03 0.1 0.3 1 3 10)
#   EMP_BC_ALPHA = BC_ALPHAS[SLURM_ARRAY_TASK_ID]
# Each task writes to its own bc_<alpha> subdir so runs do not collide.
# ============================================================================

set -euo pipefail

# ── Repo + scratch locations (must match the pipeline script) ───────────────
JAXGCRL_ROOT=/global/home/users/ishirgarg/JaxGCRL         # dir containing run.py (submit dir)
OGBENCH_ROOT=/global/home/users/ishirgarg/ogbench   # dir containing impls/
SCRATCH_BASE=/global/scratch/users/ishirgarg/jaxgcrl

# Local wandb run data + mujoco rendering (home quota is small).
export WANDB_DIR=$SCRATCH_BASE
mkdir -p "$WANDB_DIR"
export MUJOCO_GL=egl

# Optional: activate a conda env for ogbench/impls. Leave empty to use whatever
# environment the job was submitted with (matches the pipeline script).
OGBENCH_CONDA_ENV=""
maybe_activate () {
  if [ -n "$1" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
  fi
}

# ── Experiment config (identical to the pipeline script) ────────────────────
SEED=0
ENV_TAG=asoc_ar1g_s2
# OGBench env/dataset name this JaxGCRL env maps to. Stage 3 builds the env from
# this name and loads OUR collected data.
OGBENCH_ENV=antsoccer-arena-navigate-v0

# Stage 3 — OGBench empowerment_skill (mirrors empowerment_skill_navigate_50.sh).
EMP_NUM_SKILLS=50
EMP_TRAIN_STEPS=1500000

# bc_alpha sweep — one array task per value.
BC_ALPHAS=(0 0.003 0.01 0.03 0.1 0.3 1 3 10)
EMP_BC_ALPHA=${BC_ALPHAS[$SLURM_ARRAY_TASK_ID]}

# ── Derived paths — must resolve to the dataset produced earlier ────────────
EXP_NAME="${ENV_TAG}__mega__rlpdon__s${SEED}"
PIPE_DIR=$SCRATCH_BASE/mega_emp_pipeline/${EXP_NAME}
DATASET_DIR=$PIPE_DIR/dataset
DATASET=$DATASET_DIR/${ENV_TAG}_mega_line.npz
# Per-bc_alpha save dir so concurrent array tasks do not overwrite each other.
EMP_SAVE_DIR=$PIPE_DIR/empowerment_skill/bc_${EMP_BC_ALPHA}
mkdir -p "$EMP_SAVE_DIR"

echo "=============================================================="
echo "Empowerment-only (Stage 3) bc_alpha sweep: $EXP_NAME"
echo "  task idx    : $SLURM_ARRAY_TASK_ID"
echo "  bc_alpha    : $EMP_BC_ALPHA"
echo "  ogbench env : $OGBENCH_ENV"
echo "  dataset     : $DATASET"
echo "  save dir    : $EMP_SAVE_DIR"
echo "=============================================================="

# ── Verify the collected dataset is present before launching ────────────────
[ -f "$DATASET" ] && [ -f "${DATASET%.npz}-val.npz" ] || {
    echo "ERROR: expected dataset not found at $DATASET (+ -val.npz)."
    echo "       Run the collection pipeline (Stages 1-2) first."
    exit 1; }

# ── Stage 3: recover empowerment with OGBench empowerment_skill on our data ──
echo "[emp-only] STAGE 3: empowerment_skill on the collected dataset"
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

echo "[emp-only] DONE. empowerment_skill outputs -> $EMP_SAVE_DIR"
