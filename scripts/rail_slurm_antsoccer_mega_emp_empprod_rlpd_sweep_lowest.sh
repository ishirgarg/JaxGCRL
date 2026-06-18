#!/bin/bash
#SBATCH --job-name=asoc_emp_lowest
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-227

# 4D antsoccer (1-goal + 3-goal) MEGA + empowerment + empowerment_density_product
# sweep, run entirely on lowest priority. SINGLE SEED (0).
#
# Two envs:
#   asoc_sq1g : ant_ball_4d_ogbench_small_easy_square_1g   (1-goal)
#   asoc_sq   : ant_ball_4d_ogbench_small_easy_square      (3-goal)
#
# Two empowerment checkpoints (used by the empowerment / density_product
# proposers; MEGA ignores them):
#   34594770 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594770.0.20260527_234149
#   34594769 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594769.0.20260527_234149
#
# Per-proposer sweeps (mimicking the existing scripts):
#   * empowerment              — alpha∈{1,3,10} × temp∈{0,0.03,0.1,0.01} = 12
#                                (mimics rail_slurm_emp_alpha_temp_sweep_*)
#   * empowerment_density_prod — alpha∈{0.33,0.5,1,2} × temp∈{0,0.01,0.03,0.1}
#                                = 16 (mimics rail_slurm_antmaze_emp_product_*)
#   * MEGA                     — a single config (no temp sweep), 1 per env.
#
# RLPD (offline data mixing) is SWEPT {off, on} for ALL runs, including MEGA
# (--no_use_rlpd / --use_rlpd). MEGA is therefore 2 runs per env (one per RLPD
# setting); it still ignores the empowerment checkpoint.
#
# Episode partition mimics the antsoccer slice of emp_alpha_temp
# (episode_length - 1 == num_gcp_steps + num_ep_steps == 300 + 300 -> 601).
#
# ── Run accounting (single seed) ────────────────────────────────────────────
#   MEGA     : 2 envs × 2 rlpd                     =   4
#   emp      : 2 envs × 2 ckpts × 2 rlpd × 12      =  96
#   empprod  : 2 envs × 2 ckpts × 2 rlpd × 16      = 128
#   TOTAL                                          = 228   -> --array=0-227
#
# ── Index decoding ──────────────────────────────────────────────────────────
#   GLOBAL_IDX = SLURM_ARRAY_TASK_ID                       (0..227)
#   MEGA block : GLOBAL_IDX 0..3   -> ENV_IDX  = GLOBAL_IDX / 2 (0..1)
#                                     RLPD_IDX = GLOBAL_IDX % 2 (0..1)
#   Main block : GLOBAL_IDX 4..227 -> R = GLOBAL_IDX - 4   (0..223)
#       ENV_IDX  = R / 112         (0..1)        [112 = 2 ckpts × 2 rlpd × 28]
#       R1       = R % 112
#       CKPT_IDX = R1 / 56         (0..1)        [ 56 = 2 rlpd × 28]
#       R2       = R1 % 56
#       RLPD_IDX = R2 / 28         (0..1)
#       CFG      = R2 % 28         (0..27)
#         CFG 0..11  -> empowerment:               A_IDX = CFG/4 (0..2)
#                                                  T_IDX = CFG%4 (0..3)
#         CFG 12..27 -> empowerment_density_product:
#                       E_IDX = CFG-12 (0..15)     A_IDX = E_IDX/4 (0..3)
#                                                  T_IDX = E_IDX%4 (0..3)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# ── Envs (parallel arrays, indexed by ENV_IDX) ──────────────────────────────
ENVS=(
  "ant_ball_4d_ogbench_small_easy_square_1g"
  "ant_ball_4d_ogbench_small_easy_square"
)
ENV_TAGS=(asoc_sq1g asoc_sq)
# Antsoccer episode partition (same for both square variants).
EP_LEN=601
GCP_STEPS=300
EP_STEPS=300

# ── Empowerment checkpoints (indexed by CKPT_IDX) ───────────────────────────
CKPT_PREFIX=/global/home/users/ishirgarg/ogbench/impls/ckpts/antsoccer-arena-navigate
EMP_DIRS=(
  "${CKPT_PREFIX}/sd000_s_34594770.0.20260527_234149"
  "${CKPT_PREFIX}/sd000_s_34594769.0.20260527_234149"
)
CKPT_TAGS=(34594770 34594769)

# ── RLPD sweep (indexed by RLPD_IDX) ────────────────────────────────────────
RLPD_FLAGS=("--no_use_rlpd" "--use_rlpd")
RLPD_TAGS=(rlpdoff rlpdon)

# ── Per-proposer alpha/temp grids ───────────────────────────────────────────
EMP_ALPHAS=(1 3 10)
EMP_TEMPS=(0 0.03 0.1 0.01)
PROD_ALPHAS=(0.33 0.5 1 2)
PROD_TEMPS=(0 0.01 0.03 0.1)

SEED=0

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$SLURM_ARRAY_TASK_ID

if [ "$GLOBAL_IDX" -lt 4 ]; then
  # MEGA: one per env per RLPD setting (ignores the empowerment ckpt).
  ENV_IDX=$((GLOBAL_IDX / 2))
  RLPD_IDX=$((GLOBAL_IDX % 2))
  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  RLPD_FLAG=${RLPD_FLAGS[$RLPD_IDX]}
  RLPD_TAG=${RLPD_TAGS[$RLPD_IDX]}
  PROPOSER_ARGS="--goal_proposer_name mega"
  EXP_NAME="${TAG}__mega__${RLPD_TAG}__s${SEED}"
else
  R=$((GLOBAL_IDX - 4))
  ENV_IDX=$((R / 112))
  R1=$((R % 112))
  CKPT_IDX=$((R1 / 56))
  R2=$((R1 % 56))
  RLPD_IDX=$((R2 / 28))
  CFG=$((R2 % 28))

  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  EMP_DIR=${EMP_DIRS[$CKPT_IDX]}
  CKPT_TAG=${CKPT_TAGS[$CKPT_IDX]}
  RLPD_FLAG=${RLPD_FLAGS[$RLPD_IDX]}
  RLPD_TAG=${RLPD_TAGS[$RLPD_IDX]}

  if [ "$CFG" -lt 12 ]; then
    A_IDX=$((CFG / 4))
    T_IDX=$((CFG % 4))
    ALPHA=${EMP_ALPHAS[$A_IDX]}
    TEMP=${EMP_TEMPS[$T_IDX]}
    PROPOSER_ARGS="--goal_proposer_name empowerment \
        --empowerment_alpha $ALPHA \
        --goal_proposer_temperature $TEMP \
        --empowerment_run_dir $EMP_DIR"
    EXP_NAME="${TAG}__emp_a${ALPHA}_t${TEMP}__${RLPD_TAG}__c${CKPT_TAG}__s${SEED}"
  else
    E_IDX=$((CFG - 12))
    A_IDX=$((E_IDX / 4))
    T_IDX=$((E_IDX % 4))
    ALPHA=${PROD_ALPHAS[$A_IDX]}
    TEMP=${PROD_TEMPS[$T_IDX]}
    PROPOSER_ARGS="--goal_proposer_name empowerment_density_product \
        --empowerment_alpha $ALPHA \
        --goal_proposer_temperature $TEMP \
        --empowerment_run_dir $EMP_DIR"
    EXP_NAME="${TAG}__empprod_a${ALPHA}_t${TEMP}__${RLPD_TAG}__c${CKPT_TAG}__s${SEED}"
  fi
fi

echo "GLOBAL_IDX=$GLOBAL_IDX  ENV=$ENV  RLPD=$RLPD_TAG  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --env $ENV \
        --total_env_steps 80000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP_STEPS \
        --num_ep_steps $EP_STEPS \
        --n_critics 1 \
        $RLPD_FLAG \
        --seed $SEED \
        $PROPOSER_ARGS \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group antsoccer_mega_emp_empprod_rlpd_sweep \
        --log_wandb
