#!/bin/bash
#SBATCH --job-name=asoc_emp_lowest
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-386

# 4D antsoccer (1-goal + 3-goal + arena 1-goal) MEGA + empowerment +
# empowerment_density_product sweep, run entirely on lowest priority.
# THREE SEEDS (0, 1, 2).
#
# Three envs:
#   asoc_sq1g : ant_ball_4d_ogbench_small_easy_square_1g   (1-goal)
#   asoc_sq   : ant_ball_4d_ogbench_small_easy_square      (3-goal)
#   asoc_ar1g : ant_ball_4d_ogbench_arena_1g               (open arena, 1-goal)
#
# Two empowerment checkpoints (used by the empowerment / density_product
# proposers; MEGA ignores them):
#   34594770 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594770.0.20260527_234149
#   34594769 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594769.0.20260527_234149
#
# Per-proposer sweeps (mimicking the existing scripts):
#   * empowerment              — alpha∈{1,3,10} × temp∈{0,0.01,0.03} = 9
#   * empowerment_density_prod — alpha∈{0.33,0.5,1,2} × temp∈{0,0.01,0.03}
#                                = 12
#   * MEGA                     — a single config (no temp sweep), 1 per env.
#
# RLPD (offline data mixing) is ALWAYS ON (--use_rlpd) for ALL runs, including
# MEGA. MEGA is therefore 1 run per env; it still ignores the empowerment
# checkpoint.
#
# Episode partition mimics the antsoccer slice of emp_alpha_temp
# (episode_length - 1 == num_gcp_steps + num_ep_steps == 300 + 300 -> 601).
#
# ── Run accounting (per seed × 3 seeds) ─────────────────────────────────────
#   MEGA     : 3 envs                              =   3
#   emp      : 3 envs × 2 ckpts × 9               =  54
#   empprod  : 3 envs × 2 ckpts × 12              =  72
#   PER SEED                                       = 129
#   TOTAL    : 129 × 3 seeds                       = 387   -> --array=0-386
#
# ── Index decoding ──────────────────────────────────────────────────────────
#   GLOBAL_IDX = SLURM_ARRAY_TASK_ID                       (0..386)
#   SEED_IDX = GLOBAL_IDX / 129   (0..2)    [seed is the outermost dimension]
#   LOCAL    = GLOBAL_IDX % 129   (0..128)  [decoded exactly as one seed's run]
#   MEGA block : LOCAL 0..2   -> ENV_IDX  = LOCAL (0..2)
#   Main block : LOCAL 3..128 -> R = LOCAL - 3   (0..125)
#       ENV_IDX  = R / 42          (0..2)        [42 = 2 ckpts × 21]
#       R1       = R % 42
#       CKPT_IDX = R1 / 21         (0..1)
#       CFG      = R1 % 21         (0..20)
#         CFG 0..8   -> empowerment:               A_IDX = CFG/3 (0..2)
#                                                  T_IDX = CFG%3 (0..2)
#         CFG 9..20  -> empowerment_density_product:
#                       E_IDX = CFG-9 (0..11)      A_IDX = E_IDX/3 (0..3)
#                                                  T_IDX = E_IDX%3 (0..2)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# ── Envs (parallel arrays, indexed by ENV_IDX) ──────────────────────────────
ENVS=(
  "ant_ball_4d_ogbench_small_easy_square_1g"
  "ant_ball_4d_ogbench_small_easy_square"
  "ant_ball_4d_ogbench_arena_1g"
)
ENV_TAGS=(asoc_sq1g asoc_sq asoc_ar1g)
# Antsoccer episode partition (same for all variants).
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

# ── RLPD is always on for every run in this script. ─────────────────────────
RLPD_FLAG="--use_rlpd"
RLPD_TAG=rlpdon

# ── Per-proposer alpha/temp grids ───────────────────────────────────────────
EMP_ALPHAS=(1 3 10)
EMP_TEMPS=(0 0.01 0.03)
PROD_ALPHAS=(0.33 0.5 1 2)
PROD_TEMPS=(0 0.01 0.03)

# ── Seeds (outermost sweep dimension) ───────────────────────────────────────
SEEDS=(0 1 2)
PER_SEED=129

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((GLOBAL_IDX / PER_SEED))
LOCAL=$((GLOBAL_IDX % PER_SEED))
SEED=${SEEDS[$SEED_IDX]}

if [ "$LOCAL" -lt 3 ]; then
  # MEGA: one per env (ignores the empowerment ckpt).
  ENV_IDX=$LOCAL
  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  PROPOSER_ARGS="--goal_proposer_name mega"
  EXP_NAME="${TAG}__mega__${RLPD_TAG}__s${SEED}"
else
  R=$((LOCAL - 3))
  ENV_IDX=$((R / 42))
  R1=$((R % 42))
  CKPT_IDX=$((R1 / 21))
  CFG=$((R1 % 21))

  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  EMP_DIR=${EMP_DIRS[$CKPT_IDX]}
  CKPT_TAG=${CKPT_TAGS[$CKPT_IDX]}

  if [ "$CFG" -lt 9 ]; then
    A_IDX=$((CFG / 3))
    T_IDX=$((CFG % 3))
    ALPHA=${EMP_ALPHAS[$A_IDX]}
    TEMP=${EMP_TEMPS[$T_IDX]}
    PROPOSER_ARGS="--goal_proposer_name empowerment \
        --empowerment_alpha $ALPHA \
        --goal_proposer_temperature $TEMP \
        --empowerment_run_dir $EMP_DIR"
    EXP_NAME="${TAG}__emp_a${ALPHA}_t${TEMP}__${RLPD_TAG}__c${CKPT_TAG}__s${SEED}"
  else
    E_IDX=$((CFG - 9))
    A_IDX=$((E_IDX / 3))
    T_IDX=$((E_IDX % 3))
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
        --total_env_steps 200000000 \
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
