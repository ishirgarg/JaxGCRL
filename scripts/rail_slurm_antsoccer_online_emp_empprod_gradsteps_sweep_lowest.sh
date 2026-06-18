#!/bin/bash
#SBATCH --job-name=asoc_onemp_lowest
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-111

# 4D antsoccer (1-goal + 3-goal) ONLINE empowerment + empowerment_density_product
# sweep, run entirely on lowest priority. SINGLE SEED (0).
#
# Difference from rail_slurm_antsoccer_mega_emp_empprod_rlpd_sweep_lowest:
#   * NO MEGA runs.
#   * RLPD is ALWAYS OFF (--no_use_rlpd).
#   * Empowerment is computed FULLY ONLINE (--online_empowerment): a fresh
#     OGBench empowerment_skill agent is trained in lockstep, so there is no
#     pretrained checkpoint / --empowerment_run_dir. Only --ogbench_root (the
#     OGBench repo root containing impls/) is needed to import the agent class.
#   * Instead of sweeping RLPD/checkpoints, we sweep the number of online
#     empowerment gradient steps per update loop: --online_empowerment_num_grad_steps
#     in {1, 2}.
#
# Two envs:
#   asoc_sq1g : ant_ball_4d_ogbench_small_easy_square_1g   (1-goal)
#   asoc_sq   : ant_ball_4d_ogbench_small_easy_square      (3-goal)
#
# Per-proposer sweeps (identical grids to the offline script):
#   * empowerment              — alpha∈{1,3,10} × temp∈{0,0.03,0.1,0.01} = 12
#   * empowerment_density_prod — alpha∈{0.33,0.5,1,2} × temp∈{0,0.01,0.03,0.1} = 16
#
# Episode partition mimics the antsoccer slice of emp_alpha_temp
# (episode_length - 1 == num_gcp_steps + num_ep_steps == 300 + 300 -> 601).
#
# ── Run accounting (single seed) ────────────────────────────────────────────
#   emp      : 2 envs × 2 gradsteps × 12 =  48
#   empprod  : 2 envs × 2 gradsteps × 16 =  64
#   TOTAL                                = 112   -> --array=0-111
#
# ── Index decoding ──────────────────────────────────────────────────────────
#   GLOBAL_IDX = SLURM_ARRAY_TASK_ID                       (0..111)
#       ENV_IDX = GLOBAL_IDX / 56     (0..1)        [56 = 2 gradsteps × 28]
#       R1      = GLOBAL_IDX % 56
#       GS_IDX  = R1 / 28             (0..1)
#       CFG     = R1 % 28             (0..27)
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

# ── OGBench repo root (the dir containing impls/) — needed to import the
#    empowerment_skill agent class for online training. No checkpoint is loaded.
OGBENCH_ROOT=/global/home/users/ishirgarg/ogbench

# ── Online empowerment gradient-steps sweep (indexed by GS_IDX) ─────────────
GRAD_STEPS=(1 2)

# ── Per-proposer alpha/temp grids ───────────────────────────────────────────
EMP_ALPHAS=(1 3 10)
EMP_TEMPS=(0 0.03 0.1 0.01)
PROD_ALPHAS=(0.33 0.5 1 2)
PROD_TEMPS=(0 0.01 0.03 0.1)

SEED=0

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$SLURM_ARRAY_TASK_ID

ENV_IDX=$((GLOBAL_IDX / 56))
R1=$((GLOBAL_IDX % 56))
GS_IDX=$((R1 / 28))
CFG=$((R1 % 28))

ENV=${ENVS[$ENV_IDX]}
TAG=${ENV_TAGS[$ENV_IDX]}
GS=${GRAD_STEPS[$GS_IDX]}

if [ "$CFG" -lt 12 ]; then
  A_IDX=$((CFG / 4))
  T_IDX=$((CFG % 4))
  ALPHA=${EMP_ALPHAS[$A_IDX]}
  TEMP=${EMP_TEMPS[$T_IDX]}
  PROPOSER_ARGS="--goal_proposer_name empowerment \
      --empowerment_alpha $ALPHA \
      --goal_proposer_temperature $TEMP"
  EXP_NAME="${TAG}__onemp_a${ALPHA}_t${TEMP}__gs${GS}__s${SEED}"
else
  E_IDX=$((CFG - 12))
  A_IDX=$((E_IDX / 4))
  T_IDX=$((E_IDX % 4))
  ALPHA=${PROD_ALPHAS[$A_IDX]}
  TEMP=${PROD_TEMPS[$T_IDX]}
  PROPOSER_ARGS="--goal_proposer_name empowerment_density_product \
      --empowerment_alpha $ALPHA \
      --goal_proposer_temperature $TEMP"
  EXP_NAME="${TAG}__onempprod_a${ALPHA}_t${TEMP}__gs${GS}__s${SEED}"
fi

echo "GLOBAL_IDX=$GLOBAL_IDX  ENV=$ENV  GRAD_STEPS=$GS  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --env $ENV \
        --total_env_steps 80000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP_STEPS \
        --num_ep_steps $EP_STEPS \
        --n_critics 1 \
        --no_use_rlpd \
        --seed $SEED \
        $PROPOSER_ARGS \
        --online_empowerment \
        --online_empowerment_num_grad_steps $GS \
        --ogbench_root $OGBENCH_ROOT \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group antsoccer_online_emp_empprod_gradsteps_sweep \
        --log_wandb
