#!/bin/bash
#SBATCH --job-name=asoc_onemp_lowest
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-377

# 4D antsoccer (1-goal + 3-goal + arena 1-goal) ONLINE empowerment +
# empowerment_density_product sweep, run entirely on lowest priority.
# THREE SEEDS (0, 1, 2).
#
# Difference from rail_slurm_antsoccer_mega_emp_empprod_rlpd_sweep_lowest:
#   * NO MEGA runs.
#   * RLPD is ALWAYS OFF (--no_use_rlpd).
#   * Empowerment is computed FULLY ONLINE (--online_empowerment): a fresh
#     OGBench empowerment_skill agent is trained in lockstep, so there is no
#     pretrained checkpoint / --empowerment_run_dir. Only --ogbench_root (the
#     OGBench repo root containing impls/) is needed to import the agent class.
#   * Online empowerment uses a BC coefficient of 0.001
#     (--online_empowerment_bc_alpha 0.001).
#   * Instead of sweeping RLPD/checkpoints, we sweep the number of online
#     empowerment gradient steps per update loop: --online_empowerment_num_grad_steps
#     in {1, 2}.
#
# Three envs:
#   asoc_sq1g : ant_ball_4d_ogbench_small_easy_square_1g   (1-goal)
#   asoc_sq   : ant_ball_4d_ogbench_small_easy_square      (3-goal)
#   asoc_ar1g : ant_ball_4d_ogbench_arena_1g               (open arena, 1-goal)
#
# Per-proposer sweeps (identical grids to the offline script):
#   * empowerment              — alpha∈{1,3,10} × temp∈{0,0.01,0.03} = 9
#   * empowerment_density_prod — alpha∈{0.33,0.5,1,2} × temp∈{0,0.01,0.03} = 12
#
# Episode partition mimics the antsoccer slice of emp_alpha_temp
# (episode_length - 1 == num_gcp_steps + num_ep_steps == 300 + 300 -> 601).
#
# ── Run accounting (per seed × 3 seeds) ─────────────────────────────────────
#   emp      : 3 envs × 2 gradsteps × 9  =  54
#   empprod  : 3 envs × 2 gradsteps × 12 =  72
#   PER SEED                             = 126
#   TOTAL    : 126 × 3 seeds             = 378   -> --array=0-377
#
# ── Index decoding ──────────────────────────────────────────────────────────
#   GLOBAL_IDX = SLURM_ARRAY_TASK_ID                       (0..377)
#   SEED_IDX = GLOBAL_IDX / 126   (0..2)    [seed is the outermost dimension]
#   LOCAL    = GLOBAL_IDX % 126   (0..125)  [decoded exactly as one seed's run]
#       ENV_IDX = LOCAL / 42          (0..2)        [42 = 2 gradsteps × 21]
#       R1      = LOCAL % 42
#       GS_IDX  = R1 / 21             (0..1)
#       CFG     = R1 % 21             (0..20)
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

# ── OGBench repo root (the dir containing impls/) — needed to import the
#    empowerment_skill agent class for online training. No checkpoint is loaded.
OGBENCH_ROOT=/global/home/users/ishirgarg/ogbench

# ── Online empowerment gradient-steps sweep (indexed by GS_IDX) ─────────────
GRAD_STEPS=(1 2)

# ── Per-proposer alpha/temp grids ───────────────────────────────────────────
EMP_ALPHAS=(1 3 10)
EMP_TEMPS=(0 0.01 0.03)
PROD_ALPHAS=(0.33 0.5 1 2)
PROD_TEMPS=(0 0.01 0.03)

# ── Seeds (outermost sweep dimension) ───────────────────────────────────────
SEEDS=(0 1 2)
PER_SEED=126

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((GLOBAL_IDX / PER_SEED))
LOCAL=$((GLOBAL_IDX % PER_SEED))
SEED=${SEEDS[$SEED_IDX]}

ENV_IDX=$((LOCAL / 42))
R1=$((LOCAL % 42))
GS_IDX=$((R1 / 21))
CFG=$((R1 % 21))

ENV=${ENVS[$ENV_IDX]}
TAG=${ENV_TAGS[$ENV_IDX]}
GS=${GRAD_STEPS[$GS_IDX]}

if [ "$CFG" -lt 9 ]; then
  A_IDX=$((CFG / 3))
  T_IDX=$((CFG % 3))
  ALPHA=${EMP_ALPHAS[$A_IDX]}
  TEMP=${EMP_TEMPS[$T_IDX]}
  PROPOSER_ARGS="--goal_proposer_name empowerment \
      --empowerment_alpha $ALPHA \
      --goal_proposer_temperature $TEMP"
  EXP_NAME="${TAG}__onemp_a${ALPHA}_t${TEMP}__gs${GS}__s${SEED}"
else
  E_IDX=$((CFG - 9))
  A_IDX=$((E_IDX / 3))
  T_IDX=$((E_IDX % 3))
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
        --total_env_steps 200000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP_STEPS \
        --num_ep_steps $EP_STEPS \
        --n_critics 1 \
        --no_use_rlpd \
        --seed $SEED \
        $PROPOSER_ARGS \
        --online_empowerment \
        --online_empowerment_num_grad_steps $GS \
        --online_empowerment_bc_alpha 0.001 \
        --ogbench_root $OGBENCH_ROOT \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group antsoccer_online_emp_empprod_gradsteps_sweep \
        --log_wandb
