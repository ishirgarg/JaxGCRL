#!/bin/bash
#SBATCH --job-name=asoc_s2_comb_lowest
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-199

# 4D antsoccer SCALE2 combined sweep, run entirely on lowest priority.
# Combines the offline (MEGA + empowerment + RLPD) script and the online
# empowerment script into one array job, restricted to the two new
# half-distance "scale2" envs. FOUR SEEDS (0, 1, 2, 3).
#
# The scale2 envs use a 16x16 grid at maze_size_scaling=2.0 with an
# OGBench-anchored offset (=5.0), so the world frame is identical to the
# scaling-4 originals (per-axis bounds [-6, 26]) while the ant->ball->goal
# distances are HALVED (2 units/axis instead of 4):
#   asoc_ar1g_s2 : ant_ball_4d_ogbench_arena_1g_scale2               (open arena, 1-goal)
#   asoc_sq1g_s2 : ant_ball_4d_ogbench_small_easy_square_1g_scale2   (1-goal)
#
# Changes vs the two source scripts
# (rail_slurm_antsoccer_mega_emp_empprod_rlpd_sweep_lowest.sh and
#  rail_slurm_antsoccer_online_emp_empprod_gradsteps_sweep_lowest.sh):
#   * Two new scale2 envs (instead of the 3 originals).
#   * ALL empowerment_density_product runs removed.
#   * Temperatures restricted to {0, 0.03} (dropped 0.01).
#   * empowerment_alpha restricted to {1, 3} (dropped 10).
#   * Online empowerment sweeps the BC coefficient over {0.01, 0.1, 1, 10}
#     (instead of sweeping gradient steps); gradient steps fixed at 1.
#   * 4 seeds per configuration.
#
# Run families (all kept):
#   * MEGA      — RLPD always on; 1 run per env (ignores the empowerment ckpt).
#   * emp       — OFFLINE empowerment; RLPD always on; both pretrained
#                 antsoccer-arena-navigate checkpoints; alpha x temp grid.
#   * onemp     — ONLINE empowerment; RLPD off; empowerment trained in lockstep
#                 (--online_empowerment, 1 grad step); BC-coefficient sweep.
#
# Two empowerment checkpoints (used by the OFFLINE empowerment runs only):
#   34594770 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594770.0.20260527_234149
#   34594769 : .../ckpts/antsoccer-arena-navigate/sd000_s_34594769.0.20260527_234149
#
# Episode partition mimics the antsoccer slice of emp_alpha_temp
# (episode_length - 1 == num_gcp_steps + num_ep_steps == 300 + 300 -> 601).
#
# ── Run accounting (per seed x 4 seeds) ─────────────────────────────────────
#   MEGA     : 2 envs                                =   2
#   emp      : 2 envs x 2 ckpts x (2 a x 2 t = 4)    =  16
#   onemp    : 2 envs x 4 bc x (2 a x 2 t = 4)       =  32
#   PER SEED                                          =  50
#   TOTAL    : 50 x 4 seeds                           = 200   -> --array=0-199
#
# ── Index decoding ──────────────────────────────────────────────────────────
#   GLOBAL_IDX = SLURM_ARRAY_TASK_ID                       (0..199)
#   SEED_IDX = GLOBAL_IDX / 50    (0..3)    [seed is the outermost dimension]
#   LOCAL    = GLOBAL_IDX % 50    (0..49)   [decoded exactly as one seed's run]
#     MEGA block   : LOCAL 0..1   -> ENV_IDX = LOCAL (0..1)
#     emp  block   : LOCAL 2..17  -> R = LOCAL - 2   (0..15)
#         ENV_IDX  = R / 8         (0..1)        [8 = 2 ckpts x 4 cfg]
#         R1       = R % 8
#         CKPT_IDX = R1 / 4        (0..1)
#         CFG      = R1 % 4        (0..3)  -> A_IDX = CFG/2 (0..1), T_IDX = CFG%2 (0..1)
#     onemp block  : LOCAL 18..49 -> R = LOCAL - 18  (0..31)
#         ENV_IDX  = R / 16        (0..1)        [16 = 4 bc x 4 cfg]
#         R1       = R % 16
#         BC_IDX   = R1 / 4        (0..3)
#         CFG      = R1 % 4        (0..3)  -> A_IDX = CFG/2 (0..1), T_IDX = CFG%2 (0..1)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# ── Envs (parallel arrays, indexed by ENV_IDX) ──────────────────────────────
ENVS=(
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_small_easy_square_1g_scale2"
)
ENV_TAGS=(asoc_ar1g_s2 asoc_sq1g_s2)
# Antsoccer episode partition (same for all variants).
EP_LEN=601
GCP_STEPS=300
EP_STEPS=300

# ── Empowerment checkpoints (indexed by CKPT_IDX; OFFLINE emp runs only) ─────
CKPT_PREFIX=/global/home/users/ishirgarg/ogbench/impls/ckpts/antsoccer-arena-navigate
EMP_DIRS=(
  "${CKPT_PREFIX}/sd000_s_34594770.0.20260527_234149"
  "${CKPT_PREFIX}/sd000_s_34594769.0.20260527_234149"
)
CKPT_TAGS=(34594770 34594769)

# ── OGBench repo root (the dir containing impls/) — needed to import the
#    empowerment_skill agent class for ONLINE training. No checkpoint is loaded.
OGBENCH_ROOT=/global/home/users/ishirgarg/ogbench

# ── Online empowerment BC-coefficient sweep (indexed by BC_IDX) ─────────────
ONLINE_BC=(0.01 0.1 1 10)
# Online empowerment gradient steps are fixed at 1.
ONLINE_GS=1

# ── Empowerment alpha/temp grids (shared by offline + online emp) ───────────
EMP_ALPHAS=(1 3)
EMP_TEMPS=(0 0.03)

# ── Seeds (outermost sweep dimension) ───────────────────────────────────────
SEEDS=(0 1 2 3)
PER_SEED=50

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((GLOBAL_IDX / PER_SEED))
LOCAL=$((GLOBAL_IDX % PER_SEED))
SEED=${SEEDS[$SEED_IDX]}

if [ "$LOCAL" -lt 2 ]; then
  # ── MEGA: one per env (RLPD on, ignores the empowerment ckpt). ────────────
  ENV_IDX=$LOCAL
  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  RLPD_FLAG="--use_rlpd"
  EXTRA_ARGS=""
  PROPOSER_ARGS="--goal_proposer_name mega"
  EXP_NAME="${TAG}__mega__rlpdon__s${SEED}"

elif [ "$LOCAL" -lt 18 ]; then
  # ── OFFLINE empowerment (RLPD on, pretrained checkpoint). ─────────────────
  R=$((LOCAL - 2))
  ENV_IDX=$((R / 8))
  R1=$((R % 8))
  CKPT_IDX=$((R1 / 4))
  CFG=$((R1 % 4))
  A_IDX=$((CFG / 2))
  T_IDX=$((CFG % 2))

  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  EMP_DIR=${EMP_DIRS[$CKPT_IDX]}
  CKPT_TAG=${CKPT_TAGS[$CKPT_IDX]}
  ALPHA=${EMP_ALPHAS[$A_IDX]}
  TEMP=${EMP_TEMPS[$T_IDX]}

  RLPD_FLAG="--use_rlpd"
  EXTRA_ARGS=""
  PROPOSER_ARGS="--goal_proposer_name empowerment \
      --empowerment_alpha $ALPHA \
      --goal_proposer_temperature $TEMP \
      --empowerment_run_dir $EMP_DIR"
  EXP_NAME="${TAG}__emp_a${ALPHA}_t${TEMP}__rlpdon__c${CKPT_TAG}__s${SEED}"

else
  # ── ONLINE empowerment (RLPD off, online training, BC sweep, 1 grad step). ─
  R=$((LOCAL - 18))
  ENV_IDX=$((R / 16))
  R1=$((R % 16))
  BC_IDX=$((R1 / 4))
  CFG=$((R1 % 4))
  A_IDX=$((CFG / 2))
  T_IDX=$((CFG % 2))

  ENV=${ENVS[$ENV_IDX]}
  TAG=${ENV_TAGS[$ENV_IDX]}
  BC=${ONLINE_BC[$BC_IDX]}
  ALPHA=${EMP_ALPHAS[$A_IDX]}
  TEMP=${EMP_TEMPS[$T_IDX]}

  RLPD_FLAG="--no_use_rlpd"
  EXTRA_ARGS="--online_empowerment \
      --online_empowerment_num_grad_steps $ONLINE_GS \
      --online_empowerment_bc_alpha $BC \
      --ogbench_root $OGBENCH_ROOT"
  PROPOSER_ARGS="--goal_proposer_name empowerment \
      --empowerment_alpha $ALPHA \
      --goal_proposer_temperature $TEMP"
  EXP_NAME="${TAG}__onemp_a${ALPHA}_t${TEMP}__bc${BC}__gs${ONLINE_GS}__s${SEED}"
fi

echo "GLOBAL_IDX=$GLOBAL_IDX  SEED=$SEED  ENV=$ENV  EXP=$EXP_NAME"

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
        $EXTRA_ARGS \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group antsoccer_scale2_mega_emp_online_combined_sweep \
        --log_wandb
