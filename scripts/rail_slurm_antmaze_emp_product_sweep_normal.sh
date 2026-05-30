#!/bin/bash
#SBATCH --job-name=am_mega_emp_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-23

# Antmaze MEGA + empowerment alpha×temp sweep — normal-priority slice.
#
# RLPD is ENABLED for everything (use_rlpd defaults to True; no --no_use_rlpd).
#
# MEGA is ALSO swept over temperature here. MEGA's temperature is wired through
# create_mega_goal_proposer: it selects via softmax((-log_density) / T) when
# T > 0 (lower density → higher logit → higher sampling probability), else
# argmin-density (frontier) when T == 0.
#
# Global sweep: 1 env × (4 MEGA temps + 5 alphas × 4 temps) × 3 seeds
#             = 1 × (4 + 20) × 3 = 24 × 3 = 72 runs, split across 3 priority
#   tiers of 24 runs each (nothing on high):
#     normal : OFFSET=0  → GLOBAL_IDX 0..23  → seed 0, all 24 configs
#     low    : OFFSET=24 → GLOBAL_IDX 24..47 → seed 1, all 24 configs
#     lowest : OFFSET=48 → GLOBAL_IDX 48..71 → seed 2, all 24 configs
#
# Index decoding:
#   GLOBAL_IDX = OFFSET + SLURM_ARRAY_TASK_ID
#   SEED_IDX   = GLOBAL_IDX / 24                      (0..2)
#   CFG_IDX    = GLOBAL_IDX % 24                      (0..23)
#     CFG_IDX 0..3   → MEGA at TEMPS[CFG_IDX]
#     CFG_IDX 4..23  → empowerment:
#                        E_IDX = CFG_IDX - 4          (0..19)
#                        A_IDX = E_IDX / 4            (0..4) → alpha
#                        T_IDX = E_IDX % 4            (0..3) → temperature

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=0

ENV="ant_maze_ogbench_medium_navigate"
EMP_DIR="/global/home/users/ishirgarg/ogbench/impls/ckpts/antmaze/sd000_s_33215981.0.20260406_123040"
SEEDS=(0 1 2)
ALPHAS=(0.33 0.5 1 2 3)
TEMPS=(0 0.01 0.03 0.1)
# episode_length - 1 == num_gcp_steps + num_ep_steps  (500 + 500 + 1 = 1001)
EP_LEN=1001
GCP_STEPS=500
EP_STEPS=500

GLOBAL_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
SEED_IDX=$((GLOBAL_IDX / 24))
CFG_IDX=$((GLOBAL_IDX % 24))

SEED=${SEEDS[$SEED_IDX]}

if [ "$CFG_IDX" -lt 4 ]; then
  TEMP=${TEMPS[$CFG_IDX]}
  PROPOSER_ARGS="--goal_proposer_name mega \
        --goal_proposer_temperature $TEMP"
  EXP_NAME="${ENV}__mega_t${TEMP}__s${SEED}"
else
  E_IDX=$((CFG_IDX - 4))
  A_IDX=$((E_IDX / 4))
  T_IDX=$((E_IDX % 4))
  ALPHA=${ALPHAS[$A_IDX]}
  TEMP=${TEMPS[$T_IDX]}
  PROPOSER_ARGS="--goal_proposer_name empowerment_density_product \
        --empowerment_alpha $ALPHA \
        --goal_proposer_temperature $TEMP \
        --empowerment_run_dir $EMP_DIR"
  EXP_NAME="${ENV}__empprod_a${ALPHA}_t${TEMP}__s${SEED}"
fi

echo "GLOBAL_IDX=$GLOBAL_IDX  ENV=$ENV  SEED=$SEED  CFG_IDX=$CFG_IDX  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --env $ENV \
        --total_env_steps 80000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP_STEPS \
        --num_ep_steps $EP_STEPS \
        --n_critics 1 \
        --seed $SEED \
        $PROPOSER_ARGS \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl \
        --wandb_group antmaze_mega_emp_density_product_alpha_temp_sweep \
        --log_wandb
