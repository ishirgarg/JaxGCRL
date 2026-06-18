#!/bin/bash
#SBATCH --job-name=pm1g_emp_sweep_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-33

# Sweep slice 0..33 of the 51-run pointmaze-1g sweep on rail_gpu4_high.
# Companion script covers slice 34..50 (normal). Split is proportional to
# concurrent capacity (16 high : 8 normal = 2:1).
#
# Global sweep: 1 env × (4 alphas × 4 temps + 1 MEGA) × 3 seeds = 51 runs.
# Each script defines an OFFSET so GLOBAL_IDX = OFFSET + SLURM_ARRAY_TASK_ID
# decodes the same way across both scripts:
#   SEED_IDX = GLOBAL_IDX / 17           (0..2)
#   CFG_IDX  = GLOBAL_IDX % 17           (0 = MEGA, 1..16 = empowerment alpha × temp)
# For CFG_IDX >= 1:
#   E_IDX = CFG_IDX - 1                  (0..15)
#   A_IDX = E_IDX / 4                    (0..3) → alpha
#   T_IDX = E_IDX % 4                    (0..3) → temperature

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=0

ENV="pointmaze_ogbench_teleport_1g"
EMP_DIR="/global/home/users/ishirgarg/ogbench/impls/ckpts/pointmaze/sd000_s_33421629.0.20260415_150758"
SEEDS=(0 1 2)
ALPHAS=(1 3 0.3 0.1)
TEMPS=(0 0.01 0.003 0.001)
# episode_length - 1 == num_gcp_steps + num_ep_steps  (200 + 200 + 1 = 401)
EP_LEN=401
GCP_STEPS=200
EP_STEPS=200

GLOBAL_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
SEED_IDX=$((GLOBAL_IDX / 17))
CFG_IDX=$((GLOBAL_IDX % 17))

SEED=${SEEDS[$SEED_IDX]}

if [ "$CFG_IDX" -eq 0 ]; then
  PROPOSER_ARGS="--goal_proposer_name mega"
  EXP_NAME="${ENV}__mega__s${SEED}"
else
  E_IDX=$((CFG_IDX - 1))
  A_IDX=$((E_IDX / 4))
  T_IDX=$((E_IDX % 4))
  ALPHA=${ALPHAS[$A_IDX]}
  TEMP=${TEMPS[$T_IDX]}
  PROPOSER_ARGS="--goal_proposer_name empowerment \
        --empowerment_alpha $ALPHA \
        --goal_proposer_temperature $TEMP \
        --empowerment_run_dir $EMP_DIR"
  EXP_NAME="${ENV}__emp_a${ALPHA}_t${TEMP}__s${SEED}"
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
        --wandb_project_name jaxgcrl_new \
        --wandb_group pointmaze_1g_emp_alpha_temp_sweep \
        --log_wandb
