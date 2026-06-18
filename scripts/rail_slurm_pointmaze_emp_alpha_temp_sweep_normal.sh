#!/bin/bash
#SBATCH --job-name=pm_emp_sweep_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-16

# Sweep slice 34..50 of the 51-run pointmaze sweep on rail_gpu4_normal.
# Companion script covers slice 0..33 (high). Split is proportional to
# concurrent capacity (16 high : 8 normal = 2:1).
# See rail_slurm_pointmaze_emp_alpha_temp_sweep_high.sh for the index decoding.

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=34

ENV="pointmaze_ogbench_teleport"
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
        --wandb_group pointmaze_emp_alpha_temp_sweep \
        --log_wandb
