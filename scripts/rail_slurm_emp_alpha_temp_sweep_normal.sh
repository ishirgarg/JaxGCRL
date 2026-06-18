#!/bin/bash
#SBATCH --job-name=emp_sweep_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-23

# Sweep slice 48..71 of the 96-run sweep on rail_gpu4_normal.
# Companion scripts cover slices 0..47 (high) and 72..95 (low).
# See rail_slurm_emp_alpha_temp_sweep_high.sh for the index decoding.

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl 
mkdir -p "$WANDB_DIR"

OFFSET=48

ENVS=(
  "ant_ball_4d_ogbench_small_square_1g_stitch"
  "ant_maze_ogbench_medium_navigate"
)
EMP_DIRS=(
  "/global/home/users/ishirgarg/ogbench/impls/ckpts/antsoccer/sd000_s_33708849.0.20260423_043239"
  "/global/home/users/ishirgarg/ogbench/impls/ckpts/antmaze/sd000_s_33215981.0.20260406_123040"
)
SEEDS=(0 1 2)
ALPHAS=(1 3 10)
TEMPS=(0 1 0.3 0.1 0.01)
# Per-env episode partition (must satisfy episode_length - 1 == gcp + ep).
EPISODE_LENGTHS=(601 1001)
NUM_GCP_STEPS=(300 500)
NUM_EP_STEPS=(300 500)

GLOBAL_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
ENV_IDX=$((GLOBAL_IDX / 48))
REM=$((GLOBAL_IDX % 48))
SEED_IDX=$((REM / 16))
CFG_IDX=$((REM % 16))

ENV=${ENVS[$ENV_IDX]}
EMP_DIR=${EMP_DIRS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
EP_LEN=${EPISODE_LENGTHS[$ENV_IDX]}
GCP_STEPS=${NUM_GCP_STEPS[$ENV_IDX]}
EP_STEPS=${NUM_EP_STEPS[$ENV_IDX]}

if [ "$CFG_IDX" -eq 0 ]; then
  PROPOSER_ARGS="--goal_proposer_name mega"
  EXP_NAME="${ENV}__mega__s${SEED}"
else
  E_IDX=$((CFG_IDX - 1))
  A_IDX=$((E_IDX / 5))
  T_IDX=$((E_IDX % 5))
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
        --no_use_rlpd \
        --seed $SEED \
        $PROPOSER_ARGS \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group emp_alpha_temp_sweep \
        --log_wandb
