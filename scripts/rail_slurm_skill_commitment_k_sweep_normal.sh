#!/bin/bash
#SBATCH --job-name=skill_k_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-23

# SAC-discrete hierarchical skill controller: sweep over the temporal
# commitment k = {1, 5, 10, 20} on two OGBench env families, each with two
# frozen skill-policy checkpoints (15- and 50-skill). HER (uniform-future
# relabel) is ON for all runs (--use_her --p_future_her_goal 0.8).
#
# agent_type=sac_discrete trains an online SAC-discrete high-level controller
# that picks a discrete skill every k env steps over a FROZEN OGBench
# skill-conditioned policy. NOTE for this path:
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent). The controller simply tiles
#     `episode_length` into episode_length/k macro-steps. We still pass the
#     gcp/ep values for record-keeping; only episode_length matters.
#   * check_config requires `episode_length % k == 0`. With k in {1,5,10,20}
#     episode_length must be divisible by 20, so we use gcp+ep exactly (no +1):
#       env1 (antmaze):   gcp 800 + ep 800 -> episode_length 1600  (1600 % 20 == 0)
#       env2 (antsoccer): gcp 400 + ep 400 -> episode_length  800  ( 800 % 20 == 0)
#   * --num_skills is passed explicitly per checkpoint (must match the ckpt;
#     also makes the value show up in the wandb config rather than null).
#
# Full sweep = 6 base configs x 4 k-values x 3 seeds = 72 runs, split across 3
# priority tiers of 24 (one seed each):
#     normal : OFFSET=0  -> GLOBAL_IDX 0..23  -> seed 0, all 24 configs
#     low    : OFFSET=24 -> GLOBAL_IDX 24..47 -> seed 1, all 24 configs
#     lowest : OFFSET=48 -> GLOBAL_IDX 48..71 -> seed 2, all 24 configs
#
#   Base configs (env, skill-policy ckpt, num_skills), indexed by BASE_IDX:
#     0  ant_maze_ogbench_medium_navigate          34594838 (50)
#     1  ant_maze_ogbench_medium_navigate          34594763 (15)
#     2  ant_ball_ogbench_small_easy_square_1g      34594769 (15)
#     3  ant_ball_ogbench_small_easy_square_1g      34739255 (50)
#     4  ant_ball_ogbench_small_easy_square         34594769 (15)
#     5  ant_ball_ogbench_small_easy_square         34739255 (50)
#
# Index decoding:
#   GLOBAL_IDX = OFFSET + SLURM_ARRAY_TASK_ID
#   SEED_IDX   = GLOBAL_IDX / 24      (0..2)
#   CFG_IDX    = GLOBAL_IDX % 24      (0..23)
#     BASE_IDX = CFG_IDX / 4          (0..5)
#     K_IDX    = CFG_IDX % 4          (0..3)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=0

CKPT_PREFIX=/global/scratch/users/ishirgarg/ogbench/OGBench/Debug

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_ball_ogbench_small_easy_square_1g"
  "ant_ball_ogbench_small_easy_square_1g"
  "ant_ball_ogbench_small_easy_square"
  "ant_ball_ogbench_small_easy_square"
)
CKPTS=(
  "sd000_s_34594838.0.20260527_234324"
  "sd000_s_34594763.0.20260527_234148"
  "sd000_s_34594769.0.20260527_234149"
  "sd000_s_34739255.0.20260531_064819"
  "sd000_s_34594769.0.20260527_234149"
  "sd000_s_34739255.0.20260531_064819"
)
NUM_SKILLS=(50 15 15 50 15 50)
EP_LENS=(1600 1600 800 800 800 800)
GCP_STEPS=(800 800 400 400 400 400)
EP_STEPS=(800 800 400 400 400 400)
SHORT=(am_medium am_medium asoc_sq1g asoc_sq1g asoc_sq asoc_sq)

K_VALUES=(1 5 10 20)
SEEDS=(0 1 2)

# ── Decode this array task ──────────────────────────────────────────────────
GLOBAL_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
SEED_IDX=$((GLOBAL_IDX / 24))
CFG_IDX=$((GLOBAL_IDX % 24))
BASE_IDX=$((CFG_IDX / 4))
K_IDX=$((CFG_IDX % 4))

ENV=${ENVS[$BASE_IDX]}
CKPT=${CKPTS[$BASE_IDX]}
NSKILLS=${NUM_SKILLS[$BASE_IDX]}
EP_LEN=${EP_LENS[$BASE_IDX]}
GCP=${GCP_STEPS[$BASE_IDX]}
EP=${EP_STEPS[$BASE_IDX]}
TAG=${SHORT[$BASE_IDX]}
K=${K_VALUES[$K_IDX]}
SEED=${SEEDS[$SEED_IDX]}

SKILL_DIR="${CKPT_PREFIX}/${CKPT}"
EXP_NAME="${TAG}__skilldisc_ns${NSKILLS}_k${K}__s${SEED}"

echo "GLOBAL_IDX=$GLOBAL_IDX  ENV=$ENV  CKPT=$CKPT  NUM_SKILLS=$NSKILLS  K=$K  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type sac_discrete \
        --env $ENV \
        --total_env_steps 80000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP \
        --num_ep_steps $EP \
        --skill_policy_run_dir $SKILL_DIR \
        --num_skills $NSKILLS \
        --skill_commitment_k $K \
        --use_her \
        --p_future_her_goal 0.8 \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl \
        --wandb_group skill_commitment_k_sweep \
        --log_wandb
