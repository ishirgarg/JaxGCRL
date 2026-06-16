#!/bin/bash
#SBATCH --job-name=crl_skill_k_normal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-15

# CRL (contrastive) hierarchical skill controller: sweep over the temporal
# commitment k = {1, 5, 10, 20} AND the controller target-entropy scale
# {0.25, 0.5, 0.75} (replacing the 0.98 default) on two OGBench env families,
# each with two frozen skill-policy checkpoints (15- and 50-skill). HER
# (uniform-future relabel) is ON for all runs (--use_her --p_future_her_goal
# 0.8) and is REQUIRED by the contrastive critic. SINGLE SEED (0).
#
# agent_type=crl_skill trains an online CRL (contrastive) high-level controller
# that picks a discrete skill every k env steps over a FROZEN OGBench
# skill-conditioned policy. Same SMDP setup as the SAC-discrete controller; only
# the high-level learner differs (contrastive InfoNCE critic + categorical actor
# with the SAME auto-tuned-alpha entropy term). NOTE for this path:
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent). The controller simply tiles
#     `episode_length` into episode_length/k macro-steps. We still pass the
#     gcp/ep values for record-keeping; only episode_length matters.
#   * check_config requires `episode_length % k == 0`. With k in {1,5,10,20}
#     episode_length must be divisible by 20, so we use gcp+ep exactly (no +1):
#       env1 (antmaze):   gcp 800 + ep 800 -> episode_length 1600  (1600 % 20 == 0)
#       env2 (antsoccer): gcp 400 + ep 400 -> episode_length  800  ( 800 % 20 == 0)
#   * --num_skills is passed explicitly per checkpoint (must match the ckpt).
#   * --controller_target_entropy_scale sweeps the controller target entropy
#     H_bar = scale * log(num_skills); the CRL controller keeps the same
#     auto-tuned-alpha entropy term as the SAC-discrete one.
#   * CRL hyperparameters (contrastive critic) are passed explicitly:
#     --energy_fn, --contrastive_loss_fn, --logsumexp_penalty_coeff, --repr_dim.
#
# Full sweep = 6 base configs x 4 k-values x 3 entropy-scales x 1 seed = 72 runs,
# split 16 (normal) / 16 (low) / 40 (lowest):
#     normal : OFFSET=0  --array=0-15  -> CFG_IDX 0..15
#     low    : OFFSET=16 --array=0-15  -> CFG_IDX 16..31
#     lowest : OFFSET=32 --array=0-39  -> CFG_IDX 32..71
#
#   Base configs (env, skill-policy ckpt, num_skills), indexed by BASE_IDX:
#     0  ant_maze_ogbench_medium_navigate          34594838 (50)
#     1  ant_maze_ogbench_medium_navigate          34594763 (15)
#     2  ant_ball_4d_ogbench_small_easy_square_1g   34594769 (15)
#     3  ant_ball_4d_ogbench_small_easy_square_1g   34739255 (50)
#     4  ant_ball_4d_ogbench_small_easy_square      34594769 (15)
#     5  ant_ball_4d_ogbench_small_easy_square      34739255 (50)
#
# Index decoding:
#   CFG_IDX  = OFFSET + SLURM_ARRAY_TASK_ID   (0..71)
#   ENT_IDX  = CFG_IDX % 3                     (0..2)
#   K_IDX    = (CFG_IDX / 3) % 4               (0..3)
#   BASE_IDX = CFG_IDX / 12                    (0..5)
#   SEED     = 0 (fixed)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=0

CKPT_PREFIX=/global/scratch/users/ishirgarg/ogbench/OGBench/Debug

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_ball_4d_ogbench_small_easy_square_1g"
  "ant_ball_4d_ogbench_small_easy_square_1g"
  "ant_ball_4d_ogbench_small_easy_square"
  "ant_ball_4d_ogbench_small_easy_square"
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
ENT_SCALES=(0.25 0.5 0.75)
SEED=0

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
CFG_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
ENT_IDX=$((CFG_IDX % 3))
K_IDX=$(((CFG_IDX / 3) % 4))
BASE_IDX=$((CFG_IDX / 12))

ENV=${ENVS[$BASE_IDX]}
CKPT=${CKPTS[$BASE_IDX]}
NSKILLS=${NUM_SKILLS[$BASE_IDX]}
EP_LEN=${EP_LENS[$BASE_IDX]}
GCP=${GCP_STEPS[$BASE_IDX]}
EP=${EP_STEPS[$BASE_IDX]}
TAG=${SHORT[$BASE_IDX]}
K=${K_VALUES[$K_IDX]}
ENT=${ENT_SCALES[$ENT_IDX]}

SKILL_DIR="${CKPT_PREFIX}/${CKPT}"
EXP_NAME="${TAG}__crlskill_ns${NSKILLS}_k${K}_ent${ENT}__s${SEED}"

echo "CFG_IDX=$CFG_IDX  ENV=$ENV  CKPT=$CKPT  NUM_SKILLS=$NSKILLS  K=$K  ENT=$ENT  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type crl_skill \
        --env $ENV \
        --total_env_steps 120000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP \
        --num_ep_steps $EP \
        --skill_policy_run_dir $SKILL_DIR \
        --ogbench_root /global/home/users/ishirgarg/ogbench \
        --num_skills $NSKILLS \
        --skill_commitment_k $K \
        --controller_target_entropy_scale $ENT \
        --energy_fn $ENERGY_FN \
        --contrastive_loss_fn $CONTRASTIVE_LOSS_FN \
        --logsumexp_penalty_coeff $LOGSUMEXP_PENALTY_COEFF \
        --repr_dim $REPR_DIM \
        --use_her \
        --p_future_her_goal 0.8 \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl \
        --wandb_group skill_commitment_k_sweep_crl \
        --log_wandb
