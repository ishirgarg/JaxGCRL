#!/bin/bash
#SBATCH --job-name=crl_rlpd_he
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-19

# Plain online CRL (contrastive critic, NO hierarchical skill controller),
# with RLPD offline-data mixing, on the same four OGBench env variants and
# five seeds as rail_slurm_skill_commitment_crl_dads_dds_seeds_ent05_low.sh /
# rail_slurm_skill_commitment_crl_emp_seeds_ent05_lowest.sh. This is
# `run.py crl` directly (agent_type crl_skill is NOT used) — there is no
# skill_policy checkpoint, no --skill_commitment_k, no --num_skills, and no
# controller_target_entropy_scale.
#
# Same CRL critic hyperparameters as the reference scripts (norm energy,
# fwd_infonce, logsumexp 0.1, repr 64), same total_env_steps=150M, same seeds
# {0,1,2,3,4}.
#
# RLPD (--use_rlpd) is ALWAYS ON here — mixes 50% offline OGBench data into
# every training batch (CRL.use_rlpd in jaxgcrl/agents/crl/crl.py). This
# requires an entry in JAXGCRL_TO_OGBENCH (jaxgcrl/utils/offline_buffer.py)
# for every env used; the two "_stitch" envs below needed NEW entries added
# there (mapping to antmaze-medium-stitch-v0 / antsoccer-arena-stitch-v0,
# mirroring how other stitch variants are already mapped) since they weren't
# previously wired up for RLPD.
#
# episode_length is HALF the reference scripts' values (2000->1001,
# 1000->501; rounded up by 1) to satisfy plain CRL's own check_config:
#   num_envs * (episode_length - 1) % batch_size == 0
# The crl_skill controller doesn't have this constraint (only
# episode_length % k == 0), which is why the reference scripts could use the
# exact 2000/1000. With default num_envs=64, batch_size=256, this requires
# (episode_length - 1) % 4 == 0, so exact halves 1000/500 round up to
# 1001/501 (the same off-by-one convention as the codebase's other
# non-hierarchical CRL/go-explore scripts, e.g. episode_length=1001/801/601).
#
# ── The four envs ──────────────────────────────────────────────────────────
#   jaxgcrl env                                 OGBench env
#   ant_maze_ogbench_medium_navigate            antmaze-medium-navigate-v0
#   ant_maze_ogbench_medium_stitch              antmaze-medium-stitch-v0
#   ant_ball_4d_ogbench_arena_1g_scale2         antsoccer-arena-navigate-v0
#   ant_ball_4d_ogbench_arena_1g_scale2_stitch  antsoccer-arena-stitch-v0
# The *_stitch variants are physically identical envs (the suffix maps to the
# same maze layout in create_env); the name only marks runs paired with the
# stitch RLPD dataset, for wandb categorization.
#
# 20 tasks = 4 envs x 5 seeds. Index decoding:
#   ENV_IDX  = SLURM_ARRAY_TASK_ID % 4
#   SEED_IDX = SLURM_ARRAY_TASK_ID / 4   (seeds 0, 1, 2, 3, 4)
# so tasks 0-3 are seed 0, 4-7 seed 1, 8-11 seed 2, 12-15 seed 3, 16-19 seed 4.

set -euo pipefail

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# ── Per-env arrays (parallel, indexed by ENV_IDX) ──────────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
)
EP_LENS=(1001 1001 501 501)
SHORT=(am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st)

SEEDS=(0 1 2 3 4)

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-}
NUM_ENVS=${#ENVS[@]}
NUM_TASKS=$(( NUM_ENVS * ${#SEEDS[@]} ))
if [ -z "$IDX" ] || [ "$IDX" -ge "$NUM_TASKS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range; use --array=0-$(( NUM_TASKS - 1 ))." >&2
  exit 1
fi

ENV_IDX=$(( IDX % NUM_ENVS ))
SEED_IDX=$(( IDX / NUM_ENVS ))

ENV=${ENVS[$ENV_IDX]}
EP_LEN=${EP_LENS[$ENV_IDX]}
TAG=${SHORT[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

EXP_NAME="${TAG}__crl_rlpd_halfep__s${SEED}"

echo "IDX=$IDX  ENV_IDX=$ENV_IDX  ENV=$ENV  EP_LEN=$EP_LEN  SEED=$SEED  EXP=$EXP_NAME"

python run.py crl \
        --use_rlpd \
        --env $ENV \
        --total_env_steps 150000000 \
        --episode_length $EP_LEN \
        --energy_fn $ENERGY_FN \
        --contrastive_loss_fn $CONTRASTIVE_LOSS_FN \
        --logsumexp_penalty_coeff $LOGSUMEXP_PENALTY_COEFF \
        --repr_dim $REPR_DIM \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group crl_rlpd_halfep_seeds \
        --log_wandb
