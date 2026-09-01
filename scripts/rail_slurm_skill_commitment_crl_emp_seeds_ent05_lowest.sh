#!/bin/bash
#SBATCH --job-name=crl_emp_e05
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_low
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-39

# Online CRL (contrastive) hierarchical skill controller over the FROZEN
# OGBench ``empowerment_skill`` checkpoint family, on four OGBench env
# variants, at two skill counts (15 and 50), at five seeds, at
# controller_target_entropy_scale=0.5. This is the empowerment analogue of
# rail_slurm_skill_commitment_crl_dads_dds_seeds_ent05_low.sh — same controller,
# same CRL hyperparameters, same k, same seeds — with the DADS/DDS skill
# families swapped out for empowerment_skill.
#
# Unlike the dads/dds sweep there is NO entropy sweep here: only ENT=0.5 is run.
# Priority is LOW (rail_gpu4_low).
#
# ENT is the scale on the auto-tuned α's target entropy,
# H_bar = ENT * log(num_skills) — the LOWER bound the dual enforces on the
# controller's skill distribution. Being a fraction of the uniform entropy, it
# is comparable across the 15- and 50-skill rows even though log(num_skills)
# differs.
#
# FIXED ACROSS EVERY RUN: k=10, total_env_steps=150M, seeds {0,1,2,3,4}, and
# the same CRL critic hyperparameters (norm energy, fwd_infonce, logsumexp 0.1,
# repr 64). The contrastive critic's InfoNCE positives are ALWAYS γ-discounted
# future states sampled at batch time (the flat-CRL flatten_batch data path) —
# there is no HER and no HER flags.
#
# ── The four envs ──────────────────────────────────────────────────────────
#   jaxgcrl env                                 OGBench env
#   ant_maze_ogbench_medium_navigate            antmaze-medium-navigate-v0
#   ant_maze_ogbench_medium_stitch              antmaze-medium-stitch-v0
#   ant_ball_4d_ogbench_arena_1g_scale2         antsoccer-arena-navigate-v0
#   ant_ball_4d_ogbench_arena_1g_scale2_stitch  antsoccer-arena-stitch-v0
# The *_stitch variants are physically identical envs (the suffix maps to the
# same maze layout in create_env); the name only marks runs paired with
# stitch-dataset checkpoints, for wandb categorization.
#
# ── crl_skill controller notes ─────────────────────────────────────────────
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent) and are NOT passed here. The
#     controller simply tiles `episode_length` into episode_length/k macro-steps.
#   * check_config requires `episode_length % k == 0`; with k=10:
#       antmaze:   episode_length 2000 -> 200 macro-steps
#       antsoccer: episode_length 1000 -> 100 macro-steps
#   * --skill_policy_type empowerment and --num_skills are passed explicitly and
#     asserted against the checkpoint's flags.json by load_frozen_skill_policy
#     (skill_policy_type "empowerment" -> agent_name "empowerment_skill"); the
#     config values are what gets logged to wandb, so runs can be categorized by
#     type + skills. The pre-check below reads the same values out of flags.json
#     and aborts in ~1s, rather than after the multi-GB OGBench env/dataset load.
#
# ── Checkpoint run dirs (all in the flat Debug/ group) ─────────────────────
#   CFG_IDX  env             skills  run dir
#      0  am_medium           15  sd000_s_34594763.0.20260527_234148
#      1  am_medium_st        15  sd000_s_36524714.0.20260807_020729
#      2  asoc_ar1g_s2        15  sd000_s_34594769.0.20260527_234149
#      3  asoc_ar1g_s2_st     15  sd000_s_35675533.0.20260718_020653
#      4  am_medium           50  sd000_s_34594838.0.20260527_234324
#      5  am_medium_st        50  sd000_s_37866313.0.20260821_030454
#      6  asoc_ar1g_s2        50  sd000_s_34739255.0.20260531_064819
#      7  asoc_ar1g_s2_st     50  sd000_s_35675541.0.20260718_020739
#
# 40 tasks = 8 configs x 5 seeds. Index decoding:
#   CFG_IDX  = SLURM_ARRAY_TASK_ID % 8
#   SEED_IDX = SLURM_ARRAY_TASK_ID / 8   (seeds 0, 1, 2, 3, 4)
# so tasks 0-7 are seed 0, 8-15 seed 1, 16-23 seed 2, 24-31 seed 3,
# 32-39 seed 4.

set -euo pipefail

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# Skill checkpoints live under the shared /global/scratch OGBench tree (BRC
# mirror of the NAS run dirs listed above). Empowerment runs sit in the flat
# Debug/ group.
CKPT_ROOT=/global/scratch/users/ishirgarg/ogbench/OGBench

# ── Per-config arrays (parallel, indexed by CFG_IDX) ───────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
)
CKPTS=(
  "sd000_s_34594763.0.20260527_234148"  # emp15 antmaze-medium-navigate-v0
  "sd000_s_36524714.0.20260807_020729"  # emp15 antmaze-medium-stitch-v0
  "sd000_s_34594769.0.20260527_234149"  # emp15 antsoccer-arena-navigate-v0
  "sd000_s_35675533.0.20260718_020653"  # emp15 antsoccer-arena-stitch-v0
  "sd000_s_34594838.0.20260527_234324"  # emp50 antmaze-medium-navigate-v0
  "sd000_s_37866313.0.20260821_030454"  # emp50 antmaze-medium-stitch-v0
  "sd000_s_34739255.0.20260531_064819"  # emp50 antsoccer-arena-navigate-v0
  "sd000_s_35675541.0.20260718_020739"  # emp50 antsoccer-arena-stitch-v0
)
# Checkpoint group folder per run — every empowerment run is in Debug/.
GROUPS_=(
  "Debug" "Debug" "Debug" "Debug"
  "Debug" "Debug" "Debug" "Debug"
)
NUM_SKILLS=(15 15 15 15 \
            50 50 50 50)
EP_LENS=(2000 2000 1000 1000 \
         2000 2000 1000 1000)
SHORT=(am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st \
       am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st)

SEEDS=(0 1 2 3 4)

VARIANT=empowerment

# Controller target-entropy scale: H_bar = ENT*log(num_skills). Only 0.5 here.
ENT=0.5

K=10

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-}
NUM_CFGS=${#ENVS[@]}
NUM_TASKS=$(( NUM_CFGS * ${#SEEDS[@]} ))
if [ -z "$IDX" ] || [ "$IDX" -ge "$NUM_TASKS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range; use --array=0-$(( NUM_TASKS - 1 ))." >&2
  exit 1
fi

CFG_IDX=$(( IDX % NUM_CFGS ))
SEED_IDX=$(( IDX / NUM_CFGS ))

ENV=${ENVS[$CFG_IDX]}
CKPT=${CKPTS[$CFG_IDX]}
GROUP=${GROUPS_[$CFG_IDX]}
NSKILLS=${NUM_SKILLS[$CFG_IDX]}
EP_LEN=${EP_LENS[$CFG_IDX]}
TAG=${SHORT[$CFG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# ── Resolve the checkpoint run dir ─────────────────────────────────────────
# Run dir = <group>/<sd...>; if the group folder holds flags.json directly
# (no sd subdir), fall back to the group folder itself.
SKILL_DIR="${CKPT_ROOT}/${GROUP}/${CKPT}"
if [ ! -f "${SKILL_DIR}/flags.json" ] && [ -f "${CKPT_ROOT}/${GROUP}/flags.json" ]; then
  SKILL_DIR="${CKPT_ROOT}/${GROUP}"
fi
if [ ! -f "${SKILL_DIR}/flags.json" ]; then
  echo "ERROR: no flags.json in $SKILL_DIR — not an OGBench run dir." >&2
  exit 1
fi

# ── Pre-check the checkpoint against this row, in seconds ──────────────────
# load_frozen_skill_policy asserts the same things, but only once the multi-GB
# dataset machinery is already up.
CKPT_INFO=$(python - "$SKILL_DIR" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1] + "/flags.json"))["agent"]
print(cfg.get("agent_name", ""), cfg.get("num_skills", ""))
PY
)
read -r CKPT_AGENT CKPT_NSKILLS <<< "$CKPT_INFO"
echo "ckpt: agent_name=$CKPT_AGENT num_skills=$CKPT_NSKILLS"

if [ "$CKPT_AGENT" != "empowerment_skill" ]; then
  echo "ERROR: $SKILL_DIR is agent_name='$CKPT_AGENT'; expected empowerment_skill." >&2
  exit 1
fi
if [ -n "$CKPT_NSKILLS" ] && [ "$CKPT_NSKILLS" != "$NSKILLS" ]; then
  echo "ERROR: $SKILL_DIR has num_skills=$CKPT_NSKILLS; this row expects $NSKILLS." >&2
  exit 1
fi

EXP_NAME="${TAG}__crlskill_${VARIANT}_ns${NSKILLS}_k${K}_ent${ENT}__s${SEED}"

echo "IDX=$IDX  CFG_IDX=$CFG_IDX  VARIANT=$VARIANT  ENV=$ENV  SKILL_DIR=$SKILL_DIR  NUM_SKILLS=$NSKILLS  K=$K  ENT=$ENT  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type crl_skill \
        --env $ENV \
        --total_env_steps 150000000 \
        --episode_length $EP_LEN \
        --skill_policy_run_dir $SKILL_DIR \
        --ogbench_root /global/home/users/ishirgarg/ogbench \
        --skill_policy_type $VARIANT \
        --num_skills $NSKILLS \
        --skill_commitment_k $K \
        --controller_target_entropy_scale $ENT \
        --energy_fn $ENERGY_FN \
        --contrastive_loss_fn $CONTRASTIVE_LOSS_FN \
        --logsumexp_penalty_coeff $LOGSUMEXP_PENALTY_COEFF \
        --repr_dim $REPR_DIM \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --log_wandb
