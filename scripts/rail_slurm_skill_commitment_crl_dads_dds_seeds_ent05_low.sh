#!/bin/bash
#SBATCH --job-name=crl_dd_e05
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_low
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-59

# Online CRL (contrastive) hierarchical skill controller over two frozen
# skill-policy families — DADS and DDS — on four OGBench env variants, at five
# seeds, at controller_target_entropy_scale=0.5. This is the multi-seed
# consolidation of the single-config scripts
# rail_slurm_skill_commitment_crl_{dads,dds_emp}.sh; the empowerment rows of
# the dds_emp script are NOT included here.
#
# THIS IS THE 0.5 HALF of a two-script entropy sweep, on LOW priority. The
# other half — controller_target_entropy_scale=0.25 — is
# rail_slurm_skill_commitment_crl_dads_dds_seeds_ent025_normal.sh on NORMAL
# priority.
# The two scripts are IDENTICAL except for ENT, the qos, and the job name, and
# they log to the same wandb group; the ent is in every exp_name, so the two
# arms stay distinguishable there.
#
# ENT is the scale on the auto-tuned α's target entropy,
# H_bar = ENT * log(num_skills) — the LOWER bound the dual enforces on the
# controller's skill distribution, so this arm holds the controller at a more
# uniform (more exploratory) skill mixture than the 0.25 one. Being a fraction
# of the uniform entropy, it is comparable across the 15- and 50-skill rows
# even though log(num_skills) differs.
#
# FIXED ACROSS EVERY RUN: k=10, total_env_steps=180M, seeds {0,1,2,3,4}, and
# the same CRL critic hyperparameters (norm energy, fwd_infonce, logsumexp 0.1,
# repr 64). The contrastive critic's InfoNCE positives are ALWAYS γ-discounted
# future states sampled at batch time (the flat-CRL flatten_batch data path) —
# there is no HER and no HER flags.
#
# ── The three skill-policy variants ────────────────────────────────────────
#   dads   discrete, 15 skills. OGBench's ``opal`` agent trained with
#          config["latent_type"]="discrete" — the OPAL paper's Appendix F
#          offline-DADS path (EM-clustered discrete skills + BC), NOT a
#          separate "dads" agent class or OGBench's earlier impls/agents/dads.py
#          (unused here).
#   dds15  discrete diffusion skills, 15 skills.
#   dds50  discrete diffusion skills, 50 skills.
# All three are discrete, so every row uses --num_skills with
# --controller_target_entropy_scale.
#
# ── The four envs ──────────────────────────────────────────────────────────
#   ENV_IDX  jaxgcrl env                                 OGBench env
#     0  ant_maze_ogbench_medium_navigate            antmaze-medium-navigate-v0
#     1  ant_maze_ogbench_medium_stitch              antmaze-medium-stitch-v0
#     2  ant_ball_4d_ogbench_arena_1g_scale2         antsoccer-arena-navigate-v0
#     3  ant_ball_4d_ogbench_arena_1g_scale2_stitch  antsoccer-arena-stitch-v0
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
#   * k=10 also MATCHES the DADS checkpoints' chunk_size (the option horizon
#     their VAE was trained with, run_opal_sweep.sh --agent.chunk_size=10).
#     train_crl_skill_controller logs a warning whenever k disagrees with a
#     checkpoint's recorded chunk_size.
#   * --skill_policy_type and --num_skills are passed explicitly and asserted
#     against the checkpoint's flags.json by load_frozen_skill_policy
#     (skill_policy_type {dds,dads} -> agent_name {dds,opal}, plus
#     config["latent_type"]="discrete" for dads); the config values are what
#     gets logged to wandb, so runs can be categorized by type + skills. The
#     pre-check below reads the same values out of flags.json and aborts in
#     ~1s, rather than after the multi-GB OGBench env/dataset load.
#
# ── Checkpoint run dirs (verified against the NAS source tree) ─────────────
#   CFG_IDX  variant  env             skills  run dir
#      0  dads   am_medium         15  sd000_s_36595195.0.20260807_234025
#      1  dads   am_medium_st      15  sd000_s_36595199.0.20260807_234037
#      2  dads   asoc_ar1g_s2      15  sd000_s_36595197.0.20260807_234031
#      3  dads   asoc_ar1g_s2_st   15  sd000_s_36595180.0.20260807_234044
#      4  dds    am_medium         15  sd000_s_35757533.0.20260721_190340
#      5  dds    am_medium_st      15  sd000_s_35757537.0.20260721_190341
#      6  dds    asoc_ar1g_s2      15  sd000_s_35757535.0.20260721_190341
#      7  dds    asoc_ar1g_s2_st   15  sd000_s_35757539.0.20260721_190340
#      8  dds    am_medium         50  sd000_s_35757534.0.20260721_190340
#      9  dds    am_medium_st      50  sd000_s_35757538.0.20260721_190340
#     10  dds    asoc_ar1g_s2      50  sd000_s_35757536.0.20260721_190341
#     11  dds    asoc_ar1g_s2_st   50  sd000_s_35757532.0.20260721_190340
#
# 60 tasks = 12 configs x 5 seeds. Index decoding:
#   CFG_IDX  = SLURM_ARRAY_TASK_ID % 12
#   SEED_IDX = SLURM_ARRAY_TASK_ID / 12   (seeds 0, 1, 2, 3, 4)
# so tasks 0-11 are seed 0, 12-23 seed 1, 24-35 seed 2, 36-47 seed 3,
# 48-59 seed 4.

set -euo pipefail

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# Skill checkpoints live under the shared /global/scratch OGBench tree (BRC
# mirror of the NAS run dirs listed above). DADS runs sit in the flat Debug/
# group; DDS runs sit in per-config groups named dds_<ogbench-env>_K<skills>.
CKPT_ROOT=/global/scratch/users/ishirgarg/ogbench/OGBench

# ── Per-config arrays (parallel, indexed by CFG_IDX) ───────────────────────
VARIANTS=(dads dads dads dads \
          dds  dds  dds  dds \
          dds  dds  dds  dds)
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
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
  "sd000_s_36595195.0.20260807_234025"  # dads  antmaze-medium-navigate-v0
  "sd000_s_36595199.0.20260807_234037"  # dads  antmaze-medium-stitch-v0
  "sd000_s_36595197.0.20260807_234031"  # dads  antsoccer-arena-navigate-v0
  "sd000_s_36595180.0.20260807_234044"  # dads  antsoccer-arena-stitch-v0
  "sd000_s_35757533.0.20260721_190340"  # dds15 antmaze-medium-navigate-v0
  "sd000_s_35757537.0.20260721_190341"  # dds15 antmaze-medium-stitch-v0
  "sd000_s_35757535.0.20260721_190341"  # dds15 antsoccer-arena-navigate-v0
  "sd000_s_35757539.0.20260721_190340"  # dds15 antsoccer-arena-stitch-v0
  "sd000_s_35757534.0.20260721_190340"  # dds50 antmaze-medium-navigate-v0
  "sd000_s_35757538.0.20260721_190340"  # dds50 antmaze-medium-stitch-v0
  "sd000_s_35757536.0.20260721_190341"  # dds50 antsoccer-arena-navigate-v0
  "sd000_s_35757532.0.20260721_190340"  # dds50 antsoccer-arena-stitch-v0
)
# Checkpoint group folder per run.
GROUPS_=(
  "Debug" "Debug" "Debug" "Debug"
  "dds_antmaze-medium-navigate-v0_K15"
  "dds_antmaze-medium-stitch-v0_K15"
  "dds_antsoccer-arena-navigate-v0_K15"
  "dds_antsoccer-arena-stitch-v0_K15"
  "dds_antmaze-medium-navigate-v0_K50"
  "dds_antmaze-medium-stitch-v0_K50"
  "dds_antsoccer-arena-navigate-v0_K50"
  "dds_antsoccer-arena-stitch-v0_K50"
)
NUM_SKILLS=(15 15 15 15 \
            15 15 15 15 \
            50 50 50 50)
EP_LENS=(2000 2000 1000 1000 \
         2000 2000 1000 1000 \
         2000 2000 1000 1000)
SHORT=(am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st \
       am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st \
       am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st)

SEEDS=(0 1 2 3 4)

# Controller target-entropy scale: H_bar = ENT*log(num_skills). THIS SCRIPT'S
# HALF OF THE SWEEP — see the ent025_normal script for the other value.
ENT=0.5

K=10

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-}
NUM_CFGS=${#VARIANTS[@]}
NUM_TASKS=$(( NUM_CFGS * ${#SEEDS[@]} ))
if [ -z "$IDX" ] || [ "$IDX" -ge "$NUM_TASKS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range; use --array=0-$(( NUM_TASKS - 1 ))." >&2
  exit 1
fi

CFG_IDX=$(( IDX % NUM_CFGS ))
SEED_IDX=$(( IDX / NUM_CFGS ))

VARIANT=${VARIANTS[$CFG_IDX]}
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
print(cfg.get("agent_name", ""), cfg.get("latent_type", ""))
PY
)
read -r CKPT_AGENT CKPT_LATENT <<< "$CKPT_INFO"
echo "ckpt: agent_name=$CKPT_AGENT latent_type=$CKPT_LATENT"

case $VARIANT in
  dads)
    if [ "$CKPT_AGENT" != "opal" ] || [ "$CKPT_LATENT" != "discrete" ]; then
      echo "ERROR: $SKILL_DIR is agent_name='$CKPT_AGENT' latent_type='$CKPT_LATENT'; expected opal/discrete." >&2
      exit 1
    fi
    ;;
  dds)
    if [ "$CKPT_AGENT" != "dds" ]; then
      echo "ERROR: $SKILL_DIR is agent_name='$CKPT_AGENT'; expected dds." >&2
      exit 1
    fi
    ;;
esac

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
        --wandb_group skill_commitment_crl_dads_dds_seeds \
        --log_wandb
