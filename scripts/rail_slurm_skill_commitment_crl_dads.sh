#!/bin/bash
#SBATCH --job-name=crl_skill_dads
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-3

# CRL (contrastive) hierarchical skill controller over frozen "dads" checkpoints
# — OGBench's ``opal`` agent (impls/agents/opal.py, agent_name="opal") trained
# with config["latent_type"]="discrete" (the OPAL paper's Appendix F
# offline-DADS path: EM-clustered discrete skills + BC), NOT a separate "dads"
# agent class or OGBench's earlier impls/agents/dads.py (unused here) — on four
# OGBench env variants:
#     0. ant_maze_ogbench_medium_navigate            (antmaze, medium)
#     1. ant_maze_ogbench_medium_stitch              (same maze, stitch ckpt)
#     2. ant_ball_4d_ogbench_arena_1g_scale2         (antsoccer, scale2 arena)
#     3. ant_ball_4d_ogbench_arena_1g_scale2_stitch  (same arena, stitch ckpt)
# The *_stitch variants are physically identical envs (the suffix maps to the
# same maze layout in create_env); the name only marks runs paired with
# stitch-dataset checkpoints, for wandb categorization. FIXED: k=20,
# controller_target_entropy_scale=0.25, seed 0. The contrastive critic's
# InfoNCE positives are ALWAYS γ-discounted future states sampled at batch time
# (the flat-CRL flatten_batch data path) — there is no HER and no HER flags.
# Mirrors rail_slurm_skill_commitment_crl_dds_emp.sh, but for the DADS skill-
# policy family instead of DDS/empowerment (one checkpoint per env, not two).
#
# agent_type=crl_skill trains an online CRL (contrastive) high-level controller
# that picks a discrete skill every k env steps over a FROZEN OGBench
# skill-conditioned policy. NOTE for this path:
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent) and are NOT passed here. The
#     controller simply tiles `episode_length` into episode_length/k macro-steps.
#   * check_config requires `episode_length % k == 0`; with k=20:
#       antmaze:   episode_length 2000 (2000 % 20 == 0)
#       antsoccer: episode_length 1000 (1000 % 20 == 0)
#   * --num_skills and --skill_policy_type dads are passed explicitly per
#     checkpoint and asserted against the checkpoint's flags.json (num_skills
#     vs the ckpt config; skill_policy_type=dads -> agent_name=opal AND
#     config["latent_type"]=discrete, both asserted); the config values are
#     what gets logged to wandb, so runs can be categorized by type + skills.
#   * --controller_target_entropy_scale 0.25 (H_bar = 0.25*log(num_skills)).
#   * CRL hyperparameters (contrastive critic) are passed explicitly:
#     --energy_fn, --contrastive_loss_fn, --logsumexp_penalty_coeff, --repr_dim.
#
# Checkpoint run dirs (DADS runs sit in the flat Debug/ group, same as the
# empowerment rows of rail_slurm_skill_commitment_crl_dds_emp.sh):
#
#   BASE_IDX  env                                  skills  run dir
#     0  ant_maze_ogbench_medium_navigate            15  sd000_s_36595195.0.20260807_234025
#     1  ant_maze_ogbench_medium_stitch              15  sd000_s_36595199.0.20260807_234037
#     2  ant_ball_4d_ogbench_arena_1g_scale2         15  sd000_s_36595197.0.20260807_234031
#     3  ant_ball_4d_ogbench_arena_1g_scale2_stitch  15  sd000_s_36595180.0.20260807_234044
#
# Index decoding: BASE_IDX = SLURM_ARRAY_TASK_ID (0..3). SEED = 0 (fixed).

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# Skill checkpoints live under the shared /global/scratch OGBench tree (BRC
# mirror of the NAS run dirs). DADS runs sit in the flat Debug/ group.
CKPT_ROOT=/global/scratch/users/ishirgarg/ogbench/OGBench

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
)
CKPTS=(
  "sd000_s_36595195.0.20260807_234025"
  "sd000_s_36595199.0.20260807_234037"
  "sd000_s_36595197.0.20260807_234031"
  "sd000_s_36595180.0.20260807_234044"
)
GROUPS_=(Debug Debug Debug Debug)
NUM_SKILLS=(15 15 15 15)
EP_LENS=(2000 2000 1000 1000)
SHORT=(am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st)

K=10
ENT=0.25
SEED=0
STYPE=dads

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
BASE_IDX=$SLURM_ARRAY_TASK_ID

ENV=${ENVS[$BASE_IDX]}
CKPT=${CKPTS[$BASE_IDX]}
GROUP=${GROUPS_[$BASE_IDX]}
NSKILLS=${NUM_SKILLS[$BASE_IDX]}
EP_LEN=${EP_LENS[$BASE_IDX]}
TAG=${SHORT[$BASE_IDX]}

# Run dir = <group>/<sd...>; if the group folder holds flags.json directly
# (no sd subdir), fall back to the group folder itself.
SKILL_DIR="${CKPT_ROOT}/${GROUP}/${CKPT}"
if [ ! -f "${SKILL_DIR}/flags.json" ] && [ -f "${CKPT_ROOT}/${GROUP}/flags.json" ]; then
  SKILL_DIR="${CKPT_ROOT}/${GROUP}"
fi
EXP_NAME="${TAG}__crlskill_${STYPE}_ns${NSKILLS}_k${K}_ent${ENT}__s${SEED}"

echo "BASE_IDX=$BASE_IDX  ENV=$ENV  SKILL_DIR=$SKILL_DIR  TYPE=$STYPE  NUM_SKILLS=$NSKILLS  K=$K  ENT=$ENT  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type crl_skill \
        --env $ENV \
        --total_env_steps 180000000 \
        --episode_length $EP_LEN \
        --skill_policy_run_dir $SKILL_DIR \
        --ogbench_root /global/home/users/ishirgarg/ogbench \
        --skill_policy_type $STYPE \
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
        --wandb_group skill_commitment_crl_dads \
        --log_wandb
