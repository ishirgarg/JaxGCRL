#!/bin/bash
#SBATCH --job-name=crl_skill_ddsemp
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-13

# CRL (contrastive) hierarchical skill controller over BOTH the DDS ("Discrete
# Diffusion Skills") and the empowerment_skill checkpoint families, on four
# OGBench env variants:
#     1. ant_maze_ogbench_medium_navigate            (antmaze, medium)
#     2. ant_maze_ogbench_medium_stitch              (same maze, stitch ckpts)
#     3. ant_ball_4d_ogbench_arena_1g_scale2         (antsoccer, scale2 arena)
#     4. ant_ball_4d_ogbench_arena_1g_scale2_stitch  (same arena, stitch ckpts)
# The *_stitch variants are physically identical envs (the suffix maps to the
# same maze layout in create_env); the name only marks runs paired with
# stitch-dataset checkpoints, for wandb categorization. FIXED: k=20,
# controller_target_entropy_scale=0.25, seed 0. The contrastive critic's
# InfoNCE positives are ALWAYS γ-discounted future states sampled at batch time
# (the flat-CRL flatten_batch data path) — there is no HER and no HER flags.
#
# NOTE: antmaze-stitch has NO empowerment rows for now — only slice50-dataset
# empowerment runs exist for that env and we are not using slice50 variants —
# so the sweep is 14 runs, not 16. Replaces the seven old skill_commitment
# scripts (dds_emp_normal + the six k_sweep{,_crl}_{normal,low,lowest}).
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
#   * --num_skills and --skill_policy_type are passed explicitly per checkpoint
#     and asserted against the checkpoint's flags.json (num_skills vs the ckpt
#     config, skill_policy_type {dds,empowerment} -> agent_name
#     {dds,empowerment_skill}); the config values are what gets logged to
#     wandb, so runs can be categorized by type + skills.
#   * --controller_target_entropy_scale 0.25 (H_bar = 0.25*log(num_skills)).
#   * CRL hyperparameters (contrastive critic) are passed explicitly:
#     --energy_fn, --contrastive_loss_fn, --logsumexp_penalty_coeff, --repr_dim.
#
# Checkpoint run dirs verified against the NAS source tree
# (/nas/ucb/ishirgarg/ogbench/impls/ckpts/{dds,empowerment}/<ogbench-env>/),
# with num_skills read from each run's flags.json.
#
#   BASE_IDX  env                                  type         skills  run dir
#     0  ant_maze_ogbench_medium_navigate            dds          15  sd000_s_35757533.0.20260721_190340
#     1  ant_maze_ogbench_medium_navigate            dds          50  sd000_s_35757534.0.20260721_190340
#     2  ant_maze_ogbench_medium_navigate            empowerment  15  sd000_s_34594763.0.20260527_234148
#     3  ant_maze_ogbench_medium_navigate            empowerment  50  sd000_s_34594838.0.20260527_234324
#     4  ant_maze_ogbench_medium_stitch              dds          15  sd000_s_35757537.0.20260721_190341
#     5  ant_maze_ogbench_medium_stitch              dds          50  sd000_s_35757538.0.20260721_190340
#     6  ant_ball_4d_ogbench_arena_1g_scale2         dds          15  sd000_s_35757535.0.20260721_190341
#     7  ant_ball_4d_ogbench_arena_1g_scale2         dds          50  sd000_s_35757536.0.20260721_190341
#     8  ant_ball_4d_ogbench_arena_1g_scale2         empowerment  15  sd000_s_34594769.0.20260527_234149
#     9  ant_ball_4d_ogbench_arena_1g_scale2         empowerment  50  sd000_s_34739255.0.20260531_064819
#    10  ant_ball_4d_ogbench_arena_1g_scale2_stitch  dds          15  sd000_s_35757539.0.20260721_190340
#    11  ant_ball_4d_ogbench_arena_1g_scale2_stitch  dds          50  sd000_s_35757532.0.20260721_190340
#    12  ant_ball_4d_ogbench_arena_1g_scale2_stitch  empowerment  15  sd000_s_35675533.0.20260718_020653
#    13  ant_ball_4d_ogbench_arena_1g_scale2_stitch  empowerment  50  sd000_s_35675541.0.20260718_020739
#
# Index decoding: BASE_IDX = SLURM_ARRAY_TASK_ID (0..13). SEED = 0 (fixed).

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# Skill checkpoints live under the shared /global/scratch OGBench tree (BRC
# mirror of the NAS run dirs listed above). Empowerment runs sit in the Debug/
# group; DDS runs sit in per-config groups named dds_<ogbench-env>_K<skills>.
CKPT_ROOT=/global/scratch/users/ishirgarg/ogbench/OGBench

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
)
CKPTS=(
  "sd000_s_35757533.0.20260721_190340"
  "sd000_s_35757534.0.20260721_190340"
  "sd000_s_34594763.0.20260527_234148"
  "sd000_s_34594838.0.20260527_234324"
  "sd000_s_35757537.0.20260721_190341"
  "sd000_s_35757538.0.20260721_190340"
  "sd000_s_35757535.0.20260721_190341"
  "sd000_s_35757536.0.20260721_190341"
  "sd000_s_34594769.0.20260527_234149"
  "sd000_s_34739255.0.20260531_064819"
  "sd000_s_35757539.0.20260721_190340"
  "sd000_s_35757532.0.20260721_190340"
  "sd000_s_35675533.0.20260718_020653"
  "sd000_s_35675541.0.20260718_020739"
)
# Checkpoint group folder per run (DDS: dds_<env>_K<skills>; empowerment: Debug).
GROUPS_=(
  "dds_antmaze-medium-navigate-v0_K15"
  "dds_antmaze-medium-navigate-v0_K50"
  "Debug"
  "Debug"
  "dds_antmaze-medium-stitch-v0_K15"
  "dds_antmaze-medium-stitch-v0_K50"
  "dds_antsoccer-arena-navigate-v0_K15"
  "dds_antsoccer-arena-navigate-v0_K50"
  "Debug"
  "Debug"
  "dds_antsoccer-arena-stitch-v0_K15"
  "dds_antsoccer-arena-stitch-v0_K50"
  "Debug"
  "Debug"
)
TYPES=(dds dds empowerment empowerment \
       dds dds \
       dds dds empowerment empowerment \
       dds dds empowerment empowerment)
NUM_SKILLS=(15 50 15 50 15 50 15 50 15 50 15 50 15 50)
EP_LENS=(2000 2000 2000 2000 2000 2000 \
         1000 1000 1000 1000 1000 1000 1000 1000)
SHORT=(am_medium am_medium am_medium am_medium \
       am_medium_st am_medium_st \
       asoc_ar1g_s2 asoc_ar1g_s2 asoc_ar1g_s2 asoc_ar1g_s2 \
       asoc_ar1g_s2_st asoc_ar1g_s2_st asoc_ar1g_s2_st asoc_ar1g_s2_st)

K=20
ENT=0.25
SEED=0

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
STYPE=${TYPES[$BASE_IDX]}
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
        --wandb_group skill_commitment_crl_dds_emp \
        --log_wandb
