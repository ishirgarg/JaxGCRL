#!/bin/bash
#SBATCH --job-name=skill_k_ddsemp_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-15

# SAC-discrete hierarchical skill controller over BOTH the DDS ("Discrete
# Diffusion Skills") and the empowerment_skill checkpoint families, sweeping the
# temporal commitment horizon k = {20, 50, 100} on two OGBench envs:
#     1. ant_maze_ogbench_medium_navigate        (antmaze, medium)
#     2. ant_ball_4d_ogbench_arena_1g_scale2     (antsoccer, scale2 arena)
# Each env is run against four frozen skill checkpoints: {dds, empowerment} x
# {15, 50}-skill. HER (uniform-future relabel) is ON for all runs
# (--use_her --p_future_her_goal 0.8). Controller target-entropy scale is FIXED
# at 0.5 (only the discrete-SAC controller is run here). SINGLE SEED (0).
#
# agent_type=sac_discrete trains an online SAC-discrete high-level controller
# that picks a discrete skill every k env steps over a FROZEN OGBench
# skill-conditioned policy. Both checkpoint families share the same interface:
# the DDS adapter selects a VQ codebook code per skill index, the empowerment
# adapter feeds a one-hot skill to policy(obs, z). NOTE for this path:
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent). The controller simply tiles
#     `episode_length` into episode_length/k macro-steps. We still pass the
#     gcp/ep values for record-keeping; only episode_length matters.
#   * check_config requires `episode_length % k == 0`. With k in {20,50,100}
#     episode_length must be divisible by 100:
#       antmaze:   gcp 800 + ep 800 -> episode_length 1600 (1600 % 100 == 0)
#       antsoccer: gcp 200 + ep 200 -> episode_length  400 ( 400 % 100 == 0)
#   * --num_skills and --skill_policy_type are passed explicitly per checkpoint.
#     Both are ASSERTED against the checkpoint's flags.json (num_skills vs the
#     ckpt config, skill_policy_type {dds,empowerment} -> agent_name
#     {dds,empowerment_skill}) rather than derived from it; the config values
#     are what gets logged to wandb, so runs can be categorized by type + skills.
#   * --controller_target_entropy_scale is fixed at 0.5 (H_bar = 0.5*log(K)).
#
# Full sweep = 8 base configs x 3 k-values x 1 entropy x 1 seed = 24 runs,
# split across two priority tiers by CFG_IDX:
#     high   : OFFSET=0  --array=0-15  -> CFG_IDX 0..15  (16 runs, this file)
#     normal : OFFSET=16 --array=0-7   -> CFG_IDX 16..23 ( 8 runs)
# This file is the HIGH tier.
#
#   Base configs (env, skill-policy ckpt, type, num_skills), indexed by BASE_IDX:
#     0  ant_maze_ogbench_medium_navigate      dds          15  35176626
#     1  ant_maze_ogbench_medium_navigate      dds          50  35176627
#     2  ant_maze_ogbench_medium_navigate      empowerment  15  34594763
#     3  ant_maze_ogbench_medium_navigate      empowerment  50  34594838
#     4  ant_ball_4d_ogbench_arena_1g_scale2   dds          15  35176629
#     5  ant_ball_4d_ogbench_arena_1g_scale2   dds          50  35176630
#     6  ant_ball_4d_ogbench_arena_1g_scale2   empowerment  15  34594769
#     7  ant_ball_4d_ogbench_arena_1g_scale2   empowerment  50  34739255
#
# Index decoding:
#   CFG_IDX  = OFFSET + SLURM_ARRAY_TASK_ID   (0..15 for this high tier)
#   K_IDX    = CFG_IDX % 3                     (0..2)
#   BASE_IDX = CFG_IDX / 3                     (0..7)
#   SEED     = 0 (fixed)

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

OFFSET=0

# Skill checkpoints live under the shared /global/scratch OGBench tree.
CKPT_PREFIX=/global/scratch/users/ishirgarg/ogbench/OGBench/Debug

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_navigate"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2"
)
CKPTS=(
  "sd000_s_35176626.0.20260625_024651"
  "sd000_s_35176627.0.20260625_024654"
  "sd000_s_34594763.0.20260527_234148"
  "sd000_s_34594838.0.20260527_234324"
  "sd000_s_35176629.0.20260625_024657"
  "sd000_s_35176630.0.20260625_024700"
  "sd000_s_34594769.0.20260527_234149"
  "sd000_s_34739255.0.20260531_064819"
)
TYPES=(dds dds empowerment empowerment dds dds empowerment empowerment)
NUM_SKILLS=(15 50 15 50 15 50 15 50)
EP_LENS=(1600 1600 1600 1600 400 400 400 400)
GCP_STEPS=(800 800 800 800 200 200 200 200)
EP_STEPS=(800 800 800 800 200 200 200 200)
SHORT=(am_medium am_medium am_medium am_medium \
       asoc_ar1g_s2 asoc_ar1g_s2 asoc_ar1g_s2 asoc_ar1g_s2)

K_VALUES=(20 50 100)
ENT=0.5
SEED=0

# ── Decode this array task ──────────────────────────────────────────────────
CFG_IDX=$((OFFSET + SLURM_ARRAY_TASK_ID))
K_IDX=$((CFG_IDX % 3))
BASE_IDX=$((CFG_IDX / 3))

ENV=${ENVS[$BASE_IDX]}
CKPT=${CKPTS[$BASE_IDX]}
STYPE=${TYPES[$BASE_IDX]}
NSKILLS=${NUM_SKILLS[$BASE_IDX]}
EP_LEN=${EP_LENS[$BASE_IDX]}
GCP=${GCP_STEPS[$BASE_IDX]}
EP=${EP_STEPS[$BASE_IDX]}
TAG=${SHORT[$BASE_IDX]}
K=${K_VALUES[$K_IDX]}

SKILL_DIR="${CKPT_PREFIX}/${CKPT}"
EXP_NAME="${TAG}__skilldisc_${STYPE}_ns${NSKILLS}_k${K}_ent${ENT}__s${SEED}"

echo "CFG_IDX=$CFG_IDX  ENV=$ENV  CKPT=$CKPT  TYPE=$STYPE  NUM_SKILLS=$NSKILLS  K=$K  ENT=$ENT  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type sac_discrete \
        --env $ENV \
        --total_env_steps 120000000 \
        --episode_length $EP_LEN \
        --num_gcp_steps $GCP \
        --num_ep_steps $EP \
        --skill_policy_run_dir $SKILL_DIR \
        --ogbench_root /global/home/users/ishirgarg/ogbench \
        --skill_policy_type $STYPE \
        --num_skills $NSKILLS \
        --skill_commitment_k $K \
        --controller_target_entropy_scale $ENT \
        --reset_on_explore_goal_reached \
        --use_her \
        --p_future_her_goal 0.8 \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group skill_commitment_dds_emp \
        --log_wandb
