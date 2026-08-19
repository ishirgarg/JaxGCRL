#!/bin/bash
#SBATCH --job-name=crl_skill_opal
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-31

# CRL (contrastive) hierarchical skill controller over frozen "opal" checkpoints
# — OGBench's ``opal`` agent (impls/agents/opal.py, agent_name="opal") trained
# with config["latent_type"]="continuous": the OPAL/SUPE VAE path, whose skill
# latent z is a CONTINUOUS skill_dim vector rather than a one-hot. This is the
# continuous sibling of rail_slurm_skill_commitment_crl_dads.sh, which drives
# the discrete (Appendix F) checkpoints from the very same OGBench sweep
# (impls/scripts/run_opal_sweep.sh: one continuous + one discrete run per env).
#
# Four env variants x four target divergences:
#     BASE_IDX  jaxgcrl env                                  OGBench env
#       0  ant_maze_ogbench_medium_navigate            antmaze-medium-navigate-v0
#       1  ant_maze_ogbench_medium_stitch              antmaze-medium-stitch-v0
#       2  ant_ball_4d_ogbench_arena_1g_scale2         antsoccer-arena-navigate-v0
#       3  ant_ball_4d_ogbench_arena_1g_scale2_stitch  antsoccer-arena-stitch-v0
# The *_stitch variants are physically identical envs (the suffix maps to the
# same maze layout in create_env); the name only marks runs paired with
# stitch-dataset checkpoints, for wandb categorization.
#
# WHAT IS DIFFERENT FROM THE DADS SCRIPT (beyond the checkpoints):
#   * --continuous_skill: the high-level controller's action space is the raw
#     skill_dim Gaussian latent, so the actor is an UNSQUASHED Gaussian and the
#     contrastive critic consumes the skill vector directly (no one-hot). The
#     frozen low-level policy is OPAL's ``decoder`` submodule, called on a
#     single concatenated [obs, z].
#   * --skill_dim 10 replaces --num_skills. It is ASSERTED against the
#     checkpoint's flags.json config["skill_dim"] (a mismatch aborts before
#     training), and skill_policy_type=opal asserts agent_name="opal" AND
#     config["latent_type"]="continuous". NOTE: the local copy of
#     run_opal_sweep.sh shows --agent.skill_dim=8; if these checkpoints predate
#     the bump to 10, the run will fail loudly on the assert — set SKILL_DIM=8.
#   * --use_skill_prior_kl: the controller's max-entropy bonus is REPLACED by
#     SPiRL's skill-prior divergence (Pertsch, Lee & Lim, CoRL 2020, §3.3).
#     Max-ent RL regularizes toward a uniform reference; this swaps that
#     reference for OPAL's own frozen state-conditioned Gaussian prior p(z|s),
#     so the actor objective carries −α·D_KL(π(·|s) ‖ p(·|s)) (computed
#     analytically, both being diagonal Gaussians). Only the actor and the α
#     dual change: SPiRL also folds the divergence into the value backup, but
#     that presumes a bootstrapped critic and this one is trained purely by
#     InfoNCE over γ-discounted future-state positives.
#   * --skill_prior_target_kl sweeps δ, the target divergence that supersedes
#     the target entropy H̄ in the auto-tuned α (an UPPER bound where the
#     entropy constraint is a lower bound, so KL > δ drives α up). δ is in nats
#     SUMMED over latent dims, hence tied to skill_dim=10 here. SPiRL used δ=1
#     (maze navigation) and δ=5 (block stacking, kitchen) at |Z|=10; this sweep
#     brackets both. --controller_target_entropy_scale is therefore unused and
#     not passed.
#
# k is SWEPT over {10, 20}; seed 0, same CRL critic hyperparameters and step
# budget as the DADS sweep.
#   k=10 MATCHES the checkpoints' OPAL chunk_size (the option horizon the VAE
#     was trained with, run_opal_sweep.sh --agent.chunk_size=10), so the prior
#     p(z|s) the KL anchors to describes options of exactly the length the
#     controller executes.
#   k=20 runs each skill for twice its training horizon, matching the DADS
#     sweep's k for cross-sweep comparability. Note those DADS checkpoints are
#     the discrete half of the SAME OGBench sweep and were ALSO trained with
#     chunk_size=10, so that sweep likewise runs its BC decoder at twice its
#     training horizon; the difference is that it never touches a prior, whereas
#     here the prior is load-bearing (δ is enforced against it directly).
# train_crl_skill_controller logs a warning whenever k disagrees with the
# checkpoint's recorded chunk_size, so the k=20 arm is self-annotating.
# The contrastive critic's InfoNCE positives are ALWAYS γ-discounted future
# states sampled at batch time (the flat-CRL flatten_batch data path) — there
# is no HER and no HER flags.
#
#   * num_gcp_steps / num_ep_steps are NOT used by the controller (those are
#     go/explore-phase knobs for the flat agent) and are NOT passed here. The
#     controller simply tiles `episode_length` into episode_length/k macro-steps.
#   * check_config requires `episode_length % k == 0`; both swept k divide both
#     episode lengths:
#       antmaze:   episode_length 2000 -> 200 macro-steps (k=10), 100 (k=20)
#       antsoccer: episode_length 1000 -> 100 macro-steps (k=10),  50 (k=20)
#
# 32 tasks = 4 envs x 4 deltas x 2 k. Index decoding:
#   BASE_IDX  = SLURM_ARRAY_TASK_ID % 4        (env + its checkpoint)
#   DELTA_IDX = (SLURM_ARRAY_TASK_ID / 4) % 4  (target divergence δ)
#   K_IDX     = SLURM_ARRAY_TASK_ID / 16       (skill commitment k)
# so tasks 0-15 are the k=10 arm and 16-31 the k=20 arm. SEED = 0 (fixed).

set -euo pipefail

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/jaxgcrl
mkdir -p "$WANDB_DIR"

# Skill checkpoints live under the shared /global/scratch OGBench tree (BRC
# mirror of the NAS run dirs). Unlike the DADS script we do not hardcode the
# <wandb project>/<run_group> folder: the run names below are globally unique,
# so we resolve each one by globstar exactly the way ogbench's own
# impls/scripts/run_opal_resume.sh does, and fail loudly on 0 or >1 match.
CKPT_ROOT=/global/scratch/users/ishirgarg/ogbench

# ── Base configs (parallel arrays, indexed by BASE_IDX) ─────────────────────
ENVS=(
  "ant_maze_ogbench_medium_navigate"
  "ant_maze_ogbench_medium_stitch"
  "ant_ball_4d_ogbench_arena_1g_scale2"
  "ant_ball_4d_ogbench_arena_1g_scale2_stitch"
)
# Continuous-latent (latent_type=continuous) opal runs, one per env.
CKPTS=(
  "sd000_s_36595192.0.20260807_234023"  # antmaze-medium-navigate-v0
  "sd000_s_36595198.0.20260807_234034"  # antmaze-medium-stitch-v0
  "sd000_s_36595196.0.20260807_234028"  # antsoccer-arena-navigate-v0
  "sd000_s_36595201.0.20260807_234040"  # antsoccer-arena-stitch-v0
)
EP_LENS=(2000 2000 1000 1000)
SHORT=(am_medium am_medium_st asoc_ar1g_s2 asoc_ar1g_s2_st)

# ── Target divergence δ sweep (nats summed over the 10 latent dims) ─────────
DELTAS=(1 5 10 20)

# ── Skill commitment k sweep (env steps one skill is held for) ──────────────
# 10 = the checkpoints' OPAL chunk_size; 20 = the DADS sweep's k.
KS=(10 20)

SEED=0
STYPE=opal
SKILL_DIM=10

# ── CRL (contrastive critic) hyperparameters ────────────────────────────────
ENERGY_FN=norm
CONTRASTIVE_LOSS_FN=fwd_infonce
LOGSUMEXP_PENALTY_COEFF=0.1
REPR_DIM=64

# ── Decode this array task ──────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-}
NUM_TASKS=$(( ${#ENVS[@]} * ${#DELTAS[@]} * ${#KS[@]} ))
if [ -z "$IDX" ] || [ "$IDX" -ge "$NUM_TASKS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID='$IDX' out of range; use --array=0-$(( NUM_TASKS - 1 ))." >&2
  exit 1
fi

BASE_IDX=$(( IDX % ${#ENVS[@]} ))
DELTA_IDX=$(( (IDX / ${#ENVS[@]}) % ${#DELTAS[@]} ))
K_IDX=$(( IDX / (${#ENVS[@]} * ${#DELTAS[@]}) ))

ENV=${ENVS[$BASE_IDX]}
CKPT=${CKPTS[$BASE_IDX]}
EP_LEN=${EP_LENS[$BASE_IDX]}
TAG=${SHORT[$BASE_IDX]}
DELTA=${DELTAS[$DELTA_IDX]}
K=${KS[$K_IDX]}

# Resolve the checkpoint folder wherever it lives under $CKPT_ROOT. Fail loudly
# on 0 or >1 match rather than silently training against the wrong checkpoint.
shopt -s nullglob globstar
MATCHES=("$CKPT_ROOT"/**/"$CKPT")
shopt -u nullglob globstar
if [ ${#MATCHES[@]} -ne 1 ]; then
  echo "ERROR: expected 1 match for $CKPT_ROOT/**/$CKPT, got ${#MATCHES[@]}: ${MATCHES[*]}" >&2
  exit 1
fi
SKILL_DIR=${MATCHES[0]}
if [ ! -f "${SKILL_DIR}/flags.json" ]; then
  echo "ERROR: no flags.json in $SKILL_DIR — not an OGBench run dir." >&2
  exit 1
fi

# Fail in seconds, with the checkpoint's TRUE values, rather than after the
# OGBench env/dataset load. load_frozen_skill_policy asserts the same three
# things, but only once the multi-GB dataset machinery is already up.
CKPT_INFO=$(python - "$SKILL_DIR" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1] + "/flags.json"))["agent"]
print(cfg.get("agent_name", ""), cfg.get("latent_type", ""), cfg.get("skill_dim", ""))
PY
)
read -r CKPT_AGENT CKPT_LATENT CKPT_SKILL_DIM <<< "$CKPT_INFO"
echo "ckpt: agent_name=$CKPT_AGENT latent_type=$CKPT_LATENT skill_dim=$CKPT_SKILL_DIM"
if [ "$CKPT_AGENT" != "opal" ] || [ "$CKPT_LATENT" != "continuous" ]; then
  echo "ERROR: $SKILL_DIR is agent_name='$CKPT_AGENT' latent_type='$CKPT_LATENT'; expected opal/continuous." >&2
  exit 1
fi
if [ "$CKPT_SKILL_DIM" != "$SKILL_DIM" ]; then
  echo "ERROR: SKILL_DIM=$SKILL_DIM but $SKILL_DIR was trained with skill_dim=$CKPT_SKILL_DIM." >&2
  echo "       Set SKILL_DIM=$CKPT_SKILL_DIM above, and note that the DELTAS grid is in nats" >&2
  echo "       SUMMED over latent dims, so it is tied to |Z|=$CKPT_SKILL_DIM rather than 10." >&2
  exit 1
fi

EXP_NAME="${TAG}__crlskill_${STYPE}_sd${SKILL_DIM}_k${K}_kl${DELTA}__s${SEED}"

echo "IDX=$IDX  BASE_IDX=$BASE_IDX  ENV=$ENV  SKILL_DIR=$SKILL_DIR  TYPE=$STYPE  SKILL_DIM=$SKILL_DIM  K=$K  DELTA=$DELTA  SEED=$SEED  EXP=$EXP_NAME"

python run.py go-explore-simple \
        --agent_type crl_skill \
        --env $ENV \
        --total_env_steps 180000000 \
        --episode_length $EP_LEN \
        --skill_policy_run_dir $SKILL_DIR \
        --ogbench_root /global/home/users/ishirgarg/ogbench \
        --skill_policy_type $STYPE \
        --continuous_skill \
        --skill_dim $SKILL_DIM \
        --use_skill_prior_kl \
        --skill_prior_target_kl $DELTA \
        --skill_commitment_k $K \
        --energy_fn $ENERGY_FN \
        --contrastive_loss_fn $CONTRASTIVE_LOSS_FN \
        --logsumexp_penalty_coeff $LOGSUMEXP_PENALTY_COEFF \
        --repr_dim $REPR_DIM \
        --seed $SEED \
        --exp_name $EXP_NAME \
        --wandb_project_name jaxgcrl_new \
        --wandb_group skill_commitment_crl_opal \
        --log_wandb
