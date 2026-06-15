# Hierarchical Discrete-Skill Controller (`agent_type="sac_discrete"`)

Goal: freeze a pretrained **skill-conditioned** policy π(a | s, z) (z = discrete one-hot
skill) trained offline in OGBench (e.g. `empowerment_skill`), and learn an **online
high-level controller** that selects discrete skills z to maximize task reward. This is
classic hierarchical RL over a fixed low-level skill set (DIAYN-style "using learned
skills"), implemented inside the existing `GoExploreSimple` framework.

This file is the spec the implementation and the code reviewers check against. **Code is
ground truth** — if this doc disagrees with the code, fix whichever is wrong and note it.

## Locked design decisions (from a prior-work survey; do not silently change)

- **Controller RL algorithm: SAC-discrete** (Christodoulou 2019, arXiv:1910.07207).
  - Discrete actor: MLP(controller_obs) → logits over `num_skills` → `distrax.Categorical`.
  - Twin Q critics: MLP(controller_obs) → **vector of Q-values, one per skill** (no action
    input). `n_critics=2`, take elementwise `min` for targets (clipped double-Q).
  - Exact soft value (no sampling): `V(s) = Σ_z π(z|s)·(Q(s)[z] − α·log π(z|s))`.
  - Actor loss: `J = E_s Σ_z π(z|s)·(α·log π(z|s) − Q_min(s)[z])`.
  - Critic target: `y = R + γ_high·(1−done)·Σ_z π(z|s')·(Q_target_min(s')[z] − α·log π(z|s'))`.
  - Auto-tuned temperature α: target entropy `H̄ = target_entropy_scale · log(num_skills)`,
    default `target_entropy_scale = 0.98`. `alpha_loss = −log_alpha · E[(entropy − H̄)]`
    style (standard discrete-SAC auto-temp). Polyak target Q with τ.
- **Temporal commitment: fixed open-loop k.** The controller picks a skill, the frozen
  policy executes it for exactly `skill_commitment_k` env steps, then the controller
  re-selects. Default `k=10` (configurable; will be swept {5,10,25,50,100}).
- **SMDP transition** stored per macro-step: `(s_t, z, R, s_{t+k}, done)` where
  `R = Σ_{i=0..k-1} γ_low^i · r_{t+i}` accumulated **only up to the first termination**
  inside the window, `done = any termination in the window`, `s_{t+k}` = state at window
  end (or at termination). Default `γ_low = 1.0` (undiscounted sum within a macro-step);
  high-level discount `γ_high = self.discounting` (0.99) applied between macro-steps.
- **Controller observation = full env obs `[state, goal]`** (so the controller can pick
  goal-reaching skills). The **frozen skill policy** sees **state only**, mapped to the
  OGBench obs layout (reuse `make_empowerment_full_obs_builder` /
  `make_empowerment_obs_builder` exactly as the empowerment scorer does).
- **Low-level frozen**: no gradients ever flow into the skill policy.
- **Skills discrete**, count = `num_skills` inferred from the loaded checkpoint's config
  (`emp_agent.config['num_skills']`) unless overridden.

## New `GoExploreSimple` config fields

```
skill_policy_run_dir: Optional[str] = None   # OGBench skill-agent checkpoint dir (flags.json + params_*.pkl)
skill_policy_epoch: Optional[int] = None      # None -> latest
num_skills: Optional[int] = None              # None -> infer from checkpoint
skill_commitment_k: int = 10
use_full_skill_obs: bool = True               # full state row -> skill net (else override-index template)
deterministic_skill_actions: bool = True      # frozen policy uses dist.mode() (vs sample)
controller_target_entropy_scale: float = 0.98
gamma_low: float = 1.0                         # intra-macro-step reward discount
controller_replay_size: int = 200000
```

`discounting` (γ_high), `policy_lr`, `critic_lr`, `alpha_lr`, `tau`, `batch_size`,
`n_critics` (force 2 for the controller), `h_dim`, `n_hidden` are reused from existing
fields.

## Loading the frozen skill policy (reuse, do NOT duplicate networks)

Use `jaxgcrl.agents.go_explore.empowerment.load_offline_empowerment_agent(run_dir,
jax_env, template_rng, epoch, use_full_obs=use_full_skill_obs)` to get `emp_agent`. Build a
generic adapter:

```
skill_obs_builder = make_empowerment_full_obs_builder()  # or override-index variant
def skill_action_fn(states, skill_indices, key, deterministic):
    # states: [B, state_size] (jaxgcrl layout); skill_indices: [B] int
    emp_obs = skill_obs_builder(states)
    z_onehot = jax.nn.one_hot(skill_indices, num_skills)
    dist = emp_agent.network.select('policy')(emp_obs, z_onehot)   # frozen params
    a = dist.mode() if deterministic else dist.sample(seed=key)
    return jnp.clip(a, -1, 1)
```

This is generic over any OGBench skill-conditioned agent whose policy submodule has
signature `policy(obs, skills_onehot)` (true for `empowerment_skill`). Keep the adapter
in one place so other skill agents can be supported by swapping the builder.

## Training loop (`_train_skill_controller`, branched from `train_fn`)

- Env: `VmapWrapper` → `EpisodeWrapper(episode_length, action_repeat)` → an auto-reset
  wrapper for training. **Do NOT use `GoExploreWrapper` and do NOT use goal proposers** —
  task goals come from the env reset (sample with `create_random_env_goals_proposer`).
  Resolve how goals are (re)assigned on auto-reset by reading `envs/wrappers.py`; if the
  episode wrapper does not resample goals on auto-reset, carry the goal in env state and
  resample on done. **Document the resolved behavior.**
- Collection unit = one **macro-step** = `k` env steps under a fixed per-env skill:
  1. controller actor samples `z ~ π(·|controller_obs)` per env (stochastic during
     training; deterministic/argmax at eval).
  2. `jax.lax.scan` over `k` env steps: each step `a = skill_action_fn(state, z)`, step
     env, accumulate `R += alive_mask · γ_low^i · r_i`, track `done = done or step_done`,
     and freeze `next_state` at the first done via masking (alive_mask goes 0 after done).
  3. emit one macro-transition per env → flat replay buffer.
- Replay: a flat (s, z, R, s', done) buffer. The existing `TrajectoryUniformSamplingQueue`
  is trajectory/episode-structured and assumes per-step trajectories — using it for
  i.i.d. SMDP transitions is allowed only if you store macro-transitions per env-column
  and sample without the future-relabel path. Prefer a small dedicated flat uniform
  buffer for clarity; justify whichever you choose.
- Updates per gradient step (standard discrete-SAC order): update α, update twin-Q toward
  the soft target, update discrete actor, Polyak-update target Q.
- Eval: deterministic controller (argmax skill) + deterministic frozen policy, re-select
  every k steps. Reuse `ActorEvaluator` if it fits; otherwise a dedicated eval rollout that
  reports OGBench task success. Match the env's eval task goals.

## Resolved env-goal / auto-reset behavior (read from `envs/wrappers.py`)

Implemented in `skill_controller.py`. Env stacks:

- **Train**: `TrajectoryIdWrapper → VmapWrapper → EpisodeWrapper(episode_length,
  action_repeat) → TrainAutoResetWrapper`. (No `GoExploreWrapper`, no goal proposers.)
  `TrajectoryIdWrapper` is required innermost: `TrainAutoResetWrapper.step` reads
  `info['traj_id']`, which only `TrajectoryIdWrapper` creates.
- **Eval**: `TrajectoryIdWrapper → VmapWrapper → EpisodeWrapper →
  EvalAutoResetWrapper → EvalWrapper` (same stack the rest of the repo evals on).

How goals are assigned / re-assigned:

- The task goal is baked into the env at `reset(rng, goal=...)`: the env writes it
  into `obs[state_size:]` and into the physics (`q[-2:]`). `possible_goals` is a
  fixed discrete set; `create_random_env_goals_proposer` samples from it.
- `EpisodeWrapper`/`VmapWrapper` do **not** resample goals on auto-reset.
  `TrainAutoResetWrapper.step(state, action, rng)` resets any `done` env to the
  goal currently stored in `info['proposed_goals']` (and refreshes that env's
  `first_obs`/`first_pipeline_state` from a fresh `env.reset` with that goal).
  `info['proposed_goals']` is otherwise carried forward unchanged.
- **Resolution**: to get a *new* random task goal on every episode boundary, the
  controller refreshes `info['proposed_goals']` with freshly sampled goals
  (`create_random_env_goals_proposer`) at the **start of every macro-step**.
  Non-`done` envs are unaffected (their goal persists in the physics/obs until
  they terminate); only auto-resetting envs pick up the fresh goal. This yields
  "one fixed goal per episode, a new random goal each reset".
- **Eval** resets with `goal=None` ⇒ the env samples its own eval target
  (`_random_target`), matching how CRL/GoExplore eval via `ActorEvaluator`. The
  train (`possible_goals`) vs eval (`_random_target`) goal-distribution split is
  pre-existing repo behavior, mirrored here intentionally.
- On the termination step the auto-reset wrapper overwrites `obs` with the
  fresh-episode obs, so the *true* terminal next-obs is not observable. This is
  harmless: the SMDP critic target multiplies the bootstrap by `(1−done)`, so the
  stored `next_obs` for a `done` macro-transition is never used.

## Implemented decisions / deviations from the original spec

- **α loss sign**: implemented as `alpha · E[stop_grad(entropy − H̄)]` (with
  `alpha = exp(log_alpha)`), the discrete analogue of the continuous-SAC loss in
  `losses.py` (`−alpha·(log_prob + H̄)` with `log_prob → −entropy`). This pushes α
  **up** when entropy < H̄, which is the correct direction; the doc's earlier
  `−log_alpha·E[entropy−H̄]` had the opposite sign and was not used.
- **OLD α for actor/critic**: the controller captures `alpha = exp(log_alpha)`
  *before* the α update and uses it in both the critic target and the actor loss
  (spec-literal; the existing SAC code's stale comment claims OLD but actually
  reads the post-update α — we follow the spec, not that quirk).
- **Episode-length truncation is treated as a terminal `done`** (bootstrap
  zeroed), matching the rest of the codebase's `1−done` convention. `episode_length
  % k == 0` is asserted so macro-steps tile the episode exactly.
- **Replay**: a dedicated flat uniform buffer (`ControllerReplay` in
  `sac_discrete.py`) for i.i.d. SMDP transitions, not the trajectory-structured
  `TrajectoryUniformSamplingQueue`.
- **Eval**: a dedicated deterministic macro-rollout (argmax skill, deterministic
  frozen policy, re-select every `k` steps) rather than `ActorEvaluator`, because
  `ActorEvaluator` steps once per call and cannot carry the committed skill across
  `k` steps without mutating the scan carry structure.
- **Gradient steps per macro-step** = `train_step_multiplier` (default 1); twin Q
  is forced (`n_critics=2`) for the controller regardless of the shared `n_critics`.
- **Rendering** is disabled for the controller (`do_render=False`); a thin
  per-step `make_policy` (argmax skill → frozen action, no k-commitment) is still
  returned for API compatibility / optional rendering.

## Things the reviewer must check

- No gradient leaks into `emp_agent` params (frozen).
- SMDP reward accumulation stops at the first in-window termination; `done`/`next_obs`
  correct across the auto-reset boundary.
- `γ_high` applied between macro-steps, `γ_low` within; not double-applied.
- Target entropy uses `log(num_skills)` (natural log) and the configured scale.
- Twin-Q `min` used for both the actor loss and the critic target; α uses the OLD value
  for actor/critic within a step (match existing SAC convention in `losses.py`).
- Controller obs is `[state, goal]`; skill-policy obs is the OGBench-mapped state only.
- `num_skills` matches the checkpoint; one-hot dim is correct.
- Episode length / k divisibility assumptions are asserted in `check_config`.
