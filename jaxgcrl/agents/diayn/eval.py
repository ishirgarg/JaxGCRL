"""DIAYN evaluation utilities.

Contains:
- Policy wrapper for evaluation (replaces goal with skill)
- Multi-skill evaluation function
- Multi-skill rendering function
"""

import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.types import PolicyParams

from jaxgcrl.utils.evaluator import Evaluator


def make_diayn_eval_policy(
    base_make_policy: Callable,
    state_dim: int,
    num_skills: int,
    skill_idx: int,
) -> Callable:
    """Wraps ``make_policy`` so it works with raw environment observations.

    During evaluation the environment returns ``[state | goal]`` observations.
    This wrapper replaces the goal part with a specified skill (one-hot vector)
    so that the trained DIAYN policy (which expects ``[state | z]``) can be used
    without modification.

    Args:
        base_make_policy: Base policy factory function.
        state_dim: Dimension of the raw state (without goal/skill).
        num_skills: Total number of skills.
        skill_idx: Index of the skill to use (0 to num_skills-1).

    Returns:
        Wrapped make_policy function that accepts raw env observations.
    """
    fixed_skill = jax.nn.one_hot(skill_idx, num_skills, dtype=jnp.float32)

    def wrapped_make_policy(params: PolicyParams, deterministic: bool = False):
        base_policy = base_make_policy(params, deterministic=deterministic)

        def augmented_policy(obs, key):
            # obs: (batch, env_obs_size) where env_obs_size = state_dim + goal_dim
            # We discard the goal and append the fixed skill instead.
            state_part = obs[..., :state_dim]  # (..., state_dim)
            batch_shape = obs.shape[:-1]
            skill = jnp.broadcast_to(fixed_skill, batch_shape + (num_skills,))
            aug_obs = jnp.concatenate([state_part, skill], axis=-1)
            return base_policy(aug_obs, key)

        return augmented_policy

    return wrapped_make_policy


def run_multi_skill_evaluation(
    make_policy: Callable,
    state_dim: int,
    num_skills: int,
    eval_env_wrapped,
    deterministic_eval: bool,
    num_eval_envs: int,
    episode_length: int,
    action_repeat: int,
    eval_key: jax.Array,
    params: PolicyParams,
    training_metrics: Dict,
) -> Dict:
    """Run evaluation for each skill separately and aggregate metrics.

    Returns aggregated metrics with per-skill metrics prefixed by skill index.
    Follows the same pattern as other agents in the repo.

    Args:
        make_policy: Base policy factory function.
        state_dim: Dimension of the raw state (without skill).
        num_skills: Number of skills to evaluate.
        eval_env_wrapped: Wrapped evaluation environment.
        deterministic_eval: Whether to use deterministic policy.
        num_eval_envs: Number of parallel evaluation environments.
        episode_length: Maximum episode length.
        action_repeat: Action repeat factor.
        eval_key: JAX random key for evaluation.
        params: Policy parameters (normalizer_params, policy_params).
        training_metrics: Training metrics to include.

    Returns:
        Dictionary of aggregated metrics with per-skill and mean/std statistics.
    """
    all_metrics = {}
    
    # Collect all metric values per base key for aggregation
    skill_metric_values = {}
    per_skill_success = {}

    for skill_idx in range(num_skills):
        # Create evaluator for this specific skill
        skill_eval_make_policy = make_diayn_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=skill_idx
        )
        skill_evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(skill_eval_make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=jax.random.fold_in(eval_key, skill_idx),
        )

        # Run evaluation for this skill (only once per skill)
        skill_metrics = skill_evaluator.run_evaluation(params, training_metrics)

        # Extract success rates per skill (with new key format: skill_0_eval, skill_1_eval, etc.)
        for key, value in skill_metrics.items():
            if key.startswith("eval/episode_success"):
                success_type = key.replace("eval/episode_", "")
                per_skill_success[f"skill_{skill_idx}_eval/{success_type}"] = value
        
        # Collect values for aggregation (excluding std metrics and success which we handle separately)
        for key, value in skill_metrics.items():
            if key.startswith("eval/episode_") and not key.endswith("_std") and not key.startswith("eval/episode_success"):
                base_key = key.replace("eval/episode_", "")
                if base_key not in skill_metric_values:
                    skill_metric_values[base_key] = []
                skill_metric_values[base_key].append(value)

    # Add per-skill success rates
    all_metrics.update(per_skill_success)

    # Add aggregated metrics (mean only, no std, no per-skill breakdowns)
    for base_key, values in skill_metric_values.items():
        if values:
            all_metrics[f"eval/episode_{base_key}"] = np.mean(values)

    return all_metrics


def render_all_skills(
    make_policy: Callable,
    state_dim: int,
    num_skills: int,
    params: PolicyParams,
    unwrapped_env,
    render_dir: str,
    exp_name: str,
    step: int,
):
    """Render videos for all skills.

    Args:
        make_policy: Base policy factory function.
        state_dim: Dimension of the raw state (without skill).
        num_skills: Number of skills to render.
        params: Policy parameters (normalizer_params, policy_params).
        unwrapped_env: Unwrapped environment for rendering.
        render_dir: Directory to save render files.
        exp_name: Experiment name for file naming.
        step: Current training step (for file naming).
    """
    from jaxgcrl.utils.env import render

    for skill_idx in range(num_skills):
        skill_eval_make_policy = make_diayn_eval_policy(
            make_policy, state_dim, num_skills, skill_idx=skill_idx
        )
        render(
            skill_eval_make_policy,
            params,
            unwrapped_env,
            render_dir,
            f"{exp_name}_skill_{skill_idx}",
            step,
        )
