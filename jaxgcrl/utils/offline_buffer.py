"""Offline trajectory buffer for RLPD (Reinforcement Learning with Prior Data).

Loads OGBench datasets and provides a JIT-compatible sampler that returns
trajectories in the same (num_envs, episode_length, ...) shape as the online
replay buffer, so they can be concatenated and fed through process_transitions.

Episodes are stored end-to-end (no padding) with unique traj_ids per episode,
matching how the online replay buffer stores data.  When a sampled window
spans multiple episodes, flatten_batch uses traj_id equality to avoid sampling
future states across episode boundaries.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import flatten_util

from jaxgcrl.agents.go_explore.types import Transition


# Mapping from JaxGCRL environment names to OGBench dataset names.
JAXGCRL_TO_OGBENCH = {
    "ant_maze_ogbench_medium_navigate": "antmaze-medium-navigate-v0",
    "ant_maze_ogbench_medium_explore": "antmaze-medium-explore-v0",
    "ant_maze_ogbench_medium_1g": "antmaze-medium-explore-v0",
    "ant_maze_ogbench_arena": "antmaze-arena-v0",
    "pointmaze_ogbench_teleport": "pointmaze-teleport-navigate-v0",
    "humanoidmaze_ogbench_giant_stitch": "humanoidmaze-giant-stitch-v0",
    "ant_ball_ogbench_arena": "antsoccer-arena-navigate-v0",
    "ant_ball_ogbench_medium": "antsoccer-arena-navigate-v0",
    "ant_ball_ogbench_small_square": "antsoccer-arena-navigate-v0",
    "ant_ball_ogbench_easy_square": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_small_square": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_easy_square": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_medium": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_arena": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_small_easy_square": "antsoccer-arena-navigate-v0",
    "ant_ball_4d_ogbench_small_easy_square_stitch": "antsoccer-arena-stitch-v0",
    "ant_ball_ogbench_small_easy_square_1g": "antsoccer-arena-stitch-v0",
    "ant_ball_ogbench_small_square_1g": "antsoccer-arena-stitch-v0",
    "ant_ball_4d_ogbench_small_easy_square_1g": "antsoccer-arena-stitch-v0",
    "ant_ball_4d_ogbench_small_square_1g": "antsoccer-arena-stitch-v0",
    "ant_ball_ogbench_arena_stitch": "antsoccer-arena-stitch-v0",
    "ant_ball_4d_medium_stitch": "antsoccer-medium-stitch-v0",
    "cube_single": "cube-single-play-v0",
}


def map_env_to_ogbench(env_name: str):
    """Return the OGBench dataset name for a JaxGCRL env, or None."""
    return JAXGCRL_TO_OGBENCH.get(env_name)


def load_ogbench_dataset(dataset_name: str) -> dict:
    """Load an OGBench dataset (train split only).

    Calls ogbench's load_dataset directly (avoids gymnasium import from
    ogbench.__init__).  Auto-downloads the dataset if not present.

    Returns a dict with keys: observations, actions, next_observations, terminals.
    """
    import sys
    import os

    # Add the ogbench repo (sibling to JaxGCRL) to sys.path so we can import it.
    ogbench_root = os.path.expanduser("~/ishir/ogbench")
    if ogbench_root not in sys.path:
        sys.path.insert(0, ogbench_root)

    from ogbench.utils import load_dataset, download_datasets, DEFAULT_DATASET_DIR

    dataset_dir = os.path.expanduser(DEFAULT_DATASET_DIR)
    download_datasets([dataset_name], dataset_dir)

    dataset_path = os.path.join(dataset_dir, f"{dataset_name}.npz")
    return load_dataset(dataset_path, compact_dataset=False)


def segment_into_episodes(dataset: dict) -> list:
    """Split a flat OGBench dataset into a list of episode dicts.

    Episode boundaries are at terminals == 1.0.  Each returned dict has the
    same keys as the input but sliced to one episode.
    """
    terminals = dataset["terminals"]
    end_indices = np.where(terminals == 1.0)[0]

    episodes = []
    start = 0
    for end in end_indices:
        ep = {k: v[start : end + 1] for k, v in dataset.items()}
        episodes.append(ep)
        start = end + 1

    # Handle trailing data without a terminal flag
    if start < len(terminals):
        ep = {k: v[start:] for k, v in dataset.items()}
        episodes.append(ep)

    return episodes


class OfflineTrajectoryBuffer:
    """Static buffer of offline trajectories with JIT-compatible sampling.

    Mirrors how TrajectoryUniformSamplingQueue works: episodes are concatenated
    end-to-end across multiple "slots" (analogous to num_envs in the online
    buffer).  Each slot is a long stream of transitions.  Sampling picks a
    random starting position per slot and takes episode_length consecutive
    steps.  traj_id ensures flatten_batch respects episode boundaries.
    """

    def __init__(
        self,
        episodes: list,
        episode_length: int,
        num_slots: int,
        obs_size: int,
        action_size: int,
        state_size: int,
        agent_type: str = "crl",
        include_phase: bool = False,
    ):
        # Goal dim is whatever remains after the env's true state_size. For the
        # default 2D-goal envs this is 2; for ant_ball_4d_ogbench it is 4.
        goal_dim = obs_size - state_size

        # Assign a unique traj_id per episode (negative to avoid online collision)
        traj_id_counter = -1_000_000

        # Convert all episodes to Transition-field numpy arrays and concatenate
        all_obs = []
        all_act = []
        all_reward = []
        all_discount = []
        all_next_obs = []
        all_traj_id = []
        all_truncation = []

        for ep in episodes:
            ep_len = len(ep["observations"])
            if ep_len == 0:
                continue

            # Pad observation: [ogbench_obs(state_size), goal_zeros(goal_dim)]
            goal_pad = np.zeros((ep_len, goal_dim), dtype=np.float32)
            obs = np.concatenate([ep["observations"], goal_pad], axis=-1)
            act = ep["actions"]
            reward = np.zeros(ep_len, dtype=np.float32)
            discount = 1.0 - ep["terminals"].astype(np.float32)
            traj_ids = np.full(ep_len, traj_id_counter, dtype=np.float32)
            truncation = np.zeros(ep_len, dtype=np.float32)
            truncation[-1] = 1.0

            all_obs.append(obs)
            all_act.append(act)
            all_reward.append(reward)
            all_discount.append(discount)
            all_traj_id.append(traj_ids)
            all_truncation.append(truncation)

            if "next_observations" in ep:
                next_obs = np.concatenate([ep["next_observations"], goal_pad], axis=-1)
                all_next_obs.append(next_obs)

            traj_id_counter -= 1

        # Concatenate all episodes into one long stream
        flat_obs = np.concatenate(all_obs, axis=0)
        flat_act = np.concatenate(all_act, axis=0)
        flat_reward = np.concatenate(all_reward, axis=0)
        flat_discount = np.concatenate(all_discount, axis=0)
        flat_traj_id = np.concatenate(all_traj_id, axis=0)
        flat_truncation = np.concatenate(all_truncation, axis=0)
        flat_next_obs = np.concatenate(all_next_obs, axis=0) if all_next_obs else None

        total_steps = len(flat_obs)
        print(f"[RLPD] Total offline transitions: {total_steps}")

        # Distribute across slots (like num_envs in the online buffer).
        # Truncate so each slot has the same length.
        steps_per_slot = total_steps // num_slots
        usable = steps_per_slot * num_slots

        def to_slots(arr):
            return arr[:usable].reshape(num_slots, steps_per_slot, *arr.shape[1:])

        slot_obs = to_slots(flat_obs)
        slot_act = to_slots(flat_act)
        slot_reward = to_slots(flat_reward)
        slot_discount = to_slots(flat_discount)
        slot_traj_id = to_slots(flat_traj_id)
        slot_truncation = to_slots(flat_truncation)
        slot_next_obs = to_slots(flat_next_obs) if flat_next_obs is not None else None

        # Build a single dummy Transition for ravel_pytree shape inference
        state_extras = {
            "truncation": jnp.float32(0.0),
            "traj_id": jnp.float32(0.0),
        }
        if include_phase:
            state_extras["phase"] = jnp.zeros((), dtype=jnp.int32)
        dummy_transition = Transition(
            observation=jnp.zeros(obs_size),
            action=jnp.zeros(action_size),
            reward=jnp.float32(0.0),
            discount=jnp.float32(0.0),
            next_observation=jnp.zeros(obs_size) if slot_next_obs is not None else None,
            extras={"state_extras": state_extras},
        )
        dummy_flat, self._unflatten_fn = flatten_util.ravel_pytree(dummy_transition)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flat)

        # Build full Transition arrays per slot, then flatten via ravel_pytree
        phase_arr = np.zeros((num_slots, steps_per_slot), dtype=np.int32) if include_phase else None

        # Construct Transition for all slots at once: (num_slots, steps_per_slot, ...)
        slot_state_extras = {
            "truncation": jnp.array(slot_truncation),
            "traj_id": jnp.array(slot_traj_id),
        }
        if include_phase:
            slot_state_extras["phase"] = jnp.array(phase_arr)

        slot_transitions = Transition(
            observation=jnp.array(slot_obs),
            action=jnp.array(slot_act),
            reward=jnp.array(slot_reward),
            discount=jnp.array(slot_discount),
            next_observation=jnp.array(slot_next_obs) if slot_next_obs is not None else None,
            extras={"state_extras": slot_state_extras},
        )

        # Flatten each timestep: (num_slots, steps_per_slot) -> (num_slots, steps_per_slot, data_size)
        flatten_single = lambda x: flatten_util.ravel_pytree(x)[0]
        self._data = jax.vmap(jax.vmap(flatten_single))(slot_transitions)
        # shape: (num_slots, steps_per_slot, data_size)

        self._num_slots = num_slots
        self._steps_per_slot = steps_per_slot
        self._episode_length = episode_length

        print(f"[RLPD] Offline buffer: {num_slots} slots x {steps_per_slot} steps, "
              f"data_size={data_size}")
        print(f"[RLPD] Sampling window: episode_length={episode_length}, "
              f"max_start={steps_per_slot - episode_length}")

    def sample(self, key, num_envs):
        """Sample num_envs random trajectory windows from the offline buffer.

        Mimics TrajectoryUniformSamplingQueue.sample(): picks random slots and
        random starting positions, returns episode_length consecutive steps.

        Returns:
            Transition with shape (num_envs, episode_length, ...).
        """
        slot_key, start_key = jax.random.split(key)

        # Pick which slots to sample from
        slot_indices = jax.random.choice(
            slot_key, self._num_slots, shape=(num_envs,), replace=True
        )

        # Pick random starting positions within each slot
        max_start = self._steps_per_slot - self._episode_length
        start_positions = jax.random.randint(
            start_key, shape=(num_envs,), minval=0, maxval=max_start
        )

        # Extract episode_length consecutive steps per sampled slot
        offsets = jnp.arange(self._episode_length)  # (episode_length,)
        # indices: (num_envs, episode_length)
        indices = start_positions[:, None] + offsets[None, :]

        # Gather: for each env, index into its slot's data
        def gather_one(slot_idx, time_indices):
            return self._data[slot_idx][time_indices]  # (episode_length, data_size)

        batch = jax.vmap(gather_one)(slot_indices, indices)
        # (num_envs, episode_length, data_size)

        return self._unflatten_fn(batch)


def load_and_prepare_offline_buffer(
    env_name: str,
    episode_length: int,
    num_slots: int,
    obs_size: int,
    action_size: int,
    state_size: int,
    agent_type: str = "crl",
    include_phase: bool = False,
) -> OfflineTrajectoryBuffer:
    """Load OGBench data and create an offline trajectory buffer.

    Args:
        env_name: JaxGCRL environment name (e.g. "ant_maze_ogbench_medium_navigate").
        episode_length: Target episode length for sampling windows.
        num_slots: Number of parallel slots (like num_envs in online buffer).
        obs_size: Full observation size (state + goal).
        action_size: Action dimension.
        state_size: State dimension (obs_size - goal_dim).
        agent_type: "crl" or "sac".
        include_phase: Whether to include phase field in extras.

    Returns:
        OfflineTrajectoryBuffer ready for sampling.

    Raises:
        ValueError: If the env_name has no OGBench mapping.
    """
    ogbench_name = map_env_to_ogbench(env_name)
    if ogbench_name is None:
        raise ValueError(
            f"No OGBench dataset mapping for environment '{env_name}'. "
            f"Known mappings: {list(JAXGCRL_TO_OGBENCH.keys())}"
        )

    print(f"[RLPD] Loading OGBench dataset: {ogbench_name}")
    dataset = load_ogbench_dataset(ogbench_name)
    print(f"[RLPD] Dataset loaded: {dataset['observations'].shape[0]} transitions, "
          f"obs_dim={dataset['observations'].shape[1]}, "
          f"act_dim={dataset['actions'].shape[1]}")

    # Verify observation dimensions match
    ogbench_obs_dim = dataset["observations"].shape[1]
    if ogbench_obs_dim != state_size:
        raise ValueError(
            f"OGBench obs dim ({ogbench_obs_dim}) != JaxGCRL state_size ({state_size}). "
            f"Observation mapping is incompatible."
        )

    episodes = segment_into_episodes(dataset)
    print(f"[RLPD] Segmented into {len(episodes)} episodes")

    return OfflineTrajectoryBuffer(
        episodes=episodes,
        episode_length=episode_length,
        num_slots=num_slots,
        obs_size=obs_size,
        action_size=action_size,
        state_size=state_size,
        agent_type=agent_type,
        include_phase=include_phase,
    )
