import functools
import pickle
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def sample_trajectories_from_buffer(
    replay_buffer,
    buffer_state,
    state_size: int,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample trajectories from the replay buffer and extract positions.
    
    Args:
        replay_buffer: The replay buffer instance
        buffer_state: Current buffer state (will be modified by sampling)
        state_size: Size of state dimension
        goal_indices: Indices for x, y positions (typically [0, 1])
        rng_key: Random key for sampling
        
    Returns:
        Tuple of (all_positions, final_positions, goal_positions) where:
        - all_positions: (N, 2) array of [x, y] positions from all states
        - final_positions: (M, 2) array of [x, y] positions from final states
        - goal_positions: (N, 2) array of [x, y] goal positions from all observations
    """
    # Check buffer size
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Sample from buffer - use whatever it gives us
    current_buffer_state, transitions = replay_buffer.sample(buffer_state)
    
    # transitions.observation shape: (num_envs, episode_length, obs_size)
    # transitions.extras["state_extras"]["traj_id"] shape: (num_envs, episode_length)
    # transitions.extras["state_extras"]["truncation"] shape: (num_envs, episode_length)
    
    # Flatten to (num_envs * episode_length, obs_size)
    obs_flat = jnp.reshape(transitions.observation, (-1, transitions.observation.shape[-1]))
    traj_id_flat = jnp.reshape(transitions.extras["state_extras"]["traj_id"], (-1,))
    truncation_flat = jnp.reshape(transitions.extras["state_extras"]["truncation"], (-1,))
    
    # Extract x, y positions from observations (first state_size elements contain state)
    positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, 2)
    
    # Extract goal positions from latter part of observation
    # Goals are at state_size:state_size+goal_size (observation = [state, goal])
    goal_size = len(goal_indices)
    goal_positions = obs_flat[:, -goal_size:]  # (N, goal_size)
    
    # Convert to numpy for easier processing
    positions_np = np.array(positions)
    goal_positions_np = np.array(goal_positions)
    traj_ids_np = np.array(traj_id_flat)
    truncations_np = np.array(truncation_flat)
    
    # Get all positions (for all states plot)
    all_positions = positions_np
    
    # Get final positions (where truncation is True, or last state of each trajectory)
    final_positions = []
    unique_traj_ids = np.unique(traj_ids_np)
    
    for traj_id in unique_traj_ids:
        traj_mask = traj_ids_np == traj_id
        traj_positions = positions_np[traj_mask]
        traj_truncations = truncations_np[traj_mask]
        
        # Find final state: either where truncation is True, or last state
        final_idx = np.where(traj_truncations)[0]
        if len(final_idx) > 0:
            # Use first truncation point as final state
            final_positions.append(traj_positions[final_idx[0]])
        else:
            # Use last state if no truncation found
            if len(traj_positions) > 0:
                final_positions.append(traj_positions[-1])
    
    if len(final_positions) == 0:
        final_positions = np.array([]).reshape(0, 2)
    else:
        final_positions = np.array(final_positions)
    
    # Randomly sample 512 points from all states (keep all final states)
    if len(all_positions) > 512:
        rng = np.random.RandomState(seed=42)  # Deterministic sampling
        indices = rng.choice(len(all_positions), 512, replace=False)
        all_positions = all_positions[indices]
        goal_positions_np = goal_positions_np[indices]
    
    return all_positions, final_positions, goal_positions_np