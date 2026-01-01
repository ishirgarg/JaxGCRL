"""
CPU-backed Replay Buffer for JAX

Stores data on CPU to avoid GPU OOM, transfers only sampled batches to GPU.
Key design decisions:
1. buffer_state.data lives on CPU (storage only)
2. All compute happens on GPU
3. Insert: flatten on GPU -> transfer to CPU for storage
4. Sample: transfer slice to GPU -> compute on GPU
"""

import functools
from typing import Generic, Tuple, TypeVar

import numpy as np
import flax
import jax
import jax.numpy as jnp
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import PRNGKey
from jax import flatten_util

# Get CPU and GPU devices for explicit placement
CPU_DEVICE = jax.devices("cpu")[0]

def _get_default_device():
    """Get default device (GPU if available, else CPU)."""
    devices = jax.devices()
    # Prefer GPU/TPU over CPU
    for d in devices:
        if d.platform != "cpu":
            return d
    return devices[0]

GPU_DEVICE = _get_default_device()


def to_cpu(x):
    """Move array to CPU."""
    return jax.device_put(x, CPU_DEVICE)


def to_gpu(x):
    """Move array to default device (GPU if available)."""
    return jax.device_put(x, GPU_DEVICE)


Sample = TypeVar("Sample")


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray          # Stored on CPU
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: PRNGKey


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
    """Base class for limited-size FIFO reply buffers."""

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))

        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=to_cpu(jnp.zeros(self._data_shape, self._data_dtype)),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert_internal(self, buffer_state: ReplayBufferState, samples: Sample) -> ReplayBufferState:
        raise NotImplementedError()

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f"{self.__class__}.sample() is not implemented.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )


class TrajectoryUniformSamplingQueue:
    """
    CPU-backed replay buffer that stores data on CPU and only transfers 
    sampled batches to GPU.
    
    IMPORTANT: All compute happens on GPU. Only storage is on CPU.
    - Insert: flatten on GPU -> transfer flattened data to CPU
    - Sample: transfer data slice to GPU -> gather/unflatten on GPU
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        # JIT compile flatten to run on GPU
        self._flatten_fn = jax.jit(jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])))
        
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        # JIT compile unflatten to run on GPU
        self._unflatten_fn = jax.jit(jax.vmap(jax.vmap(self._unflatten_fn)))
        
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length
        
        # Pre-compile gather function
        self._gather_fn = jax.jit(self._gather_impl)

    def _gather_impl(self, data_slice, matrix):
        """Gather trajectories from data slice. Runs on GPU."""
        def gather_single_env(env_data, indices):
            return jnp.take(env_data, indices, axis=0, mode="wrap")
        return jax.vmap(gather_single_env, in_axes=(1, 0))(data_slice, matrix)

    def init(self, key) -> ReplayBufferState:
        """Initialize buffer state with data on CPU."""
        return ReplayBufferState(
            data=to_cpu(jnp.zeros(self._data_shape, self._data_dtype)),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state: ReplayBufferState, samples) -> ReplayBufferState:
        """Insert data into the replay buffer."""
        self.check_can_insert(buffer_state, samples, 1)
        return self.insert_internal(buffer_state, samples)

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def check_can_sample(self, buffer_state, shards):
        """Checks whether sampling can be performed. Do not JIT this method."""
        pass

    def insert_internal(self, buffer_state: ReplayBufferState, samples) -> ReplayBufferState:
        """
        Insert data in the replay buffer.
        
        Flatten happens on GPU (where samples live), then transfer to CPU for storage.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        # Flatten on GPU (samples are on GPU), this is JIT compiled
        update = self._flatten_fn(samples)
        # Block until computation is done, then transfer to CPU
        update_cpu = np.array(update.block_until_ready())  # Convert to numpy on CPU
        
        # Get positions as Python ints
        position = int(buffer_state.insert_position)
        data_len = self._data_shape[0]
        update_len = update_cpu.shape[0]
        
        # Work with numpy array on CPU for efficiency
        data_np = np.array(buffer_state.data)
        
        # Calculate roll if needed
        roll = min(0, data_len - position - update_len)
        
        if roll < 0:
            data_np = np.roll(data_np, roll, axis=0)
            position = position + roll
        
        # Update using numpy (CPU)
        data_np[position:position + update_len] = update_cpu
        
        new_position = (position + update_len) % (data_len + 1)
        sample_position = max(0, int(buffer_state.sample_position) + roll)

        return buffer_state.replace(
            data=to_cpu(jnp.array(data_np)),
            insert_position=jnp.array(new_position, dtype=jnp.int32),
            sample_position=jnp.array(sample_position, dtype=jnp.int32),
        )

    def sample(self, buffer_state: ReplayBufferState):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state: ReplayBufferState):
        """
        Sample from buffer.
        
        Strategy:
        1. Generate random indices
        2. Slice data on CPU using numpy (fast memory access)
        3. Transfer slice to GPU
        4. Gather and unflatten on GPU (compute-heavy)
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        
        key, sample_key = jax.random.split(buffer_state.key)
        shape = self.num_envs

        # Generate random indices (small arrays, device doesn't matter much)
        sample_key, subkey1, subkey2 = jax.random.split(sample_key, 3)
        envs_idxs = jax.random.choice(
            subkey1, jnp.arange(self.num_envs), shape=(shape,), replace=False
        )
        envs_idxs_np = np.array(envs_idxs)
        
        sample_pos = int(buffer_state.sample_position)
        insert_pos = int(buffer_state.insert_position)
        
        # Create trajectory start indices
        min_val = sample_pos
        max_val = max(insert_pos - self.episode_length, min_val + 1)
        start_values = jax.random.randint(
            subkey2, shape=(shape,), minval=min_val, maxval=max_val
        )
        row_indices = jnp.arange(self.episode_length)
        matrix = start_values[:, jnp.newaxis] + row_indices

        # Slice data on CPU using numpy - fast memory access
        data_np = np.array(buffer_state.data)
        data_slice_np = data_np[:, envs_idxs_np, :]
        
        # Transfer slice to GPU for compute
        data_slice_gpu = to_gpu(jnp.array(data_slice_np))
        matrix_gpu = to_gpu(matrix)
        
        # Gather on GPU (JIT compiled)
        batch_gpu = self._gather_fn(data_slice_gpu, matrix_gpu)
        
        # Unflatten on GPU (JIT compiled)
        transitions = self._unflatten_fn(batch_gpu)
        
        return buffer_state.replace(key=key), transitions

    def size(self, buffer_state: ReplayBufferState) -> int:
        return int(buffer_state.insert_position - buffer_state.sample_position)