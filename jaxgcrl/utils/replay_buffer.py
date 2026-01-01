"""
CPU-backed Replay Buffer for JAX

Stores data on CPU to avoid GPU OOM, transfers only sampled batches to GPU.
Key design decisions:
1. buffer_state.data lives on CPU
2. Insert operations happen on CPU (not JIT-compiled to avoid device issues)
3. Sampling: indices computed, batch gathered on CPU, then transferred to GPU
"""

import functools
from typing import Generic, Tuple, TypeVar

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
    """Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

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
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)
        update = to_cpu(update)
        data = to_cpu(buffer_state.data)

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=to_cpu(data),
            insert_position=position,
            sample_position=sample_position,
        )

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
    
    This avoids GPU OOM for large replay buffers while maintaining efficient 
    GPU computation for training.
    
    Key differences from standard implementation:
    1. Data is explicitly placed on CPU during init and insert
    2. Insert is NOT JIT-compiled (to avoid device placement issues)
    3. Sampling gathers data on CPU, then transfers the batch to GPU
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
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
        
        # Pre-compile the GPU-side unflatten function
        self._unflatten_fn_jit = jax.jit(self._unflatten_fn)

    def init(self, key) -> ReplayBufferState:
        """Initialize buffer state with data on CPU."""
        return ReplayBufferState(
            data=to_cpu(jnp.zeros(self._data_shape, self._data_dtype)),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state: ReplayBufferState, samples) -> ReplayBufferState:
        """Insert data into the replay buffer (CPU operation)."""
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
        
        NOT JIT-compiled to avoid device placement issues.
        All operations happen on CPU.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        # Flatten samples and move to CPU
        update = self._flatten_fn(samples)
        update = to_cpu(update)
        
        # Ensure data is on CPU
        data = to_cpu(buffer_state.data)

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=to_cpu(data),
            insert_position=position,
            sample_position=sample_position,
        )

    def sample(self, buffer_state: ReplayBufferState):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state: ReplayBufferState):
        """
        Sample from buffer: compute indices, gather batch on CPU, transfer to GPU.
        
        Strategy:
        1. Generate random indices
        2. Gather the batch on CPU  
        3. Transfer only the sampled batch to GPU
        4. Unflatten on GPU
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(
            sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False
        )

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            # Handle edge case where max_val <= min_val
            safe_max_val = jnp.maximum(max_val, min_val + 1)
            start_values = jax.random.randint(
                subkey, shape=(rows,), minval=min_val, maxval=safe_max_val
            )
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        # Gather batch on CPU first
        data_cpu = to_cpu(buffer_state.data[:, envs_idxs, :])
        batch_cpu = create_batch_vmaped(data_cpu, matrix)
        
        # Transfer only the sampled batch to GPU and unflatten
        batch_gpu = to_gpu(batch_cpu)
        transitions = self._unflatten_fn_jit(batch_gpu)
        
        return buffer_state.replace(key=key), transitions

    def size(self, buffer_state: ReplayBufferState) -> int:
        return int(buffer_state.insert_position - buffer_state.sample_position)