from typing import Generic, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import PRNGKey
from jax import flatten_util
from jax.experimental import multihost_utils

# TODO: make only single type of Replay Buffer (for CRL and baselines)
Sample = TypeVar("Sample")

# Cache device references to avoid repeated lookups
_CPU_DEVICE = None
_GPU_DEVICE = None

def _get_cpu_device():
    global _CPU_DEVICE
    if _CPU_DEVICE is None:
        _CPU_DEVICE = jax.devices('cpu')[0]
    return _CPU_DEVICE

def _get_gpu_device():
    global _GPU_DEVICE
    if _GPU_DEVICE is None:
        gpu_devices = jax.devices('gpu')
        _GPU_DEVICE = gpu_devices[0] if gpu_devices else jax.devices()[0]
    return _GPU_DEVICE


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
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
        # Pin only the large data tensor to CPU to avoid OOM on GPU
        # Keep scalars (positions, key) on GPU for compatibility with jax.lax.scan
        cpu_device = _get_cpu_device()
        gpu_device = _get_gpu_device()
        return ReplayBufferState(
            data=jax.device_put(jnp.zeros(self._data_shape, self._data_dtype), cpu_device),
            sample_position=jax.device_put(jnp.zeros((), jnp.int32), gpu_device),
            insert_position=jax.device_put(jnp.zeros((), jnp.int32), gpu_device),
            key=jax.device_put(key, gpu_device),
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

        cpu_device = _get_cpu_device()
        gpu_device = _get_gpu_device()
        
        # Flatten samples on GPU (where they likely are), then move to CPU for storage
        update = self._flatten_fn(samples)
        update = jax.device_put(update, cpu_device)
        data = buffer_state.data

        # Move position scalars to CPU for the buffer update operations
        position = jax.device_put(buffer_state.insert_position, cpu_device)
        sample_pos = jax.device_put(buffer_state.sample_position, cpu_device)
        
        # Compute indices on CPU (scalar operations, minimal overhead)
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update buffer on CPU (this is just memory copy, not compute)
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_pos = jnp.maximum(0, sample_pos + roll)

        # Move position scalars back to GPU for compatibility with jax.lax.scan
        return buffer_state.replace(
            data=data,
            insert_position=jax.device_put(position, gpu_device),
            sample_position=jax.device_put(sample_pos, gpu_device),
        )

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f"{self.__class__}.sample() is not implemented.")

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )  # pytype: disable=bad-return-type  # jax-ndarray


class TrajectoryUniformSamplingQueue:
    """
    Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
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

    def init(self, key):
        # Pin only the large data tensor to CPU to avoid OOM on GPU
        # Keep scalars (positions, key) on GPU for compatibility with jax.lax.scan
        cpu_device = _get_cpu_device()
        gpu_device = _get_gpu_device()
        return ReplayBufferState(
            data=jax.device_put(jnp.zeros(self._data_shape, self._data_dtype), cpu_device),
            sample_position=jax.device_put(jnp.zeros((), jnp.int32), gpu_device),
            insert_position=jax.device_put(jnp.zeros((), jnp.int32), gpu_device),
            key=jax.device_put(key, gpu_device),
        )

    def insert(self, buffer_state, samples):
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

    def insert_internal(self, buffer_state, samples):
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

        cpu_device = _get_cpu_device()
        gpu_device = _get_gpu_device()
        
        # Flatten samples on GPU (where they likely are), then move to CPU for storage
        update = self._flatten_fn(samples)  # shape: (unroll_len, num_envs, data_size)
        update = jax.device_put(update, cpu_device)
        data = buffer_state.data  # shape: (max_replay_size, num_envs, data_size)

        # Move position scalars to CPU for the buffer update operations
        position = jax.device_put(buffer_state.insert_position, cpu_device)
        sample_pos = jax.device_put(buffer_state.sample_position, cpu_device)
        
        # Compute indices on CPU (scalar operations, minimal overhead)
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update buffer on CPU (this is just memory copy, not compute)
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_pos = jnp.maximum(0, sample_pos + roll)

        # Move position scalars back to GPU for compatibility with jax.lax.scan
        return buffer_state.replace(
            data=data,
            insert_position=jax.device_put(position, gpu_device),
            sample_position=jax.device_put(sample_pos, gpu_device),
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        
        cpu_device = _get_cpu_device()
        gpu_device = _get_gpu_device()
        
        # Generate random keys on GPU for faster random number generation
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # Note: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Get position scalars on CPU for index computation (they're small)
        sample_position_cpu = jax.device_put(buffer_state.sample_position, cpu_device)
        insert_position_cpu = jax.device_put(buffer_state.insert_position, cpu_device)

        # Sampling envs idxs - generate on CPU since we'll use them for CPU indexing
        sample_key_cpu = jax.device_put(sample_key, cpu_device)
        envs_idxs = jax.random.choice(sample_key_cpu, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        # Generate sampling matrix on CPU (random indices for CPU data)
        sample_key_cpu, subkey = jax.random.split(sample_key_cpu)
        start_values = jax.random.randint(
            subkey, 
            shape=(shape,), 
            minval=sample_position_cpu, 
            maxval=insert_position_cpu - self.episode_length
        )
        row_indices = jnp.arange(self.episode_length)
        matrix = start_values[:, jnp.newaxis] + row_indices
        # Wrap indices to handle circular buffer
        matrix = matrix % len(buffer_state.data)

        # Select subset of environments from CPU data
        # This is a simple slice/gather - minimal computation
        data_subset = buffer_state.data[:, envs_idxs, :]  # shape: (max_replay_size, shape, data_size)
        
        # Gather data on CPU (just memory access, indices already computed)
        # Use advanced indexing which is efficient for this pattern
        def gather_trajectories(data, indices):
            """Gather trajectories using advanced indexing."""
            # data: (max_replay_size, data_size)
            # indices: (episode_length,)
            return data[indices]
        
        # Vectorize over environments
        batch = jax.vmap(gather_trajectories, in_axes=(1, 0))(data_subset, matrix)
        # batch shape: (shape, episode_length, data_size)
        
        # Transfer to GPU IMMEDIATELY after gathering, before any computation
        batch = jax.device_put(batch, gpu_device)
        
        # Unflatten on GPU - this is where the actual computation happens
        transitions = self._unflatten_fn(batch)
        
        # Key stays on GPU
        return buffer_state.replace(key=key), transitions

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position
