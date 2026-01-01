import functools
from typing import Generic, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import PRNGKey
from jax import flatten_util

# TODO: make only single type of Replay Buffer (for CRL and baselines)
Sample = TypeVar("Sample")


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
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
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
        data = buffer_state.data

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
            data=data,
            insert_position=position,
            sample_position=sample_position,
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
    
    Args:
        max_replay_size: Maximum number of transitions to store.
        dummy_data_sample: A sample transition used to determine data shape.
        sample_batch_size: Batch size for sampling.
        num_envs: Number of parallel environments.
        episode_length: Length of episodes for trajectory sampling.
        use_cpu: If True, store replay buffer data on CPU to avoid GPU OOM.
            Data is transferred to GPU only when sampled. Default False.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
        use_cpu: bool = False,
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
        self._use_cpu = use_cpu
        
        # CPU data storage (only used if use_cpu=True)
        self._cpu_data = None

    def init(self, key):
        if self._use_cpu:
            # Store data on CPU as numpy array
            self._cpu_data = np.zeros(self._data_shape, dtype=np.float32)
            # Use minimal GPU memory - just a small placeholder
            gpu_data = jnp.zeros((1, 1, 1), self._data_dtype)
        else:
            gpu_data = jnp.zeros(self._data_shape, self._data_dtype)
            
        return ReplayBufferState(
            data=gpu_data,
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
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
        if self._use_cpu:
            return self._insert_internal_cpu(buffer_state, samples)
        else:
            return self._insert_internal_gpu(buffer_state, samples)

    def _insert_internal_gpu(self, buffer_state, samples):
        """GPU-based insert (original implementation)."""
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)  # Updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data  # shape = (max_replay_size, num_envs, data_size)

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (
            (position + len(update)) % (len(data) + 1)
        )  # so whenever roll happens, position becomes len(data), else it is increased by len(update), what is the use of doing % (len(data) + 1)??
        sample_position = jnp.maximum(
            0, buffer_state.sample_position + roll
        )  # what is the use of this line? sample_position always remains 0 as roll can never be positive

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def _insert_internal_cpu(self, buffer_state, samples):
        """CPU-based insert for memory efficiency.
        
        Uses jax.experimental.io_callback to escape tracing and perform CPU operations.
        """
        # Flatten on GPU first
        update = self._flatten_fn(samples)  # shape: (unroll_len, num_envs, data_size)
        
        def cpu_insert(update_arr, insert_pos, sample_pos):
            """Pure Python/NumPy function for CPU buffer insertion."""
            update_np = np.asarray(update_arr)
            position = int(insert_pos)
            max_size = self._data_shape[0]
            update_len = update_np.shape[0]
            
            # If needed, roll the buffer to make room
            roll = min(0, max_size - position - update_len)
            if roll < 0:
                self._cpu_data = np.roll(self._cpu_data, roll, axis=0)
                position = position + roll
            
            # Update CPU buffer
            self._cpu_data[position:position + update_len] = update_np
            
            # Compute new positions
            new_position = (position + update_len) % (max_size + 1)
            new_sample_position = max(0, int(sample_pos) + roll)
            
            return np.array(new_position, dtype=np.int32), np.array(new_sample_position, dtype=np.int32)
        
        # Use experimental.io_callback since we're modifying external state (self._cpu_data)
        new_insert_pos, new_sample_pos = jax.experimental.io_callback(
            cpu_insert,
            (jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32)),
            update,
            buffer_state.insert_position,
            buffer_state.sample_position,
        )
        
        return buffer_state.replace(
            insert_position=new_insert_pos,
            sample_position=new_sample_pos,
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state):
        if self._use_cpu:
            return self._sample_internal_cpu(buffer_state)
        else:
            return self._sample_internal_gpu(buffer_state)

    def _sample_internal_gpu(self, buffer_state):
        """GPU-based sampling (original implementation)."""
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # Note: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
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

        """
        The function create_batch will be called for every envs_idxs of buffer_state.data and every row of matrix.
        Because every row of matrix has consecutive indices of self.episode_length, for every
        envs_idx of envs_idxs, we will sample a random self.episode_length length sequence from 
        buffer_state.data[:, envs_idx, :]. But I don't think the code ensures that this sequence 
        won't be across episodes?

        flatten_batch takes care of this
        """
        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    def _sample_internal_cpu(self, buffer_state):
        """CPU-based sampling for memory efficiency.
        
        Uses jax.experimental.pure_callback to escape tracing and gather from CPU buffer.
        """
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        shape = self.num_envs

        # Generate random indices on GPU (these operations are JAX-traceable)
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)
        start_values = jax.random.randint(
            sample_key, 
            shape=(shape,), 
            minval=buffer_state.sample_position, 
            maxval=buffer_state.insert_position - self.episode_length
        )
        
        def cpu_gather(start_vals, env_idxs):
            """Pure Python/NumPy function for CPU buffer gathering."""
            start_values_np = np.asarray(start_vals)
            envs_idxs_np = np.asarray(env_idxs)
            
            # Create index matrix on CPU
            row_indices = np.arange(self.episode_length)
            matrix = start_values_np[:, np.newaxis] + row_indices
            
            # Wrap indices for circular buffer
            matrix = matrix % self._data_shape[0]
            
            # Gather from CPU buffer using advanced indexing
            batch_np = self._cpu_data[matrix, envs_idxs_np[:, np.newaxis], :]
            
            return batch_np
        
        # Use pure_callback to gather from CPU (no side effects)
        batch = jax.experimental.pure_callback(
            cpu_gather,
            jax.ShapeDtypeStruct((shape, self.episode_length, self._data_shape[2]), jnp.float32),
            start_values,
            envs_idxs,
        )
        
        # Unflatten on GPU
        transitions = self._unflatten_fn(batch)
        
        return buffer_state.replace(key=key), transitions

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position
