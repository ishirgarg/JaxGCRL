import jax
from brax.envs import PipelineEnv, State, Wrapper, Env
from jax import numpy as jnp
from typing import Callable, Any, Optional

class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


class EvalAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

class TrainAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def __init__(
        self, 
        env: PipelineEnv, 
        goal_proposer: Callable[[jax.Array, Any], jnp.ndarray],
        initial_dummy_transitions_sample: Optional[Any] = None,
    ):
        super().__init__(env)
        self._goal_proposer = goal_proposer
        self._initial_dummy_transitions_sample = initial_dummy_transitions_sample

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        done = state.done
        
        # For all envs where done is True, propose goal and reset
        # Use jax.lax.cond to only reset when any env is done (JIT-compatible)
        transitions_sample = state.info.get('transitions_sample', self._initial_dummy_transitions_sample)
        rng = state.info['rng']
        
        def reset_done_envs(state, done, rng, transitions_sample):
            # Propose goals for all envs (will be used only for done ones)
            goals = jax.vmap(self._goal_proposer)(rng, transitions_sample)
            
            # Reset all envs (but we'll only use the reset states for done ones)
            reset_state = self.env.reset(rng, goal=goals)
            
            # Update first_pipeline_state and first_obs for done envs
            def where_done_reset(x_reset, x_current):
                if x_reset.shape and x_reset.shape[0] != done.shape[0]:
                    return x_current
                if done.shape:
                    done_reshaped = jnp.reshape(done, [done.shape[0]] + [1] * (len(x_reset.shape) - 1))
                else:
                    done_reshaped = done
                return jnp.where(done_reshaped, x_reset, x_current)
            
            state.info['first_pipeline_state'] = jax.tree.map(
                where_done_reset, reset_state.pipeline_state, state.info.get('first_pipeline_state', state.pipeline_state)
            )
            state.info['first_obs'] = jax.tree.map(
                where_done_reset, reset_state.obs, state.info.get('first_obs', state.obs)
            )
            
            # Always preserve transitions_sample in state.info
            info = dict(state.info)
            info['transitions_sample'] = transitions_sample
            return state.replace(info=info)
        
        def no_reset(state, done, rng, transitions_sample):
            # Just preserve transitions_sample
            info = dict(state.info)
            info['transitions_sample'] = transitions_sample
            return state.replace(info=info)
        
        # Only reset if any env is done (JIT-compatible)
        state = jax.lax.cond(
            jnp.any(done),
            reset_done_envs,
            no_reset,
            state, done, rng, transitions_sample
        )

        def where_done(x, y):
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done_reshaped = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            else:
                done_reshaped = done
            return jnp.where(done_reshaped, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )

        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        def reset_fn(r, g):
            return self.env.reset(r, goal=g)
        if goal is not None:
            return jax.vmap(reset_fn, in_axes=(0, 0))(rng, goal)
        else:
            return jax.vmap(reset_fn, in_axes=(0, None))(rng, goal)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['sum_reward'] += jnp.sum(rewards, axis=0)
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_done'] = done
        return state.replace(done=done)
