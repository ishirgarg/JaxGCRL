import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp
from typing import Callable, Any, Optional


class GoalProposerWrapper(Wrapper):
    """Wrapper that intercepts reset calls and uses a goal proposer to set goals.
    
    The goal proposer can be a simple function `(rng) -> goal` or a more complex
    function `(rng, goal_proposer_state) -> goal` that takes additional state.
    
    For goal proposers that need training state (buffer, critic params, etc.),
    use `update_goal_proposer_state()` to update the state, and the goal proposer
    should be a closure that captures this state.
    """
    
    def __init__(
        self, 
        env: PipelineEnv, 
        goal_proposer: Callable[[jax.Array], jnp.ndarray],
        goal_proposer_state: Optional[Any] = None,
    ):
        super().__init__(env)
        self._goal_proposer = goal_proposer
        self._goal_proposer_state = goal_proposer_state

    def reset(self, rng: jax.Array) -> State:
        """Proposes goal and resets environment.
        
        Note: reset() is called by training.wrap, so we can't change its signature.
        We get transitions_sample from goal_proposer_state (Python dict read),
        but pass it as an explicit JAX argument to the goal proposer. This ensures JAX sees
        the current value on every call, not a trace-time constant from the closure.
        """
        # Get transitions_sample from goal_proposer_state (Python dict read)
        transitions_sample = self._goal_proposer_state.get('transitions_sample') if self._goal_proposer_state else None
        
        # #region agent log
        import json
        log_data = {
            "location": "wrappers.py:37",
            "message": "GoalProposerWrapper.reset called",
            "data": {
                "transitions_sample_is_none": transitions_sample is None,
                "goal_proposer_state_is_none": self._goal_proposer_state is None,
            },
            "timestamp": int(__import__('time').time() * 1000),
            "runId": "debug",
            "hypothesisId": "F"
        }
        if transitions_sample is not None:
            try:
                obs_shape = transitions_sample.observation.shape if hasattr(transitions_sample, 'observation') else None
                log_data["data"]["obs_shape"] = str(obs_shape) if obs_shape else None
                # Check if it's a dummy (all zeros)
                if obs_shape:
                    import jax.numpy as jnp
                    obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
                    is_all_zeros = bool(jnp.all(obs_flat == 0))
                    log_data["data"]["is_dummy_transition"] = is_all_zeros
            except: pass
        try:
            with open('/home/ishirgarg/JaxGCRL/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        
        # Pass transitions_sample as explicit JAX argument to goal proposer
        # This is critical: it must be a JAX argument, not read from Python dict inside closure
        goal = self._goal_proposer(rng, transitions_sample)
        
        # Reset the environment with the proposed goal
        state = self.env.reset(rng, goal=goal)
        
        return state
    
    def update_goal_proposer_state(self, new_state: Any) -> "GoalProposerWrapper":
        """Create a new wrapper with updated goal proposer state.
        
        Since wrappers are typically immutable in JAX, this returns a new wrapper
        instance. The goal_proposer function should be a closure that captures
        the state, so updating the state and recreating the closure will work.
        
        Args:
            new_state: New goal proposer state (buffer_state, critic_params, etc.)
            
        Returns:
            New wrapper instance with updated state
        """
        return GoalProposerWrapper(
            self.env,
            self._goal_proposer,  # Closure should capture new_state
            goal_proposer_state=new_state,
        )


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array, goal: jax.Array = None) -> State:
        if goal is not None:
            state = self.env.reset(rng, goal=goal)
        else:
            state = self.env.reset(rng)
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
