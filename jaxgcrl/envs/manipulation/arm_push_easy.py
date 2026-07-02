import jax
from brax import base
from jax import numpy as jnp

from .arm_envs import ArmEnvs

"""
Push-Easy: Move a cube from a random location on the blue region to a random goal on the adjacent red region. The regions are very small.
- Observation space: 18-dim obs + 3-dim goal.
- Action space:      5-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""


class ArmPushEasy(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_push_easy.xml"

    @property
    def action_size(self) -> int:
        return 5  # Override default (actuator count)

    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_push_easy"
        self.episode_length = 150

        self.goal_indices = jnp.array([0, 1, 2])  # Cube position
        self.completion_goal_indices = jnp.array([0, 1, 2])  # Identical
        self.state_dim = 18
        self.goal_reach_thresh = 0.1

        self.arm_noise_scale = 0
        self.cube_noise_scale = 0.1
        self.goal_noise_scale = 0.1

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.1, 0.6, 0.03]) + jnp.array(
            [self.goal_noise_scale, self.goal_noise_scale, 0]
        ) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos
