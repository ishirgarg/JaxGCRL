import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py

RESET = R = "r"
GOAL = G = "g"
BALL = B = "b"

SQUARE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, B, R, B, R, B, 1],
    [1, B, B, B, B, B, B, 1],
    [1, R, B, G, G, B, R, 1],
    [1, B, B, G, G, B, B, 1],
    [1, R, B, G, G, B, R, 1],
    [1, B, R, B, R, B, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

EASY_SQUARE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 0, B, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

SMALL_SQUARE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, G, 0, 0, 1],
    [1, 0, B, 0, G, 0, 0, 1],
    [1, 0, 0, 0, G, 0, 0, 1],
    [1, G, G, G, G, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, B, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]


BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, G, G, G, 1],
    [1, 1, B, G, B, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


MAZE_HEIGHT = 0.5
XY_OFFSET = 4.0


def find(structure, size_scaling, obj):
    """Return (x,y) positions for cells equal to obj with a fixed -4,-4 offset."""
    objects = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == obj:
                x = i * float(size_scaling) - XY_OFFSET
                y = j * float(size_scaling) - XY_OFFSET
                objects.append([x, y])
    return jnp.array(objects) if objects else jnp.zeros((0, 2))


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "square_maze":
        maze_layout = SQUARE_MAZE
    elif maze_layout_name == "easy_square_maze":
        maze_layout = EASY_SQUARE_MAZE
    elif maze_layout_name == "small_square_maze":
        maze_layout = SMALL_SQUARE_MAZE
    elif maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant_ball.xml")

    possible_starts = find(maze_layout, maze_size_scaling, RESET)
    possible_goals = find(maze_layout, maze_size_scaling, GOAL)
    possible_balls = find(maze_layout, maze_size_scaling, BALL)

    # Compute bounds with fixed -4,-4 world-frame offset and half-cell padding
    rows = len(maze_layout)
    cols = len(maze_layout[0])
    scaling = float(maze_size_scaling)
    half = 0.5 * scaling
    x_bounds = (-XY_OFFSET - half, (rows - 1) * scaling - XY_OFFSET + half)
    y_bounds = (-XY_OFFSET - half, (cols - 1) * scaling - XY_OFFSET + half)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(rows):
        for j in range(cols):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        i * scaling - XY_OFFSET,
                        j * scaling - XY_OFFSET,
                        MAZE_HEIGHT / 2 * scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * scaling,
                        0.5 * scaling,
                        MAZE_HEIGHT / 2 * scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals, possible_balls, x_bounds, y_bounds


class AntBallMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="spring",
        maze_layout_name="big_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals, possible_balls, x_bounds, y_bounds = make_maze(
            maze_layout_name, maze_size_scaling
        )

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.possible_balls = possible_balls
        self.x_bounds = tuple(float(v) for v in x_bounds)
        self.y_bounds = tuple(float(v) for v in y_bounds)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._object_idx = self.sys.link_names.index("object")
        self.dense_reward = dense_reward
        # Metadata
        self.maze_layout_name = maze_layout_name
        self.maze_size_scaling = float(maze_size_scaling)

        self.state_dim = 31
        self.goal_indices = jnp.array([28, 29])
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array, goal: jax.Array | None = None, start: jax.Array | None = None) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        start_xy = self._random_start(rng2) if start is None else jnp.asarray(start, dtype=q.dtype)
        q = q.at[:2].set(start_xy)

        target = self._random_target(rng) if goal is None else jnp.asarray(goal, dtype=q.dtype)
        obj = self._random_ball(rng)

        q = q.at[-4:].set(jnp.concatenate([obj, target]))

        qd = qd.at[-4:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        old_obs = self._get_obs(pipeline_state0)
        # Distance between goal and object
        old_dist = jnp.linalg.norm(old_obs[-2:] - old_obs[-4:-2])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[-2:] - obs[-4:-2])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target and object q, qd
        qpos = pipeline_state.q[:-4]
        qvel = pipeline_state.qd[:-4]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        object_position = pipeline_state.x.pos[self._object_idx][:2]

        return jnp.concatenate([qpos] + [qvel] + [object_position] + [target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]

    def _random_ball(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_balls))
        return jnp.array(self.possible_balls[idx])[0]
