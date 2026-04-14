"""Ant Soccer environment with OGBench-compatible observation structure.

The ball uses a free joint (7 qpos: x,y,z,qw,qx,qy,qz; 6 qvel: 3 linvel + 3 angvel)
matching the OGBench antsoccer environment, instead of the 2D slide joints in AntBall.

Observation layout (matches OGBench exactly for the state portion):
  state[0:15]  = ant qpos  (root x,y,z, quat w,x,y,z, 8 joint angles)
  state[15:22] = ball qpos (x,y,z, quat w,x,y,z)
  state[22:36] = ant qvel  (3 linvel, 3 angvel, 8 joint vel)
  state[36:42] = ball qvel (3 linvel, 3 angvel)
  goal[0:2]    = target x,y (appended for GCRL goal conditioning)

Total: state_dim=42, goal_dim=2, obs_size=44
goal_indices=[15, 16] (ball x,y position in state)
"""

import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

RESET = R = "r"
GOAL = G = "g"
BALL = B = "b"

MAZE_HEIGHT = 0.5

# OGBench arena: 8x8 open space
ARENA = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# Small square: ball in a quarter of the maze (ported from ant_ball_maze.py)
SMALL_SQUARE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, G, 0, 0, 1],
    [1, 0, B, 0, G, 0, 0, 1],
    [1, 0, 0, 0, G, 0, 0, 1],
    [1, G, G, G, G, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# Easy square: ball spawn fixed near center, goals along right/bottom edges
# (ported from ant_ball_maze.py EASY_SQUARE_MAZE)
EASY_SQUARE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 0, B, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# OGBench medium: transposed so physical layout matches OGBench
# (OGBench uses pos_x=j, pos_y=i; JaxGCRL uses pos_x=i, pos_y=j)
MEDIUM = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def _find_cells(maze_layout, maze_size_scaling, obj):
    """Return (x,y) positions of cells matching obj."""
    xy_offset = float(maze_size_scaling)
    cells = []
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == obj:
                cells.append([i * maze_size_scaling - xy_offset,
                              j * maze_size_scaling - xy_offset])
    return jnp.array(cells) if cells else jnp.zeros((0, 2))


def _open_cells(maze_layout, maze_size_scaling):
    """Return (x,y) positions of all open cells (non-wall)."""
    xy_offset = float(maze_size_scaling)
    cells = []
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] != 1:
                cells.append([i * maze_size_scaling - xy_offset,
                              j * maze_size_scaling - xy_offset])
    return jnp.array(cells) if cells else jnp.zeros((0, 2))


def make_ball_maze(maze_layout_name, maze_size_scaling):
    """Build XML string with maze walls from the OGBench-compatible ball XML template."""
    if maze_layout_name == "arena":
        maze_layout = ARENA
    elif maze_layout_name == "medium":
        maze_layout = MEDIUM
    elif maze_layout_name == "small_square":
        maze_layout = SMALL_SQUARE
    elif maze_layout_name == "easy_square":
        maze_layout = EASY_SQUARE
    else:
        raise ValueError(f"Unknown OGBench ball maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "assets", "ant_ball_ogbench.xml")
    xy_offset = float(maze_size_scaling)

    # Check if the layout uses R/G/B markers for separate spawn regions
    has_markers = any(
        cell in (R, G, B)
        for row in maze_layout
        for cell in row
    )
    if has_markers:
        starts = _find_cells(maze_layout, maze_size_scaling, R)
        goals = _find_cells(maze_layout, maze_size_scaling, G)
        balls = _find_cells(maze_layout, maze_size_scaling, B)
    else:
        open_cells = _open_cells(maze_layout, maze_size_scaling)
        starts = open_cells
        goals = open_cells
        balls = open_cells

    rows = len(maze_layout)
    cols = len(maze_layout[0])
    half = 0.5 * maze_size_scaling
    x_bounds = (-xy_offset - half, (rows - 1) * maze_size_scaling - xy_offset + half)
    y_bounds = (-xy_offset - half, (cols - 1) * maze_size_scaling - xy_offset + half)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(rows):
        for j in range(cols):
            if maze_layout[i][j] == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (
                        i * maze_size_scaling - xy_offset,
                        j * maze_size_scaling - xy_offset,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f" % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    return xml_string, starts, goals, balls, x_bounds, y_bounds


class AntBallOGBench(PipelineEnv):
    """Ant Soccer with OGBench-compatible observation vector.

    The ball has a full free joint (7 qpos, 6 qvel) instead of 2D slides.
    Arena and medium maze layouts match OGBench exactly.

    q  layout: ant_root(7) + ant_hinges(8) + ball_root(7) + target_slides(2) = 24
    qd layout: ant_root(6) + ant_hinges(8) + ball_root(6) + target_slides(2) = 22
    """

    # Number of qpos/qvel elements belonging to the target (slide joints at the end)
    _TARGET_Q = 2
    _TARGET_QD = 2
    # Ball free joint dimensions
    _BALL_Q = 7   # x,y,z,qw,qx,qy,qz
    _BALL_QD = 6  # linvel(3), angvel(3)
    # Ant root free joint + 8 hinges
    _ANT_Q = 15   # root(7) + hinges(8)
    _ANT_QD = 14  # root(6) + hinges(8)

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
        maze_layout_name="arena",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        **kwargs,
    ):
        xml_string, starts, goals, balls, x_bounds, y_bounds = make_ball_maze(
            maze_layout_name, maze_size_scaling
        )

        sys = mjcf.loads(xml_string)
        self.possible_goals = goals
        self.possible_starts = starts
        self.possible_balls = balls
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
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear))
            )

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
        self.dense_reward = dense_reward
        self.maze_layout_name = maze_layout_name
        self.maze_size_scaling = float(maze_size_scaling)

        self._ball_idx = self.sys.link_names.index("ball")

        # OGBench-compatible dimensions:
        # state = [qpos(22), qvel(20)] = 42,  goal = target_xy(2)
        self.state_dim = self._ANT_Q + self._BALL_Q + self._ANT_QD + self._BALL_QD  # 42
        self.goal_indices = jnp.array([15, 16])  # ball x,y in qpos portion
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_ant, rng_ball, rng_goal = jax.random.split(rng, 6)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Ant start position
        if start is None:
            start_xy = self._random_start(rng_ant)
        else:
            start_xy = jnp.asarray(start, dtype=q.dtype)
        q = q.at[:2].set(start_xy)

        # Ball position (free joint qpos at indices _ANT_Q : _ANT_Q + _BALL_Q)
        ball_xy = self._random_ball(rng_ball)
        ball_q_start = self._ANT_Q  # 15
        q = q.at[ball_q_start].set(ball_xy[0])      # ball x
        q = q.at[ball_q_start + 1].set(ball_xy[1])   # ball y
        q = q.at[ball_q_start + 2].set(0.5)          # ball z (on ground)
        q = q.at[ball_q_start + 3].set(1.0)          # ball qw
        q = q.at[ball_q_start + 4].set(0.0)          # ball qx
        q = q.at[ball_q_start + 5].set(0.0)          # ball qy
        q = q.at[ball_q_start + 6].set(0.0)          # ball qz

        # Zero ball velocity
        ball_qd_start = self._ANT_QD  # 14
        qd = qd.at[ball_qd_start:ball_qd_start + self._BALL_QD].set(0.0)

        # Target position (last 2 elements of q)
        if goal is None:
            target_xy = self._random_goal(rng_goal)
        else:
            target_xy = jnp.asarray(goal, dtype=q.dtype)
        q = q.at[-self._TARGET_Q:].set(target_xy)
        qd = qd.at[-self._TARGET_QD:].set(0.0)

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
        return State(pipeline_state, obs, reward, done, metrics)

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

        obs = self._get_obs(pipeline_state)
        # Ball xy = state[15:17], target xy = obs[-2:]
        ball_xy = obs[15:17]
        target_xy = obs[-2:]
        dist = jnp.linalg.norm(ball_xy - target_xy)

        old_obs = self._get_obs(pipeline_state0)
        old_ball_xy = old_obs[15:17]
        old_target_xy = old_obs[-2:]
        old_dist = jnp.linalg.norm(old_ball_xy - old_target_xy)

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
        """Build observation matching OGBench layout: [qpos(22), qvel(20), target_xy(2)].

        qpos = q[:-2]  → ant_root(7) + ant_hinges(8) + ball_root(7) = 22
        qvel = qd[:-2] → ant_root(6) + ant_hinges(8) + ball_root(6) = 20
        target = pipeline_state.x.pos[-1][:2]  → 2
        """
        qpos = pipeline_state.q[:-self._TARGET_Q]   # 22
        qvel = pipeline_state.qd[:-self._TARGET_QD]  # 20
        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos, qvel, target_pos])

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return self.possible_starts[idx][0]

    def _random_goal(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return self.possible_goals[idx][0]

    def _random_ball(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_balls))
        return self.possible_balls[idx][0]
