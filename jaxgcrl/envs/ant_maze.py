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
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = "r"
GOAL = G = "g"



U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_HARD = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

# OGBench arena, laid out in OGBench's native row/col orientation.
# Symmetric across the diagonal, so the grid is identical to OGBench's.
OGBENCH_ARENA = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, G, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# OGBench medium maze in OGBench's native row/col orientation.
# Wall cells exactly mirror ogbench/locomaze/maze.py's medium maze_map;
# R cells at (1,1) and (2,1) cover OGBench's init_ij=(1,1) and a second
# start near the same corner.
OGBENCH_MEDIUM_NAVIGATE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, R, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

CROSS_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, R, 1, 1, 1, 1],
    [1, 1, 1, R, G, R, 1, 1, 1],
    [1, 1, R, 1, G, 1, R, 1, 1],
    [1, R, G, G, G, G, G, R, 1],
    [1, 1, R, 1, G, 1, R, 1, 1],
    [1, 1, 1, R, G, R, 1, 1, 1],
    [1, 1, 1, 1, R, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_HARD = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, 0, G, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, G, 0, 0, 1, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, G, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


MAZE_HEIGHT = 0.5


def _cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention):
    # OGBench places wall geoms at (j*unit - offset_x, i*unit - offset_y), so
    # the grid's column-axis becomes physical x and the row-axis becomes y.
    # Legacy JaxGCRL mazes (u_maze, big_maze, etc.) use the opposite mapping.
    if use_ogbench_convention:
        return [j * size_scaling - xy_offset, i * size_scaling - xy_offset]
    return [i * size_scaling - xy_offset, j * size_scaling - xy_offset]


def find_starts(structure, size_scaling, use_ogbench_convention=False):
    xy_offset = float(size_scaling)  # one maze cell width
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append(_cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention))

    return jnp.array(starts)


def find_goals(structure, size_scaling, use_ogbench_convention=False):
    xy_offset = float(size_scaling)  # one maze cell width
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append(_cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention))

    return jnp.array(goals)


# Maze layouts whose walls and physical constants are aligned to OGBench so
# that RLPD with antmaze-*-navigate-v0 offline datasets sees consistent
# (s, a, s') transitions. These layouts use `ant_maze_ogbench.xml`, which
# mirrors ogbench/locomaze/assets/ant.xml (timestep=0.02, gear=30) with
# Euler integration (matching ant_ball_ogbench.xml for speed) instead of
# the legacy `ant_maze.xml` (timestep=0.01, gear=150).
OGBENCH_MAZE_LAYOUTS = {
    "maze_ogbench_arena",
    "maze_ogbench_medium_navigate",
}


def is_ogbench_maze(maze_layout_name: str) -> bool:
    return maze_layout_name in OGBENCH_MAZE_LAYOUTS


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_hard":
        maze_layout = U_MAZE_HARD
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "maze_ogbench_arena":
        maze_layout = OGBENCH_ARENA
    elif maze_layout_name == "maze_ogbench_medium_navigate":
        maze_layout = OGBENCH_MEDIUM_NAVIGATE
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_hard":
        maze_layout = BIG_MAZE_HARD
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "cross_maze":
        maze_layout = CROSS_MAZE
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    use_ogbench_convention = is_ogbench_maze(maze_layout_name)
    xml_name = "ant_maze_ogbench.xml" if use_ogbench_convention else "ant_maze.xml"
    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", xml_name)
    xy_offset = float(maze_size_scaling)  # one maze cell width

    possible_starts = find_starts(maze_layout, maze_size_scaling, use_ogbench_convention)
    possible_goals = find_goals(maze_layout, maze_size_scaling, use_ogbench_convention)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                wall_x, wall_y = _cell_xy(
                    i, j, maze_size_scaling, xy_offset, use_ogbench_convention
                )
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        wall_x,
                        wall_y,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
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

    return xml_string, possible_starts, possible_goals


class AntMaze(PipelineEnv):
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
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals = make_maze(maze_layout_name, maze_size_scaling)

        # Get maze layout to calculate bounds
        if maze_layout_name == "u_maze":
            maze_layout = U_MAZE
        elif maze_layout_name == "u_maze_hard":
            maze_layout = U_MAZE_HARD
        elif maze_layout_name == "u_maze_eval":
            maze_layout = U_MAZE_EVAL
        elif maze_layout_name == "maze_ogbench_arena":
            maze_layout = OGBENCH_ARENA
        elif maze_layout_name == "maze_ogbench_medium_navigate":
            maze_layout = OGBENCH_MEDIUM_NAVIGATE
        elif maze_layout_name == "big_maze":
            maze_layout = BIG_MAZE
        elif maze_layout_name == "big_maze_hard":
            maze_layout = BIG_MAZE_HARD
        elif maze_layout_name == "big_maze_eval":
            maze_layout = BIG_MAZE_EVAL
        elif maze_layout_name == "cross_maze":
            maze_layout = CROSS_MAZE
        elif maze_layout_name == "hardest_maze":
            maze_layout = HARDEST_MAZE
        else:
            raise ValueError(f"Unknown maze layout: {maze_layout_name}")

        # Calculate x and y bounds based on maze layout dimensions.
        # Under the OGBench convention grid columns map to x and rows to y;
        # legacy JaxGCRL mazes use the opposite mapping.
        num_rows = len(maze_layout)
        num_cols = len(maze_layout[0])
        half = 0.5 * maze_size_scaling
        xy_offset = float(maze_size_scaling)  # one maze cell width
        if is_ogbench_maze(maze_layout_name):
            x_cells, y_cells = num_cols, num_rows
        else:
            x_cells, y_cells = num_rows, num_cols
        self.x_bounds = jnp.array([
            -xy_offset - half,
            (x_cells - 1) * maze_size_scaling - xy_offset + half,
        ])
        self.y_bounds = jnp.array([
            -xy_offset - half,
            (y_cells - 1) * maze_size_scaling - xy_offset + half,
        ])

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self._is_ogbench_layout = is_ogbench_maze(maze_layout_name)

        if self._is_ogbench_layout:
            # OGBench antmaze control: mujoco timestep=0.02 with frame_skip=5
            # -> dt=0.1 s (10 Hz). The ogbench XML carries timestep=0.02 with
            # Euler (RK4 dropped for speed, matching ant_ball_ogbench.xml —
            # diverges slightly from OGBench's RK4 dynamics).
            n_frames = 5
            if backend in ["spring", "positional"]:
                # Brax approximate backends: sub-step more aggressively so the
                # control dt still lands at 0.1 s. These will never exactly
                # match mujoco dynamics — prefer mjx when training against
                # OGBench offline data.
                sys = sys.tree_replace({"opt.timestep": 0.005})
                n_frames = 20
        else:
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
        self.dense_reward = dense_reward
        self.state_dim = 29
        self.goal_indices = jnp.array([0, 1])
        # MISC controllable_indices: directly-actuated DOFs only.
        # state[0:15]  = ant qpos  (root xyz, root quat, 8 hinge angles)
        # state[15:29] = ant qvel  (root linvel, root angvel, 8 hinge vels)
        # Hinges sit at qpos[7:15] and qvel[6:14] -> obs indices [7..14] and [21..28].
        self.controllable_indices = jnp.array(
            list(range(7, 15)) + list(range(21, 29))
        )
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        """Resets the environment to an initial state.

        Args:
            rng:   Random key.
            goal:  Optional (2,) goal position.  If None, sampled randomly.
            start: Optional (2,) start xy position.  If None, sampled randomly.
                   Pass a fixed value from GoExploreWrapper so the ant always
                   returns to the same starting cell across go phases.
        """

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # Use provided start position or sample a random one
        if start is None:
            start_pos = self._random_start(rng2)
        else:
            start_pos = start
        q = q.at[:2].set(start_pos)

        if goal is None:
            target = self._random_target(rng3)
        else:
            target = goal
        
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        # Return metrics
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
        old_dist = jnp.linalg.norm(old_obs[:2] - old_obs[-2:])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_forward=forward_reward,
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
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        """Returns a random start location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]
