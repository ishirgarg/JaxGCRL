import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import actuator, base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# This is based on original Humanoid environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/humanoid.py

# This is chosen to be very close to the z coordinate of the humanoid torso, when it is standing straight
TARGET_Z_COORD = 1.25

# Maze creation adapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py
RESET = R = "r"
GOAL = G = "g"

U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
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

# OGBench humanoidmaze-giant layout (12 rows x 16 cols), aligned with
# ogbench/locomaze/maze.py's giant maze_map. R cells mark the five task
# init_ij positions OGBench uses for `humanoidmaze-giant-*` evaluations:
# (1,1), (1,14), (8,14), (8,3), (5,9). Every other open cell is G so that
# any reachable cell can be sampled as a training goal.
OGBENCH_GIANT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, G, G, G, G, G, G, 1, 1, G, G, G, R, 1],
    [1, G, 1, G, 1, 1, G, 1, G, 1, G, G, 1, 1, G, 1],
    [1, G, G, G, 1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, G, 1, 1, 1, 1, 1, 1, G, 1, G, 1],
    [1, G, G, G, 1, G, G, G, 1, R, G, G, G, 1, G, 1],
    [1, 1, 1, G, 1, G, 1, G, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, G, 1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, G, 1, R, 1, G, 1, 1, 1, 1, 1, 1, G, 1, R, 1],
    [1, G, 1, 1, 1, G, G, G, 1, G, G, G, 1, 1, G, 1],
    [1, G, G, G, G, G, 1, G, G, G, 1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_HEIGHT = 0.5

# Layouts whose walls and physical constants are aligned with OGBench so
# that RLPD with humanoidmaze-* offline datasets sees consistent
# (s, a, s') transitions. These layouts use `humanoid_maze_ogbench.xml`
# (DMC humanoid, 21 hinges) and OGBench's column→x grid convention so
# wall positions, init_ij/goal_ij, and observations match
# ogbench/locomaze/assets/humanoid.xml + locomaze/maze.py.
OGBENCH_MAZE_LAYOUTS = {
    "ogbench_giant_stitch",
}


def is_ogbench_maze(maze_layout_name: str) -> bool:
    return maze_layout_name in OGBENCH_MAZE_LAYOUTS


def _cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention):
    # OGBench places wall geoms at (j*unit - offset_x, i*unit - offset_y), so
    # the grid's column-axis becomes physical x and the row-axis becomes y.
    # Legacy JaxGCRL mazes use the opposite mapping.
    if use_ogbench_convention:
        return [j * size_scaling - xy_offset, i * size_scaling - xy_offset]
    return [i * size_scaling, j * size_scaling]


def find_starts(structure, size_scaling, use_ogbench_convention=False):
    xy_offset = float(size_scaling)
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append(_cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention))

    return jnp.array(starts)


def find_goals(structure, size_scaling, use_ogbench_convention=False):
    xy_offset = float(size_scaling)
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append(_cell_xy(i, j, size_scaling, xy_offset, use_ogbench_convention))

    return jnp.array(goals)


def _layout_for_name(maze_layout_name):
    if maze_layout_name == "u_maze":
        return U_MAZE
    if maze_layout_name == "u_maze_eval":
        return U_MAZE_EVAL
    if maze_layout_name == "big_maze":
        return BIG_MAZE
    if maze_layout_name == "big_maze_eval":
        return BIG_MAZE_EVAL
    if maze_layout_name == "hardest_maze":
        return HARDEST_MAZE
    if maze_layout_name == "ogbench_giant_stitch":
        return OGBENCH_GIANT
    raise ValueError(f"Unknown maze layout: {maze_layout_name}")


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    maze_layout = _layout_for_name(maze_layout_name)
    use_ogbench_convention = is_ogbench_maze(maze_layout_name)
    xml_name = (
        "humanoid_maze_ogbench.xml" if use_ogbench_convention else "humanoid_maze.xml"
    )
    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", xml_name)
    xy_offset = float(maze_size_scaling)

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


class HumanoidMaze(PipelineEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=2.0,  # Was 4.0 for antmaze -- just trying to make it tractable
        dense_reward: bool = False,
        **kwargs,
    ):
        self._is_ogbench_layout = is_ogbench_maze(maze_layout_name)
        # OGBench humanoidmaze datasets use maze_unit=4.0 (giant: 16x12 grid).
        if self._is_ogbench_layout:
            maze_size_scaling = 4.0

        xml_string, possible_starts, possible_goals = make_maze(
            maze_layout_name, maze_size_scaling
        )
        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

        if self._is_ogbench_layout:
            # OGBench humanoidmaze: mujoco timestep=0.005 with frame_skip=5
            # -> dt=0.025 s (40 Hz).
            n_frames = 5
            if backend in ["spring", "positional"]:
                # Brax approximate backends: keep the smaller timestep so the
                # control dt still lands at 0.025 s. Spring/positional dynamics
                # will diverge from OGBench's mujoco — prefer mjx for tighter
                # parity with the offline dataset.
                sys = sys.tree_replace({"opt.timestep": 0.0015})
                n_frames = 17
                # DMC humanoid actuator gears (21 motors). Slightly stiffer
                # gains than ogbench so spring backend keeps the agent upright.
                gear = jnp.array(
                    [
                        80.0,   # abdomen_y
                        80.0,   # abdomen_z
                        80.0,   # abdomen_x
                        80.0,   # right_hip_x
                        80.0,   # right_hip_z
                        240.0,  # right_hip_y
                        160.0,  # right_knee
                        40.0,   # right_ankle_x
                        40.0,   # right_ankle_y
                        80.0,   # left_hip_x
                        80.0,   # left_hip_z
                        240.0,  # left_hip_y
                        160.0,  # left_knee
                        40.0,   # left_ankle_x
                        40.0,   # left_ankle_y
                        40.0,   # right_shoulder1
                        40.0,   # right_shoulder2
                        80.0,   # right_elbow
                        40.0,   # left_shoulder1
                        40.0,   # left_shoulder2
                        80.0,   # left_elbow
                    ]
                )
                sys = sys.replace(actuator=sys.actuator.replace(gear=gear))
        else:
            n_frames = 5
            if backend in ["spring", "positional"]:
                sys = sys.tree_replace({"opt.timestep": 0.0015})
                n_frames = 10
                gear = jnp.array(
                    [
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        350.0,
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ]
                )  # pyformat: disable
                sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._target_ind = self.sys.link_names.index("target")
        self.dense_reward = dense_reward

        if self._is_ogbench_layout:
            # 2D goals (xy) — matches ant_maze_ogbench and OGBench's locomaze.
            # Cache link indices used by the OGBench-style observation so we
            # avoid string lookup inside jitted step()/reset().
            #
            # Brax collapses rigid bodies without their own joint into the
            # parent link, so `head`, `left_hand`, and `right_hand` (which
            # only carry a geom) do not appear in `sys.link_names`. We
            # reconstruct their world positions from the parent's pose plus
            # the body-local offset declared in humanoid_maze_ogbench.xml.
            self._torso_idx = self.sys.link_names.index("torso")
            self._left_foot_idx = self.sys.link_names.index("left_foot")
            self._right_foot_idx = self.sys.link_names.index("right_foot")
            self._left_lower_arm_idx = self.sys.link_names.index("left_lower_arm")
            self._right_lower_arm_idx = self.sys.link_names.index("right_lower_arm")
            # Body-local offsets matching humanoid_maze_ogbench.xml.
            self._head_local = jnp.array([0.0, 0.0, 0.19])  # head pos in torso frame
            self._right_hand_local = jnp.array([0.18, 0.18, 0.18])  # in right_lower_arm frame
            self._left_hand_local = jnp.array([0.18, -0.18, 0.18])  # in left_lower_arm frame
            # 69 = xy(2) + qpos[7:](21) + head_height(1) + extremities(12)
            #      + torso_z_axis(3) + com_lin_vel(3) + qvel(27)
            self.state_dim = 69
            self.goal_indices = jnp.array([0, 1])
            self.goal_reach_thresh = 0.5
        else:
            self.state_dim = 268
            self.goal_indices = jnp.array([0, 1, 2])
            self.goal_reach_thresh = 0.5

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        """Resets the environment to an initial state.

        Args:
            rng:   Random key.
            goal:  Optional (2,) goal position.  If None, sampled randomly.
            start: Optional (2,) start xy position.  If None, sampled randomly.
                   Pass a fixed value from GoExploreWrapper so the humanoid
                   always returns to the same starting cell across go phases.
        """
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(rng1, [self.sys.q_size()], minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, [self.sys.qd_size()], minval=low, maxval=hi)

        if start is None:
            start_xy = self._random_start(rng3)
        else:
            start_xy = start
        qpos = qpos.at[:2].set(start_xy)

        if goal is None:
            target = self._random_target(rng)
        else:
            target = goal
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-2:].set(0)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "dist": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "success": zero,
            "success_easy": zero,
        }

        state = State(pipeline_state, obs, reward, done, metrics)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state, action)

        # Distance to goal: 2D xy for OGBench layouts, 3D xyz for legacy ones.
        if self._is_ogbench_layout:
            distance_to_target = jnp.linalg.norm(obs[:2] - obs[-2:])
        else:
            distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        success = jnp.array(distance_to_target < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(distance_to_target < 2.0, dtype=float)

        if self.dense_reward:
            reward = -distance_to_target + healthy_reward - ctrl_cost
        else:
            reward = success

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jnp.linalg.norm(com_after),
            dist=distance_to_target,
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """Observe humanoid body position, velocities, and angles.

        For OGBench layouts, returns the 69-dim state vector that
        humanoidmaze-* offline datasets use, followed by the 2D target xy.
        Format mirrors ogbench/locomaze/humanoid.py:HumanoidEnv.get_ob().
        """
        if self._is_ogbench_layout:
            return self._get_obs_ogbench(pipeline_state)

        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jnp.hstack([cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]])

        xd_i = base.Transform.create(pos=x_i.pos - pipeline_state.x.pos).vmap().do(pipeline_state.xd)
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jnp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd)

        target_pos = pipeline_state.x.pos[-1][:2]
        # external_contact_forces are excluded
        return jnp.concatenate(
            [
                position,
                velocity,
                com_inertia.ravel(),
                com_velocity.ravel(),
                qfrc_actuator,
                target_pos,
                jnp.array([TARGET_Z_COORD]),  # Height of the target is fixed
            ]
        )

    def _get_obs_ogbench(self, pipeline_state: base.State) -> jax.Array:
        """Brax port of ogbench.locomaze.humanoid.HumanoidEnv.get_ob().

        Layout (matches the ``humanoidmaze-*`` offline datasets):
          xy(2) + qpos[7:](21) + head_height(1) + extremities(12)
          + torso_vertical_orientation(3) + com_linear_velocity(3)
          + qvel(27) = 69 dims, then target_xy(2) appended for GCRL.
        """
        # Strip the two target slide joints from q/qd so we get
        # humanoid-only qpos(28) and qvel(27) — matches OGBench dimensions.
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        xy = qpos[:2]
        joint_angles = qpos[7:]  # 21 hinges (skip 7 freejoint dofs)

        torso_pos = pipeline_state.x.pos[self._torso_idx]
        torso_quat = pipeline_state.x.rot[self._torso_idx]

        # head/hand world positions reconstructed from parent + local offset
        # because brax collapses joint-less child bodies into the parent.
        head_world = torso_pos + math.rotate(self._head_local, torso_quat)
        head_height = head_world[2]

        right_lower_arm_pos = pipeline_state.x.pos[self._right_lower_arm_idx]
        right_lower_arm_quat = pipeline_state.x.rot[self._right_lower_arm_idx]
        right_hand_world = right_lower_arm_pos + math.rotate(
            self._right_hand_local, right_lower_arm_quat
        )
        left_lower_arm_pos = pipeline_state.x.pos[self._left_lower_arm_idx]
        left_lower_arm_quat = pipeline_state.x.rot[self._left_lower_arm_idx]
        left_hand_world = left_lower_arm_pos + math.rotate(
            self._left_hand_local, left_lower_arm_quat
        )

        # Limb positions in torso frame. OGBench computes
        # (limb_pos - torso_pos) @ torso_xmat, which projects the world-frame
        # offset onto the torso body axes — equivalent to rotating by the
        # torso quaternion's inverse.
        torso_quat_inv = math.quat_inv(torso_quat)

        def to_torso_frame(world_pos):
            return math.rotate(world_pos - torso_pos, torso_quat_inv)

        extremities = jnp.concatenate(
            [
                to_torso_frame(left_hand_world),
                to_torso_frame(pipeline_state.x.pos[self._left_foot_idx]),
                to_torso_frame(right_hand_world),
                to_torso_frame(pipeline_state.x.pos[self._right_foot_idx]),
            ]
        )

        # Torso z-axis expressed in world frame (xmat[1, [6,7,8]] in mujoco).
        torso_z_axis_world = math.rotate(jnp.array([0.0, 0.0, 1.0]), torso_quat)

        # subtreelinvel(torso) ≈ COM linear velocity of the entire kinematic
        # tree because the torso is the only direct child of world.
        com_lin_vel = self._com_linear_velocity(pipeline_state)

        state = jnp.concatenate(
            [
                xy,
                joint_angles,
                jnp.array([head_height]),
                extremities,
                torso_z_axis_world,
                com_lin_vel,
                qvel,
            ]
        )

        target_xy = pipeline_state.x.pos[self._target_ind][:2]
        return jnp.concatenate([state, target_xy])

    def _com_linear_velocity(self, pipeline_state: base.State) -> jax.Array:
        """Mass-weighted COM linear velocity across all bodies (3 dims)."""
        _, inertia, mass_sum, x_i = self._com(pipeline_state)
        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        return jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, xd_i.vel), axis=0) / mass_sum

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(
                    jax.vmap(jnp.diagonal)(inertia.i) ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return (
            com,
            inertia,
            mass_sum,
            x_i,
        )  # pytype: disable=bad-return-type  # jax-ndarray

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]
