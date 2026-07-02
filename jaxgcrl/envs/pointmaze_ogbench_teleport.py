"""OGBench-aligned pointmaze with teleports for JaxGCRL.

Mirrors ogbench/locomaze/maze.py's `MazeEnv(loco='point', maze='teleport')`:
the maze layout, teleport ij positions, task pairs, and OGBench's coordinate
convention (column->x, row->y) are reproduced exactly so that
`pointmaze-teleport-navigate-v0` offline transitions are consistent with
this env.

Step semantics follow OGBench's PointEnv:
    qpos[:2] += 0.2 * action; qvel = 0
followed by a zero-ctrl Brax pipeline_step that resolves wall contacts via
the underlying physics solver. After dynamics, if the agent is within
1.5 * teleport_radius of any inbound teleport, it is jumped to a uniformly
random outbound teleport (rng threaded through `state.info['rng']`).
"""

import os
import xml.etree.ElementTree as ET

import jax
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

from jaxgcrl.envs._locomotion_common import apply_mjx_solver_flags


# Symbolic markers used inside TELEPORT_MAP. They are all open cells to the
# wall builder (which only treats `1` as a wall). Their (i, j) positions
# drive the reset/teleport logic via TASK_PAIRS / TELEPORT_IN_IJS.
RESET = R = "r"
GOAL = G = "g"
T = "t"


# OGBench teleport maze layout (rows=9, cols=12).
# 1 = wall, 0 = open, R = reset (start) cell, G = goal cell, T = inbound
# teleport cell. R/G/T are annotations only — see TASK_PAIRS for the
# enforced (start, goal) pairing and TELEPORT_IN_IJS for teleport behavior.
TELEPORT_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, G, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, G, 0, 0, 0, 1, T, 1, 0, 1, 0, 1],
    [1, T, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, R, 1, 0, 1, 0, 1],
    [1, R, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Inbound and outbound teleport (i, j) cells (OGBench).
TELEPORT_IN_IJS = [(4, 6), (5, 1)]
TELEPORT_OUT_IJS = [(1, 7), (6, 1), (6, 10)]
TELEPORT_RADIUS = 1.0

# Paired (start_ij, goal_ij) tasks. Each reset samples ONE pair and uses
# both its start and goal — we never combine the start of one pair with
# the goal of another. The pairs correspond to the R/G annotations in
# TELEPORT_MAP: the left column (col 1) and the right column (col 6).
TASK_PAIRS = [
    ((7, 1), (4, 1)),  # left R/G column
    ((6, 6), (2, 6)),  # right R/G column
]

# Single-goal variant: same maze geometry as TELEPORT_MAP but only the left
# R/G column is retained as an annotated task; the right R/G cells become
# plain open cells. Used by pointmaze_ogbench_teleport_1g.
TELEPORT_MAP_1G = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, G, 0, 0, 0, 1, T, 1, 0, 1, 0, 1],
    [1, T, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, R, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
TASK_PAIRS_1G = [
    ((7, 1), (4, 1)),  # left R/G column
]

MAZE_HEIGHT = 0.5
DEFAULT_MAZE_SIZE_SCALING = 4.0


def _ij_to_xy(i, j, size_scaling, xy_offset):
    """OGBench convention: column index -> x, row index -> y."""
    return (j * size_scaling - xy_offset, i * size_scaling - xy_offset)


def _build_xml(
    maze_size_scaling: float,
    teleport_map,
    teleport_in_ijs,
    teleport_out_ijs,
) -> bytes:
    xml_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "assets",
        "pointmaze_ogbench_teleport.xml",
    )
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")
    xy_offset = float(maze_size_scaling)

    # Walls.
    for i in range(len(teleport_map)):
        for j in range(len(teleport_map[0])):
            if teleport_map[i][j] == 1:
                wall_x, wall_y = _ij_to_xy(i, j, maze_size_scaling, xy_offset)
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (
                        wall_x,
                        wall_y,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f" % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    contype="1",
                    conaffinity="1",
                    material="wall",
                )

    # Teleport markers (visualization only — non-colliding).
    for idx, (i, j) in enumerate(teleport_in_ijs):
        x, y = _ij_to_xy(i, j, maze_size_scaling, xy_offset)
        ET.SubElement(
            worldbody, "geom",
            name=f"teleport_in_{idx}",
            type="cylinder",
            size=f"{TELEPORT_RADIUS} 0.05",
            pos=f"{x} {y} 0.05",
            material="teleport_in",
            contype="0", conaffinity="0",
        )
    for idx, (i, j) in enumerate(teleport_out_ijs):
        x, y = _ij_to_xy(i, j, maze_size_scaling, xy_offset)
        ET.SubElement(
            worldbody, "geom",
            name=f"teleport_out_{idx}",
            type="cylinder",
            size=f"{TELEPORT_RADIUS} 0.05",
            pos=f"{x} {y} 0.05",
            material="teleport_out",
            contype="0", conaffinity="0",
        )

    return ET.tostring(tree.getroot())


class PointMazeOGBenchTeleport(PipelineEnv):
    """OGBench-aligned 2D point-mass teleport maze.

    Observation: [agent_x, agent_y, target_x, target_y]. The first two dims
    (state) match OGBench's pointmaze observation (qpos = [ballx, bally]);
    RLPD pads the goal portion with zeros when loading the
    `pointmaze-teleport-navigate-v0` dataset.
    """

    def __init__(
        self,
        backend: str = "mjx",
        maze_size_scaling: float = DEFAULT_MAZE_SIZE_SCALING,
        reset_noise_scale: float = 0.1,
        dense_reward: bool = False,
        action_scale: float = 0.2,
        teleport_map=None,
        task_pairs=None,
        teleport_in_ijs=None,
        teleport_out_ijs=None,
        **kwargs,
    ):
        if teleport_map is None:
            teleport_map = TELEPORT_MAP
        if task_pairs is None:
            task_pairs = TASK_PAIRS
        if teleport_in_ijs is None:
            teleport_in_ijs = TELEPORT_IN_IJS
        if teleport_out_ijs is None:
            teleport_out_ijs = TELEPORT_OUT_IJS

        xml_string = _build_xml(
            maze_size_scaling, teleport_map, teleport_in_ijs, teleport_out_ijs
        )
        sys = mjcf.loads(xml_string)

        # Match ant_maze_ogbench timing: timestep 0.02 * frame_skip 5 = 0.1 s.
        n_frames = 5
        if backend in ("spring", "positional"):
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 20

        if backend == "mjx":
            sys = apply_mjx_solver_flags(sys)

        if backend == "positional":
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear))
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._reset_noise_scale = reset_noise_scale
        self._action_scale = action_scale
        self.dense_reward = dense_reward
        # state_dim = 2 (agent qpos) -> matches OGBench pointmaze obs dim.
        self.state_dim = 2
        self.goal_indices = jnp.array([0, 1])
        self.controllable_indices = jnp.array([0, 1])
        # OGBench point goal_tol == 1.0.
        self.goal_reach_thresh = 1.0

        xy_offset = float(maze_size_scaling)
        active_pairs = task_pairs
        # Per-pair (start, goal) xy arrays — row k = the k-th task pair.
        # Reset draws ONE pair index and uses both its start and goal so
        # starts/goals never get mixed across pairs.
        self._pair_starts = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset))
             for (i, j), _ in active_pairs]
        )
        self._pair_goals = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset))
             for _, (i, j) in active_pairs]
        )
        self._num_pairs = len(active_pairs)
        # Backward-compat: unique start/goal cells exposed for go_explore
        # goal proposers that read env.possible_goals / env.possible_starts.
        unique_starts = sorted({init for init, _ in active_pairs})
        unique_goals = sorted({goal for _, goal in active_pairs})
        self.possible_starts = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset)) for i, j in unique_starts]
        )
        self.possible_goals = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset)) for i, j in unique_goals]
        )

        # Teleport metadata (jax arrays for jit compatibility).
        self._teleport_in_xys = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset)) for i, j in teleport_in_ijs]
        )
        self._teleport_out_xys = jnp.array(
            [list(_ij_to_xy(i, j, maze_size_scaling, xy_offset)) for i, j in teleport_out_ijs]
        )
        self._teleport_radius = float(TELEPORT_RADIUS)
        self._teleport_threshold = self._teleport_radius * 1.5

        # Maze bounds for plotting / clipping (OGBench convention).
        num_rows = len(teleport_map)
        num_cols = len(teleport_map[0])
        half = 0.5 * maze_size_scaling
        self.x_bounds = jnp.array([
            -xy_offset - half,
            (num_cols - 1) * maze_size_scaling - xy_offset + half,
        ])
        self.y_bounds = jnp.array([
            -xy_offset - half,
            (num_rows - 1) * maze_size_scaling - xy_offset + half,
        ])

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        rng, rng_q, rng_qd, rng_pair, rng_teleport = jax.random.split(rng, 5)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng_q, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng_qd, (self.sys.qd_size(),))

        # Sample a single (start, goal) pair so start and goal stay matched
        # to the same task. Caller-provided overrides win individually.
        pair_idx = jax.random.randint(rng_pair, (), 0, self._num_pairs)
        pair_start = self._pair_starts[pair_idx]
        pair_goal = self._pair_goals[pair_idx]

        start_pos = pair_start if start is None else start
        q = q.at[:2].set(start_pos)

        target = pair_goal if goal is None else goal
        q = q.at[-2:].set(target)
        # OGBench point env keeps the agent qvel at zero between steps.
        qd = qd.at[:2].set(0.0)
        qd = qd.at[-2:].set(0.0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        zero = jnp.zeros(())
        metrics = {
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), metrics)
        state.info["rng"] = rng_teleport
        return state

    def step(self, state: State, action: jax.Array) -> State:
        ps0 = state.pipeline_state
        action = jnp.clip(action, -1.0, 1.0)

        # OGBench point step rule: qpos[:2] += 0.2 * action; qvel = 0.
        new_q = ps0.q.at[:2].add(self._action_scale * action)
        new_qd = jnp.zeros_like(ps0.qd)

        # Re-init the pipeline so the underlying mjx Data's qpos/qvel reflect
        # the manual override (brax mjx.pipeline.step reads data.qpos, not the
        # high-level .q field). Then advance with zero ctrl so contact
        # resolution mirrors OGBench's mj_step after the qpos override.
        ps_init = self.pipeline_init(new_q, new_qd)
        zero_ctrl = jnp.zeros(self.sys.act_size())
        pipeline_state = self.pipeline_step(ps_init, zero_ctrl)

        # Teleport check.
        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, sub = jax.random.split(rng)
        new_xy = self._maybe_teleport(pipeline_state.q[:2], sub)
        post_q = pipeline_state.q.at[:2].set(new_xy)
        post_qd = pipeline_state.qd.at[:2].set(0.0)
        pipeline_state = pipeline_state.replace(q=post_q, qd=post_qd)

        obs = self._get_obs(pipeline_state)
        target_xy = obs[-2:]
        dist = jnp.linalg.norm(obs[:2] - target_xy)
        success = jnp.array(dist < self.goal_reach_thresh, dtype=jnp.float32)
        success_easy = jnp.array(dist < 2.0, dtype=jnp.float32)
        reward = -dist if self.dense_reward else success
        done = jnp.zeros_like(reward)

        state.info["rng"] = rng
        state.metrics.update(
            x_position=pipeline_state.q[0],
            y_position=pipeline_state.q[1],
            distance_from_origin=jnp.linalg.norm(pipeline_state.q[:2]),
            dist=dist,
            success=success,
            success_easy=success_easy,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        # State is the agent's qpos (2D xy, matches OGBench point obs).
        agent_qpos = pipeline_state.q[:-2]
        target_xy = pipeline_state.x.pos[-1][:2]
        return jnp.concatenate([agent_qpos, target_xy])

    def _maybe_teleport(self, xy: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        dists = jnp.linalg.norm(xy[None, :] - self._teleport_in_xys, axis=-1)
        in_radius = jnp.any(dists <= self._teleport_threshold)
        idx = jax.random.randint(rng, (), 0, self._teleport_out_xys.shape[0])
        out_xy = self._teleport_out_xys[idx]
        return jnp.where(in_radius, out_xy, xy)
