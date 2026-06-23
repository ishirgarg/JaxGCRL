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
RESET_GOAL = M = "m"  # cell that is both an agent reset AND a goal candidate

MAZE_HEIGHT = 0.5

# OGBench arena: 8x8 open space. Rows index y (OGBench convention), so the
# grid is the transpose of the legacy JaxGCRL layout; physical R/B/G
# positions are unchanged.
ARENA = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, B, R, B, R, B, 1],
    [1, B, B, B, B, B, R, 1],
    [1, R, B, G, G, B, B, 1],
    [1, B, B, G, G, B, R, 1],
    [1, R, B, B, B, B, B, 1],
    [1, B, B, R, B, R, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

ARENA_1G = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, R, 0, 0, 0, 1],
    [1, 0, 0, 0, B, 0, 0, 1],
    [1, 0, 0, 0, 0, G, 0, 1],
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

SMALL_EASY_SQUARE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, G, 0, 0, 0, 1],
    [1, 0, B, 0, 0, 0, 0, 1],
    [1, G, 0, G, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
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

# OGBench arena (open 8x8) structured as concentric rings for stitch training:
# ant resets on the outer ring of open cells, ball goals are the next inner
# ring, and ball spawns are the 4 innermost cells. Pairs with the
# antsoccer-arena-stitch RLPD dataset.
ARENA_STITCH = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, R, R, R, R, R, 1],
    [1, R, G, G, G, G, R, 1],
    [1, R, G, B, B, G, R, 1],
    [1, R, G, B, B, G, R, 1],
    [1, R, G, G, G, G, R, 1],
    [1, R, R, R, R, R, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# OGBench medium maze in OGBench's native row/col orientation.
# Mirrors ogbench/locomaze/maze.py's medium maze_map exactly.
MEDIUM = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# Medium maze with the agent always reset at the bottom-left open cell.
# Ball spawn is fixed at (4, 2) — the closest open cell to the strict
# "2 squares diagonally up-right" target (4, 3), which is a wall in the
# OGBench medium maze. Every other open cell is a ball goal.
MEDIUM_STITCH = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, B, G, 1, 1, 1],
    [1, G, B, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

SMALL_EASY_SQUARE_1G= [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 0, 1],
    [1, 0, B, 0, 0, 0, 0, 1],
    [1, 0, 0, G, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
SMALL_SQUARE_1G= [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, B, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, G, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# ── "scale2" variants ───────────────────────────────────────────────────────
# These mirror ARENA_1G / SMALL_EASY_SQUARE_1G but on a 16x16 grid intended to
# be built with maze_size_scaling=2.0 (half the default cell size) and the
# OGBench-anchored offset from `_default_xy_offset` (= 5.0 at scaling 2). That
# combination reproduces the *exact* same world frame as the scaling-4 8x8
# originals — per-axis wall bounds [-6, 26] — because doubling the grid while
# halving the scaling preserves the total physical extent (16*2 == 8*4 == 32).
#
# R/B/G remain ONE cell apart (same "distance in units" / cells as the
# originals), so at scaling 2.0 the physical ant->ball->goal gaps are HALVED:
# 2 units/axis instead of 4. Placements are chosen on the (odd-integer) scale2
# lattice to sit in the same region as the originals:
#   arena_1g:      orig R(8,8)  B(12,12) G(16,16) -> scale2 R(9,9)  B(11,11) G(13,13)
#   small_easy_sq: orig R(0,0)  B(4,4)   G(8,8)   -> scale2 R(1,1)  B(3,3)   G(5,5)
ARENA_1G_SCALE2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, B, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, G, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

SMALL_EASY_SQUARE_1G_SCALE2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, G, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


# Canonical OGBench antsoccer arena reference: an 8-cell grid at scaling 4 with
# an offset of 4. Its outer wall low edge sits at -(offset + scaling/2) = -6,
# giving per-axis world bounds [-6, 26].
_REF_SCALING = 4.0
_REF_OFFSET = 4.0
_CANONICAL_LOW_EDGE = _REF_OFFSET + 0.5 * _REF_SCALING  # 6.0


def _default_xy_offset(maze_size_scaling):
    """OGBench-anchored coordinate offset, derived dynamically from scaling.

    OGBench's locomaze maze.py uses a fixed offset of 4 at maze_unit=4, which
    coincides with the legacy `offset == maze_size_scaling` convention used here
    (4 == 4). To keep the maze anchored to that exact world frame when the cell
    size changes, we instead hold the outer wall's LOW edge fixed at
    -_CANONICAL_LOW_EDGE (= -6):

        offset = _CANONICAL_LOW_EDGE - 0.5 * maze_size_scaling

    At the canonical scaling of 4 this returns 4 (identical to legacy behaviour,
    so every existing 8x8 env is byte-for-byte unchanged). At scaling 2 it
    returns 5, which — together with a grid whose total extent stays 32 units
    (e.g. 16 cells * 2) — reproduces the original [-6, 26] frame exactly.
    """
    return _CANONICAL_LOW_EDGE - 0.5 * float(maze_size_scaling)


def _cell_xy(i, j, maze_size_scaling, xy_offset):
    # OGBench convention: columns map to x, rows map to y. See
    # ogbench/locomaze/maze.py `ij_to_xy` (x=j*unit-offset, y=i*unit-offset).
    return [j * maze_size_scaling - xy_offset,
            i * maze_size_scaling - xy_offset]


def _find_cells(maze_layout, maze_size_scaling, obj):
    """Return (x,y) positions of cells matching obj (or any marker if obj is a tuple/list)."""
    xy_offset = _default_xy_offset(maze_size_scaling)
    if isinstance(obj, (tuple, list, set)):
        markers = set(obj)
    else:
        markers = {obj}
    cells = []
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] in markers:
                cells.append(_cell_xy(i, j, maze_size_scaling, xy_offset))
    return jnp.array(cells) if cells else jnp.zeros((0, 2))


def _open_cells(maze_layout, maze_size_scaling):
    """Return (x,y) positions of all open cells (non-wall)."""
    xy_offset = _default_xy_offset(maze_size_scaling)
    cells = []
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] != 1:
                cells.append(_cell_xy(i, j, maze_size_scaling, xy_offset))
    return jnp.array(cells) if cells else jnp.zeros((0, 2))


def make_ball_maze(maze_layout_name, maze_size_scaling):
    """Build XML string with maze walls from the OGBench-compatible ball XML template."""
    if maze_layout_name == "arena":
        maze_layout = ARENA
    elif maze_layout_name == "arena_1g":
        maze_layout = ARENA_1G
    elif maze_layout_name == "arena_stitch":
        maze_layout = ARENA_STITCH
    elif maze_layout_name == "medium":
        maze_layout = MEDIUM
    elif maze_layout_name == "medium_stitch":
        maze_layout = MEDIUM_STITCH
    elif maze_layout_name == "small_square":
        maze_layout = SMALL_SQUARE
    elif maze_layout_name == "easy_square":
        maze_layout = EASY_SQUARE
    elif maze_layout_name == "small_easy_square":
        maze_layout = SMALL_EASY_SQUARE
    elif maze_layout_name == "small_easy_square_1g":
        maze_layout = SMALL_EASY_SQUARE_1G
    elif maze_layout_name == "small_square_1g":
        maze_layout = SMALL_SQUARE_1G
    elif maze_layout_name == "arena_1g_scale2":
        maze_layout = ARENA_1G_SCALE2
    elif maze_layout_name == "small_easy_square_1g_scale2":
        maze_layout = SMALL_EASY_SQUARE_1G_SCALE2
    else:
        raise ValueError(f"Unknown OGBench ball maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "assets", "ant_ball_ogbench.xml")
    xy_offset = _default_xy_offset(maze_size_scaling)

    # Check if the layout uses R/G/B/M markers for separate spawn regions.
    # M cells count as both an agent reset and a goal (used by arena_stitch
    # where several OGBench arena tasks share an init/goal cell).
    has_markers = any(
        cell in (R, G, B, M)
        for row in maze_layout
        for cell in row
    )
    if has_markers:
        starts = _find_cells(maze_layout, maze_size_scaling, (R, M))
        goals = _find_cells(maze_layout, maze_size_scaling, (G, M))
        balls = _find_cells(maze_layout, maze_size_scaling, B)
        # Layouts without B markers (e.g. medium_stitch) sample ball spawns
        # from any open cell, matching OGBench stitch datasets where
        # ball_init_ij is drawn freely from the open maze.
        if balls.shape[0] == 0:
            balls = _open_cells(maze_layout, maze_size_scaling)
    else:
        open_cells = _open_cells(maze_layout, maze_size_scaling)
        starts = open_cells
        goals = open_cells
        balls = open_cells

    # Paired (agent_xy, ball_xy, goal_xy) tuples — only set for layouts that
    # define a fixed task list. Otherwise None and the env samples R/G/B
    # independently at reset.
    paired_tasks = None

    rows = len(maze_layout)
    cols = len(maze_layout[0])
    half = 0.5 * maze_size_scaling
    # OGBench convention: columns index x, rows index y.
    x_bounds = (-xy_offset - half, (cols - 1) * maze_size_scaling - xy_offset + half)
    y_bounds = (-xy_offset - half, (rows - 1) * maze_size_scaling - xy_offset + half)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(rows):
        for j in range(cols):
            if maze_layout[i][j] == 1:
                wall_x, wall_y = _cell_xy(i, j, maze_size_scaling, xy_offset)
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
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    return xml_string, starts, goals, balls, x_bounds, y_bounds, paired_tasks


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
        add_ant_to_goal: bool = False,
        **kwargs,
    ):
        xml_string, starts, goals, balls, x_bounds, y_bounds, paired_tasks = make_ball_maze(
            maze_layout_name, maze_size_scaling
        )

        sys = mjcf.loads(xml_string)
        if add_ant_to_goal:
            # Each G cell is interpreted as (ant_goal, ball_goal) at the same
            # square, so possible_goals are 4D and every proposer / HER path
            # consistently sees length-4 goals.
            goals = jnp.concatenate([goals, goals], axis=-1)
        self.possible_goals = goals
        self.possible_starts = starts
        self.possible_balls = balls
        # Paired (agent, ball, goal) tuples for layouts with a fixed task list.
        # When set, default reset samples one tuple jointly instead of drawing
        # the three positions independently from possible_*.
        self.paired_tasks = paired_tasks
        self.x_bounds = tuple(float(v) for v in x_bounds)
        self.y_bounds = tuple(float(v) for v in y_bounds)

        # OGBench antsoccer runs mujoco at timestep=0.02 with frame_skip=5
        # (10 Hz control). The XML specifies timestep=0.02 with Euler (RK4
        # dropped for speed — diverges slightly from OGBench's dynamics).
        n_frames = 5

        if backend in ["spring", "positional"]:
            # Brax approximate backends: sub-step more aggressively so the
            # control dt still lands at 0.1 s (matching OGBench 10 Hz).
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 20

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
        self.add_ant_to_goal = add_ant_to_goal
        self.maze_layout_name = maze_layout_name
        self.maze_size_scaling = float(maze_size_scaling)

        self._ball_idx = self.sys.link_names.index("ball")

        # OGBench-compatible dimensions:
        # state = [qpos(22), qvel(20)] = 42.
        # goal = target_xy(2) if add_ant_to_goal is False, else
        #        target_ant_xy + target_ball_xy = 4 (both set to the same G cell).
        self.state_dim = self._ANT_Q + self._BALL_Q + self._ANT_QD + self._BALL_QD  # 42
        if self.add_ant_to_goal:
            # Ant xy lives at state[0:2] (free root joint).
            self.goal_indices = jnp.array([0, 1, 15, 16])
        else:
            self.goal_indices = jnp.array([15, 16])  # ball x,y in qpos portion
        # MISC controllable_indices: directly-actuated DOFs of the *ant* only.
        # The ball is unactuated and is part of s_g (or downstream state); the
        # ant's free root pose is downstream of joint torques + contact.
        # state[0:15]  = ant qpos  (root xyz, root quat, 8 hinge angles -> 7..14)
        # state[15:22] = ball qpos
        # state[22:36] = ant qvel  (root linvel, root angvel, 8 hinge vels -> 28..35)
        # state[36:42] = ball qvel
        self.controllable_indices = jnp.array(
            list(range(7, 15)) + list(range(28, 36))
        )
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_ant, rng_ball, rng_goal, rng_task = jax.random.split(rng, 7)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Default sampling: when paired_tasks is set, draw one of the
        # predefined (agent, ball, goal) tuples jointly so the env only ever
        # resets into one of OGBench's evaluation tasks. Otherwise fall back
        # to independent R/G/B sampling.
        if self.paired_tasks is not None:
            idx = jax.random.randint(rng_task, (), 0, self.paired_tasks.shape[0])
            paired = self.paired_tasks[idx]            # (3, 2)
            paired_start_xy = paired[0]
            paired_ball_xy = paired[1]
            paired_target_xy = paired[2]
            if self.add_ant_to_goal:
                paired_goal_arr = jnp.concatenate([paired_target_xy, paired_target_xy])
            else:
                paired_goal_arr = paired_target_xy
        else:
            paired_start_xy = None
            paired_ball_xy = None
            paired_goal_arr = None

        # Start position. `start` can be either length-2 (ant xy only) or
        # length-4 (ant xy + ball xy); the latter is used by GoExploreWrapper to
        # restore the full ant+ball pose at the start of each go phase in the
        # 4D-goal variant.
        if start is None:
            if paired_start_xy is not None:
                start_xy = paired_start_xy
                ball_xy = paired_ball_xy
            else:
                start_xy = self._random_start(rng_ant)
                ball_xy = self._random_ball(rng_ball)
        else:
            start_arr = jnp.asarray(start, dtype=q.dtype)
            if start_arr.shape[-1] == 4:
                start_xy = start_arr[:2]
                ball_xy = start_arr[2:4]
            else:
                # 2D-goal variant: goal_indices=[15,16] are ball xy, so the
                # GoExploreWrapper-stored first_start is the ball's initial
                # position. Treat it as such and randomize the ant start.
                start_xy = self._random_start(rng_ant)
                ball_xy = start_arr[:2]
        q = q.at[:2].set(start_xy)

        # Ball position (free joint qpos at indices _ANT_Q : _ANT_Q + _BALL_Q)
        ball_q_start = self._ANT_Q  # 15
        # ball_xy was resolved above from the `start` argument (4D case) or
        # randomly sampled (2D or None case).
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

        # Target marker position. The XML's "target" body has 2 slide joints
        # and lives at q[-2:] — it is a non-physical sphere used to store the
        # GCRL goal location (the ball's target xy). In 4D-goal mode the goal
        # is [ant_goal_xy, ball_goal_xy]; we bake the ball-target xy into the
        # marker and rely on _get_obs duplicating it for the ant-target slot.
        if goal is None:
            if paired_goal_arr is not None:
                goal_arr = paired_goal_arr
            else:
                goal_arr = self._random_goal(rng_goal)
        else:
            goal_arr = jnp.asarray(goal, dtype=q.dtype)
        target_xy = goal_arr[-2:]
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
        # Ball xy = state[15:17]. The target xy (G cell) is always the last
        # two elements of obs, and in 4D-goal mode the preceding two are the
        # ant-goal copy (same value).
        ball_xy = obs[15:17]
        target_xy = obs[-2:]
        dist = jnp.linalg.norm(ball_xy - target_xy)

        old_obs = self._get_obs(pipeline_state0)
        old_ball_xy = old_obs[15:17]
        old_target_xy = old_obs[-2:]
        old_dist = jnp.linalg.norm(old_ball_xy - old_target_xy)

        vel_to_target = (old_dist - dist) / self.dt
        if self.add_ant_to_goal:
            ant_xy = obs[0:2]
            ant_dist = jnp.linalg.norm(ant_xy - target_xy)
            success = jnp.array(
                (dist < self.goal_reach_thresh) & (ant_dist < self.goal_reach_thresh),
                dtype=float,
            )
            success_easy = jnp.array((dist < 2.0) & (ant_dist < 2.0), dtype=float)
        else:
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

        if self.add_ant_to_goal:
            # Duplicate the G cell for ant and ball targets so that obs[-4:]
            # aligns with goal_indices=[0,1,15,16] (ant_xy, ball_xy).
            target_pos = jnp.concatenate([target_pos, target_pos])

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
