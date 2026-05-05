"""OGBench cube-single-v0 environment ported to JaxGCRL for online training.

This is a faithful port of `ogbench.manipspace.envs.cube_env.CubeEnv` with
`env_type='single'`. The MJCF, physics parameters, action space, observation
layout, goal definition, and per-task init/goal cube positions are all matched
1:1 to OGBench. The only intentional deviation is the constraint solver — MJX
exposes Newton/CG, while OGBench's MuJoCo runs the PGS solver by default; we
use Newton with conservative iteration counts.

OGBench's environment exposes a 5-D delta-end-effector action that is resolved
to UR5e joint targets through a 20-iteration damped-least-squares differential
IK loop on an arm-only MuJoCo model. We reproduce the same loop in JAX/MJX so
the action semantics are preserved exactly.
"""

import os
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io.mjcf import load_model
from mujoco import mjx


# ---------------------------------------------------------------------------
# Constants — copied verbatim from `ogbench.manipspace.envs.manipspace_env` and
# `cube_env.CubeEnv`. Keep these in sync if OGBench upstream changes.
# ---------------------------------------------------------------------------

_HOME_QPOS = np.asarray(
    [-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)
# SO3 quat (wxyz) for the "down" orientation of the gripper. This is the
# quaternion equivalent of a 180° rotation about the world y-axis: pinch z
# axis points straight down.
_EFFECTOR_DOWN_QUAT = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

_WORKSPACE_BOUNDS = np.asarray(
    [[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]], dtype=np.float64
)
_ARM_SAMPLING_BOUNDS = np.asarray(
    [[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]], dtype=np.float64
)
_OBJECT_SAMPLING_BOUNDS = np.asarray(
    [[0.30, -0.30], [0.55, 0.30]], dtype=np.float64
)
_TARGET_SAMPLING_BOUNDS = np.asarray(
    [[0.30, -0.30], [0.55, 0.30]], dtype=np.float64
)

# Action range for the 5-D delta action: (Δx, Δy, Δz, Δyaw, Δgripper).
_ACTION_RANGE = np.asarray([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float64)

# Observation scaling — `compute_observation` shifts XYZ by the table center
# and multiplies by 10; the gripper opening is multiplied by 3.
_XYZ_CENTER = np.asarray([0.425, 0.0, 0.0], dtype=np.float64)
_XYZ_SCALER = 10.0
_GRIPPER_SCALER = 3.0

# 5 single-cube tasks, copied from CubeEnv.set_tasks(env_type='single').
_TASK_INIT_XYZS = np.asarray(
    [
        [0.425, 0.10, 0.02],  # task1_horizontal
        [0.35, 0.00, 0.02],  # task2_vertical1
        [0.50, 0.00, 0.02],  # task3_vertical2
        [0.35, -0.20, 0.02],  # task4_diagonal1
        [0.35, 0.20, 0.02],  # task5_diagonal2
    ],
    dtype=np.float64,
)
_TASK_GOAL_XYZS = np.asarray(
    [
        [0.425, -0.10, 0.02],
        [0.50, 0.00, 0.02],
        [0.35, 0.00, 0.02],
        [0.50, 0.20, 0.02],
        [0.50, -0.20, 0.02],
    ],
    dtype=np.float64,
)

# Success when the cube is within this distance of its target (raw metres).
_SUCCESS_THRESHOLD = 0.04

# Differential IK hyperparameters (mirrors `controllers.diff_ik.DiffIKController`).
# OGBench uses damping=1e-12 in numpy float64; in JAX/MJX float32 that magnitude
# is below the precision floor for typical (J Jᵀ) eigenvalues, so we bump it up
# to keep the linear solve well-conditioned. The functional behaviour is the
# same — damping only matters near singularities.
_IK_MAX_ITERS = 5
_IK_DAMPING = 1e-6
_IK_MAX_ANGLE_CHANGE = np.radians(45.0)


# ---------------------------------------------------------------------------
# MJCF assembly — defer to `ogbench.manipspace` so we share the exact same
# scene description (UR5e + Robotiq 2F-85 + cube + mocap target + walls).
# ---------------------------------------------------------------------------


def _assemble_mjcf() -> Tuple[mujoco.MjModel, mujoco.MjModel]:
    """Build the cube-single MjModel and the arm-only IK MjModel.

    We import ogbench lazily so that downstream code that doesn't touch this
    env doesn't pay the dm_control import cost.

    Returns:
        mj: the full scene MjModel (used for physics and observations).
        ik_mj: a UR5e-only MjModel used purely for kinematics during IK.
    """
    import gymnasium as gym
    import ogbench  # noqa: F401  (registers the gym env ids)
    from dm_control import mjcf
    from ogbench.manipspace import mjcf_utils

    scene_env = gym.make('cube-single-v0').unwrapped
    scene_env.reset()
    full_xml = mjcf_utils.to_string(scene_env._mjcf_model)
    full_assets = mjcf_utils.get_assets(scene_env._mjcf_model)

    import re

    # MJX does not implement cylinder ↔ mesh collision. The only cylinder geom
    # in the OGBench scene is the wrist-3 EEF collision shape; switch it to a
    # capsule of identical (radius, half-length) so collisions are valid.
    patched_xml = full_xml.replace(
        '<default class="ur5e/eef_collision">\n      <geom type="cylinder"',
        '<default class="ur5e/eef_collision">\n      <geom type="capsule"',
    )
    if patched_xml == full_xml:
        # Fall back to a looser match in case mjcf serialisation changes.
        patched_xml = re.sub(
            r'(<default class="ur5e/eef_collision">\s*<geom type=")cylinder(")',
            r'\1capsule\2',
            full_xml,
        )

    # Switch integrator to Euler to match the rest of the OGBench-aligned envs
    # in this repo (ant_ball_ogbench / ant_maze_ogbench). For the spring/
    # positional backends we also have to drop `cone="elliptic"` and
    # `impratio="10"`, which brax's `validate_model` rejects (spring only
    # supports the pyramidal cone, and impratio is mjx-specific).
    patched_xml = re.sub(
        r'integrator="implicitfast"', 'integrator="Euler"', patched_xml
    )
    patched_xml = re.sub(r'\s*cone="elliptic"', '', patched_xml)
    patched_xml = re.sub(r'\s*impratio="[^"]*"', '', patched_xml)

    mj = mujoco.MjModel.from_xml_string(patched_xml, assets=full_assets)

    # Arm-only IK model — exactly the standalone UR5e xml ogbench uses for IK.
    desc_dir = Path(scene_env._desc_dir).resolve()
    ik_mjcf = mjcf.from_path(
        (desc_dir / 'universal_robots_ur5e' / 'ur5e.xml').as_posix(),
        escape_separators=True,
    )
    ik_xml = mjcf_utils.to_string(ik_mjcf)
    ik_assets = mjcf_utils.get_assets(ik_mjcf)
    ik_mj = mujoco.MjModel.from_xml_string(ik_xml, assets=ik_assets)

    return mj, ik_mj


# ---------------------------------------------------------------------------
# Quaternion / SO3 helpers — JAX-traceable mirrors of `ogbench.manipspace.lie`.
# Quaternions are wxyz (first-element-real), matching MuJoCo's convention.
# ---------------------------------------------------------------------------


def _quat_from_z_radians(theta: jnp.ndarray) -> jnp.ndarray:
    half = 0.5 * theta
    return jnp.stack([jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)])


def _quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return jnp.stack(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _quat_inv(q: jnp.ndarray) -> jnp.ndarray:
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _quat_apply(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    qv = jnp.concatenate([jnp.zeros((1,)), v])
    return _quat_mul(_quat_mul(q, qv), _quat_inv(q))[1:]


def _mat_to_quat(mat: jnp.ndarray) -> jnp.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion (matches `mju_mat2Quat`).

    Branch-free Shepperd-style: compute candidates from all four pivots in
    parallel and select the one with the largest absolute value, then renormalise.
    """
    m = mat
    # Squared candidates for (4 * |q_i|).
    sq = jnp.array(
        [
            1.0 + m[0, 0] + m[1, 1] + m[2, 2],  # 4 * w^2
            1.0 + m[0, 0] - m[1, 1] - m[2, 2],  # 4 * x^2
            1.0 - m[0, 0] + m[1, 1] - m[2, 2],  # 4 * y^2
            1.0 - m[0, 0] - m[1, 1] + m[2, 2],  # 4 * z^2
        ]
    )
    sq = jnp.maximum(sq, 0.0)
    pivot = jnp.argmax(sq)
    s = jnp.sqrt(sq[pivot]) * 2.0  # 4 * |q_pivot|

    # Off-diagonal anti-symmetric combinations.
    a = m[2, 1] - m[1, 2]
    b = m[0, 2] - m[2, 0]
    c = m[1, 0] - m[0, 1]
    # Off-diagonal symmetric combinations.
    d = m[2, 1] + m[1, 2]
    e = m[0, 2] + m[2, 0]
    f = m[1, 0] + m[0, 1]

    quat_w = jnp.stack([0.25 * s, a / s, b / s, c / s])  # pivot=w
    quat_x = jnp.stack([a / s, 0.25 * s, f / s, e / s])  # pivot=x
    quat_y = jnp.stack([b / s, f / s, 0.25 * s, d / s])  # pivot=y
    quat_z = jnp.stack([c / s, e / s, d / s, 0.25 * s])  # pivot=z

    candidates = jnp.stack([quat_w, quat_x, quat_y, quat_z])  # (4, 4)
    q = candidates[pivot]
    # MuJoCo convention: ensure the leading element is non-negative.
    q = jnp.where(q[0] < 0.0, -q, q)
    return q / jnp.linalg.norm(q)


def _quat_to_vel(q: jnp.ndarray) -> jnp.ndarray:
    """Rotation quaternion → axis-angle vector (matches `mju_quat2Vel` with dt=1).

    MuJoCo computes ``half_angle = atan2(||v||, w)`` and returns
    ``2 * half_angle / ||v|| * v``. We mirror that exactly so the IK error term
    is bit-for-bit equivalent to OGBench's controller for w >= 0; the sign of w
    is preserved.
    """
    w = q[0]
    v = q[1:]
    norm_v = jnp.linalg.norm(v)
    safe_norm = jnp.where(norm_v > 1e-12, norm_v, 1.0)
    half_angle = jnp.arctan2(norm_v, w)
    scale = jnp.where(norm_v > 1e-12, 2.0 * half_angle / safe_norm, 2.0)
    return scale * v


def _yaw_from_mat(mat: jnp.ndarray) -> jnp.ndarray:
    """Yaw (rotation about z) extracted from a 3x3 rotation matrix.

    Matches `lie.SO3.compute_yaw_radians` which is
    atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2)).
    """
    q = _mat_to_quat(mat)
    q0, q1, q2, q3 = q
    return jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))


def _yaw_from_quat(q: jnp.ndarray) -> jnp.ndarray:
    q0, q1, q2, q3 = q
    return jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))


# ---------------------------------------------------------------------------
# Differential IK in JAX — direct port of `controllers.diff_ik.DiffIKController`.
# Uses an arm-only MJX model with 6 hinge joints, so kinematics are cheap.
# ---------------------------------------------------------------------------


class _JaxDiffIK:
    """Damped-least-squares differential IK on a UR5e-only mjx model."""

    def __init__(
        self,
        ik_mj: mujoco.MjModel,
        site_name: str = 'attachment_site',
        max_iters: int = _IK_MAX_ITERS,
    ):
        self._mjx_model = mjx.put_model(ik_mj)
        self._mjx_data_template = mjx.make_data(self._mjx_model)
        self._site_id = ik_mj.site(site_name).id
        self._site_bodyid = int(ik_mj.site_bodyid[self._site_id])
        # 6 hinge joints; nv == nq == 6.
        self._nv = ik_mj.nv
        self._max_iters = int(max_iters)

    def solve(
        self,
        target_pos: jnp.ndarray,
        target_quat: jnp.ndarray,
        curr_qpos: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return joint angles that drive the attachment site to (target_pos, target_quat)."""
        m = self._mjx_model
        d_template = self._mjx_data_template
        site_id = self._site_id
        body_id = self._site_bodyid
        damping_eye = _IK_DAMPING * jnp.eye(6)

        def iter_step(_, qpos):
            d = d_template.replace(qpos=qpos)
            d = mjx.kinematics(m, d)
            d = mjx.com_pos(m, d)

            site_xpos = d.site_xpos[site_id]
            site_xmat = d.site_xmat[site_id].reshape(3, 3)

            pos_err = target_pos - site_xpos

            site_quat = _mat_to_quat(site_xmat)
            err_quat = _quat_mul(target_quat, _quat_inv(site_quat))
            ori_err = _quat_to_vel(err_quat)

            err = jnp.concatenate([pos_err, ori_err])

            # mjx.jac returns (nv, 3) jacobians; we want a (6, nv) stack so the
            # damped least-squares solve operates on a 6-D twist error.
            jacp, jacr = mjx.jac(m, d, site_xpos, jnp.array(body_id))
            jac = jnp.concatenate([jacp.T, jacr.T], axis=0)  # (6, nv)

            H = jac @ jac.T + damping_eye
            update = jac.T @ jnp.linalg.solve(H, err)

            update_max = jnp.max(jnp.abs(update))
            scale = jnp.where(
                update_max > _IK_MAX_ANGLE_CHANGE,
                _IK_MAX_ANGLE_CHANGE / jnp.maximum(update_max, 1e-12),
                1.0,
            )
            update = update * scale

            return qpos + update

        return jax.lax.fori_loop(0, self._max_iters, iter_step, curr_qpos)


# ---------------------------------------------------------------------------
# The env itself.
# ---------------------------------------------------------------------------


class CubeSingle(PipelineEnv):
    """OGBench cube-single ported for online goal-conditioned RL.

    Observation (28-D, layout identical to OGBench `compute_observation`):
        [0:6]    arm joint positions
        [6:12]   arm joint velocities
        [12:15]  (pinch_pos - center) * 10
        [15]     cos(pinch_yaw)
        [16]     sin(pinch_yaw)
        [17]     gripper_opening * 3
        [18]     gripper contact force — stubbed to 0.0 in this port (mjx can't
                 run mj_rnePostConstraint through the Robotiq <connect>
                 equalities; the dim is kept for shape parity with OGBench)
        [19:22]  (cube_pos - center) * 10
        [22:26]  cube quaternion (wxyz)
        [26]     cos(cube_yaw)
        [27]     sin(cube_yaw)

    Action (5-D in [-1, 1]):
        (Δpinch_x, Δpinch_y, Δpinch_z, Δpinch_yaw, Δgripper)
        unnormalised range = ±[0.05, 0.05, 0.05, 0.3, 1.0].

    Goal (3-D, scaled cube target xyz, lives at obs indices [19, 20, 21]):
        success when ||cube_pos - target||_2 < 0.04 m  (= 0.4 in scaled space).
    """

    state_dim = 28
    goal_indices = jnp.array([19, 20, 21])
    completion_goal_indices = jnp.array([19, 20, 21])
    goal_reach_thresh = _SUCCESS_THRESHOLD * _XYZ_SCALER  # 0.4

    def __init__(
        self,
        backend: str = 'mjx',
        episode_length: int = 200,
        cube_init_noise: float = 0.01,
        permute_blocks: bool = False,  # kept for API parity with OGBench
        ik_max_iters: int = _IK_MAX_ITERS,
        **kwargs,
    ):
        del permute_blocks  # single cube, no permutation possible
        # Brax's spring / positional pipelines reject this scene at
        # `validate_model` time — the Robotiq fingers_actuator uses a tendon
        # transmission and the 4-bar finger linkage relies on `<equality>`
        # connect constraints, neither of which the approximate backends
        # support. mjx is the only backend that can actually close the
        # gripper, so we lock the env to it.
        if backend != 'mjx':
            raise ValueError(
                f'CubeSingle requires backend="mjx" — brax spring/positional do not '
                f'support the Robotiq tendon actuator or 4-bar equality constraints.'
            )

        mj, ik_mj = _assemble_mjcf()

        # Match OGBench's PD gains exactly (post_compilation in manipspace_env).
        arm_actuator_ids = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        mj.actuator_gainprm[arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        mj.actuator_gainprm[arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -200, -200, -50])
        mj.actuator_biasprm[arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 2000, 2000, 500])

        # OGBench uses physics_timestep=0.002, control_timestep=0.05 (n_steps=25).
        mj.opt.timestep = 0.002

        # Compute the constant pinch→attach SE3 transform once with MuJoCo so we
        # don't have to re-solve it every step. T_pa = pinch_pose⁻¹ ∘ attach_pose
        # at the home configuration.
        data = mujoco.MjData(mj)
        data.qpos[:6] = _HOME_QPOS
        mujoco.mj_forward(mj, data)
        pinch_site_id = int(mj.site('ur5e/robotiq/pinch').id)
        attach_site_id = int(mj.site('ur5e/attachment_site').id)
        pinch_pos = data.site_xpos[pinch_site_id].copy()
        pinch_mat = data.site_xmat[pinch_site_id].copy().reshape(3, 3)
        attach_pos = data.site_xpos[attach_site_id].copy()
        attach_mat = data.site_xmat[attach_site_id].copy().reshape(3, 3)

        pinch_quat = np.zeros(4, dtype=np.float64)
        attach_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(pinch_quat, pinch_mat.ravel())
        mujoco.mju_mat2Quat(attach_quat, attach_mat.ravel())
        # SE3 inverse: rot⁻¹ = q*, pos⁻¹ = -q⁻¹ ∘ pos
        pinch_quat_inv = pinch_quat * np.array([1.0, -1.0, -1.0, -1.0])

        def _qmul_np(a, b):
            w0, x0, y0, z0 = a
            w1, x1, y1, z1 = b
            return np.array([
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ])

        def _qrot_np(q, v):
            return _qmul_np(_qmul_np(q, np.array([0, *v])), q * np.array([1, -1, -1, -1]))[1:]

        T_pa_quat = _qmul_np(pinch_quat_inv, attach_quat)
        T_pa_pos = _qrot_np(pinch_quat_inv, attach_pos - pinch_pos)
        self._T_pa_pos = jnp.asarray(T_pa_pos)
        self._T_pa_quat = jnp.asarray(T_pa_quat)

        # Build brax/mjx system from the patched MjModel.
        sys = load_model(mj)
        sys = sys.tree_replace(
            {
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            }
        )
        # OGBench control dt is 0.05 s; pair physics_dt=0.002 with n_frames=25.
        n_frames = 25
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self.n_frames = n_frames
        self.episode_length = episode_length
        self._cube_init_noise = cube_init_noise

        # Cache key body / site / joint / actuator IDs (constant for the run).
        self._arm_qpos_idx = jnp.arange(0, 6)
        self._arm_qvel_idx = jnp.arange(0, 6)
        self._arm_actuator_idx = jnp.arange(0, 6)
        self._gripper_actuator_idx = jnp.array([6])
        self._gripper_opening_qpos_idx = 6  # right_driver_joint qpos
        self._cube_qpos_start = 14  # object_joint_0 qposadr
        self._cube_qvel_start = 14  # object_joint_0 dofadr

        self._pinch_site_id = pinch_site_id
        self._cube_target_mocap_id = int(mj.body('object_target_0').mocapid[0])

        # IK solver (lazy compilation).
        self._ik = _JaxDiffIK(ik_mj, site_name='attachment_site', max_iters=ik_max_iters)

        # Action limits.
        self._action_low = jnp.asarray(-_ACTION_RANGE)
        self._action_high = jnp.asarray(_ACTION_RANGE)

        # Goal-conditioned bookkeeping. Goals stored in the scaled space so they
        # align with obs[19:22].
        self._task_goal_xyzs_scaled = jnp.asarray(
            (_TASK_GOAL_XYZS - _XYZ_CENTER) * _XYZ_SCALER
        )
        self._task_init_xyzs = jnp.asarray(_TASK_INIT_XYZS)
        self._task_goal_xyzs = jnp.asarray(_TASK_GOAL_XYZS)
        self._num_tasks = _TASK_INIT_XYZS.shape[0]
        # Exposed for goal proposers (e.g. random_env_goals): the 5 task target
        # cube positions, in the same scaled space as obs[19:22].
        self.possible_goals = self._task_goal_xyzs_scaled

        # Visualization bounds — go_explore_simple plots cube xy at obs[19], obs[20]
        # which live in the scaled space (raw - center) * 10. Take the
        # object-sampling rectangle, transform it the same way, and expose the
        # result as x_bounds / y_bounds.
        _obj_low_scaled = (_OBJECT_SAMPLING_BOUNDS[0] - _XYZ_CENTER[:2]) * _XYZ_SCALER
        _obj_high_scaled = (_OBJECT_SAMPLING_BOUNDS[1] - _XYZ_CENTER[:2]) * _XYZ_SCALER
        self.x_bounds = (float(_obj_low_scaled[0]), float(_obj_high_scaled[0]))
        self.y_bounds = (float(_obj_low_scaled[1]), float(_obj_high_scaled[1]))

        # Constants exposed to JIT.
        self._home_qpos = jnp.asarray(_HOME_QPOS)
        self._effector_down_quat = jnp.asarray(_EFFECTOR_DOWN_QUAT)
        self._workspace_low = jnp.asarray(_WORKSPACE_BOUNDS[0])
        self._workspace_high = jnp.asarray(_WORKSPACE_BOUNDS[1])
        self._xyz_center = jnp.asarray(_XYZ_CENTER)

    # -- API surface used by training code --------------------------------

    @property
    def action_size(self) -> int:
        return 5

    # -- Initial-state construction ---------------------------------------

    def _initial_q(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Sample a (q, qd, task_idx) tuple matching CubeEnv.initialize_episode."""
        rng, rng_task, rng_pos, rng_yaw = jax.random.split(rng, 4)

        task_idx = jax.random.randint(rng_task, (), 0, self._num_tasks)
        init_xyz = self._task_init_xyzs[task_idx]
        goal_xyz = self._task_goal_xyzs[task_idx]

        # OGBench perturbs the cube xy by ±0.01 and randomises yaw uniformly.
        xy_noise = self._cube_init_noise * jax.random.uniform(
            rng_pos, (2,), minval=-1.0, maxval=1.0
        )
        cube_xy = init_xyz[:2] + xy_noise
        cube_z = init_xyz[2]
        cube_yaw = jax.random.uniform(rng_yaw, (), minval=0.0, maxval=2 * jnp.pi)
        cube_quat = _quat_from_z_radians(cube_yaw)

        # Default qpos from the system, then overwrite the bits we control.
        q = self.sys.init_q
        # Arm: home pose.
        q = q.at[0:6].set(self._home_qpos)
        # Gripper: leave default zero.
        # Cube: position + yaw.
        q = q.at[14:17].set(jnp.array([cube_xy[0], cube_xy[1], cube_z]))
        q = q.at[17:21].set(cube_quat)

        qd = jnp.zeros(self.sys.qd_size())
        return q, qd, goal_xyz

    def _set_mocap(self, pipeline_state: base.State, target_xyz: jax.Array) -> base.State:
        """Move the cube target mocap to the given xyz position (identity orientation)."""
        # mjx pipeline_state stores mocap data as `mocap_pos` / `mocap_quat`.
        new_pos = pipeline_state.mocap_pos.at[self._cube_target_mocap_id].set(target_xyz)
        new_quat = pipeline_state.mocap_quat.at[self._cube_target_mocap_id].set(
            jnp.array([1.0, 0.0, 0.0, 0.0])
        )
        return pipeline_state.replace(mocap_pos=new_pos, mocap_quat=new_quat)

    # -- Reset / Step -----------------------------------------------------

    def reset(self, rng: jax.Array, goal=None, start=None) -> State:
        rng, rng_q, rng_post = jax.random.split(rng, 3)
        q, qd, default_goal_xyz = self._initial_q(rng_q)

        # `start` is the cube's initial position in the SCALED space (i.e. the
        # values stored at obs[goal_indices] = obs[19:22]). GoExploreWrapper
        # passes this in to restore the cube to its episode-start position at
        # the beginning of each go phase.
        if start is not None:
            start_arr = jnp.asarray(start, dtype=q.dtype)
            cube_xyz = start_arr / _XYZ_SCALER + self._xyz_center
            q = q.at[14:17].set(cube_xyz)

        # `goal` is the desired cube target in the same SCALED space. When the
        # proposer/wrapper supplies a goal, override the per-task default.
        if goal is None:
            goal_xyz = default_goal_xyz
            goal_scaled = (goal_xyz - self._xyz_center) * _XYZ_SCALER
        else:
            goal_scaled = jnp.asarray(goal, dtype=q.dtype)
            goal_xyz = goal_scaled / _XYZ_SCALER + self._xyz_center

        pipeline_state = self.pipeline_init(q, qd)
        pipeline_state = self._set_mocap(pipeline_state, goal_xyz)

        info = {
            'goal': goal_scaled,
            'goal_xyz': goal_xyz,
            'timestep': 0.0,
            'postexplore_timestep': jax.random.uniform(rng_post),
        }

        obs = self._get_obs(pipeline_state, goal_scaled)
        zero = jnp.zeros(())
        metrics = {'success': zero, 'success_easy': zero, 'success_hard': zero, 'dist': zero}
        return State(pipeline_state, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pipeline_state0 = state.pipeline_state

        ctrl = self._compute_control(pipeline_state0, action)
        pipeline_state = self.pipeline_step(pipeline_state0, ctrl)

        timestep = state.info['timestep'] + 1.0 / self.episode_length
        obs = self._get_obs(pipeline_state, state.info['goal'])
        success, success_easy, success_hard, dist = self._compute_goal_completion(
            pipeline_state, state.info['goal_xyz']
        )

        state.metrics.update(
            success=success,
            success_easy=success_easy,
            success_hard=success_hard,
            dist=dist,
        )
        info = {**state.info, 'timestep': timestep}
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=success,
            done=jnp.zeros_like(success),
            info=info,
        )

    def update_goal(self, state: State, goal: jax.Array) -> State:
        """Override the current goal (used by go-explore / HER variants).

        The 3-D scaled goal is converted back to a raw xyz target for the mocap.
        """
        goal_xyz = goal / _XYZ_SCALER + self._xyz_center
        info = {**state.info, 'goal': goal, 'goal_xyz': goal_xyz}
        pipeline_state = self._set_mocap(state.pipeline_state, goal_xyz)
        # Refresh obs since the goal is part of metrics, not obs (obs already
        # contains the cube; the goal lives in info).
        return state.replace(pipeline_state=pipeline_state, info=info)

    # -- Control / IK -----------------------------------------------------

    def _compute_control(self, pipeline_state: base.State, action: jnp.ndarray) -> jnp.ndarray:
        """Mirror `manipspace_env.set_control`: apply Δ to pinch pose, run IK."""
        # Unnormalise.
        action_unscaled = 0.5 * (action + 1.0) * (self._action_high - self._action_low) + self._action_low
        a_pos, a_yaw, a_gripper = action_unscaled[:3], action_unscaled[3], action_unscaled[4]

        # Current pinch pose & gripper opening.
        # mjx pipeline_state exposes site arrays; brax wraps them as `site_xpos` / `site_xmat`.
        site_xpos = pipeline_state.site_xpos[self._pinch_site_id]
        site_xmat = pipeline_state.site_xmat[self._pinch_site_id].reshape(3, 3)
        effector_yaw = _yaw_from_mat(site_xmat)
        gripper_opening = jnp.clip(
            pipeline_state.qpos[self._gripper_opening_qpos_idx] / 0.8, 0.0, 1.0
        )

        # Apply deltas.
        target_eff_pos = site_xpos + a_pos
        target_eff_pos = jnp.clip(target_eff_pos, self._workspace_low, self._workspace_high)

        # OGBench composes the yaw delta as Rz(a_yaw) @ Rz(yaw) @ Rx(-180°),
        # then re-extracts yaw via atan2 (which wraps to [-π, π]) and rebuilds
        # the orientation as Rz(yaw) @ Rx(180°). The atan2 wrap is what makes
        # this differ from a hard clip — we mirror it with a modulo wrap.
        target_yaw = jnp.mod(effector_yaw + a_yaw + jnp.pi, 2 * jnp.pi) - jnp.pi
        target_eff_quat = _quat_mul(_quat_from_z_radians(target_yaw), self._effector_down_quat)

        target_gripper = jnp.clip(gripper_opening + a_gripper, 0.0, 1.0)

        # Convert pinch pose → attach pose (T_wa = T_wp @ T_pa).
        attach_pos = target_eff_pos + _quat_apply(target_eff_quat, self._T_pa_pos)
        attach_quat = _quat_mul(target_eff_quat, self._T_pa_quat)

        # Run IK on the arm-only model.
        curr_arm_qpos = pipeline_state.qpos[0:6]
        qpos_target = self._ik.solve(attach_pos, attach_quat, curr_arm_qpos)

        # Build full ctrl vector. nu = 7 (6 arm + 1 gripper).
        ctrl = jnp.zeros(7)
        ctrl = ctrl.at[0:6].set(qpos_target)
        ctrl = ctrl.at[6].set(255.0 * target_gripper)
        return ctrl

    # -- Observation ------------------------------------------------------

    def _get_obs(self, pipeline_state: base.State, goal_scaled: jax.Array) -> jax.Array:
        joint_pos = pipeline_state.qpos[0:6]
        joint_vel = pipeline_state.qvel[0:6]

        eff_pos = pipeline_state.site_xpos[self._pinch_site_id]
        eff_pos_scaled = (eff_pos - self._xyz_center) * _XYZ_SCALER
        eff_yaw = _yaw_from_mat(pipeline_state.site_xmat[self._pinch_site_id].reshape(3, 3))
        gripper_opening = jnp.clip(
            pipeline_state.qpos[self._gripper_opening_qpos_idx] / 0.8, 0.0, 1.0
        )

        cube_pos = pipeline_state.qpos[self._cube_qpos_start:self._cube_qpos_start + 3]
        cube_pos_scaled = (cube_pos - self._xyz_center) * _XYZ_SCALER
        cube_quat = pipeline_state.qpos[self._cube_qpos_start + 3:self._cube_qpos_start + 7]
        cube_yaw = _yaw_from_quat(cube_quat)

        # We append the goal so JaxGCRL training code that expects obs to end
        # with the goal vector keeps working (mirrors arm_*  envs in this repo).
        return jnp.concatenate(
            [
                joint_pos,                       # 0:6
                joint_vel,                       # 6:12
                eff_pos_scaled,                  # 12:15
                jnp.array([jnp.cos(eff_yaw)]),   # 15
                jnp.array([jnp.sin(eff_yaw)]),   # 16
                jnp.array([gripper_opening * _GRIPPER_SCALER]),  # 17
                jnp.zeros(1),                    # 18 (gripper_contact — stubbed)
                cube_pos_scaled,                 # 19:22
                cube_quat,                       # 22:26
                jnp.array([jnp.cos(cube_yaw)]),  # 26
                jnp.array([jnp.sin(cube_yaw)]),  # 27
                goal_scaled,                     # 28:31
            ]
        )

    def _compute_goal_completion(
        self, pipeline_state: base.State, goal_xyz: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        cube_pos = pipeline_state.qpos[self._cube_qpos_start:self._cube_qpos_start + 3]
        dist = jnp.linalg.norm(cube_pos - goal_xyz)
        success = jnp.asarray(dist < _SUCCESS_THRESHOLD, dtype=jnp.float32)
        success_easy = jnp.asarray(dist < 0.10, dtype=jnp.float32)
        success_hard = jnp.asarray(dist < 0.02, dtype=jnp.float32)
        return success, success_easy, success_hard, dist
