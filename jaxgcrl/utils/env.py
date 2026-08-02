import csv
import logging
import math
import os
from datetime import datetime
from typing import List

import jax
import wandb_osh
from brax.io import html
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from jaxgcrl.envs.ant import Ant
from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_ball_maze import AntBallMaze
from jaxgcrl.envs.ant_ball_ogbench import AntBallOGBench
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.ant_push import AntPush
from jaxgcrl.envs.half_cheetah import Halfcheetah
from jaxgcrl.envs.humanoid import Humanoid
from jaxgcrl.envs.humanoid_maze import HumanoidMaze
from jaxgcrl.envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from jaxgcrl.envs.manipulation.arm_binpick_hard import ArmBinpickHard
from jaxgcrl.envs.manipulation.arm_grasp import ArmGrasp
from jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy
from jaxgcrl.envs.manipulation.arm_push_hard import ArmPushHard
from jaxgcrl.envs.manipulation.arm_reach import ArmReach
from jaxgcrl.envs.manipulation.cube_single import CubeSingle
from jaxgcrl.envs.pointmaze_ogbench_teleport import (
    PointMazeOGBenchTeleport,
    TELEPORT_MAP_1G,
    TASK_PAIRS_1G,
)
from jaxgcrl.envs.pusher import Pusher, PusherReacher
from jaxgcrl.envs.pusher2 import Pusher2
from jaxgcrl.envs.reacher import Reacher
from jaxgcrl.envs.simple_maze import SimpleMaze

legal_envs = (
    "ant",
    "ant_random_start",
    "ant_ball",
    "ant_push",
    "humanoid",
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "arm_reach",
    "arm_grasp",
    "arm_push_easy",
    "arm_push_hard",
    "arm_binpick_easy",
    "arm_binpick_hard",
    "cube_single",
    "ant_ball_maze",
    "ant_ball_square_maze",
    "ant_ball_easy_square_maze",
    "ant_ball_small_square_maze",
    "ant_ball_ogbench_arena",
    "ant_ball_ogbench_medium",
    "ant_ball_ogbench_small_square",
    "ant_ball_ogbench_easy_square",
    "ant_ball_ogbench_small_easy_square",
    "ant_ball_ogbench_small_easy_square_1g",
    "ant_ball_ogbench_small_square_1g",
    "ant_ball_4d_ogbench_small_easy_square",
    "ant_ball_4d_ogbench_small_easy_square_stitch",
    "ant_ball_4d_ogbench_small_easy_square_1g",
    "ant_ball_4d_ogbench_small_square_1g_stitch",
    "ant_ball_4d_ogbench_small_square_1g",
    "ant_ball_4d_ogbench_arena",
    "ant_ball_ogbench_arena_1g",
    "ant_ball_4d_ogbench_arena_1g",
    "ant_ball_4d_ogbench_arena_1g_scale2",
    "ant_ball_4d_ogbench_arena_1g_scale2_stitch",
    "ant_ball_4d_ogbench_arena_1g_scale2_stitch_slice50",
    "ant_ball_4d_ogbench_small_easy_square_1g_scale2",
    "ant_ball_4d_ogbench_arena_1g_scale1",
    "ant_ball_ogbench_arena_stitch",
    "ant_ball_4d_ogbench_medium",
    "ant_ball_4d_ogbench_small_square",
    "ant_ball_4d_ogbench_easy_square",
    "ant_ball_4d_medium_stitch",
    "ant_maze_ogbench_arena",
    "ant_maze_ogbench_medium_navigate",
    "ant_maze_ogbench_medium_explore",
    "ant_maze_ogbench_medium_stitch",
    "ant_maze_ogbench_medium_stitch_slice50",
    "ant_maze_ogbench_medium_1g",
    "ant_maze_ogbench_u",
    "pointmaze_ogbench_teleport",
    "pointmaze_ogbench_teleport_1g",
    "ant_u_maze",
    "ant_u_maze_hard",
    "ant_big_maze",
    "ant_big_maze_hard",
    "ant_cross_maze",
    "ant_cross_maze_hard",
    "ant_hardest_maze",
    "humanoid_u_maze",
    "humanoid_big_maze",
    "humanoid_hardest_maze",
    "humanoidmaze_ogbench_giant_stitch",
    "simple_u_maze",
    "simple_big_maze",
    "simple_hardest_maze",
)


def create_env(env_name: str, backend: str = None, **kwargs) -> object:
    """
    This function creates and returns an appropriate environment object based on the specified environment name and
    backend.

    Args:
        env_name (str): Name of the environment.
        backend (str): Backend to be used for the environment.

    Returns:
        object: The instantiated environment object.

    Raises:
        ValueError: If the specified environment name is unknown.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring")
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_ball_4d_medium_stitch":
        # OGBench antsoccer-medium-stitch task: ant resets at the bottom-left
        # cell, ball spawns anywhere on the open medium maze, and the goal is
        # any open cell except the reset square. 4D goal = (ant_xy, ball_xy).
        env = AntBallOGBench(
            backend=backend or "mjx",
            maze_layout_name="medium_stitch",
            add_ant_to_goal=True,
        )
    elif env_name.startswith("ant_ball_4d_ogbench_"):
        # 4D-goal variant: goal = [ant_x, ant_y, ball_x, ball_y] where both
        # ant and ball must reach the same G cell for default goals.
        layout = env_name[len("ant_ball_4d_ogbench_"):]
        # "_stitch" / "_stitch_slice50" are dataset/categorization tags only —
        # they map to the same physical layout as the base name.
        if layout.endswith("_slice50"):
            layout = layout[: -len("_slice50")]
        if layout.endswith("_stitch"):
            layout = layout[: -len("_stitch")]
        # "_scale2" layouts use a 16x16 grid at maze_size_scaling=2.0 (half the
        # default cell size). "_scale1" layouts use a 32x32 grid at
        # maze_size_scaling=1.0 (quarter cell size) so the ant and ball start
        # right next to each other. The OGBench-anchored offset is computed
        # inside AntBallOGBench from the scaling, so the world frame still
        # matches the scaling-4 originals ([-6, 26]); only the ant->ball->goal
        # gaps shrink (4 units at scale4, 2 at scale2, 1 at scale1).
        if layout.endswith("_scale2"):
            maze_size_scaling = 2.0
        elif layout.endswith("_scale1"):
            maze_size_scaling = 1.0
        else:
            maze_size_scaling = 4.0
        env = AntBallOGBench(
            backend=backend or "mjx",
            maze_layout_name=layout,
            maze_size_scaling=maze_size_scaling,
            add_ant_to_goal=True,
        )
    elif env_name.startswith("ant_ball_ogbench_"):
        # 2D-goal variant (ball-only goal). Includes ant_ball_ogbench_arena,
        # ant_ball_ogbench_medium, and ant_ball_ogbench_arena_stitch (which
        # mirrors OGBench's antsoccer-arena tasks: 5 paired (agent, ball,
        # goal) tuples and ball-only success).
        layout = env_name[len("ant_ball_ogbench_"):]
        env = AntBallOGBench(backend=backend or "mjx", maze_layout_name=layout)
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        # Order matters: "humanoid" must be matched before "ant" because the
        # humanoidmaze layout names contain "giant" (which contains "ant").
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "humanoid" in env_name:
            # Possible env_name:
            #   'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'
            #     -> legacy Brax humanoid + humanoid_maze.xml (spring backend)
            #   'humanoidmaze_ogbench_giant_stitch'
            #     -> OGBench-aligned DMC humanoid + humanoid_maze_ogbench.xml,
            #        69-dim obs that matches the humanoidmaze-giant-stitch
            #        dataset. Defaults to `mjx` for parity with the OGBench
            #        offline dataset (mujoco-based); pass `--backend spring`
            #        if you need to trade dataset parity for memory.
            if env_name.startswith("humanoidmaze_"):
                layout = env_name[len("humanoidmaze_"):]
                env = HumanoidMaze(
                    backend=backend or "mjx", maze_layout_name=layout
                )
            else:
                layout = env_name[len("humanoid_"):]
                env = HumanoidMaze(
                    backend=backend or "spring", maze_layout_name=layout
                )
        elif "ant" in env_name:
            # Any "ogbench"-tagged layout loads the ogbench-aligned XML and
            # defaults to mjx so physics match the OGBench offline dataset
            # (timestep=0.02, gear=30). Legacy layouts stay on spring.
            layout = env_name[4:]
            # The "_explore" / "_stitch" / "_stitch_slice50" variants reuse the
            # medium maze layout; only the offline dataset (and hence which
            # skill checkpoint is paired with the run) differs.
            if layout in (
                "maze_ogbench_medium_explore",
                "maze_ogbench_medium_stitch",
                "maze_ogbench_medium_stitch_slice50",
            ):
                layout = "maze_ogbench_medium_navigate"
            if "ogbench" in layout:
                env = AntMaze(backend=backend or "mjx", maze_layout_name=layout)
            else:
                env = AntMaze(backend=backend or "spring", maze_layout_name=layout)
        elif env_name == "pointmaze_ogbench_teleport":
            # OGBench-aligned 2D point teleport maze. Defaults to mjx so the
            # contact resolution after the qpos override matches OGBench's
            # mj_step semantics that produced pointmaze-teleport-navigate-v0.
            env = PointMazeOGBenchTeleport(backend=backend or "mjx")
        elif env_name == "pointmaze_ogbench_teleport_1g":
            # Same teleport maze geometry but with a dedicated 1-goal map and
            # task-pair set (left R/G column only: start (7,1) -> goal (4,1)).
            env = PointMazeOGBenchTeleport(
                backend=backend or "mjx",
                teleport_map=TELEPORT_MAP_1G,
                task_pairs=TASK_PAIRS_1G,
            )
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    elif env_name == "cube_single":
        env = CubeSingle(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


class MetricsRecorder:
    """
    Initialize the MetricsRecorder with the specified number of timesteps
    and the metrics to be collected.

    Parameters:
    metrics_to_collect (List[str]): List of metric names that are to be collected.
    exp_dir (str): Directory to save renders to.
    exp_name (str): Experiment name for naming rendered trajectory visualizations.
    """

    def __init__(
        self,
        metrics_to_collect: List[str],
        exp_dir,
        exp_name,
        mode,
    ):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]
        self.metrics_to_collect = metrics_to_collect
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.mode = mode

        if mode == "offline":
            wandb_osh.set_log_level("ERROR")
        self.trigger_sync = TriggerWandbSyncHook()

        self._csv_path = None
        self._csv_columns = None

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(int(num_steps))

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self, media=None):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        # Non-scalar wandb objects (histograms / images / html) are merged into
        # the SAME log call so we only ever emit one wandb.log per step. Logging
        # them separately at the same step collides with wandb's monotonic-step
        # semantics and silently drops this scalar row online.
        if media:
            data_to_log.update(media)
        wandb.log(data_to_log, step=self.x_data[-1])

        if self.mode == "offline":
            self.trigger_sync()

    def log_csv(self):
        run_dir = wandb.run.dir if wandb.run is not None else self.exp_dir
        if run_dir is None:
            return
        if self._csv_path is None:
            self._csv_path = os.path.join(run_dir, "metrics.csv")
            self._csv_columns = ["step"] + list(self.metrics_to_collect)
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_columns)

        row = [self.x_data[-1]]
        for key in self.metrics_to_collect:
            values = self.y_data.get(key)
            row.append(values[-1] if values else "")
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def print_progress(self):
        for key, y_values in self.y_data.items():
            logging.info(
                f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}"
            )

    def progress(self, num_steps, metrics, make_policy, params, env, do_render=True):
        # Optional non-scalar wandb media (e.g. the skill controller's histogram /
        # renders) ride along under a reserved key so they log in the single
        # log_wandb call rather than a separate, step-colliding wandb.log.
        media = metrics.pop("_wandb_media", None)
        for key in self.metrics_to_collect:
            self.ensure_metric(metrics, key)

        if do_render:
            render(make_policy, params, env, self.exp_dir, self.exp_name, num_steps)

        self.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in self.metrics_to_collect},
        )
        self.log_wandb(media)
        self.log_csv()
        self.print_progress()

    @staticmethod
    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is NaN in metrics: {metrics}")


def render(make_policy, params, env, exp_dir, exp_name, num_steps):
    """
    Renders a given environment over a series of steps and stores the resulting
    HTML file to a specified directory. Logs the rendered HTML using wandb.

    This function initializes the environment and the inference function, then
    runs the environment for a fixed number of steps, periodically resetting.
    It collects the state of the environment at each step, renders the HTML,
    stores the result, and logs it.

    Parameters:
    inf_fun_factory : Callable
        A factory function that returns an inference function when provided with
        'params'.
    params : any
        Parameters for the 'inf_fun_factory'.
    env : object
        The environment object with 'reset', 'step', and 'sys.tree_replace' methods.
    exp_dir : str
        The directory where the rendered HTML file will be saved.
    exp_name : str
        The file name to be used for the saved HTML (without extension).
    num_steps : int
        The number of environment steps taken so far (used for naming the file).

    Returns:
    None
    """
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rollout = []
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key)
    state = jit_env_reset(rng=subkey)
    for i in range(5000):
        rollout.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = jit_policy(state.obs[None], subkey)  # Policy requires batched dimension
        action = action[0]  # Remove batch dimension
        state = jit_env_step(state, action)
        if i % 1000 == 0:
            key, subkey = jax.random.split(key)
            state = jit_env_reset(rng=subkey)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    if exp_dir is not None:
        with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
            file.write(url)
    wandb.log({"render": wandb.Html(url)}, step=int(num_steps))
