import mujoco


def apply_mjx_solver_flags(sys, iterations=1, ls_iterations=4):
    return sys.tree_replace(
        {
            "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
            "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            "opt.iterations": iterations,
            "opt.ls_iterations": ls_iterations,
        }
    )


def ant_initial_metrics(zero):
    return {
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
