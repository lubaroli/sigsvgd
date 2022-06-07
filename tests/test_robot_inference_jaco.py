import math
import pathlib
import sys
from copy import deepcopy

import numpy as np
import tqdm
from tqdm import trange

from stein_mpc.controllers import DuSt
from stein_mpc.kernels import ScaledGaussianKernel
from stein_mpc.models.robot.manipulator_loss import JacoEndEffectorTerminateOnlyLoss
from stein_mpc.models.robot.pybullet_model import (
    PyBullteVisualiser,
    assert_visualiser_and_differentiable_robot_align,
)

THIS_FILE_DIR = pathlib.Path(__file__).parent.resolve()

sys.path.insert(0, f"{THIS_FILE_DIR}/../")

# import diff_robot_data
import torch as th

from stein_mpc.models.robot.differentiable_robot import DifferentiableRobot

# from stein_mpc.utils.background_plotter import BackgroundProcessRunner

th.manual_seed(0)


def run_robot_simulation(
    init_state,
    controller,
    model,
    sim_env,
    steps=200,
    opt_steps=2,
    seed=None,
    visualiser=None,
    target_ee=None,
    debug=False,
):
    if seed is not None:
        th.manual_seed(seed)

    theta_dof = 6

    # create the aggregating tensors and set them to 'nan' to check if
    # simulation breaks
    sim_dict = dict()
    sim_dict["states"] = th.full(
        (steps + 1, controller.dim_s), fill_value=float("nan"), dtype=th.float,
    )
    sim_dict["actions"] = th.full(
        (steps, controller.dim_a), fill_value=float("nan"), dtype=th.float,
    )
    sim_dict["costs"] = th.full((steps, 1), fill_value=float("nan"), dtype=th.float)
    if hasattr(controller, "n_pol"):
        pol_shape = [steps, controller.n_pol]
    else:
        pol_shape = [steps]
    sim_dict["policies"] = th.full(
        pol_shape + [controller.hz_len, controller.dim_a],
        fill_value=float("nan"),
        dtype=th.float,
    )
    sim_dict["trace"] = []
    # main iteration loop
    # if env_name.lower() == "pendulum":
    #     # model = PendulumModel()
    #     # sim_env = gym.make("Pendulum-v1")
    #     # sim_env.reset()
    #     # sim_env.unwrapped.l = experiment_params[i]["length"]
    #     # sim_env.unwrapped.m = experiment_params[i]["mass"]
    #     sim_env.unwrapped.state = init_state
    # else:
    #     raise ValueError("Invalid environment name: {}.".format(env_name))
    state = init_state
    sim_dict["states"][0] = init_state

    if visualiser:
        visualiser.set_target_joints_states(state[:theta_dof])
    if debug and visualiser:
        assert_visualiser_and_differentiable_robot_align(
            state[:theta_dof], sim_env, visualiser
        )

    with tqdm.trange(steps, desc="SimSteps") as pbar:
        for step in pbar:

            a_seq, trace = controller.forward(
                state, model, params_dist=None, steps=opt_steps, debug=debug
            )
            action = a_seq[0]
            state = sim_env.step(state.unsqueeze(0), action.squeeze()).squeeze(0)
            cost = controller.term_cost_fn(state.view(1, -1))
            ee = sim_env.robot_model.compute_forward_kinematics(
                state[:theta_dof], target_ee
            )

            if visualiser:
                visualiser.set_target_joints_states(state[:theta_dof])
            if debug and visualiser:
                assert_visualiser_and_differentiable_robot_align(
                    state[:theta_dof], sim_env, visualiser
                )

            if step % 10 == 0:
                pbar.write(
                    "Step {0}: action taken {1}, cost {2:.2f}".format(
                        step, action, float(cost.mean())
                    )
                )
                pbar.write("Current state: theta={0}".format(state.squeeze()))
                pbar.write("Current ee={0}".format(ee))

            sim_dict["policies"][step] = controller.pol_mean.detach().clone()
            sim_dict["trace"].append(trace)
            sim_dict["actions"][step] = action
            sim_dict["costs"][step] = cost
            sim_dict["states"][step + 1] = state

    # End of episode
    # sim_env.close()
    return sim_dict


def setup_and_run():
    # ================================================
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    # ================================================
    PI = math.pi
    ONE_DEG = 2 * PI / 360
    # HORIZON = 20
    HORIZON = 5
    N_POLICIES = 1
    ACTION_SAMPLES = 0
    PARAMS_SAMPLES = 0
    ALPHA = 1.0
    LEARNING_RATE = 1.0e-1
    CTRL_SIGMA = 0.1
    # CTRL_DIM = 1
    PRIOR_SIGMA = 0.1
    SIM_STEPS = 200
    RENDER = True
    OPTIMISER = "adam"
    OPT_STEPS = 10

    # ACTION_SAMPLES = 5
    SIM_STEPS = 10000
    HORIZON = 10
    OPT_STEPS = 2

    n_particles = 100

    TARGET_LINK = "j2n6s300_end_effector"
    urdf_path = f"{THIS_FILE_DIR}/../jaco_data/urdf/jaco_without_finger.urdf"

    # ======================================
    # ========== SIMULATION SETUP ==========
    # ======================================

    # Define environment...

    # model = DifferentiableRobot(urdf_path, uncertain_params=["j2n6s300_link_6__mass"])
    model = DifferentiableRobot(urdf_path)
    system = DifferentiableRobot(urdf_path)
    # system = DifferentiableRobot(
    #     urdf_path, link_param_mappings={"j2n6s300_link_6__mass": th.tensor(1.8), },
    #     # urdf_path, link_param_mappings={"j2n6s300_link_6__mass": th.tensor(3.2), },
    # )

    system.print_learnable_params()

    # init_state = th.zeros(model.dof * 2)
    init_state = th.tensor([0, np.pi, np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0.0])
    # init_particles = (
    #     dist.Normal(loc=th.tensor(1.0), scale=th.tensor(0.9))
    #         .sample([n_particles])
    #         .unsqueeze(-1)
    # )
    init_policies = th.distributions.Normal(0, PRIOR_SIGMA ** 2).sample(
        [N_POLICIES, HORIZON, model.dof]
    )
    # likelihood = GaussianLikelihood(
    #     th.zeros(model.dof * 2, ), 0.1, model, log_space=True,
    # )

    kernel = ScaledGaussianKernel(bandwidth_fn=lambda *args: model.dof ** 0.5)

    if OPTIMISER.lower() == "sgd":
        # Params for Adam
        opt = th.optim.SGD
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif OPTIMISER.lower() == "adam":
        # Params for Adam
        opt = th.optim.Adam
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif OPTIMISER.lower() == "adagrad":
        # Params for Adam
        opt = th.optim.Adam
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif OPTIMISER.lower() == "lbfgs":
        # Params for LBFGS
        opt = th.optim.LBFGS
        opt_kwargs = {
            "lr": LEARNING_RATE,
            "max_iter": 1,
            "line_search_fn": None,
        }
    else:
        raise ValueError("Invalid optimizer: {}".format(OPTIMISER))

    # loss_func = JacoJointsLoss(
    #     # target_q=th.tensor([2,0,2,2,2,2]),
    #     target_q=th.tensor([0, 2, 2, 2, 2, 2]),
    #     # target_q=th.tensor([1,1,-1,1,1,0]),
    #     mul_factor_q=8000,
    # )
    loss_func = JacoEndEffectorTerminateOnlyLoss(
        differentiable_robot=system,
        # target_ee_pos=th.Tensor([0.5,0.5,0.5]),
        target_ee_pos=th.Tensor([-0.4, -0.3, 0.98]),
        target_ee_name=TARGET_LINK,
    )

    # ========== DUAL SVMPC ==========
    case = "DuSt-MPC"
    print("\nRunning {} simulation:".format(case))
    print(system.observation_space)
    print(system.action_space)

    controller = DuSt(
        observation_space=system.observation_space,
        action_space=system.action_space,
        hz_len=HORIZON,  # control horizon
        n_pol=N_POLICIES,
        n_action_samples=ACTION_SAMPLES,  # sampled trajectories
        n_params_samples=PARAMS_SAMPLES,  # sampled params
        pol_mean=init_policies,
        pol_cov=0.1 ** 2 * th.eye(model.dof),
        pol_hyper_prior=False,
        stein_sampler="SVGD",
        kernel=kernel,
        temperature=ALPHA,  # temperature
        inst_cost_fn=loss_func.inst_cost,
        term_cost_fn=loss_func.term_cost,
        optimizer_class=opt,
        **opt_kwargs,
    )

    bullet_visualiser = PyBullteVisualiser(urdf_path)
    # denote the state joints idx
    bullet_visualiser.set_target_joints([1, 2, 3, 4, 5, 6])

    run_robot_simulation(
        deepcopy(init_state),
        controller,
        model,
        system,
        steps=SIM_STEPS,
        opt_steps=OPT_STEPS,
        seed=None,
        visualiser=bullet_visualiser,
        target_ee=TARGET_LINK,
        debug=True,
    )


if __name__ == "__main__":
    setup_and_run()
