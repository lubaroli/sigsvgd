import math

import torch
from src.controllers import DuSt, DISCO
from src.kernels import ScaledGaussianKernel
from src.models import PendulumModel
from src.utils._experiments import run_gym_simulation


def inst_cost(states, controls=None, n_pol=1, debug=None):
    # Note that theta may range beyond 2*pi
    theta, theta_d = states.chunk(2, dim=1)
    control_cost = 0
    if controls is not None:
        control_cost = 0.01 * controls ** 2
    return 100.0 * (theta.cos() - 1) ** 2 + 1.0 * theta_d ** 2 + control_cost


def term_cost(states, n_pol=1, debug=None):
    return inst_cost(states).squeeze()


if __name__ == "__main__":
    # ================================================
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    # ================================================
    PI = math.pi
    ONE_DEG = 2 * PI / 360
    HORIZON = 20
    N_POLICIES = 1
    ACTION_SAMPLES = 0
    PARAMS_SAMPLES = 0
    ALPHA = 1.0
    LEARNING_RATE = 1.0e-1
    CTRL_SIGMA = 0.1
    CTRL_DIM = 1
    PRIOR_SIGMA = 0.1
    SIM_STEPS = 200
    RENDER = True

    # ======================================
    # ========== SIMULATION SETUP ==========
    # ======================================

    # Define environment...
    env = PendulumModel()

    init_state = torch.tensor([PI * 2 / 3, 0])
    init_policies = torch.distributions.Normal(0, PRIOR_SIGMA ** 2).sample(
        [N_POLICIES, HORIZON, CTRL_DIM]
    )

    kernel = ScaledGaussianKernel(bandwidth_fn=lambda *args: CTRL_DIM ** 0.5)

    optimizer = "adam"
    if optimizer.lower() == "sgd":
        # Params for Adam
        opt = torch.optim.SGD
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif optimizer.lower() == "adam":
        # Params for Adam
        opt = torch.optim.Adam
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif optimizer.lower() == "adagrad":
        # Params for Adam
        opt = torch.optim.Adam
        opt_kwargs = {
            "lr": LEARNING_RATE,
        }
    elif optimizer.lower() == "lbfgs":
        # Params for LBFGS
        opt = torch.optim.LBFGS
        opt_kwargs = {
            "lr": LEARNING_RATE,
            "max_iter": 1,
            "line_search_fn": None,
        }
    else:
        raise ValueError("Invalid optimizer: {}".format(optimizer))

    # ========== DUAL SVMPC ==========
    case = "DuSt-MPC"
    print("\nRunning {} simulation:".format(case))
    controller = DuSt(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hz_len=HORIZON,  # control horizon
        n_pol=N_POLICIES,
        n_action_samples=ACTION_SAMPLES,  # sampled trajectories
        n_params_samples=PARAMS_SAMPLES,  # sampled params
        pol_mean=init_policies,
        pol_cov=CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
        pol_hyper_prior=False,
        stein_sampler="SVGD",
        kernel=kernel,
        temperature=ALPHA,  # temperature
        inst_cost_fn=inst_cost,
        term_cost_fn=term_cost,
        optimizer_class=opt,
        **opt_kwargs
    )

    # ========== DISCO ==========
    # case = "DISCO-MPC"
    # print("\nRunning {} simulation:".format(case))
    # controller = DISCO(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     hz_len=HORIZON,  # control horizon
    #     init_policy=init_policies[0],
    #     pol_cov=CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
    #     pol_samples=ACTION_SAMPLES,  # sampled trajectories
    #     temperature=ALPHA,  # temperature
    #     inst_cost_fn=inst_cost,
    #     term_cost_fn=term_cost,
    #     params_sampling=False,
    #     params_samples=PARAMS_SAMPLES,  # sampled params
    # )
    run_gym_simulation(
        "pendulum", init_state, controller, steps=SIM_STEPS, render=RENDER
    )
