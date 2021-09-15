import math

import torch
import gym
from stein_mpc.controllers import DuSt
from stein_mpc.kernels import ScaledGaussianKernel
from stein_mpc.models import PendulumModel


def inst_cost(states, controls=None, n_pol=1, debug=None):
    # Note that theta may range beyond 2*pi
    theta, theta_d = states.chunk(2, dim=1)
    return 50.0 * (theta.cos() - 1) ** 2 + 1.0 * theta_d ** 2


def term_cost(states, n_pol=1, debug=None):
    return inst_cost(states).squeeze()


if __name__ == "__main__":
    # ================================================
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    # ================================================
    PI = math.pi
    ONE_DEG = 2 * PI / 360
    HORIZON = 15
    N_POLICIES = 10
    ACTION_SAMPLES = 50
    PARAMS_SAMPLES = 5
    ALPHA = 1.0
    LEARNING_RATE = 0.05
    CTRL_SIGMA = 1.0
    CTRL_DIM = 1
    PRIOR_SIGMA = 1.0
    SIM_STEPS = 200
    RENDER = True

    # ======================================
    # ========== SIMULATION SETUP ==========
    # ======================================

    # Define environment...
    env = PendulumModel()

    init_state = torch.tensor([PI / 2, 0])
    init_policies = torch.distributions.Normal(0, PRIOR_SIGMA ** 2).sample(
        [N_POLICIES, HORIZON, CTRL_DIM]
    )

    # fix kernel bandwidth
    kernel = ScaledGaussianKernel(lambda x: 1.0)
    # ... or use median heuristic
    # kernel = ScaledGaussianKernel()
    optimizer = torch.optim.LBFGS
    opt_kwargs = {
        "lr": 0.5,
        "max_iter": 1,
        "line_search_fn": None,
    }

    # ========== DUAL SVMPC ==========
    case = "DuSt-MPC"
    print("\nRunning {} simulation:".format(case))
    controller = DuSt(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hz_len=HORIZON,  # control horizon
        n_pol=N_POLICIES,
        init_pol_mean=init_policies,
        pol_cov=CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
        pol_samples=ACTION_SAMPLES,  # sampled trajectories
        stein_sampler="ScaledSVGD",
        kernel=kernel,
        temperature=1 / ALPHA,  # temperature
        inst_cost_fn=inst_cost,
        term_cost_fn=term_cost,
        params_sampling=False,
        params_samples=PARAMS_SAMPLES,  # sampled params
        optimizer_class=optimizer,
        **opt_kwargs
    )

    # ===== SIM LOOP =====

    # create the aggregating tensors and set them to 'nan' to check if
    # simulation breaks
    states = torch.full(
        (SIM_STEPS, controller.dim_s), fill_value=float("nan"), dtype=torch.float,
    )
    actions = torch.full(
        (SIM_STEPS, controller.dim_a), fill_value=float("nan"), dtype=torch.float,
    )
    costs = torch.full((SIM_STEPS, 1), fill_value=float("nan"), dtype=torch.float)
    pol_particles = torch.full(
        (SIM_STEPS, controller.n_pol, controller.hz_len, controller.dim_a),
        fill_value=float("nan"),
        dtype=torch.float,
    )
    weights = torch.full(
        (SIM_STEPS, controller.n_pol), fill_value=float("nan"), dtype=torch.float,
    )

    # main iteration loop
    sim_env = gym.make("Pendulum-v0")
    sim_env.reset()
    # sim_env.unwrapped.l = experiment_params[i]["length"]
    # sim_env.unwrapped.m = experiment_params[i]["mass"]
    sim_env.unwrapped.state = init_state
    state = init_state
    for step in range(SIM_STEPS):
        if RENDER:
            sim_env.render()

        a_seq, p_weights = controller.forward(state, env, params_dist=None, steps=2)
        action = a_seq[0]
        _, _, done, _ = sim_env.step(action)
        state = torch.as_tensor(sim_env.state, dtype=torch.float).unsqueeze(0)
        cost = controller.inst_cost_fn(state.view(1, -1))

        if not step % 10:
            print(
                "Step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                    step, float(action), float(cost)
                )
            )
            print(
                "Current state: theta={0[0]}, theta_dot={0[1]}".format(state.squeeze())
            )
        pol_particles[step] = controller.pol_mean.detach().clone()
        weights[step] = p_weights
        actions[step] = action
        states[step] = state
        costs[step] = cost
        if done:
            sim_env.close()
            break

    # End of episode
    sim_env.close()
