from matplotlib import pyplot as plt
from stein_mpc.models.skid_steer_robot import SkidSteerRobot

import torch as th
from tqdm import trange

from stein_mpc.controllers.disco import MultiDISCO
from stein_mpc.inference.likelihoods import ExponentiatedUtility
from stein_mpc.kernels.base_kernels import RBF
from stein_mpc.kernels.composite_kernels import iid_mp
from stein_mpc.inference.svgd import get_gmm
from stein_mpc.inference.svmpc import SVMPC


def state_cost(traj, target_pos, target_speed, *args, **kwargs):
    # state is: x, y, theta, linear speed, angular speed
    sq_dists = (traj[:, :2] - target_pos).pow(2).sum(dim=1, keepdim=True)
    if traj.shape[0] > 1:
        velocity_penalty = (traj[:, 2] - target_speed).pow(2).view(-1, 1)
    else:
        velocity_penalty = 0
    cost = (
        (sq_dists.sqrt() - target_pos).pow(2).sum(dim=1, keepdim=True)
        + (traj[:, 3].view(-1, 1) - target_speed).pow(2).sum(dim=1, keepdim=True)
        + velocity_penalty
    )
    return cost


def run_sim(steps):
    init_state = th.zeros(5)
    target_pos = th.tensor([0.2, 0.4])
    integration_step = 0.02
    model = SkidSteerRobot(integration_step)

    # construct controller
    hz_len = 10
    n_pol = 1
    act_samples = 50
    temperature = 0.01
    max_linear_speed = model.action_space.high[0]
    max_wheel_speed = max_linear_speed / model.params_dict["wheel_radius"]
    actions_std = max_wheel_speed
    actions_cov = th.eye(2) * actions_std ** 2

    def cost_function(x, *args, **kwargs):
        return state_cost(x, target_pos, max_linear_speed, *args, **kwargs)

    controller = MultiDISCO(
        model.observation_space,
        model.action_space,
        hz_len,
        n_pol,
        act_samples,
        pol_cov=actions_cov,
        inst_cost_fn=cost_function,
        term_cost_fn=cost_function,
        params_sampling=False,
        temperature=temperature,
    )

    # construct optimizer
    learning_rate = 1
    pol_sigma = 1
    pol_prior = get_gmm(
        th.randn(n_pol, hz_len, controller.dim_a),
        th.ones(n_pol),
        pol_sigma ** 2 * th.eye(controller.dim_a),
    )
    init_particles = pol_prior.sample([n_pol]) * 0
    likelihood = ExponentiatedUtility(
        1 / temperature, controller=controller, model=model, n_samples=act_samples,
    )
    base_kernel = RBF(bandwidth=-1, minimum_bw=0.01)
    kernel = iid_mp(
        base_kernel=base_kernel, ctrl_dim=controller.dim_a, indep_controls=True,
    )
    svmpc_kwargs = {
        "init_particles": init_particles.detach().clone(),
        "prior": pol_prior,
        "likelihood": likelihood,
        "kernel": kernel,
        "n_particles": n_pol,
        "bw_scale": 1.0,
        "n_steps": 2,
        "optimizer_class": th.optim.SGD,
        "lr": learning_rate,
    }
    svmpc = SVMPC(**svmpc_kwargs)

    # simulation loop
    state = init_state
    trajectory = state.unsqueeze(0)
    rollouts_cache = th.empty(0, 1, act_samples, n_pol, hz_len + 1, state.shape[0])
    costs_cache = th.empty((0, 1))
    actions_cache = th.empty((0, controller.dim_a))
    warm_up = 10
    for step in trange(steps + warm_up):
        # # use these lines for DISCO
        # _, rollouts, _, _, _ = controller.forward(state, model)
        # action = controller.step(strategy="argmax")

        # or use these lines for DSVMPC
        svmpc.optimize(state, params_dist=None)
        a_seq, _ = svmpc.forward(state, params_dist=None)
        if step < warm_up:
            action = th.zeros_like(a_seq[0]).unsqueeze(0)
        else:
            action = a_seq[0].unsqueeze(0)
        rollouts = svmpc.likelihood.last_states.detach().clone()

        state = model.step(state.unsqueeze(0), action, params_dict=None).squeeze()

        # store data
        trajectory = th.cat([trajectory, state.unsqueeze(0)], dim=0)
        actions_cache = th.cat([actions_cache, action])
        rollouts_cache = th.cat([rollouts_cache, rollouts.unsqueeze(0)], dim=0)
        inst_cost = controller.inst_cost_fn(state.view(1, -1))
        costs_cache = th.cat([costs_cache, inst_cost], dim=0)
        if (target_pos - state[:2]).norm() <= 1.0e-3:
            break

    def cost_map(xx, yy, target):
        return (xx - target[0]).pow(2) + (yy - target[1]).pow(2)

    n_pts = 50
    x = th.linspace(init_state[0], target_pos[0] * 1.2, n_pts)
    y = th.linspace(init_state[1], target_pos[1] * 1.2, n_pts)
    xx, yy = th.meshgrid(x, y)
    zz = cost_map(xx, yy, target_pos)
    plt.pcolor(xx, yy, zz)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.scatter(init_state[0], init_state[1])
    plt.scatter(target_pos[0], target_pos[1])
    plt.show()


if __name__ == "__main__":
    th.manual_seed(0)
    run_sim(400)
