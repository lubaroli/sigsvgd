from copy import copy, deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import yaml
from stein_mpc.controllers import DISCO, DuSt
from stein_mpc.inference import likelihoods
from stein_mpc.inference.mpf import MPF
from stein_mpc.kernels import GaussianKernel, ScaledGaussianKernel
from stein_mpc.models import ParticleModel
from stein_mpc.utils.helper import create_video_from_plots, save_progress
from stein_mpc.utils.math import to_gmm

from tqdm import trange


def main(sim_params, exp_params, env_params):
    # ========== SIMULATION HYPERPARAMETERS ==========
    WARM_UP = sim_params["warm_up"]
    STEPS = sim_params["steps"]
    EPISODES = sim_params["episodes"]
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    HORIZON = exp_params["horizon"]
    N_PARTICLES = exp_params["n_particles"]
    ACTION_SAMPLES = exp_params["action_samples"]
    PARAMS_SAMPLES = exp_params["params_samples"]
    ALPHA = exp_params["alpha"]
    LEARNING_RATE = exp_params["learning_rate"]
    BANDWIDTH_SCALE = exp_params["bandwidth_scaling"]
    CTRL_SIGMA = exp_params["ctrl_sigma"]
    CTRL_DIM = exp_params["ctrl_dim"]
    LIKELIHOOD = exp_params["likelihood"]
    USE_SVMPC = exp_params["use_svmpc"]
    USE_MPF = exp_params["use_mpf"]
    PRIOR_SIGMA = exp_params["prior_sigma"]
    WEIGHTED_PRIOR = exp_params["weighted_prior"]
    DYN_PRIOR = exp_params["dyn_prior"]
    DYN_PRIOR_ARG1 = exp_params["dyn_prior_arg1"]
    DYN_PRIOR_ARG2 = exp_params["dyn_prior_arg2"]
    LOAD = exp_params["extra_load"]
    SAMPLING = exp_params["sampling"]
    # ========== Experiment Setup ==========
    # Initial state
    state = torch.as_tensor(env_params["init_state"]).clone()
    policies_prior = to_gmm(
        torch.randn(N_PARTICLES, HORIZON, CTRL_DIM),
        torch.ones(N_PARTICLES),
        PRIOR_SIGMA ** 2 * torch.eye(CTRL_DIM),
    )
    init_policies = policies_prior.sample([N_PARTICLES])
    dynamics_prior = getattr(dist, DYN_PRIOR)(DYN_PRIOR_ARG1, DYN_PRIOR_ARG2)

    system_kwargs = {
        "uncertain_params": ["mass"],
        "mass": dynamics_prior.mean,
    }
    # Model is used for the internal rollouts, system is the simulator,
    # they may have different parameters.
    base_system = ParticleModel(**env_params, **system_kwargs)
    base_model = ParticleModel(**env_params, **system_kwargs)

    kernel_type = exp_params["kernel"]
    if kernel_type == "rbf":
        kernel = ScaledGaussianKernel()
    elif kernel_type == "rbf_fixed_bw":
        kernel = ScaledGaussianKernel(
            bandwidth_fn=lambda *args: (CTRL_DIM + HORIZON) ** 0.5
        )
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))

    opt = torch.optim.LBFGS
    opt_kwargs = {
        "lr": LEARNING_RATE,
    }

    controller = DuSt(
        observation_space=base_model.observation_space,
        action_space=base_model.action_space,
        hz_len=HORIZON,
        n_pol=N_PARTICLES,
        n_pol_samples=ACTION_SAMPLES,
        n_params_samples=PARAMS_SAMPLES,
        pol_mean=init_policies,
        pol_cov=torch.eye(CTRL_DIM) * CTRL_SIGMA ** 2,
        pol_hyper_prior=True,
        stein_sampler="ScaledSVGD",
        kernel=kernel,
        temperature=ALPHA,
        params_sampling=False,
        params_log_space=exp_params["mpf_log_space"],
        inst_cost_fn=base_model.default_inst_cost,
        term_cost_fn=base_model.default_term_cost,
        optimizer_class=opt,
        **opt_kwargs
    )

    mpf_n_part = exp_params["mpf_n_particles"]
    mpf_steps = exp_params["mpf_steps"]
    mpf_log_space = exp_params["mpf_log_space"]
    mpf_bw = exp_params["mpf_bandwidth"]
    mpf_init = dynamics_prior.sample([mpf_n_part, 1]).clamp(min=1e-6)
    mpf_init = mpf_init.log() if mpf_log_space else mpf_init
    dynamics_lik = likelihoods.GaussianLikelihood(
        initial_obs=state,
        obs_std=exp_params["mpf_obs_std"],
        model=base_model,
        log_space=exp_params["mpf_log_space"],
    )
    base_mpf = MPF(
        init_particles=mpf_init,
        kernel=GaussianKernel(),
        likelihood=dynamics_lik,
        optimizer_class=torch.optim.SGD,
        lr=exp_params["mpf_learning_rate"],
        bw=(2 * exp_params["dyn_prior_arg2"]) ** 1 / 2,
        bw_scale=exp_params["mpf_bandwidth_scaling"],
    )

    base_path = save_progress(params=config_data)
    # x_lim = system_kwargs["mass"] + 2 + LOAD
    x_lim = 5
    lin_s = torch.linspace(0, x_lim, 100)
    log_s = lin_s.log()

    # ===== Experiment Loop =====
    for ep in range(EPISODES):
        # Reset state
        state = torch.as_tensor(env_params["init_state"]).clone()
        tau = state.unsqueeze(0)
        rollouts = torch.empty(
            0,
            controller.params_samples,
            controller.pol_samples,
            controller.n_pol,
            controller.hz_len + 1,
            state.shape[0],
        )
        costs = torch.empty((0, 1))
        actions = torch.empty((0, CTRL_DIM))
        dyn_particles = torch.empty((0, mpf_n_part))
        iterator = trange(STEPS)
        system = deepcopy(base_system)
        model = deepcopy(base_model)
        if USE_MPF:
            mpf = copy(base_mpf)
            dynamics_dist = mpf.prior
        else:
            mpf = None
            dynamics_dist = dynamics_prior
        controller = deepcopy(controller)
        save_path = save_progress(folder_name=base_path.stem + "/ep{}".format(ep))
        for step in iterator:
            # if step == STEPS // 4:  # Changes the simulator mass
            #     system.params_dict["mass"] = system.params_dict["mass"].clone() + LOAD
            # Rollout dynamics, get costs
            # ----------------------------
            # selects next action and makes sure the controller plan is
            # the same as svmpc
            a_seq, history = controller.forward(state, model, params_dist=None, steps=2)
            # fetch trajectories of last iteration
            s_seq = history[-1]["trajectories"]
            action = a_seq[0]
            state = system.step(state, action.squeeze())
            grad_dyn = torch.zeros(mpf_steps)
            if step >= WARM_UP and mpf is not None:
                # optimize will automatically update `dynamics_dist`
                grad_dyn, _ = mpf.optimize(
                    action.squeeze(), state, bw=mpf_bw, n_steps=mpf_steps
                )

            tau = torch.cat([tau, state.unsqueeze(0)], dim=0)
            actions = torch.cat([actions, action.unsqueeze(0)])
            rollouts = torch.cat([rollouts, s_seq.unsqueeze(0)], dim=0)
            inst_cost = controller.inst_cost_fn(state.view(1, -1))
            costs = torch.cat([costs, inst_cost.unsqueeze(0)], dim=0)
            system.render(
                path=save_path / "plots/{0:03d}.png".format(step),
                states=tau[:, :2],
                rollouts=s_seq.flatten(0, 1),  # flattens actions and params samples
            )
            plt.close()
            if USE_MPF:
                dyn_particles = torch.cat(
                    [dyn_particles, mpf.x.detach().clone().t()], dim=0
                )

            if system.with_obstacle:
                if system.obst_map.get_collisions(state[:2]):
                    print("\nCrashed at step {}".format(step))
                    # print("Last particles state:")
                    # print(svmpc.theta)
                    break
            if (system.target - state).norm() <= 1.0:
                break

        episode_data = {
            "costs": costs,
            "trajectory": tau,
            "actions": actions,
            "rollouts": rollouts,
            "dyn_particles": dyn_particles,
        }
        save_progress(
            folder_name=save_path.relative_to(base_path.parent), data=episode_data
        )
        create_video_from_plots(save_path)


if __name__ == "__main__":
    config_file = Path("tests/particle_config.yaml")
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

    sim_params = {}
    try:
        sim_params = config_data["sim_params"]
    except KeyError:
        print("Invalid key for simulation params!")

    exp_params = {}
    try:
        exp_params = config_data["exp_params"]
    except KeyError:
        print("Invalid key for experiment params!")

    env_params = {}
    try:
        env_params = config_data["env_params"]
    except KeyError:
        print("Invalid key for environment params!")

    main(sim_params, exp_params, env_params)
