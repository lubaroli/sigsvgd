import time
from pathlib import Path

import torch
import torch.distributions as dist
import yaml
from sigkernel import RBFKernel, SigKernel
from src.controllers import DuSt
from src.inference import likelihoods
from src.inference.mpf import MPF
from src.kernels import GaussianKernel, ScaledGaussianKernel
from src.models import ParticleModel
from src.utils.helper import generate_seeds, save_progress, set_seed
from src.utils.plots import create_video_from_plots, plot_particles
from tqdm import trange


def run_exp(sim_hp, exp_hp, env_hp, ctx):
    # Initial state
    state = torch.tensor(env_hp["init_state"], **ctx)

    # Define the system and the simulator they may have different parameters.
    dynamics_prior = getattr(dist, exp_hp["dyn_prior"])(
        exp_hp["dyn_prior_arg1"], exp_hp["dyn_prior_arg2"]
    )
    system_kwargs = {
        "uncertain_params": ["mass"],
        "mass": dynamics_prior.mean,
    }
    system = ParticleModel(**env_hp, **system_kwargs)
    simulator = ParticleModel(**env_hp, **system_kwargs)

    # Define controller Kernel
    def fixed_bandwidth(*args, **kwargs):
        return (exp_hp["ctrl_dim"] + exp_hp["horizon"]) ** 0.5

    kernel_type = exp_hp["kernel"]
    if kernel_type == "rbf":
        kernel = ScaledGaussianKernel()
    elif kernel_type == "rbf_fixed_bw":
        kernel = ScaledGaussianKernel(bandwidth_fn=fixed_bandwidth)
    elif kernel_type == "signature":
        static_kernel = RBFKernel(sigma=fixed_bandwidth())
        kernel = SigKernel(static_kernel, exp_hp["dyadic_order"])
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))

    # Define optimizer
    opt = torch.optim.Adam
    opt_kwargs = {"lr": exp_hp["learning_rate"]}

    # Define controller
    controller = DuSt(
        observation_space=simulator.observation_space,
        action_space=simulator.action_space,
        hz_len=exp_hp["horizon"],
        n_pol=exp_hp["n_policies"],
        n_action_samples=exp_hp["action_samples"],
        n_params_samples=exp_hp["params_samples"],
        pol_mean=None,  # let the controller randomize initial actions
        pol_cov=torch.eye(exp_hp["ctrl_dim"]) * exp_hp["ctrl_sigma"] ** 2,
        pol_hyper_prior=True,
        stein_sampler=exp_hp["stein_sampler"],
        kernel=kernel,
        temperature=exp_hp["alpha"],
        params_log_space=exp_hp["mpf_log_space"],
        inst_cost_fn=simulator.default_inst_cost,
        term_cost_fn=simulator.default_term_cost,
        action_primitives=exp_hp["primitives"],
        optimizer_class=opt,
        **opt_kwargs,
    )

    mpf_n_part = exp_hp["mpf_n_particles"]
    mpf_steps = exp_hp["mpf_steps"]
    mpf_log_space = exp_hp["mpf_log_space"]
    mpf_bw = exp_hp["mpf_bandwidth"]
    mpf_init = dynamics_prior.sample([mpf_n_part, 1]).clamp(min=1e-6)
    mpf_init = mpf_init.log() if mpf_log_space else mpf_init
    dynamics_lik = likelihoods.GaussianLikelihood(
        initial_obs=state,
        obs_std=exp_hp["mpf_obs_std"],
        model=simulator,
        log_space=exp_hp["mpf_log_space"],
    )
    dyn_sampler = MPF(
        init_particles=mpf_init,
        kernel=GaussianKernel(),
        likelihood=dynamics_lik,
        optimizer_class=torch.optim.SGD,
        lr=exp_hp["mpf_learning_rate"],
        bw=(2 * exp_hp["dyn_prior_arg2"]) ** 1 / 2,
        bw_scale=exp_hp["mpf_bandwidth_scaling"],
    )

    # Variables to aggregate results
    tau = state.unsqueeze(0).to(**ctx)
    rollouts = torch.empty(
        (0,) + controller.rollout_shape + (controller.hz_len + 1, state.shape[0])
    ).to(**ctx)
    costs = torch.empty((0, 1)).to(**ctx)
    actions = torch.empty((0, exp_hp["ctrl_dim"])).to(**ctx)
    dyn_particles = torch.empty((0, mpf_n_part)).to(**ctx)

    # ===== Experiment Loop =====
    iterator = trange(sim_hp["steps"])
    if exp_hp["use_mpf"]:
        mpf = dyn_sampler
        dynamics_dist = mpf.prior
    else:
        mpf = None
        dynamics_dist = dynamics_prior
    for step in iterator:
        action_mat, step_data = controller.forward(
            state, simulator, dynamics_dist, opt_steps=exp_hp["opt_steps"]
        )
        # fetch trajectories of last iteration
        state_mat = step_data[exp_hp["opt_steps"] - 1]["trajectories"]
        action = action_mat[0]
        state = system.step(state, action.squeeze())
        tau = torch.cat([tau, state.unsqueeze(0)], dim=0)
        actions = torch.cat([actions, action.unsqueeze(0)])
        rollouts = torch.cat([rollouts, state_mat.unsqueeze(0)], dim=0)
        inst_cost = controller.inst_cost_fn(state.view(1, -1))
        costs = torch.cat([costs, inst_cost.unsqueeze(0)], dim=0)
        if exp_hp["use_mpf"]:
            grad_dyn = torch.zeros(mpf_steps)
            if step >= sim_hp["warm_up"]:
                # optimize will automatically update `dynamics_dist`
                grad_dyn, _ = mpf.optimize(
                    action.squeeze(), state, bw=mpf_bw, n_steps=mpf_steps
                )
            dyn_particles = torch.cat(
                [dyn_particles, mpf.x.detach().clone().t()], dim=0
            )

        if system.with_obstacle:
            if system.obst_map.get_collisions(state[:2]):
                print("\nCrashed at step {}".format(step))
                break
        if (system.target - state).norm() <= 1.0:
            break

    return (
        system,
        {
            "costs": costs.detach().cpu(),
            "trajectory": tau.detach().cpu(),
            "actions": actions.detach().cpu(),
            "rollouts": rollouts.detach().cpu(),
            "dyn_particles": dyn_particles.detach().cpu(),
        },
    )


if __name__ == "__main__":
    """Runs the simulation script

    Parameters:
        sim_hp: Simulation hyper-parameters
        exp_hp: Experiment hyper-parameters
        env_hp: Environment hyper-parameters
    """
    config_file = Path(__file__).parent / "particle_maze_config.yaml"
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

    sim_hp = {}
    try:
        sim_hp = config_data["sim_params"]
    except KeyError:
        print("Invalid key for simulation params!")

    exp_hp = {}
    try:
        exp_hp = config_data["exp_params"]
    except KeyError:
        print("Invalid key for experiment params!")

    env_hp = {}
    try:
        env_hp = config_data["env_params"]
    except KeyError:
        print("Invalid key for environment params!")

    if torch.cuda.is_available():
        ctx = {"device": "cuda:0", "dtype": torch.float32}
    else:
        ctx = {"device": "cpu", "dtype": torch.float32}

    # ========== Experiment Setup ==========
    print("=== Start of particle maze experiment ===")
    seeds = generate_seeds(sim_hp["episodes"])
    ymd_time = time.strftime("%Y%m%d-%H%M%S")

    # Define motion primitives (this should probably be a more elaborate function)
    exp_hp["primitives"] = torch.zeros(5, exp_hp["horizon"], 2)
    exp_hp["primitives"][1] = -10.0
    exp_hp["primitives"][2] = 10.0
    exp_hp["primitives"][3, ..., :] = torch.tensor([-10.0, 10.0])
    exp_hp["primitives"][4, ..., :] = torch.tensor([10.0, -10.0])

    for ep_num, seed in enumerate(seeds):
        set_seed(seed)
        print(f"Running episode {ep_num + 1} with seed {seed}")
        print("SVMPC baseline")
        ep_path = Path(f"data/local/maze-{ymd_time}/{seed}/svmpc")
        exp_hp["kernel"] = "rbf"
        system, episode_data = run_exp(sim_hp, exp_hp, env_hp, ctx)
        save_progress(folder_name=ep_path, data=episode_data)
        plot_particles(system, episode_data, ep_path.joinpath("plots"))
        create_video_from_plots(ep_path.joinpath("plots"), ep_path)
        print("PathSig controller")
        ep_path = Path(f"data/local/maze-{ymd_time}/{seed}/pathsig")
        exp_hp["kernel"] = "signature"
        system, episode_data = run_exp(sim_hp, exp_hp, env_hp, ctx)
        save_progress(folder_name=ep_path, data=episode_data)
        plot_particles(system, episode_data, ep_path.joinpath("plots"))
        create_video_from_plots(ep_path.joinpath("plots"), ep_path)
