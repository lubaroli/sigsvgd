"""
Questions:
- How to choose lengthscale for Path Signature?
"""

import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import torch
import torch.distributions as dist
from stein_mpc.inference import SVGD, PlanningEstimator
from stein_mpc.kernels import GaussianKernel, SignatureKernel
from stein_mpc.utils.helper import generate_seeds, set_seed
from stein_mpc.utils.scheduler import CosineScheduler
from torch.autograd import grad as ag
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import trange


def create_spline_trajectory(knots, timesteps=100, ctx={"device": "cpu"}):
    t = torch.linspace(0, 1, timesteps, **ctx)
    t_knots = torch.linspace(0, 1, knots.shape[-2], **ctx)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


def plot_2d_dist(
    log_p,
    l_bounds=(0.0, 0.0),
    u_bounds=(5.0, 5.0),
    grid_size=100,
    ctx={"device": "cpu"},
):
    x_lim, y_lim = zip(l_bounds, u_bounds)
    x = torch.linspace(*x_lim, grid_size, **ctx)
    y = torch.linspace(*y_lim, grid_size, **ctx)
    X, Y = torch.meshgrid(x, y)
    Z = torch.exp(log_p(torch.stack((X.flatten(), Y.flatten()), dim=0).T)).reshape(
        grid_size, grid_size
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.contourf(X.cpu(), Y.cpu(), Z.data.cpu().numpy(), 30, cmap=plt.get_cmap("Greys"))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    return fig, ax


def create_2d_movie(
    data,
    log_p,
    start,
    target,
    l_bounds=(0.0, 5.0),
    u_bounds=(0.0, 5.0),
    grid_size=100,
    n_iter=100,
    title="Trajectory Optimization with Path Signatures",
    save_path="movie.mp4",
    step=1,
    ctx={},
):
    fig, ax = plot_2d_dist(log_p, l_bounds, u_bounds, grid_size, ctx)
    cycler = plt.cycler("color", plt.cm.tab20(torch.linspace(0, 1, batch)))
    ax.set_prop_cycle(cycler)
    plt.title(title)
    plt.tight_layout()
    ax.scatter(*start, c="red", marker="x", s=20)
    ax.scatter(*target, c="red", marker="x", s=20)
    container = []
    print("Creating movie...")
    for i in trange(0, n_iter, step):
        im = []
        # list of [batch] lines
        if "traj" in data:
            im.extend(
                ax.plot(
                    data["traj"][i, ..., 0].T.cpu(),
                    data["traj"][i, ..., 1].T.cpu(),
                    animated=True,
                )
            )
        # list of [batch] knots
        if "knots" in data:
            im.extend(
                ax.plot(
                    data["knots"][i, ..., 1:-1, 0].cpu(),
                    data["knots"][i, ..., 1:-1, 1].cpu(),
                    "ob",
                    animated=True,
                )
            )
        caption = ax.text(
            0.5,
            0.1,
            "Current iteration: {}".format(i + 1),
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=ax.transAxes,
            ha="center",
        )
        im.extend([caption])
        container.append(im)

    ani = animation.ArtistAnimation(fig, container, interval=50, blit=True)
    ani.save(filename=save_path)
    plt.close()


def run_exp(hp, log_p, x, path, id, render=False, step=1):
    experiment_path = Path(f"{path}/{id}")
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    def batch_cost_fn(x, log_p, start_pose, target_pose, timesteps=100, w=[1, 1]):
        batch = x.shape[0]
        if hp["use_splines"] is True:
            knots = torch.cat(
                (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
            )
            traj = create_spline_trajectory(knots, timesteps, ctx=ctx)
        else:
            traj = torch.cat(
                (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
            )
        obst_cost = (w[0] * log_p(traj).exp()).sum(-1)
        len_cost = torch.norm(w[1] * (traj[:, 1:] - traj[:, :-1]), dim=[-2, -1])
        return (obst_cost + len_cost), {"trajectories": traj}

    # Set initial state
    initial_knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    if hp["use_splines"] is True:
        x = initial_knots[:, 1:-1, :].clone().requires_grad_(True)
    else:
        initial_traj = create_spline_trajectory(initial_knots, hp["waypts"], ctx=ctx)
        x = initial_traj[:, 1:-1, :].clone().requires_grad_(True)

    # Set hyperparams
    if hp["scheduler"] is not None:
        plot_title = "Trajectory Optimization with Scheduled "
    else:
        plot_title = "Trajectory Optimization with "

    cost_fn_params = [
        log_p,
        start_pose,
        target_pose,
        100,  # timesteps
        [1, 1],  # cost weights for collision and length
    ]

    # Run Path Signature optimization
    ####################################################################################
    print(f"\nRunning PathSig with lengthscale: {hp['ps_lengthscale'](0)}")
    particles = x.clone()
    kernel = SignatureKernel(
        bandwidth_fn=hp["ps_lengthscale"], depth=hp["dyadic_order"]
    )
    stein_sampler = SVGD(kernel, **hp["optimizer_args"])
    estimator = PlanningEstimator(
        kernel, batch_cost_fn, cost_fn_params, hp["scheduler"], ctx
    )
    data_dict, _ = stein_sampler.optimize(
        particles, estimator.score, n_steps=hp["n_iter"], debug=True
    )
    torch.save(data_dict, f=experiment_path / "pathsig_data.pt")

    # Create episode movie
    if render:
        trace = data_dict["trace"]
        if hp["use_splines"] is True:
            all_knots = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            all_traj = create_spline_trajectory(all_knots)
            plot_data = {"traj": all_traj, "knots": all_knots}
        else:
            all_traj = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            plot_data = {"traj": all_traj}
        create_2d_movie(
            plot_data,
            log_p,
            start_pose.cpu(),
            target_pose.cpu(),
            l_bounds=limits[0],
            u_bounds=limits[1],
            n_iter=hp["n_iter"],
            title=plot_title + "Path Signature Kernel",
            save_path=experiment_path / "pathsig.mp4",
            step=step,
            ctx=ctx,
        )

    # Run SVMP comparison
    ####################################################################################
    print(f"\nRunning SVMP with lengthscale: {hp['lengthscale'](0)}")
    particles = x.clone()
    kernel = GaussianKernel(bandwidth_fn=hp["lengthscale"])
    stein_sampler = SVGD(kernel, **hp["optimizer_args"])
    estimator = PlanningEstimator(
        kernel, batch_cost_fn, cost_fn_params, hp["scheduler"], ctx
    )
    data_dict, _ = stein_sampler.optimize(
        particles, estimator.score, n_steps=hp["n_iter"], debug=True
    )
    torch.save(data_dict, f=experiment_path / "svmp_data.pt")

    # Create episode movie
    if render:
        trace = data_dict["trace"]
        if hp["use_splines"] is True:
            all_knots = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            all_traj = create_spline_trajectory(all_knots)
            plot_data = {"traj": all_traj, "knots": all_knots}
        else:
            all_traj = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            plot_data = {"traj": all_traj}
        create_2d_movie(
            plot_data,
            log_p,
            start_pose.cpu(),
            target_pose.cpu(),
            l_bounds=limits[0],
            u_bounds=limits[1],
            n_iter=hp["n_iter"],
            title=plot_title + "Gaussian Kernel",
            save_path=experiment_path / "svmp.mp4",
            step=step,
            ctx=ctx,
        )

    # Run SGD comparison
    ####################################################################################
    print("\nRunning SGD")
    particles = x.clone()
    data_dict, _ = stein_sampler.optimize(
        particles, estimator.sgd_score, n_steps=hp["n_iter"], debug=True
    )
    torch.save(data_dict, f=experiment_path / "sgd_data.pt")

    # Create episode movie
    if render:
        trace = data_dict["trace"]
        if hp["use_splines"] is True:
            all_knots = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            all_traj = create_spline_trajectory(all_knots)
            plot_data = {"traj": all_traj, "knots": all_knots}
        else:
            all_traj = torch.cat(
                (
                    start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                    trace,
                    target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1).cpu(),
                ),
                dim=-2,
            )
            plot_data = {"traj": all_traj}
        create_2d_movie(
            plot_data,
            log_p,
            start_pose.cpu(),
            target_pose.cpu(),
            l_bounds=limits[0],
            u_bounds=limits[1],
            n_iter=hp["n_iter"],
            title="Trajectory Optimization with SGD",
            save_path=experiment_path / "sgd.mp4",
            step=step,
            ctx=ctx,
        )


if __name__ == "__main__":
    print("=== Test script for SVGD with Path signatures ===")
    if torch.cuda.is_available():
        ctx = {"device": "cuda:0", "dtype": torch.float32}
    else:
        ctx = {"device": "cpu", "dtype": torch.float32}

    # Confiure experiment hyperparameters
    N_EXP = 1
    hyperparameters = {
        "use_splines": True,
        "n_iter": 500,
        "waypts": 80,
        # [obstacle penalty, length penalty]
        "w_penalty": [1.0, 2.0],
        "ps_lengthscale": lambda _: 0.03,
        "lengthscale": lambda _: 0.2,
        # "ps_lengthscale": lambda _: 0.2,
        # "lengthscale": lambda _: 0.5,
        "dyadic_order": 5,
        "optimizer_args": {
            "optimizer_class": torch.optim.Adam,
            "adaptive_gradient": True,
            "lr": 0.05,
        },
    }
    hyperparameters["scheduler"] = CosineScheduler(
        1, 0, hyperparameters["n_iter"] // 2, hyperparameters["n_iter"] // 1.5
    )
    hyperparameters["scheduler"] = None

    seeds = generate_seeds(N_EXP)
    batch, length, channels = 20, 5, 2
    obstacles = [50, 10]
    obstacles = [1]
    ps_results = {}
    sgd_results = {}
    ymd_time = time.strftime("%Y%m%d-%H%M%S")
    for seed in seeds:
        set_seed(seed)
        path = Path(f"data/local/path-plan-{ymd_time}/{seed}")
        if not path.exists():
            path.mkdir(parents=True)

        # Samples points for multiple splines with the same start and target positions
        start_pose = torch.tensor([0.25, 0.75], **ctx)
        target_pose = torch.tensor([4.75, 4.5], **ctx)
        # offset = torch.tensor([1.0, 1.0], **ctx)
        # x = 4 * torch.rand(batch, length - 2, channels, **ctx) + offset
        x = torch.linspace(0.25, 4.75, length) + 0.4 * torch.randn(batch, length)
        y = torch.linspace(0.75, 4.5, length) + 0.4 * torch.randn(batch, length)
        wp = torch.cat([x[:, 1:-1][..., None], y[:, 1:-1][..., None]], dim=-1).to(**ctx)

        ps_results[seed], sgd_results[seed] = ({}, {})
        for n_obst in obstacles:
            # values = torch.logspace(-3, 0, 10)
            # for bw in values:
            # hyperparameters["ps_lengthscale"] = lambda _: bw
            # hyperparameters["lengthscale"] = lambda _: bw
            # n_obst = 1
            # Gaussian obstacles
            limits = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
            w = torch.ones(n_obst, **ctx)
            samples = qmc.Halton(2, seed=seed).random(n_obst)
            mean = qmc.scale(samples, limits[0] + 0.5, limits[1] - 0.5)
            mean = torch.as_tensor(mean, **ctx) + 100
            var = 0.05 * torch.ones([n_obst, 2], **ctx)
            mix = dist.Categorical(w)
            comp = dist.Independent(dist.Normal(mean, var), 1)
            log_p = dist.MixtureSameFamily(mix, comp).log_prob

            run_exp(hyperparameters, log_p, wp, path, n_obst, render=True, step=2)
            # res = run_exp(hyperparameters, log_p, x, path, n_obst)
            # ps_results[seed][n_obst], sgd_results[seed][n_obst] = res

    # results = compile_results(seeds, obstacles, ps_results, sgd_results)
    # torch.save(results, path.parent.joinpath("pp_results.pt"))
