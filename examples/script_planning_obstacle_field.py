"""
Questions:
- How to choose lengthscale for Path Signature?
"""

import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc
import sigkernel
import torch
import torch.distributions as dist
from stein_mpc.inference import SVGD
from stein_mpc.utils.helper import generate_seeds, set_seed
from torch.autograd import grad as ag
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import trange


def create_spline_trajectory(knots, timesteps=100, ctx={}):
    t = torch.linspace(0, 1, timesteps, **ctx)
    t_knots = torch.linspace(0, 1, knots.shape[-2], **ctx)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


def plot_2d_dist(
    log_p, l_bounds=(0.0, 0.0), u_bounds=(5.0, 5.0), grid_size=100, ctx={}
):
    x_lim, y_lim = zip(l_bounds, u_bounds)
    x = torch.linspace(*x_lim, grid_size, **ctx)
    y = torch.linspace(*y_lim, grid_size, **ctx)
    X, Y = torch.meshgrid(x, y)
    Z = torch.exp(log_p(torch.stack((X.flatten(), Y.flatten()), dim=0).T)).reshape(
        grid_size, grid_size
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(X.cpu(), Y.cpu(), Z.data.cpu().numpy(), 30, cmap=plt.get_cmap("Greys"))
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
    ctx={},
):
    fig, ax = plot_2d_dist(log_p, l_bounds, u_bounds, grid_size, ctx)
    cycler = plt.cycler("color", plt.cm.tab20b.colors[:batch])
    ax.set_prop_cycle(cycler)
    plt.title(title)
    ax.scatter(*start, c="red", marker="x", s=20)
    ax.scatter(*target, c="red", marker="x", s=20)
    container = []
    print("Creating movie...")
    for i in trange(n_iter):
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


def run_exp(hp, log_p, x, path, id):
    static_kernel = sigkernel.RBFKernel(sigma=hp["ps_lengthscale"])
    kernel = sigkernel.SigKernel(static_kernel, hp["dyadic_order"])
    stein_sampler = SVGD(kernel, **hp["optimizer_args"])

    def batch_cost_function(log_p, x, start_pose, target_pose, timesteps=100, w=[1, 1]):
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
        return (obst_cost + len_cost), traj

    # need score estimator to include trajectory length regularization cost
    def sgd_score_estimator(x):
        cost, traj = batch_cost_function(
            log_p, x, start_pose, target_pose, hp["waypts"], hp["w_penalty"]
        )
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0].flatten(1)
        k_xx = torch.eye(batch, **ctx)
        grad_k = torch.zeros_like(grad_log_p, **ctx)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": cost}
        return grad_log_p, score_dict

    def score_estimator(x):
        cost, traj = batch_cost_function(
            log_p, x, start_pose, target_pose, hp["waypts"], hp["w_penalty"]
        )
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0].flatten(1)
        k_xx = kernel.compute_Gram(traj.double(), traj.double())
        k_xx = k_xx.float()
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx.detach(),
            "grad_k": grad_k.detach(),
            "loss": cost,
        }
        return grad_log_p, score_dict

    # Run Path Signature optimization
    ####################################################################################
    # Set initial state
    initial_knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    if hp["use_splines"] is True:
        x = initial_knots[:, 1:-1, :].clone().requires_grad_(True)
    else:
        initial_traj = create_spline_trajectory(initial_knots, hp["waypts"], ctx=ctx)
        x = initial_traj[:, 1:-1, :].clone().requires_grad_(True)

    # Optimization loop
    if hp["sgd_refine"] is True:
        # SGD + PathSig
        wu_data_dict, _ = stein_sampler.optimize(
            x, score_estimator, n_steps=(hp["n_iter"] - hp["n_iter"] // 4), debug=True
        )
        ps_data_dict, _ = stein_sampler.optimize(
            x, sgd_score_estimator, n_steps=hp["n_iter"] // 4, debug=True
        )
        trace = torch.cat([wu_data_dict["trace"][:-1], ps_data_dict["trace"]], dim=0)
        filename = f"{path}/pp_pathsig_sgd_{id}.mp4"
        plot_title = (
            "Trajectory Optimization with Path Signature Kernel and SGD refinement"
        )
    else:
        # Just PathSig
        ps_data_dict, _ = stein_sampler.optimize(
            x, score_estimator, n_steps=hp["n_iter"], debug=True
        )
        trace = ps_data_dict["trace"]
        filename = f"{path}/pp_pathsig_{id}.mp4"
        plot_title = "Trajectory Optimization with Path Signature Kernel"

    # Create episode movie
    if hp["use_splines"] is True:
        all_knots = torch.cat(
            (
                start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
                trace,
                target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
            ),
            dim=-2,
        )
        all_traj = create_spline_trajectory(all_knots, ctx=ctx)
        plot_data = {"traj": all_traj, "knots": all_knots}
    else:
        all_traj = torch.cat(
            (
                start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
                trace,
                target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
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
        title=plot_title,
        save_path=filename,
        ctx=ctx,
    )

    # Run SGD comparison
    ####################################################################################
    # Set initial state
    if hp["use_splines"] is True:
        x = initial_knots[:, 1:-1, :].clone().requires_grad_(True)
    else:
        initial_traj = create_spline_trajectory(initial_knots, hp["waypts"], ctx=ctx)
        x = initial_traj[:, 1:-1, :].clone().requires_grad_(True)

    # Run optimization
    stein_sampler.opt_inertia = 0
    sgd_data_dict, _ = stein_sampler.optimize(
        x, sgd_score_estimator, n_steps=hp["n_iter"], debug=True
    )
    trace = sgd_data_dict["trace"]
    filename = f"{path}/pp_sgd_{id}.mp4"

    # Create episode movie
    if hp["use_splines"] is True:
        all_knots = torch.cat(
            (
                start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
                trace,
                target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
            ),
            dim=-2,
        )
        all_traj = create_spline_trajectory(all_knots, ctx=ctx)
        plot_data = {"traj": all_traj, "knots": all_knots}
    else:
        all_traj = torch.cat(
            (
                start_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
                trace,
                target_pose.repeat(hp["n_iter"] + 1, batch, 1, 1),
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
        save_path=filename,
        ctx=ctx,
    )
    return ps_data_dict, sgd_data_dict


def compile_results(seeds, obstacles, ps_res, sgd_res):
    ps_costs = torch.tensor([])
    sgd_costs = torch.tensor([])

    for s in seeds:
        for n in obstacles:
            last_key = list(ps_res[s][n].keys())[-2]
            ep_cost = ps_res[s][n][last_key]["loss"].cpu()
            ps_costs = torch.cat([ps_costs, ep_cost[None, :]])

            last_key = list(sgd_res[s][n].keys())[-2]
            ep_cost = sgd_res[s][n][last_key]["loss"].cpu()
            sgd_costs = torch.cat([sgd_costs, ep_cost[None, :]])

    results = {
        "sgd_costs": sgd_costs.reshape(len(seeds), len(obstacles), -1),
        "ps_costs": ps_costs.reshape(len(seeds), len(obstacles), -1),
        "seeds": seeds,
        "n_obstacles": obstacles,
    }
    return results


if __name__ == "__main__":
    print("=== Test script for SVGD with Path signatures ===")
    if torch.cuda.is_available():
        ctx = {"device": "cuda:0", "dtype": torch.float32}
    else:
        ctx = {"device": "cpu", "dtype": torch.float32}

    # Confiure experiment hyperparameters
    N_EXP = 5
    hyperparameters = {
        "sgd_refine": True,
        "use_splines": True,
        "n_iter": 500,
        "waypts": 30,
        # [obstacle penalty, length penalty]
        "w_penalty": [1.0, 10.0],
        "ps_lengthscale": 1.0,
        "dyadic_order": 5,
        "optimizer_args": {
            "optimizer_class": torch.optim.Adam,
            "adaptive_gradient": True,
            "lr": 0.1,
        },
    }

    seeds = generate_seeds(N_EXP)
    obstacles = torch.arange(1, 40, 3).tolist()
    ps_results = {}
    sgd_results = {}
    for seed in seeds:
        set_seed(seed)
        ymd_time = time.strftime("%Y%m%d-%H%M%S")
        path = Path(f"data/local/path-plan-{ymd_time}/{seed}")
        if not path.exists():
            path.mkdir(parents=True)

        # Samples points for multiple splines with the same start and target positions
        batch, length, channels = 10, 7, 2
        start_pose = torch.tensor([0.25, 0.75], **ctx)
        target_pose = torch.tensor([4.75, 4.5], **ctx)
        offset = torch.tensor([1.0, 1.0], **ctx)
        x = 4 * torch.rand(batch, length - 2, channels, **ctx) + offset

        ps_results[seed], sgd_results[seed] = ({}, {})
        for n_obst in obstacles:
            # Gaussian obstacles
            limits = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
            w = torch.ones(n_obst, **ctx)
            samples = qmc.Halton(2, seed=seed).random(n_obst)
            mean = qmc.scale(samples, limits[0] + 0.5, limits[1] - 0.5)
            mean = torch.as_tensor(mean, **ctx)
            # mean = dist.uniform.Uniform(limits[0], limits[1]).sample([n_obst])
            var = 0.05 * torch.ones([n_obst, 2], **ctx)
            mix = dist.Categorical(w)
            comp = dist.Independent(dist.Normal(mean, var), 1)
            log_p = dist.MixtureSameFamily(mix, comp).log_prob

            res = run_exp(hyperparameters, log_p, x, path, n_obst)
            ps_results[seed][n_obst], sgd_results[seed][n_obst] = res

    results = compile_results(seeds, obstacles, ps_results, sgd_results)
    torch.save(results, path.parent.joinpath("pp_results.pt"))
