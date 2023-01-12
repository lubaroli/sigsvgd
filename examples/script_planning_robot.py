import time
from pathlib import Path

import numpy
import torch
from torch.autograd import grad as ag
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from gpytorch.priors import SmoothedBoxPrior

from stein_mpc.inference import SVGD
from stein_mpc.kernels import SignatureKernel, GaussianKernel
from stein_mpc.models.robot import robot_scene, robot_visualizer
from stein_mpc.models.robot import robot_visualizer, robot_scene
from stein_mpc.models.robot.robot_scene import PathRequest
from stein_mpc.models.robot.robot_simulator import PandaRobot
from stein_mpc.models.robot_learning import (
    continuous_occupancy_map,
    continuous_self_collision_pred,
)
from stein_mpc.utils.helper import generate_seeds, get_project_root, set_seed
from stein_mpc.utils.scheduler import CosineScheduler


class ScoreEstimator:
    def __init__(self, q_initial, q_target, kernel, cost_fn_params, scheduler=None):
        self.q_initial = q_initial
        self.q_target = q_target
        self.cost_fn_params = cost_fn_params
        self.kernel = kernel
        if not scheduler:
            self.scheduler = lambda: 1
        else:
            self.scheduler = scheduler

    # need score estimator to include trajectory length regularization cost
    def sgd_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(
            x, self.q_initial, self.q_target, *self.cost_fn_params
        )
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = torch.eye(x.shape[0], device=device)
        grad_k = torch.zeros_like(grad_log_p, device=device)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": cost, **cost_dict}
        return grad_log_p, score_dict

    def svgd_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(
            x, self.q_initial, self.q_target, *self.cost_fn_params
        )
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx, grad_k = self.kernel(
            x.flatten(1), x.flatten(1).detach(), compute_grad=True
        )
        grad_k = grad_k.sum(0)
        score_dict = {
            "k_xx": k_xx,
            "grad_k": self.scheduler() * grad_k,
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict

    def pathsig_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(
            x, self.q_initial, self.q_target, *self.cost_fn_params
        )
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0].flatten(1)
        k_xx = self.kernel(x, x)
        grad_k = -1 * ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx.detach(),
            "grad_k": self.scheduler() * grad_k.detach(),
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps).to(knots.device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(knots.device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


def color_generator():
    import plotly.express as px

    while True:
        yield from px.colors.qualitative.Plotly


def plot_all_trajectory_end_effector_from_knot(
    q_initial, q_target, x, title="initial trajectory"
):
    with torch.no_grad():
        # creates a figure that contains the occupancy map with occ > x%
        fig = continuous_occupancy_map.visualise_model_pred(
            occmap,
            prob_threshold=0.8,
            marker_showscale=False,
            marker_colorscale="viridis",
        )

        batch = x.shape[0]
        knots = torch.cat(
            (q_initial.repeat(batch, 1, 1), x, q_target.repeat(batch, 1, 1)), 1
        )
        traj = create_spline_trajectory(knots, timesteps=100)

        _original_shape = traj.shape
        traj = traj.reshape(-1, _original_shape[-1])

        # convert the qs of arm into joint pose
        xs_all_joints = robot.qs_to_joints_xs(traj)
        traj_of_end_effector = xs_all_joints[-1, ...]
        traj_of_end_effector = traj_of_end_effector.reshape(
            _original_shape[0], _original_shape[1], -1
        ).cpu()

        # convert the qs of knot into world space
        _original_knot_shape = x.shape
        xs_of_knot = robot.qs_to_joints_xs(x.reshape(-1, _original_knot_shape[-1]))
        traj_of_knot = xs_of_knot[-1, ...]
        traj_of_knot = traj_of_knot.reshape(
            _original_knot_shape[0], _original_knot_shape[1], -1
        ).cpu()

        # use consistent coloring from our deterministic color generator
        color_gen = color_generator()
        for i, color in zip(range(traj_of_end_effector.shape[0]), color_gen):
            # plot arm end-effector traj as lines
            fig.add_traces(
                robot_viz.plot_xs(
                    traj_of_end_effector[i, ...],
                    color=color,
                    showlegend=False,
                    mode="lines",
                    line_width=8,
                )
            )
            # plot knot of the arm end effector
            fig.add_traces(
                robot_viz.plot_xs(
                    traj_of_knot[i, ...],
                    color=color,
                    showlegend=False,
                    mode="markers",
                    marker_size=10,
                )
            )

        # plot start/end
        for qs, name, color in [
            (q_initial, "q_initial", "red"),
            (q_target, "q_target", "cyan"),
        ]:
            fig.add_traces(
                robot_viz.plot_arms(
                    qs.detach(),
                    highlight_end_effector=True,
                    name=name,
                    color=color,
                    mode="lines",
                )
            )

    fig.update_layout(title=title)
    return fig


def plot_ee_trajectories_from_knots(output_fname, *args, **kwargs):
    kwargs["title"] = None
    fig = plot_all_trajectory_end_effector_from_knot(*args, **kwargs)

    # this eyevector FIXES the camera viewing angles.
    eyevector = numpy.array([-1.8, -1.4, 1.7]) * 0.6

    camera = dict(
        eye=dict(x=eyevector[0], y=eyevector[1], z=eyevector[2]),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )
    fig.update_layout(scene_camera=camera)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1200,
        height=800,
        margin=dict(t=0, r=0, l=0, b=0),
    )
    fig.write_image(output_fname)


def create_body_points(x, n_pts=10):
    inter_pts = torch.arange(0, 1, 1 / n_pts)
    device = x.device
    body_points = x[:-1, None] + inter_pts[:, None, None].to(device) * x[1:, None]
    return body_points.flatten(0, 1)


def batch_cost_function(
    x,
    start_pose,
    target_pose,
    timesteps=100,
    w_collision=1.0,
    w_self_collision=1.0,
    w_trajdist=1.0,
    use_ee_for_traj_dist=False,
):
    batch = x.shape[0]
    knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    qs = create_spline_trajectory(knots, timesteps)

    _original_shape = qs.shape
    # xs shape is [n_dof, batch * timesteps, 3]
    xs = robot.qs_to_joints_xs(qs.reshape(-1, _original_shape[-1]))

    ee_xs = xs[-1, ...]
    ee_xs = ee_xs.reshape(_original_shape[0], _original_shape[1], -1)
    # compute piece-wise linear distance
    if use_ee_for_traj_dist:
        traj_dist = torch.linalg.norm(ee_xs[:, 1:, :] - ee_xs[:, :-1, :], dim=-1).sum(1)
    else:
        # Penalize change in joint angles
        # qs shape is [batch, timesteps, n_dof]
        q_weights = torch.linspace(1.0, 0.7, 7)[None, None, :].to(qs.device)
        # q_weights = torch.ones(7)[None, None, :].to(qs.device)
        qs_dist = torch.linalg.norm(
            q_weights * (qs[:, 1:, :] - qs[:, :-1, :]), dim=-1
        ).sum(1)
        ee_dist = torch.linalg.norm(ee_xs[:, 1:, :] - ee_xs[:, :-1, :], dim=-1).sum(1)
        traj_dist = qs_dist + 1.0 * ee_dist

        # # xs shape is [n_joints, batch, timesteps, 3]
        # _xs_all_joints = xs_all_joints.reshape(
        #     xs_all_joints.shape[0],
        #     _original_shape[0],
        #     _original_shape[1],
        #     xs_all_joints.shape[-1],
        # )
        # traj_dist = (
        #     torch.linalg.norm(
        #         _xs_all_joints[:, :, 1:, :] - _xs_all_joints[:, :, :-1, :], dim=3
        #     )
        #     .sum(0)
        #     .sum(-1)
        # )

    # collision prob is in the shape of [n_dof * (n_pts - 1) x batch * timesteps x 1]
    n_pts = 10
    body_points = create_body_points(xs, n_pts)
    collision_prob = occmap(body_points).squeeze(-1)

    # sharpenning up with a Laplacian filter
    # collision_prob = torch.exp(-4 * (1 - collision_prob))
    # collision_prob = collision_prob ** 4

    # sum up and average across body points
    collision_prob = collision_prob.sum(0) / (n_pts - 1)
    # the following is now in the shape of [batch x timesteps]
    collision_prob = collision_prob.reshape(_original_shape[0], -1)
    # sum up the collision prob across timesteps and weight by length of each step
    # collision_prob = ee_dist / _original_shape[1] * collision_prob.sum(-1)
    collision_prob = collision_prob.sum(-1)

    self_collision_prob = self_collision_pred(qs).squeeze(-1)
    # we can then sum up the collision prob across timesteps
    self_collision_prob = self_collision_prob.sum(1)

    # the following should now be in 1d shape of [batch]
    cost = (
        w_collision * collision_prob
        + w_self_collision * self_collision_prob
        + w_trajdist * traj_dist
    )
    data_dict = {
        "knots": knots,
        "trajectories": qs,
        "costs_self_col": w_self_collision * self_collision_prob,
        "costs_col": w_collision * collision_prob,
        "costs_dist": traj_dist,
        "ee_trajectories": ee_xs,
    }

    return (
        cost,
        data_dict,
    )


def run_optimisation(q_initial, q_target, method="pathsig", hp={}):
    limit_lowers, limit_uppers = robot.get_joints_limits()
    batch, length, channels = hp["batch"], hp["length"], hp["channels"]

    limit_lowers = limit_lowers[:channels]
    limit_uppers = limit_uppers[:channels]
    # inter_knots = torch.arange(0, 1, 1 / (length - 2))
    # inter_knots = (inter_knots[:, None] * q_target[None, :]) + q_initial[None, :]
    # eps = (
    #     0.3 * torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
    #     + limit_lowers
    # )
    # x = inter_knots[None, ...] + eps
    # x = torch.clamp(x, min=limit_lowers, max=limit_uppers).to(device)
    q_initial = q_initial.to(device)
    q_target = q_target.to(device)

    x = (
        torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
        + limit_lowers
    ).to(device)

    if method == "svgd":
        if "svgd_bw" in hp:
            kernel = GaussianKernel(bandwidth_fn=lambda _: hp["svgd_bw"])
        else:
            kernel = GaussianKernel(bandwidth_fn=lambda _: (length + channels) ** 0.5)
    elif "pathsig_bw" in hp:
        kernel = SignatureKernel(bandwidth=hp["pathsig_bw"], depth=hp["depth"])
    else:
        kernel = SignatureKernel(
            bandwidth=(length + channels) ** 0.5, depth=hp["depth"]
        )
    scheduler = (
        CosineScheduler(1, 0, 3 * n_iter // 4, n_iter // 4)
        if hp["scheduler"] == "default"
        else hp["scheduler"]
    )
    estimator = ScoreEstimator(
        q_initial,
        q_target,
        kernel,
        scheduler=scheduler,
        cost_fn_params=[hp["timesteps"]] + hp["cost_weights"],
    )
    hyper_prior = (
        SmoothedBoxPrior(limit_lowers.to(device), limit_uppers.to(device), 0.1).log_prob
        if not torch.isinf(limit_lowers).any() and not torch.isinf(limit_uppers).any()
        else None
    )
    stein_sampler = SVGD(
        kernel, optimizer_class=None, lr=hp["lr"], log_prior=hyper_prior
    )

    plot_ee_trajectories_from_knots(
        experiment_path / "initial.png", q_initial, q_target, x, title=None,
    )

    if method == "ps_sgd":
        wu_data_dict, _ = stein_sampler.optimize(
            x,
            estimator.pathsig_score_estimator,
            n_steps=n_iter - n_iter // 4,
            debug=True,
        )
        data_dict, _ = stein_sampler.optimize(
            x, estimator.sgd_score_estimator, n_steps=n_iter // 4, debug=True,
        )
        torch.save([wu_data_dict, data_dict], f=experiment_path / "data.pt")
    elif method == "svgd":
        data_dict, _ = stein_sampler.optimize(
            x, estimator.svgd_score_estimator, n_steps=n_iter, debug=True,
        )
        torch.save(data_dict, f=experiment_path / "data.pt")
    elif method == "sgd":
        data_dict, _ = stein_sampler.optimize(
            x, estimator.sgd_score_estimator, n_steps=n_iter, debug=True,
        )
        torch.save(data_dict, f=experiment_path / "data.pt")
    else:
        data_dict, _ = stein_sampler.optimize(
            x, estimator.pathsig_score_estimator, n_steps=n_iter, debug=True,
        )
        torch.save(data_dict, f=experiment_path / "data.pt")

    plot_ee_trajectories_from_knots(
        experiment_path / "final.png",
        q_initial,
        q_target,
        data_dict["trace"][-1].to(device),
        title=None,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_path = get_project_root()
    robot = PandaRobot(device=device)
    robot_viz = robot_visualizer.RobotVisualizer(robot)  # visualizer for plotting

    # Configure hyperparams
    n_episodes = 5
    n_requests = 4
    n_iter = 500
    methods = ["pathsig", "svgd", "sgd"]
    hyperparams = {
        "batch": 20,
        "length": 5,
        "channels": robot.dof,
        "lr": 0.001,
        "pathsig_bw": 1.5,
        # "svgd_bw": 0.7,
        "svgd_bw": 1.5,
        "depth": 6,
        "scheduler": "default",
        "timesteps": 200,
        "cost_weights": [1.0, 10.0, 2.5,],  # collision, self-collision, traj. length
        # "cost_weights": [1.5, 10.0, 4.0,],
    }

    problems = list(robot_scene.tag_names)
    problems = problems[2:3]
    print("\n")
    print(
        """
    =====================================
    === Start of robot arm experiment ===
    =====================================
    """
    )
    for problem in problems:
        scene = robot_scene.RobotScene(robot, problem)
        try:
            occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)
            occmap.to(device)
            self_collision_pred = continuous_self_collision_pred.load_trained_model(
                robot.self_collision_model_weight_path
            )
            self_collision_pred.to(device)
        except FileNotFoundError as e:
            print(
                f"\nERROR: File not found at {scene.weight_path}."
                f"\nHave you downloaded the weight file via running 'Make'?\n"
            )
            raise e

        # ========== Experiment Setup ==========
        ymd_time = time.strftime("%Y%m%d-%H%M%S")
        seeds = generate_seeds(n_episodes)
        req_inds = numpy.random.permutation(len(scene.request_paths))[:n_requests]
        req_paths = numpy.array(scene.request_paths)[req_inds]
        for req_i, req_path in zip(req_inds, req_paths):
            print(f"\n=== Scene 1 of {problem} problem with request {req_i} ===")

            req = PathRequest.from_yaml(req_path)
            q_start = torch.as_tensor(req.start_state.get(robot.target_joint_names))
            q_target = torch.as_tensor(req.target_state.get(robot.target_joint_names))
            for ep_num, seed in enumerate(seeds):
                for method in methods:
                    set_seed(seed)
                    print(
                        f"Episode {ep_num + 1} - seed {seed}: optimizing with {method}"
                    )
                    INDEX = 0
                    experiment_path = Path(
                        f"data/local/robot-{problem}-{ymd_time}/{req_i}-{seed}/{method}"
                    )
                    plot_path = experiment_path / "plots"
                    if not plot_path.exists():
                        plot_path.mkdir(parents=True)
                    run_optimisation(q_start, q_target, method, hyperparams)
