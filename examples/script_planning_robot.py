import time
from pathlib import Path

import numpy
import torch
from torch.autograd import grad as ag
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from stein_mpc.inference import SVGD
from stein_mpc.kernels import SignatureKernel, GaussianKernel
from stein_mpc.models.robot import robot_visualiser, robot_scene
from stein_mpc.models.robot.robot_scene import PathRequest
from stein_mpc.models.robot.robot_simulator import PandaRobot
from stein_mpc.models.robot_learning import continuous_occupancy_map
from stein_mpc.utils.helper import generate_seeds, get_project_root, set_seed
from stein_mpc.utils.scheduler import CosineScheduler


class ScoreEstimator:
    def __init__(self, q_initial, q_target, kernel, scheduler=None):
        self.q_initial = q_initial
        self.q_target = q_target
        self.kernel = kernel
        if not scheduler:
            self.scheduler = lambda: 1
        else:
            self.scheduler = scheduler

    # need score estimator to include trajectory length regularization cost
    def sgd_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(x, self.q_initial, self.q_target)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = torch.eye(x.shape[0], device=device)
        grad_k = torch.zeros_like(grad_log_p, device=device)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": cost, **cost_dict}
        return grad_log_p, score_dict

    def svgd_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(x, self.q_initial, self.q_target)
        ee_traj = cost_dict["ee_trajectories"]
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = self.kernel(ee_traj.flatten(1), ee_traj.flatten(1), compute_grad=False)
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx,
            "grad_k": self.scheduler() * grad_k,
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict

    def pathsig_score_estimator(self, x):
        cost, cost_dict = batch_cost_function(x, self.q_initial, self.q_target)
        ee_traj = cost_dict["ee_trajectories"]
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0].flatten(1)
        k_xx = self.kernel(ee_traj, ee_traj)
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx.detach(),
            "grad_k": self.scheduler() * grad_k.detach(),
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
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
                robot_visualiser.plot_xs(
                    traj_of_end_effector[i, ...],
                    color=color,
                    showlegend=False,
                    mode="lines",
                    line_width=8,
                )
            )
            # plot knot of the arm end effector
            fig.add_traces(
                robot_visualiser.plot_xs(
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
                robot_visualiser.plot_arms(
                    qs.detach(),
                    highlight_end_effector=True,
                    name=name,
                    color=color,
                    mode="lines",
                )
            )

    fig.update_layout(title=title)
    return fig


def save_all_trajectory_end_effector_from_knot(output_fname, *args, **kwargs):
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


def plot_callback(x, q_initial, q_target):
    """
    A callback function that is called in each iteration by the optimiser.
    We will keep track of the iteration number and plot every X iter
    """
    global INDEX
    if INDEX % PLOT_EVERY == 0:
        save_all_trajectory_end_effector_from_knot(
            plot_path / f"{INDEX:03d}.png", q_initial, q_target, x.detach(), title=None,
        )
    INDEX += 1


# def load_request(problem, req_num):
#     project_path = get_project_root()
#     req_path = project_path.joinpath(
#         f"robodata/{problem}/request{req_num:04d}.yaml"
#     )
#     with req_path.open() as f:
#         request = yaml.load(f, yaml.FullLoader)
#     q_start = request["start_state"]["joint_state"]["position"]
#     target_dict_pairs = request["goal_constraints"][0]["joint_constraints"]
#     q_target = [target_dict_pairs[i]["position"] for i in range(len(target_dict_pairs))]
#     q_target.extend(q_start[-2:])
#     return torch.as_tensor(q_start), torch.as_tensor(q_target)


def create_body_points(x, n_pts=10):
    inter_pts = torch.arange(0, 1, 1 / n_pts)
    device = x.device
    body_points = x[:-1, None] + inter_pts[:, None, None].to(device) * x[1:, None]
    return body_points.flatten(0, 1)


def batch_cost_function(
    x, start_pose, target_pose, timesteps=100, w_collision=5.0, w_trajdist=1.0,
):
    batch = x.shape[0]
    knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    traj = create_spline_trajectory(knots, timesteps)

    _original_shape = traj.shape
    # xs shape is [n_dof, batch * timesteps, 3]
    xs_all_joints = robot.qs_to_joints_xs(traj.reshape(-1, _original_shape[-1]))

    traj_of_end_effector = xs_all_joints[-1, ...]
    traj_of_end_effector = traj_of_end_effector.reshape(
        _original_shape[0], _original_shape[1], -1
    )

    # this collision prob is in the shape of [n_dof * n_pts x (batch x timesteps) x 1]
    body_points = create_body_points(xs_all_joints)
    collision_prob = occmap(body_points)
    # the following is now in the shape of [batch x timesteps]
    collision_prob = (
        collision_prob.sum(0)
        .reshape(_original_shape[0], _original_shape[1])
        .squeeze(-1)
    )
    # we can then sums up the collision prob across timesteps
    collision_prob = collision_prob.sum(1)

    # compute piece-wise linear distance
    traj_dist = torch.linalg.norm(
        traj_of_end_effector[:, 1:, :] - traj_of_end_effector[:, :-1, :], dim=2
    ).sum(1)

    # the following should now be in 1d shape of [batch]
    cost = w_collision * collision_prob + w_trajdist * traj_dist

    return (
        cost,
        {"knots": knots, "trajectories": traj, "ee_trajectories": traj_of_end_effector},
    )


def run_optimisation(
    q_initial,
    q_target,
    batch,
    length,
    channels,
    method="pathsig",
    lr=0.1,
    scheduler=None,
):
    limit_lowers, limit_uppers = robot.get_joints_limits()

    # # defining our initial and target joints configurations
    # q_initial = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    # q_initial[1] = 0
    # q_initial[2] = 0.5
    # q_target = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    # q_target[1] = 0.25

    q_initial = q_initial.to(device)
    q_target = q_target.to(device)

    limit_lowers = limit_lowers[:channels]
    limit_uppers = limit_uppers[:channels]
    x = (
        torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
        + limit_lowers
    ).to(device)

    if method == "svgd":
        kernel = GaussianKernel(bandwidth_fn=lambda _: (length + channels) ** 0.5)
    else:
        kernel = SignatureKernel(bandwidth=(length + channels) ** 0.5)
    scheduler = (
        CosineScheduler(1, 0, 3 * n_iter // 4, n_iter // 4)
        if scheduler is None
        else scheduler
    )
    estimator = ScoreEstimator(q_initial, q_target, kernel, scheduler=scheduler)
    stein_sampler = SVGD(kernel, optimizer_class=None, lr=lr)

    save_all_trajectory_end_effector_from_knot(
        experiment_path / "initial.png", q_initial, q_target, x, title=None,
    )

    if method == "ps_sgd":
        wu_data_dict, _ = stein_sampler.optimize(
            x,
            estimator.pathsig_score_estimator,
            n_steps=n_iter - n_iter // 4,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        data_dict, _ = stein_sampler.optimize(
            x,
            estimator.sgd_score_estimator,
            n_steps=n_iter // 4,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        torch.save([wu_data_dict, data_dict], f=experiment_path / "data.pt")
    elif method == "svgd":
        data_dict, _ = stein_sampler.optimize(
            x,
            estimator.svgd_score_estimator,
            n_steps=n_iter,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        torch.save(data_dict, f=experiment_path / "data.pt")
    elif method == "sgd":
        data_dict, _ = stein_sampler.optimize(
            x,
            estimator.sgd_score_estimator,
            n_steps=n_iter,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        torch.save(data_dict, f=experiment_path / "data.pt")
    else:
        data_dict, _ = stein_sampler.optimize(
            x,
            estimator.pathsig_score_estimator,
            n_steps=n_iter,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        torch.save(data_dict, f=experiment_path / "data.pt")

    save_all_trajectory_end_effector_from_knot(
        experiment_path / "final.png",
        q_initial,
        q_target,
        data_dict["trace"][-1],
        title=None,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_path = get_project_root()
    robot = PandaRobot(device=device,)

    problems = list(robot_scene.tag_names)
    for problem in problems:
        scene = robot_scene.RobotScene(robot, problem)
        try:
            occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)
            occmap.to(device)
        except FileNotFoundError as e:
            print(
                f"\nERROR: File not found at {scene.weight_path}."
                f"\nHave you downloaded the weight file via running 'Make'?\n"
            )
            raise e

        # construct the robot arm as a simulator

        # construct a visualiser for the robot for plotting
        robot_visualiser = robot_visualiser.RobotVisualiser(robot)

        # ========== Experiment Setup ==========
        print("\n=== Start of robot arm experiment ===")
        episodes = 5
        n_iter = 500
        batch, length, channels = 15, 5, robot.dof
        ymd_time = time.strftime("%Y%m%d-%H%M%S")
        seeds = generate_seeds(episodes)
        PLOT_EVERY = 50

        methods = ["svgd", "sgd", "pathsig"]
        # requests = [
        #     torch.randint(0 + 25 * i, 24 + 25 * i, (1,)).item() for i in range(4)
        # ]
        # requests = [1]

        for req_i, req_path in enumerate(scene.request_paths):
            req = PathRequest.from_yaml(req_path)

            print(f"=== Scene 1 of {problem} problem with request {req_i} ===")
            print(f"{req}\n")

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
                    run_optimisation(
                        q_start, q_target, batch, length, channels, method, lr=0.001
                    )
