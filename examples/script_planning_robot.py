import time
from pathlib import Path

import numpy
import sigkernel
import torch
import yaml
from stein_mpc.inference import SVGD
from stein_mpc.models.arm import arm_simulator, arm_visualiser
from stein_mpc.models.ros_robot import continuous_occupancy_map
from stein_mpc.utils.helper import generate_seeds, get_project_root, set_seed
from torch.autograd import grad as ag
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs


class SignatureKernel:
    def __init__(self, lengthscale=0.5, depth=3):
        static_kernel = sigkernel.RBFKernel(sigma=lengthscale)
        self.kernel = sigkernel.SigKernel(static_kernel, dyadic_order=depth)

    def __call__(self, X, Y):
        ctx = {"device": X.device, "dtype": X.dtype}
        k_xx = self.kernel.compute_Gram(X.double(), Y.double())
        return k_xx.to(**ctx)


class ScoreEstimator:
    def __init__(self, q_initial, q_target, kernel):
        self.q_initial = q_initial
        self.q_target = q_target
        self.kernel = kernel

    # need score estimator to include trajectory length regularization cost
    def sgd_score_estimator(self, x):
        cost, _ = batch_cost_function(x, self.q_initial, self.q_target)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = torch.eye(x.shape[0], device=device)
        grad_k = torch.zeros_like(grad_log_p, device=device)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k}
        return grad_log_p, score_dict

    def score_estimator(self, x):
        cost, traj = batch_cost_function(self, x)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx, grad_k = self.kernel(traj, traj)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k}
        return grad_log_p, score_dict

    def autograd_score_estimator(self, x):
        cost, traj = batch_cost_function(x, self.q_initial, self.q_target)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0].flatten(1)
        k_xx = self.kernel(traj, traj)
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx.detach(),
            "grad_k": grad_k.detach(),
            "loss": cost,
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


def load_request(problem, req_num):
    project_path = get_project_root()
    req_path = project_path.joinpath(
        f"robodata/{problem}_panda/request{req_num:04d}.yaml"
    )
    with req_path.open() as f:
        request = yaml.load(f, yaml.FullLoader)
    q_start = request["start_state"]["joint_state"]["position"]
    target_dict_pairs = request["goal_constraints"][0]["joint_constraints"]
    q_target = [target_dict_pairs[i]["position"] for i in range(len(target_dict_pairs))]
    q_target.extend(q_start[-2:])
    return torch.as_tensor(q_start), torch.as_tensor(q_target)


def batch_cost_function(
    x, start_pose, target_pose, timesteps=100, w_collision=5.0, w_trajdist=1.0,
):
    batch = x.shape[0]
    knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    traj = create_spline_trajectory(knots, timesteps)

    _original_shape = traj.shape
    traj = traj.reshape(-1, _original_shape[-1])

    xs_all_joints = robot.qs_to_joints_xs(traj)

    traj_of_end_effector = xs_all_joints[-1, ...]
    traj_of_end_effector = traj_of_end_effector.reshape(
        _original_shape[0], _original_shape[1], -1
    )

    # this collision prob is in the shape of [ndof x (batch x timesteps) x 1]
    collision_prob = occmap(xs_all_joints)
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

    return cost, traj_of_end_effector


def run_optimisation(
    q_initial, q_target, batch, length, channels, method="pathsig", lr=0.1
):
    limit_lowers, limit_uppers = robot.get_joints_limits()

    # # defining our initial and target joints configurations
    # q_initial = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    # q_initial[1] = 0
    # q_initial[2] = 0.5
    # q_target = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    # q_target[1] = 0.25

    ####################################################
    q_initial = q_initial.to(device)
    q_target = q_target.to(device)

    limit_lowers = limit_lowers[:channels]
    limit_uppers = limit_uppers[:channels]
    x = (
        torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
        + limit_lowers
    ).to(device)

    kernel = SignatureKernel(lengthscale=(length + channels) ** 0.5)
    estimator = ScoreEstimator(q_initial, q_target, kernel)
    stein_sampler = SVGD(kernel, optimizer_class=torch.optim.Adam, lr=lr)

    save_all_trajectory_end_effector_from_knot(
        experiment_path / "initial.png", q_initial, q_target, x, title=None,
    )

    if method == "pathsig":
        wu_data_dict, _ = stein_sampler.optimize(
            x,
            estimator.autograd_score_estimator,
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
    elif method == "sgd":
        data_dict, _ = stein_sampler.optimize(
            x,
            estimator.sgd_score_estimator,
            n_steps=n_iter,
            debug=True,
            callback_func=lambda _x: plot_callback(_x, q_initial, q_target),
        )
        torch.save(data_dict, f=experiment_path / "data.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_path = get_project_root()
    urdf_path = project_path.joinpath("robot_resources/panda/urdf/panda.urdf")
    problems = [
        "bookshelf_small",
        "bookshelf_tall",
        "bookshelf_thin",
        "box",
        "cage",
        "table_bars",
        "table_pick",
        "table_under_pick",
    ]
    for problem in problems:
        occmap_fname = project_path.joinpath(
            f"robodata/{problem}_panda-scene0001_continuous-occmap-weight.ckpt"
        )
        try:
            occmap = continuous_occupancy_map.load_trained_model(occmap_fname)
            occmap.to(device)
        except FileNotFoundError as e:
            print(
                f"\nERROR: File not found at {occmap_fname}."
                f"\nHave you downloaded the weight file via running 'Make'?\n"
            )
            raise e

        # choose links to operate
        target_link_names = [
            # "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
        ]

        # construct the robot arm as a simulator
        robot = arm_simulator.Robot(
            urdf_path=str(urdf_path),
            target_link_names=target_link_names,
            end_effector_link_name="panda_hand",
            device=device,
        )

        # construct a visualiser for the robot for plotting
        robot_visualiser = arm_visualiser.RobotVisualiser(robot)

        # ========== Experiment Setup ==========
        print("\n=== Start of robot arm experiment ===")
        episodes = 5
        n_iter = 500
        batch, length, channels = 15, 5, 9
        ymd_time = time.strftime("%Y%m%d-%H%M%S")
        seeds = generate_seeds(episodes)
        PLOT_EVERY = 499

        methods = ["pathsig", "sgd"]
        requests = [
            torch.randint(0 + 25 * i, 24 + 25 * i, (1,)).item() for i in range(4)
        ]
        for req in requests:
            print(f"=== Scene 1 of {problem} problem with request {req} ===")
            q_start, q_target = load_request(problem, req)
            for ep_num, seed in enumerate(seeds):
                for method in methods:
                    set_seed(seed)
                    print(
                        f"Episode {ep_num + 1} - seed {seed}: optimizing with {method}"
                    )
                    INDEX = 0
                    experiment_path = Path(
                        f"data/local/robot-{problem}-{ymd_time}/{req}-{seed}/{method}"
                    )
                    plot_path = experiment_path / "plots"
                    if not plot_path.exists():
                        plot_path.mkdir(parents=True)
                    run_optimisation(q_start, q_target, batch, length, channels, method)
