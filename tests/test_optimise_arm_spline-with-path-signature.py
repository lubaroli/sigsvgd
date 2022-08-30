import os

import numpy
import torch

from stein_mpc.inference import SVGD
from stein_mpc.kernels import PathSigKernel
from stein_mpc.models.arm import arm_simulator
from stein_mpc.models.arm import arm_visualiser
from stein_mpc.utils.helper import get_default_progress_folder_path

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

urdf_path = f"{THIS_DIR}/../robot_resources/panda/urdf/panda.urdf"

device = "cuda" if torch.cuda.is_available() else "cpu"


def color_generator():
    import plotly.express as px

    while True:
        yield from px.colors.qualitative.Plotly


############################################################

# the link to operates
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

# constructing the robot arm as a simulator
robot = arm_simulator.Robot(
    urdf_path=urdf_path,
    target_link_names=target_link_names,
    end_effector_link_name="panda_hand",
    device=device,
)
# robot.print_info()

# construct a visualiser for the robot for plotting.
robot_visualiser = arm_visualiser.RobotVisualiser(robot)

############################################################
# load NN model and display prob

from stein_mpc.models.ros_robot import continuous_occupancy_map

occupancy_map_wegiht_fname = f"{THIS_DIR}/../robodata/001_continuous-occmap-weight.ckpt"

try:
    occmap = continuous_occupancy_map.load_trained_model(occupancy_map_wegiht_fname)
    occmap.to(device)
except FileNotFoundError as e:
    print("\n")
    print(
        f"ERROR: File not found at {occupancy_map_wegiht_fname}.\nHave you "
        f"downloaded the weight file via running 'Make'?"
    )
    print("\n")
    raise e

############################################################

# defining function to construct splines
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


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


####################################################
# Defines cost functions and sets up the problem
data_output_path_name = get_default_progress_folder_path()


# # Defines the cost function as a function of x which is a 3 x 2 differentiable vector
# # defining the intermediate knots of the spline
# def cost_function(x, start_pose, target_pose, timesteps=100):
#     knots = torch.cat((start_pose.unsqueeze(0), x, target_pose.unsqueeze(0)), 0)
#     out = create_spline_trajectory(knots, timesteps)
#     return occmap(robot.qs_to_joints_xs(out))


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


####################################################


class ScoreEstimator:
    def __init__(self, q_initial, q_target, kernel):
        self.q_initial = q_initial
        self.q_target = q_target
        self.kernel = kernel

    # need score estimator to include trajectory length regularization cost
    def sgd_score_estimator(self, x):
        cost, _ = batch_cost_function(x, self.q_initial, self.q_target)
        # considering the likelihood is exp(-cost)
        grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
        k_xx = torch.eye(x.shape[0], device=device)
        grad_k = torch.zeros_like(grad_log_p, device=device)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k}
        return grad_log_p, score_dict

    def score_estimator(self, x):
        cost, traj = batch_cost_function(self, x)
        # considering the likelihood is exp(-cost)
        grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
        k_xx, grad_k = self.kernel(traj, traj, x, depth=2)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k}
        return grad_log_p, score_dict


INDEX = 0
PLOT_EVERY = 50


def callback_to_plot(x, q_initial, q_target):
    """
    A callback function that is called in each iteration by the optimiser.
    We will keep track of the iteration number and plot every X iter
    """
    global INDEX
    if INDEX % PLOT_EVERY == 0:
        save_all_trajectory_end_effector_from_knot(
            data_output_path_name / f"{INDEX:03d}.png",
            q_initial,
            q_target,
            x.detach(),
            title=None,
        )
    INDEX += 1


def run_optimisation():
    limit_lowers, limit_uppers = robot.get_joints_limits()

    # defining our initial and target joints configurations
    q_initial = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    q_initial[1] = 0
    q_initial[2] = 0.5
    q_target = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    q_target[1] = 0.25

    ####################################################
    q_initial = q_initial.to(device)
    q_target = q_target.to(device)

    batch, length, channels = 6, 5, 9
    x = (
        torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
        + limit_lowers
    ).to(device)

    n_iter = 400
    kernel = PathSigKernel()
    stein_sampler = SVGD(kernel, optimizer_class=torch.optim.Adam, lr=0.08)

    save_all_trajectory_end_effector_from_knot(
        data_output_path_name / f"initial.png", q_initial, q_target, x, title=None,
    )

    estimator = ScoreEstimator(q_initial, q_target, kernel)

    data_dict_1, _ = stein_sampler.optimize(
        x,
        estimator.sgd_score_estimator,
        n_steps=n_iter // 2,
        debug=True,
        callback_func=lambda _x: callback_to_plot(_x, q_initial, q_target),
    )
    data_dict_2, _ = stein_sampler.optimize(
        x,
        estimator.score_estimator,
        n_steps=(n_iter - n_iter // 2),
        debug=True,
        callback_func=lambda _x: callback_to_plot(_x, q_initial, q_target),
    )


if __name__ == "__main__":
    torch.manual_seed(1)

    run_optimisation()
