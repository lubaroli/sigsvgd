# %%

import os

import torch

from stein_mpc.inference import SVGD
from stein_mpc.kernels import PathSigKernel
from stein_mpc.models.arm import arm_simulator
from stein_mpc.models.arm import arm_visualiser
from stein_mpc.utils.helper import get_default_progress_folder_path

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


device = "cuda" if torch.cuda.is_available() else "cpu"


def color_generator():
    import plotly.express as px

    while True:
        yield from px.colors.qualitative.Plotly


############################################################

urdf_path = f"{THIS_DIR}/../robot_resources/panda/urdf/panda.urdf"

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

robot = arm_simulator.Robot(
    urdf_path=urdf_path,
    target_link_names=target_link_names,
    end_effector_link_name="panda_hand",
    device=device,
)

robot.print_info()

robot_visualiser = arm_visualiser.RobotVisualiser(robot)

############################################################
# load NN model and display prob

from stein_mpc.models.ros_robot import continuous_occupancy_map

occmap = continuous_occupancy_map.load_trained_model(
    f"{THIS_DIR}/../robodata/001_continuous-occmap-weight.ckpt"
)
occmap.to(device)

############################################################

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


torch.manual_seed(1)
limit_lowers, limit_uppers = robot.get_joints_limits()

q_initial = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
q_initial[1] = 0
q_initial[2] = 0.5
q_target = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
q_target[1] = 0.25


####################################################
q_initial = q_initial.to(device)
q_target = q_target.to(device)


def plot_all_trajectory_end_effector_from_knot(
    q_initial, q_target, x, title="initial trajectory"
):
    with torch.no_grad():
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

        xs_all_joints = robot.qs_to_joints_xs(traj)
        traj_of_end_effector = xs_all_joints[-1, ...]
        traj_of_end_effector = traj_of_end_effector.reshape(
            _original_shape[0], _original_shape[1], -1
        ).cpu()

        _original_knot_shape = x.shape
        xs_of_knot = robot.qs_to_joints_xs(x.reshape(-1, _original_knot_shape[-1]))
        traj_of_knot = xs_of_knot[-1, ...]
        traj_of_knot = traj_of_knot.reshape(
            _original_knot_shape[0], _original_knot_shape[1], -1
        ).cpu()

        color_gen = color_generator()
        for i, color in zip(range(traj_of_end_effector.shape[0]), color_gen):
            fig.add_traces(
                robot_visualiser.plot_xs(
                    traj_of_end_effector[i, ...],
                    color=color,
                    showlegend=False,
                    mode="lines",
                    line_width=8,
                )
            )
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

    import numpy

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


def plot_trajectory_from_knot(q_initial, q_target, knots, title="initial trajectory"):
    fig = continuous_occupancy_map.visualise_model_pred(
        occmap, prob_threshold=0.8, marker_showscale=False, marker_colorscale="viridis"
    )
    traj = create_spline_trajectory(
        torch.cat((q_initial.unsqueeze(0), knots, q_target.unsqueeze(0)), 0),
        timesteps=200,
    )
    for qs in traj:
        fig.add_traces(
            robot_visualiser.plot_arms(
                qs.detach(),
                highlight_end_effector=False,
                showlegend=False,
                color="black",
            )
        )
    # plot knots
    for qs in knots:
        fig.add_traces(
            robot_visualiser.plot_arms(
                qs.detach(), highlight_end_effector=False, name="knot", color="magenta",
            )
        )
    # plot start/end
    for qs, name, color in [
        (q_initial, "q_initial", "red"),
        (q_target, "q_target", "cyan"),
    ]:
        fig.add_traces(
            robot_visualiser.plot_arms(
                qs.detach(), highlight_end_effector=True, name=name, color=color,
            )
        )

    fig.update_layout(title=title)
    fig.show()


####################################################
# Defines cost functions and sets up the problem


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

batch, length, channels = 6, 5, 9
x = (
    torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
    + limit_lowers
).to(device)

output_path_name = get_default_progress_folder_path()

save_all_trajectory_end_effector_from_knot(
    output_path_name / f"{0:03d}.png", q_initial, q_target, x, title=None,
)

n_iter = 400
kernel = PathSigKernel()
stein_sampler = SVGD(kernel, optimizer_class=torch.optim.Adam, lr=0.08)


# need score estimator to include trajectory length regularization cost
def sgd_score_estimator(x):
    cost, _ = batch_cost_function(x, q_initial, q_target)
    # considering the likelihood is exp(-cost)
    grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
    k_xx = torch.eye(x.shape[0], device=device)
    grad_k = torch.zeros_like(grad_log_p, device=device)
    score_dict = {"k_xx": k_xx, "grad_k": grad_k}
    return grad_log_p, score_dict


def score_estimator(x):
    cost, traj = batch_cost_function(x, q_initial, q_target)
    # considering the likelihood is exp(-cost)
    grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
    k_xx, grad_k = kernel(traj, traj, x, depth=2)
    score_dict = {"k_xx": k_xx, "grad_k": grad_k}
    return grad_log_p, score_dict


INDEX = 0
PLOT_EVERY = 50


def callback_to_plot(x):
    global INDEX
    INDEX += 1
    if INDEX % PLOT_EVERY != 0:
        return
    save_all_trajectory_end_effector_from_knot(
        output_path_name / f"{INDEX:03d}.png",
        q_initial,
        q_target,
        x.detach(),
        title=None,
    )


data_dict_1, _ = stein_sampler.optimize(
    x,
    sgd_score_estimator,
    n_steps=n_iter // 2,
    debug=True,
    callback_func=callback_to_plot,
)
data_dict_2, _ = stein_sampler.optimize(
    x,
    score_estimator,
    n_steps=(n_iter - n_iter // 2),
    debug=True,
    callback_func=callback_to_plot,
)

# x = x.detach()
#
# plot_all_trajectory_end_effector_from_knot(
#     q_initial, q_target, x, title="final optimised trajectory",
# )
