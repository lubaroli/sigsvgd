# %%

import os

import torch

from stein_mpc.inference import SVGD
from stein_mpc.kernels import PathSigKernel
from stein_mpc.models.arm import arm_simulator
from stein_mpc.models.arm import arm_visualiser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

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
)

robot.print_info()

robot_visualiser = arm_visualiser.RobotVisualiser(robot)

############################################################
# load NN model and display prob

from stein_mpc.models.ros_robot import continuous_occupancy_map

occmap = continuous_occupancy_map.load_trained_model(
    f"{THIS_DIR}/../robodata/001_continuous-occmap-weight.ckpt"
)

############################################################

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps)
    t_knots = torch.linspace(0, 1, knots.shape[-2])
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


def plot_all_trajectory_from_knot(
    q_initial, q_target, knots, title="initial trajectory"
):
    fig = continuous_occupancy_map.visualise_model_pred(
        occmap, prob_threshold=0.8, marker_showscale=False, marker_colorscale="viridis"
    )
    for i in range(knots.shape[0]):
        _knots = knots[i, ...]

        traj = create_spline_trajectory(
            torch.cat((q_initial.unsqueeze(0), _knots, q_target.unsqueeze(0)), 0),
            timesteps=100,
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
        # plot _knots
        for qs in _knots:
            fig.add_traces(
                robot_visualiser.plot_arms(
                    qs.detach(),
                    highlight_end_effector=False,
                    name="knot",
                    color="magenta",
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


# Defines the cost function as a function of x which is a 3 x 2 differentiable vector
# defining the intermediate knots of the spline
def cost_function(x, start_pose, target_pose, timesteps=100):
    knots = torch.cat((start_pose.unsqueeze(0), x, target_pose.unsqueeze(0)), 0)
    out = create_spline_trajectory(knots, timesteps)
    return occmap(robot.qs_to_joints_xs(out))


def batch_cost_function(x, start_pose, target_pose, timesteps=100, w=1.0):
    batch = x.shape[0]
    knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    traj = create_spline_trajectory(knots, timesteps)
    _original_shape = traj.shape
    traj = traj.reshape(-1, _original_shape[-1])

    out = robot.qs_to_joints_xs(traj)
    # out = out.reshape(
    #     out.shape[0],
    #     _original_shape[0],
    #     _original_shape[1],
    #     out.shape[-1],
    # )
    cost = occmap(out)

    return cost, traj


####################################################

batch, length, channels = 6, 5, 9
x = (
    torch.rand(batch, length - 2, channels) * (limit_uppers - limit_lowers)
    + limit_lowers
)

plot_all_trajectory_from_knot(
    q_initial, q_target, x, title="initial trajectory",
)

n_iter = 400
kernel = PathSigKernel()
stein_sampler = SVGD(kernel, optimizer_class=torch.optim.Adam, lr=0.01)


# need score estimator to include trajectory length regularization cost
def sgd_score_estimator(x):
    cost, _ = batch_cost_function(x, q_initial, q_target)
    # considering the likelihood is exp(-cost)
    grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
    k_xx = torch.eye(x.shape[0])
    grad_k = torch.zeros_like(grad_log_p)
    score_dict = {"k_xx": k_xx, "grad_k": grad_k}
    return grad_log_p, score_dict


def score_estimator(x):
    cost, traj = batch_cost_function(x, q_initial, q_target)
    # considering the likelihood is exp(-cost)
    grad_log_p = torch.autograd.grad(-cost.sum(), x, retain_graph=True)[0]
    k_xx, grad_k = kernel(traj, traj, x, depth=2)
    score_dict = {"k_xx": k_xx, "grad_k": grad_k}
    return grad_log_p, score_dict


data_dict_1, _ = stein_sampler.optimize(
    x, sgd_score_estimator, n_steps=n_iter // 2, debug=True
)
data_dict_2, _ = stein_sampler.optimize(
    x, score_estimator, n_steps=(n_iter - n_iter // 2), debug=True
)

x = x.detach()

plot_all_trajectory_from_knot(
    q_initial, q_target, x, title="final optimised trajectory",
)
