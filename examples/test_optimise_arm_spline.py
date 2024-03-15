# %%

import os
import numpy as np
import torch
import tqdm

from src.models.robot import robot_simulator, robot_scene
from src.models.robot import robot_visualizer
from src.models.robot.robot_simulator import PandaRobot

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


############################################################


robot = PandaRobot()
robot.print_info()
scene = robot_scene.RobotScene(robot, robot_scene.tag_names[0])

robot_visualizer = robot_visualizer.RobotVisualizer(robot)

############################################################
# load NN model and display prob

from src.models.robot_learning import continuous_occupancy_map

occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)

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


q_initial = torch.rand(robot.dof) * (limit_uppers - limit_lowers) + limit_lowers
q_initial[1] = 0
q_initial[2] = 0.5
q_target = torch.rand(robot.dof) * (limit_uppers - limit_lowers) + limit_lowers
q_target[1] = 0.25

num_knots_per_traj = 5

sample_knots = (
    torch.rand(num_knots_per_traj, robot.dof) * (limit_uppers - limit_lowers)
    + limit_lowers
)


####################################################


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
            robot_visualizer.plot_arms(
                qs.detach(),
                highlight_end_effector=False,
                showlegend=False,
                color="black",
            )
        )
    # plot knots
    for qs in knots:
        fig.add_traces(
            robot_visualizer.plot_arms(
                qs.detach(), highlight_end_effector=False, name="knot", color="magenta",
            )
        )
    # plot start/end
    for qs, name, color in [
        (q_initial, "q_initial", "red"),
        (q_target, "q_target", "cyan"),
    ]:
        fig.add_traces(
            robot_visualizer.plot_arms(
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


####################################################

plot_trajectory_from_knot(
    q_initial,
    q_target,
    torch.cat((q_initial.unsqueeze(0), sample_knots, q_target.unsqueeze(0)), 0),
    title="initial trajectory",
)

# Sets up the optimization problem
x = sample_knots.clone().requires_grad_(True)
optimizer = torch.optim.Adam(params=[x], lr=0.1)
n_iter = 500

with tqdm.trange(n_iter) as pbar:
    for i in pbar:
        optimizer.zero_grad()
        cost = cost_function(x, q_initial, q_target).sum()
        pbar.set_postfix(cost=cost.detach())

        d_cost = -torch.autograd.grad(cost, x, create_graph=True)[0]
        x.grad = -d_cost
        optimizer.step()

x = x.detach()

plot_trajectory_from_knot(q_initial, q_target, x, title="final optimised trajectory")
