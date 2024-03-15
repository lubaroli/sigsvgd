import numpy as np
import torch

from src.models.robot import robot_scene, robot_visualizer
from src.models.robot.robot_simulator import PandaRobot
from src.models.robot_learning import continuous_occupancy_map
from src.utils.helper import get_project_root

probject_root = get_project_root()

robot = PandaRobot()
robot.print_info()

robot_visualizer = robot_visualizer.RobotVisualizer(robot)

############################################################
# load NN model and display prob

scene = robot_scene.RobotScene(robot, robot_scene.tag_names[0])

occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)

fig = continuous_occupancy_map.visualise_model_pred(
    occmap, prob_threshold=0.8, marker_showscale=False, marker_colorscale="viridis"
)

############################################################
target_ee = torch.Tensor([0, 1.5, 0.5]).unsqueeze(0)
qs = torch.Tensor(robot.ee_xs_to_qs(target_ee))

base_xs = robot.qs_to_joints_xs(qs)[0, :]
fig.add_traces(
    robot_visualizer.plot_xs(
        base_xs, color="red", name="fixed base", marker_symbol="x", marker_size=10,
    )
)
ee_xs = robot.qs_to_joints_xs(qs)[-1, :]
fig.add_traces(
    robot_visualizer.plot_xs(
        ee_xs, color="red", name="ee", marker_symbol="x", marker_size=10,
    )
)
############################################################

print(robot.learnable_robot_model.get_joint_limits())
limit_lowers, limit_uppers = robot.get_joints_limits()

import torch.optim as optim

print("\n\n")

torch.manual_seed(123)

# batch_size = 5


qs.requires_grad_()
optimizer = optim.Adam([qs], lr=5e-2)

for i in range(20):
    _qs = qs.detach().clone()

    optimizer.zero_grad()

    print(qs)
    cost = occmap(robot.qs_to_joints_xs(qs))

    print(f"===== step {i} =====")
    print(f"qs:   {_qs.numpy().squeeze()}")
    print(f"cost: {cost.detach().numpy().squeeze()}")
    cost = cost.sum()
    print(f"cost sum: {cost.detach().numpy().squeeze()}")
    cost.backward()
    print(f"qs-grad: {qs.grad}")

    fig.add_traces(
        robot_visualizer.plot_arms(
            qs.detach(), highlight_end_effector=True, showlegend=False
        )
    )

    optimizer.step()

print(f"\n\n\n===== final =====")
print(f"{qs.detach().numpy().squeeze()}")

fig.add_traces(
    robot_visualizer.plot_arms(
        qs.detach(), highlight_end_effector=True, color="purple", name="final arm"
    )
)

fig.show()
