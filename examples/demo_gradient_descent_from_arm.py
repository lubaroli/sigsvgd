import csv
import os

import numpy as np
import plotly.graph_objects as go
import torch

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

fig = continuous_occupancy_map.visualise_model_pred(
    occmap, prob_threshold=0.8, marker_showscale=False
)

############################################################
qs = torch.Tensor(
    [[0, 2, -np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0],]
)
base_xs = robot.qs_to_joints_xs(qs)[0, :]
fig.add_traces(
    robot_visualiser.plot_xs(
        base_xs, color="red", name="fixed base", marker_symbol="x", marker_size=10,
    )
)
############################################################

print(robot.learnable_robot_model.get_joint_limits())
limit_lowers, limit_uppers = robot.get_joints_limits()

import torch.optim as optim


print("\n\n")

torch.manual_seed(0)

for arm_i in range(5):
    qs = torch.rand(9) * (limit_uppers - limit_lowers) + limit_lowers
    qs.requires_grad_()
    optimizer = optim.Adam([qs], lr=5e-2)

    for i in range(20):
        _qs = qs.detach().clone()

        optimizer.zero_grad()

        cost = occmap(robot.qs_to_joints_xs(qs))

        print(f"===== arm {arm_i} | step {i} =====")
        print(f"qs:   {_qs.numpy().reshape(-1)}")
        print(f"cost: {cost.detach().numpy().reshape(-1)}")
        cost = cost.sum()
        print(f"cost sum: {cost.detach().numpy().reshape(-1)}")
        cost.backward()
        print(f"qs-grad: {qs.grad}")

        fig.add_traces(
            robot_visualiser.plot_arms(
                qs.detach(), highlight_end_effector=True, showlegend=False
            )
        )

        optimizer.step()

    print(f"\n\n\n===== arm {arm_i} | final =====")
    print(f"{qs.detach().numpy().reshape(-1)}")

    fig.add_traces(
        robot_visualiser.plot_arms(
            qs.detach(), highlight_end_effector=True, color="black", name="final arm"
        )
    )


fig.show()
