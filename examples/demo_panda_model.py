import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

import torch
import numpy as np

import plotly.graph_objects as go

from stein_mpc.models.arm import arm_simulator
from stein_mpc.models.arm import arm_visualiser

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

###########################################################

qs = torch.Tensor([[0, 0, 0, 0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0],])
layout = go.Layout(scene=dict(aspectmode="data"))

go.Figure(
    [*robot_visualiser.plot_arms(qs, highlight_end_effector=True)],
    layout=layout,
    layout_title="A robot arm",
).show()

###########################################################

xs = torch.Tensor(
    [
        [0.3, 0.2, 0.3],
        [0.2, 0.2, 0.34],
        [0.21, 0.23, 0.38],
        [0.61, 0.23, 0.38],
        [0.5, -0.3, 0.75],
    ]
)

_qs_from_ik = robot.ee_xs_to_qs(xs)
_qs_from_ik = torch.Tensor(_qs_from_ik)

go.Figure(
    [
        *robot_visualiser.plot_arms(_qs_from_ik, highlight_end_effector=True),
        *robot_visualiser.plot_xs(xs, name="target ik"),
    ],
    layout_title="A robot arm with a few random (infeasible) end-effector target for IK",
).show()

###########################################################

t = torch.linspace(0, 2 * np.pi, 20)
_x = torch.empty_like(t)
_y = torch.empty_like(t)
_z = torch.empty_like(t)

_x = t / 10 - 0.6
_y = t / 10 - 0.6
_z = torch.cos(t) / 5 + 0.8

xs = torch.vstack([_x, _y, _z])
xs = xs.swapaxes(0, 1)

_qs_from_ik = robot.ee_xs_to_qs(xs)
_qs_from_ik = torch.Tensor(_qs_from_ik)

go.Figure(
    [
        *robot_visualiser.plot_arms(_qs_from_ik, highlight_end_effector=True),
        *robot_visualiser.plot_xs(xs, name="target ik"),
    ],
    layout_title="A robot arm with a cos curve for IK",
).show()
