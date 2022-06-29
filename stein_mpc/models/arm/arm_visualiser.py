from typing import List

import numpy as np
from plotly import graph_objects as go
from plotly.basedatatypes import BaseTraceType

from .arm_simulator import Robot, WorkSpaceType, ConfigurationSpaceType


class RobotVisualiser:
    def __init__(self, robot: Robot):
        self.robot = robot

    @staticmethod
    def plot_xs(xs: WorkSpaceType, color="red", **kwargs) -> List[BaseTraceType]:
        return [
            go.Scatter3d(
                x=xs[..., 0].reshape(-1),
                y=xs[..., 1].reshape(-1),
                z=xs[..., 2].reshape(-1),
                marker_color=color,
                mode="markers",
                **kwargs,
            )
        ]

    def plot_arms(
        self,
        qs: ConfigurationSpaceType,
        highlight_end_effector: bool = False,
        color="green",
        name="Arm links & Joints",
        showlegend=True,
    ) -> List[BaseTraceType]:
        joints_xs = self.robot.qs_to_joints_xs(qs).numpy().swapaxes(0, 1)

        # create an array that denote the end of sequence
        assert len(joints_xs.shape) == 3
        array_of_None = np.empty(
            (joints_xs.shape[0], 1, joints_xs.shape[2]), dtype=np.object
        )
        array_of_None.fill(None)

        # joint the indicator array together
        series = np.hstack([joints_xs, array_of_None])

        # combine them into one single sequence
        series = np.concatenate(series)

        traces = [
            go.Scatter3d(
                x=series[..., 0].reshape(-1),
                y=series[..., 1].reshape(-1),
                z=series[..., 2].reshape(-1),
                marker_color=color,
                name=name,
                showlegend=showlegend,
            )
        ]

        if highlight_end_effector:
            traces.extend(
                self.plot_xs(
                    joints_xs[:, -1, :],
                    color="blue",
                    marker_symbol="x",
                    marker_size=4,
                    name="End effectors",
                    showlegend=showlegend,
                )
            )
        return traces
