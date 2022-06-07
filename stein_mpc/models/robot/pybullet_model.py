import os
import sys
from typing import Union

import pybullet as p
import pybullet_data
import numpy as np
import torch

from stein_mpc.models.robot.differentiable_robot import DifferentiableRobot

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{cur_path}/../../../external/pybullet-planning")

import pybullet_tools
import pybullet_tools.utils


class PyBullteVisualiser:
    def __init__(self, urdf_path):

        try:
            p.disconnect()
        except:
            pass

        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        pybullet_tools.utils.disable_gravity()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        #         p.setGravity(0,0,-10)
        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 1]
        startPos = [0, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # p.setAdditionalSearchPath(
        #     "/home/soraxas/pybullet/pybullet-planning/models/drake/jaco_description/urdf"
        # )  # optionally
        self.boxId = p.loadURDF(
            urdf_path, startPos, startOrientation, useFixedBase=True
        )

        self.target_joints = None

    def print_info(self):
        print(p.getBodyInfo(self.boxId))
        for i in range(p.getNumJoints(self.boxId)):
            print(p.getJointState(self.boxId, i))
            print(p.getJointInfo(self.boxId, i))

    def get_movable_joint(self):
        return pybullet_tools.utils.get_movable_joints(self.boxId)

    def print_movable_joint_info(self):
        print(f"Movable joints:\n{self.get_movable_joint()}")

        print(f"Movable joints info:")
        for jidx in self.get_movable_joint():
            print(f"Joint {jidx}")
            print(f"  limit: {pybullet_tools.utils.get_joint_limits(self.boxId, jidx)}")
            print(p.getJointInfo(self.boxId, jidx))

    def set_target_joints(self, target_joints):
        self.target_joints = target_joints

    def get_target_joints_states(self):
        return pybullet_tools.utils.get_joint_positions(self.boxId, self.target_joints)

    def set_target_joints_states(self, joint_states):
        assert len(joint_states) == len(self.target_joints)
        for jidx, state in zip(self.target_joints, joint_states):
            pybullet_tools.utils.set_joint_position(self.boxId, jidx, state)

    def get_link_pose(self, link: Union[str, int]):
        if type(link) == str:
            link_id = pybullet_tools.utils.link_from_name(self.boxId, link)
        else:
            link_id = link
        return pybullet_tools.utils.get_link_pose(self.boxId, link_id)


def assert_visualiser_and_differentiable_robot_align(
    state: torch.Tensor,
    diffrobot_model: DifferentiableRobot,
    pybullet_model: PyBullteVisualiser,
):
    """
    Given an instance of pybullet model and differentiable model, make sure that both
    model has the same joint angles and project to the same end effector
    """
    diffrobot_model.robot_model.update_kinematic_state(
        state, torch.zeros_like(state),
    )
    for link_name, v in diffrobot_model.robot_model._name_to_idx_map.items():
        pose_in_diffrobot = (
            diffrobot_model.robot_model._bodies[v].pose.translation().tolist()[0]
        )
        pose_in_pybullet = pybullet_model.get_link_pose(link_name)[0]

        if not np.isclose(pose_in_diffrobot, pose_in_pybullet, atol=1e-6).all():
            raise ValueError(
                "The pose result from diff-robot != "
                f"pybullet @ link {link_name}. Expected {pose_in_diffrobot}, "
                f"got {pose_in_pybullet}"
            )


# j_vis.set_target_joints([1, 2, 3, 4, 5, 6])

# print()

# j_vis.print_info()
# j_vis.print_movable_joint_info()

# print(j_vis.get_target_joints_states())
