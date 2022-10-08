import glob
import time
from dataclasses import dataclass
from functools import cached_property
from os import path
from pathlib import Path
from typing import List

import pybullet as p
import pybullet_tools.utils as pu
import yaml
from scipy.interpolate import interp1d

from stein_mpc.models.robot.robot_simulator import Robot
from stein_mpc.utils.helper import get_project_root

this_directory = Path(path.abspath(path.dirname(__file__)))

tag_names = [
    "bookshelf_small_panda",
    "bookshelf_tall_panda",
    "bookshelf_thin_panda",
    "box_panda",
    "cage_panda",
    "kitchen_panda",
    "table_bars_panda",
    "table_pick_panda",
    "table_under_pick_panda",
]

import numpy as np
import quaternion
import math


class Transform:
    def __init__(self, mat=None, quat: quaternion.quaternion = None, pos=None):
        if mat is None:
            self.matrix = np.identity(4)
            if quat is not None and pos is not None:
                self.matrix[:3, :3] = quaternion.as_rotation_matrix(quat)
                self.matrix[:3, 3] = pos
        else:
            self.matrix = np.copy(mat)

    def __str__(self):
        return self.matrix.__str__()

    def __repr__(self):
        return self.matrix.__repr__()

    def quat_2_mat(self, quat, pos):
        """Conversion quaternion vers matrix."""
        self.matrix[:3, :3] = quaternion.as_rotation_matrix(quat)
        self.matrix[:3, 3] = pos

    def inverse(self):
        return Transform(np.linalg.inv(self.matrix))

    def __invert__(self):
        return Transform(np.linalg.inv(self.matrix))

    def __sub__(self, other):
        return self.composition(~other)

    def __isub__(self, other):
        return self.composition(~other)

    def quaternion(self) -> quaternion.quaternion:
        return quaternion.from_rotation_matrix(self.matrix)

    def position(self) -> List[float]:
        return self.matrix[:3, 3]

    def composition(self, tr):
        return Transform(mat=np.dot(self.matrix, tr.matrix))

    def __mul__(self, other):
        return self.composition(other)

    def __imul__(self, other):
        self.matrix = self.matrix.dot(other.matrix)
        return self

    def relative_transform(self, other) -> "Transform":
        return ~other.composition(self)

    def projection(self, pt):
        if len(pt) == 3:
            return self.matrix.dot(pt + [1])
        else:
            return self.matrix.dot(pt)


def rotation_matrix(axe, angle):
    matrix = np.identity(4)
    if axe == "x":
        matrix[1, 1] = math.cos(angle)
        matrix[1, 2] = -math.sin(angle)
        matrix[2, 1] = math.sin(angle)
        matrix[2, 2] = math.cos(angle)
    elif axe == "y":
        matrix[0, 0] = math.cos(angle)
        matrix[0, 2] = math.sin(angle)
        matrix[2, 0] = -math.sin(angle)
        matrix[2, 2] = math.cos(angle)
    elif axe == "z":
        matrix[0, 0] = math.cos(angle)
        matrix[0, 1] = -math.sin(angle)
        matrix[1, 0] = math.sin(angle)
        matrix[1, 1] = math.cos(angle)
    return matrix


def translation_matrix(tr):
    matrix = np.identity(4)
    matrix[:3, 3] = np.asarray(tr)
    return matrix


@dataclass
class JointState:
    name: List[str]
    position: List[float]

    def get(self, joint_state_names: List[str]):
        return [self.position[self.name.index(name)] for name in joint_state_names]


#
# @dataclass
# class Quaternion:
#     values: List[float]
#
#     def __len__(self):
#         return len(self.values)
#
#     def __mul__(self, other: "Quaternion") -> "Quaternion":
#         x1, y1, z1, w1 = self.values
#         x2, y2, z2, w2 = other.values
#
#         w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#         x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#         y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#         z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#         return Quaternion([x, y, z, w])
#
#     def conjugate(self):
#         x, y, z, w = self.values
#         return Quaternion([-x, -y, -z, w])
#
#     def rotational_matrix(self):
#         qx, qy, qz, qw = self.values
#
#         m00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
#         m01 = 2 * qx * qy - 2 * qz * qw
#         m02 = 2 * qx * qz + 2 * qy * qw
#         m10 = 2 * qx * qy + 2 * qz * qw
#         m11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
#         m12 = 2 * qy * qz - 2 * qx * qw
#         m20 = 2 * qx * qz - 2 * qy * qw
#         m21 = 2 * qy * qz + 2 * qx * qw
#         m22 = 1 - 2 * qx ** 2 - 2 * qy ** 2
#         result = [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]
#
#         return result


@dataclass
class Pose:
    position: List[float]
    orientation: quaternion.quaternion

    def __init__(self, position, quat):
        self.position = position
        if not isinstance(quat, quaternion.quaternion):
            quat = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2])
        self.orientation = quat

    @cached_property
    def transformation_matrix(self):
        return Transform(pos=self.position, quat=self.orientation)

    def composite(self, other: "Pose"):
        trans = self.transformation_matrix.composition(other.transformation_matrix)
        return Pose(trans.position(), trans.quaternion())

    # def __add__(self, other: "Pose"):
    #     assert isinstance(other, Pose)
    #     assert len(self.position) == len(other.position)
    #     assert len(self.orientation) == len(other.orientation)
    #     return Pose(
    #         [self.position[i] + other.position[i] for i in range(len(self.position))],
    #         self.orientation * other.orientation,
    #     )

    def __iter__(self):
        yield self.position
        quat = self.orientation
        yield [quat.x, quat.y, quat.z, quat.w]


@dataclass
class PathRequest:
    start_state: JointState
    target_state: JointState

    @classmethod
    def from_yaml(cls, fname: str) -> "PathRequest":
        with open(fname, "r") as f:
            obj = yaml.safe_load(f)
        return PathRequest(
            JointState(
                obj["start_state"]["joint_state"]["name"],
                obj["start_state"]["joint_state"]["position"],
            ),
            JointState(
                [
                    j["joint_name"]
                    for j in obj["goal_constraints"][0]["joint_constraints"]
                ],
                [
                    j["position"]
                    for j in obj["goal_constraints"][0]["joint_constraints"]
                ],
            ),
        )


@dataclass
class Trajectory:
    states: List[JointState]

    @classmethod
    def from_yaml(cls, fname: str) -> "Trajectory":
        with open(fname, "r") as f:
            obj = yaml.safe_load(f)
        return Trajectory(
            [
                JointState(obj["joint_trajectory"]["joint_names"], point["positions"])
                for point in obj["joint_trajectory"]["points"]
            ]
        )

    def get(self, joint_state_names: List[str]):
        return [state.get(joint_state_names) for state in self.states]


def interpolate_trajectory(start, target):
    fst = np.array(start)
    snd = np.array(target)
    linfit = interp1d([0, 1], np.vstack([fst, snd]), axis=0)
    return linfit


class RobotScene:
    def __init__(self, robot: Robot, tag_name: str):
        self.robot = robot
        self.tag_name = tag_name
        self.added_bodies = []

    @cached_property
    def config_path(self):
        return get_project_root() / "robodata" / f"{self.tag_name}-config.yaml"

    @cached_property
    def robot_base_offset(self) -> Pose:
        with open(self.config_path, "r") as f:
            yamlobj = yaml.safe_load(f)
        return Pose(
            yamlobj["base_offset"]["position"], yamlobj["base_offset"]["orientation"],
        )

    @cached_property
    def scene_path(self):
        return get_project_root() / "robodata" / f"{self.tag_name}-scene0001.yaml"

    @cached_property
    def weight_path(self):
        return (
            get_project_root()
            / "robodata"
            / f"{self.tag_name}-scene0001_continuous-occmap-weight.ckpt"
        )

    @cached_property
    def dataset_path(self):
        return (
            get_project_root() / "robodata" / f"{self.tag_name}-scene0001_dataset.csv"
        )

    def __len__(self):
        return len(self.trajectory_paths)

    @cached_property
    def trajectory_paths(self):
        return sorted(
            glob.glob(
                str(
                    get_project_root()
                    / "robodata"
                    / f"{self.tag_name}-scene0001_path*.yaml"
                )
            )
        )

    @cached_property
    def request_paths(self):
        return sorted(
            glob.glob(
                str(
                    get_project_root()
                    / "robodata"
                    / f"{self.tag_name}-scene0001_request*.yaml"
                )
            )
        )

    def clear(self):
        while len(self.added_bodies) > 0:
            bid = self.added_bodies.pop(0)
            p.removeBody(bid)

    def build_scene(self):
        with open(self.scene_path, "r") as stream:
            yamlobj = yaml.safe_load(stream)

        if "fixed_frame_transforms" in yamlobj:
            _base_transform = yamlobj["fixed_frame_transforms"][0]["transform"]
            p.resetBasePositionAndOrientation(
                self.robot.pyb_robot_id,
                _base_transform["translation"],
                _base_transform["rotation"],
            )

        for obj in yamlobj["world"]["collision_objects"]:
            if "primitives" in obj:
                assert len(obj["primitives"]) == 1

                # primitive objects
                _type = obj["primitives"][0]["type"]
                _ = obj["primitive_poses"][0]

                # base frame
                pose = Pose(
                    obj["pose"]["position"],
                    obj["pose"]["orientation"]
                    # Quaternion([0,0,0,1])
                )
                # transform from the base frame
                pose = pose.composite(Pose(_["position"], _["orientation"]))

                dim = obj["primitives"][0]["dimensions"]
                if _type == "box":
                    self.added_bodies.append(pu.create_box(*dim, pose=pose))
                elif _type == "cylinder":
                    dim = obj["primitives"][0]["dimensions"]
                    self.added_bodies.append(
                        pu.create_cylinder(radius=dim[1], height=dim[0], pose=pose)
                    )
                else:
                    raise NotImplementedError(_type)

            elif "meshes" in obj:
                assert len(obj["meshes"]) == 1
                # meshes
                _ = obj["mesh_poses"][0]

                # base frame
                pose = Pose(
                    obj["pose"]["position"],
                    obj["pose"]["orientation"]
                    # Quaternion([0,0,0,1])
                )

                # transform from the base frame
                pose = pose.composite(Pose(_["position"], _["orientation"]))

                mesh = obj["meshes"][0]["vertices"], obj["meshes"][0]["triangles"]
                self.added_bodies.append(pu.create_mesh(mesh, pose=pose))

            else:
                raise NotImplementedError(str(obj))
        return list(self.added_bodies)

    def play(
        self,
        trajectory: Trajectory,
        target_joint_names: List[str],
        interpolate_step: int = 50,
        delay_between_interpolated_joint: float = 0.02,
        delay_between_joint: float = 2.0,
    ):
        target_joint_indexes = self.robot.joint_name_to_indexes(target_joint_names)

        last_qs = None
        for qs in trajectory.get(target_joint_names):
            if last_qs is not None:
                interp = interpolate_trajectory(last_qs, qs)
                ts = np.linspace(0, 1, num=interpolate_step)
                for t in ts:
                    self.robot.set_qs(interp(t), target_joint_indexes)
                    time.sleep(delay_between_interpolated_joint)

            last_qs = qs
            self.robot.set_qs(qs, target_joint_indexes)
            time.sleep(delay_between_joint)
