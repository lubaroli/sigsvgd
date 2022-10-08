import argparse
import time
from pathlib import Path

import pybullet as p
import pybullet_tools.utils as pu
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from stein_mpc.models.robot import robot_scene
from stein_mpc.models.robot.robot_scene import Trajectory, JointState
from stein_mpc.utils.helper import get_project_root

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", type=str)
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.005, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.005, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.75, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=1, type=float)
parser.add_argument(
    "-t", "--tags-to-vis", default="all", choices=["all"] + robot_scene.tag_names
)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)


def create_spline_trajectory(knots, timesteps=100, device="cpu"):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


if __name__ == "__main__":
    args = parser.parse_args()

    project_path = get_project_root()
    urdf_path = project_path.joinpath("robot_resources/panda/urdf/panda.urdf")
    # choose links to operate
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
    target_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        # "panda_joint8",
        # "panda_hand_joint",
    ]

    folder = Path(args.data_folder)
    tag_name = folder.name.split("-")[1] + "_panda"

    scene = robot_scene.RobotScene(tag_name=tag_name)
    print(f"Scene: {tag_name}\n")
    robot = scene.build_robot(
        urdf_path=str(urdf_path),
        target_link_names=target_link_names,
        end_effector_link_name="panda_hand",
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
    )
    scene.build_scene()

    subfolders = list(folder.glob("*"))
    i = -1
    for subfolder in subfolders:
        if subfolder.is_dir():
            i += 1
            for kernel_method in subfolder.glob("*"):
                _name = f"{folder.name.split('-')[1]} [{kernel_method.name}]"
                print(_name)

                data = torch.load(kernel_method / "data.pt", map_location="cpu")
                trace = data["trace"]

                traj = create_spline_trajectory(trace[-1, ...], timesteps=100)

                for j in range(traj.shape[0]):
                    tid = pu.add_text(
                        f"{_name}. Req: {i + 1}/{len(subfolders)}  Ep: {j + 1}/{traj.shape[0]}",
                        position=[0, 0, 1.5],
                    )
                    scene.play(
                        Trajectory(
                            [
                                JointState(
                                    name=target_joint_names, position=traj[j, t, ...]
                                )
                                for t in range(traj.shape[1])
                            ]
                        ),
                        target_joint_names,
                        interpolate_step=5,
                        delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                        delay_between_joint=args.delay_between_joint,
                    )

                    time.sleep(args.delay_between_solution)
                    pu.remove_debug(tid)

                time.sleep(args.delay_between_scene)

    scene.clear()
    robot.destroy()
