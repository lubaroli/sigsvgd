import argparse
import time

import pybullet as p

from stein_mpc.models.ros_robot import robot_scene
from stein_mpc.utils.helper import get_project_root

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.01, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.2, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.5, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=2, type=float)
parser.add_argument("-s", "--interpolate-step", default=50, type=int)
parser.add_argument(
    "-t", "--tags-to-vis", default="all", choices=["all"] + robot_scene.tag_names
)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)

if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.tags_to_vis == "all":
        args.tags_to_vis = list(robot_scene.tag_names)
    else:
        args.tags_to_vis = args.tags_to_vis.split(",")

    print("\n\n")

    for tag_name in args.tags_to_vis:
        print("=" * 40)

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
        for i, (request_fn, traj_fn) in enumerate(
                zip(scene.request_paths, scene.trajectory_paths)
        ):
            if i >= args.num_solution_to_vis:
                break

            print("-" * 20)
            request = robot_scene.PathRequest.from_yaml(request_fn)
            traj = robot_scene.Trajectory.from_yaml(traj_fn)

            print(f"Path Request:\n{request}\n")
            print(f"Example Trajectory:\n{traj}\n")

            import pybullet_tools.utils as pu

            tid = pu.add_text(
                f"{tag_name}: traj {i + 1} / {min(len(scene), args.num_solution_to_vis)}",
                position=[0, 0, 1.5]
            )
            scene.play(
                traj,
                target_joint_names,
                interpolate_step=args.interpolate_step,
                delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                delay_between_joint=args.delay_between_joint,
            )

            time.sleep(args.delay_between_solution)
            pu.remove_debug(tid)

        time.sleep(args.delay_between_scene)

        scene.clear()
        robot.destroy()
