import argparse
import time

import pybullet as p

from src.models.robot import robot_scene
from src.models.robot.robot_scene import Trajectory, JointState
from src.models.robot.robot_simulator import PandaRobot
from src.utils.helper import get_project_root

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
parser.add_argument(
    "-r",
    "--visualise-request",
    help="visualise the request start and target directly, "
    "instead of the example trajectory",
    action="store_true",
)

if __name__ == "__main__":
    project_path = get_project_root()

    args = parser.parse_args()

    if args.tags_to_vis == "all":
        args.tags_to_vis = list(robot_scene.tag_names)
    else:
        args.tags_to_vis = args.tags_to_vis.split(",")
    print("\n\n")

    robot = PandaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
        include_plane=False,
    )
    for tag_name in args.tags_to_vis:
        print("=" * 40)

        scene = robot_scene.RobotScene(robot=robot, tag_name=tag_name)
        print(f"Scene: {tag_name}\n")

        scene.build_scene()
        for i, (request_fn, traj_fn) in enumerate(
            zip(scene.request_paths, scene.trajectory_paths)
        ):
            if i >= args.num_solution_to_vis:
                break

            print("-" * 20)
            request = robot_scene.PathRequest.from_yaml(request_fn)

            if args.visualise_request:
                traj = Trajectory([request.start_state, request.target_state])
            else:
                traj = robot_scene.Trajectory.from_yaml(traj_fn)

            print(f"Path Request:\n{request}\n")
            print(f"Example Trajectory:\n{traj}\n")

            import pybullet_tools.utils as pu

            tid = pu.add_text(
                f"{tag_name}: traj {i + 1} / {min(len(scene), args.num_solution_to_vis)}",
                position=[0, 0, 1.5],
            )
            scene.play(
                traj,
                robot.target_joint_names,
                interpolate_step=args.interpolate_step,
                delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                delay_between_joint=args.delay_between_joint,
            )

            time.sleep(args.delay_between_solution)
            pu.remove_debug(tid)

        time.sleep(args.delay_between_scene)

        scene.clear()
    robot.destroy()
