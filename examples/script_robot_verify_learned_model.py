import argparse
import time

import numpy as np
import pybullet as p
import pybullet_tools.utils as pu
import torch
import torch.optim as optim
import tqdm

from src.models.robot import robot_scene
from src.models.robot.robot_simulator import PandaRobot
from src.models.robot_learning import (
    continuous_occupancy_map,
    continuous_self_collision_pred,
)
from src.utils.helper import get_project_root

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.01, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.01, type=float)
parser.add_argument("-d2", "--delay-between-solution", default=0.05, type=float)
parser.add_argument("-d3", "--delay-between-scene", default=2, type=float)
parser.add_argument("-s", "--interpolate-step", default=10, type=int)
parser.add_argument(
    "-t", "--tags-to-vis", default="all", choices=["all"] + robot_scene.tag_names
)
parser.add_argument("-n", "--num-solution-to-vis", default=100, type=int)


class PBText:
    def __init__(self, position):
        self.tid = None
        self.position = position
        self._last_text = None

    def set_text(self, text):
        if self.tid is not None:
            if self._last_text == text:
                # skip
                return
            pu.remove_debug(self.tid)
            self.tid = None
        self.tid = pu.add_text(text, position=self.position,)
        self._last_text = text


if __name__ == "__main__":
    project_path = get_project_root()

    args = parser.parse_args()

    if args.tags_to_vis == "all":
        args.tags_to_vis = list(robot_scene.tag_names)
    else:
        args.tags_to_vis = args.tags_to_vis.split(",")
    print("\n\n")

    torch.manual_seed(123)

    robot = PandaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
        include_plane=False,
    )
    limit_lowers, limit_uppers = robot.get_joints_limits()

    title = PBText([-1, -1, 2])
    title.set_text(
        "=== collision (we randomly optimise for 1 criteria to test it out) ==="
    )
    text_collision_env = PBText([0, 0, 1.75])
    text_collision_self = PBText([0, 0, 1.5])

    for tag_name in args.tags_to_vis:
        print("=" * 40)

        scene = robot_scene.RobotScene(robot=robot, tag_name=tag_name)
        print(f"Scene: {tag_name}\n")

        obstacles = scene.build_scene()
        occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)
        selfcollision_model = continuous_self_collision_pred.load_trained_model(
            robot.self_collision_model_weight_path
        )

        is_colliding_with_env = robot.get_collision_functor(
            obstacles=obstacles, self_collisions=False, check_joint_limits=False,
        )

        is_colliding_with_self = robot.get_collision_functor(
            self_collisions=True, check_joint_limits=False,
        )
        text_collision_env.set_text("")
        i = 0
        while i < 10:
            #############################################
            qs = torch.rand(robot.dof) * (limit_uppers - limit_lowers) + limit_lowers
            qs.requires_grad_()
            optimizer = optim.Adam([qs], lr=5e-2)
            robot.set_qs(qs)

            if not is_colliding_with_self(qs) and not is_colliding_with_env(qs):
                # if not is_colliding_with_env(qs):
                time.sleep(0.02)
                continue
            i += 1

            traj = robot_scene.Trajectory([])
            optimise_for_env = np.random.rand() > 0.5
            for j in tqdm.trange(40, desc="gradient descent"):
                optimizer.zero_grad()
                cost = occmap(robot.qs_to_joints_xs(qs))
                selfcol_cost = selfcollision_model(qs)

                robot.set_qs(qs)
                text_collision_env.set_text(
                    f"Env {'YES' if is_colliding_with_env(qs) else '___'} ({cost.sum().item():.2f})   {f'[optimising]' if optimise_for_env else ''}"
                )
                text_collision_self.set_text(
                    f"Self {'YES' if is_colliding_with_self(qs) else '___'} ({selfcol_cost.sum().item():.2f})    {f'[optimising]' if not optimise_for_env else ''}"
                )

                if optimise_for_env:
                    cost.sum().backward()
                else:
                    selfcol_cost.sum().backward()

                optimizer.step()
                traj.states.append(
                    robot_scene.JointState(
                        name=robot.target_joint_names, position=qs.tolist()
                    )
                )
                time.sleep(0.01)

            time.sleep(args.delay_between_solution)

        time.sleep(args.delay_between_scene)

        scene.clear()
    robot.destroy()
