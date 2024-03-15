import argparse

import numpy as np
import pybullet as p
import tqdm

from src.models.robot import robot_scene
from src.models.robot.robot_simulator import PandaRobot
from src.utils.helper import get_project_root

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    args = parser.parse_args()
    project_path = get_project_root()

    robot = PandaRobot(
        device="cpu",
        p_client=p.DIRECT,
        # p_client=p.GUI,
        include_plane=False,
    )

    tag_name = robot_scene.tag_names[0]
    scene = robot_scene.RobotScene(robot=robot, tag_name=tag_name)

    print(f"Scene: {tag_name}\n")
    limit_lowers, limit_uppers = robot.get_joints_limits(as_tensor=False)
    limit_lowers = np.array(limit_lowers)
    limit_uppers = np.array(limit_uppers)

    collision_functor = robot.get_collision_functor()

    queried_qs = []
    queried_result = []
    for i in tqdm.trange(1_000_000):
        rand_qs = (
            np.random.rand(len(limit_lowers)) * (limit_uppers - limit_lowers)
            + limit_lowers
        )
        queried_qs.append(rand_qs)
        queried_result.append(collision_functor(rand_qs))
    robot.destroy()

    np.save(
        project_path / "robodata" / "panda_self_collision_dataset.npy",
        dict(queried_qs=queried_qs, queried_result=queried_result),
    )
