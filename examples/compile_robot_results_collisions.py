import argparse
import os
import sys
import shutil
from os import path
from pathlib import Path
import pandas as pd

import numpy as np
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
import tqdm
import pickle

from stein_mpc.models.robot import robot_scene
from stein_mpc.models.robot.robot_scene import Trajectory, JointState, PathRequest
from stein_mpc.models.robot.robot_simulator import PandaRobot
from stein_mpc.utils.helper import get_project_root

this_directory = Path(path.abspath(path.dirname(__file__)))
sys.path.insert(0, str(this_directory / ".."))

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", type=str)


def create_spline_trajectory(knots, timesteps=100, device="cpu"):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


def get_stats(folder, NUM_TIMESTEP_PER_SAMPLE=100):
    tag_name = folder.name.split("-")[1]  # + "_panda"

    robot = PandaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        # p_client=p.GUI,
        include_plane=False,
    )
    scene = robot_scene.RobotScene(robot=robot, tag_name=tag_name)
    obstacles = scene.build_scene()

    is_colliding_with_env = robot.get_collision_functor(
        obstacles=obstacles,
        self_collisions=True,
    )
    get_colliding_points = robot.get_colliding_points_functor(
        obstacles=obstacles,
        self_collisions=True,
    )

    print(f"Scene: {tag_name}\n")

    rand_q = robot.get_joints_limits()

    from collections import defaultdict

    method_results = defaultdict(lambda: [])

    method_results_pct_free = defaultdict(lambda: [])

    subfolders = list(s for s in folder.glob("*") if s.is_dir())
    i = -1
    for subfolder in tqdm.tqdm(subfolders, desc=f"Scene: {tag_name}"):
        i += 1

        req_i = int(subfolder.name.split("-")[0])
        req = PathRequest.from_yaml(scene.request_paths[req_i])
        q_initial = torch.as_tensor(req.start_state.get(robot.target_joint_names))
        q_target = torch.as_tensor(req.target_state.get(robot.target_joint_names))

        for kernel_method in subfolder.glob("*"):
            _name = f"{folder.name.split('-')[1]} [{kernel_method.name}]"

            if not (kernel_method / "data.pt").is_file():
                continue

            data = torch.load(kernel_method / "data.pt", map_location="cpu")
            trace = data["trace"]

            x = trace[-1, ...]
            knots = torch.cat(
                (
                    q_initial.repeat(x.shape[0], 1, 1),
                    x,
                    q_target.repeat(x.shape[0], 1, 1),
                ),
                1,
            )

            traj = create_spline_trajectory(knots, timesteps=NUM_TIMESTEP_PER_SAMPLE)

            for j in range(traj.shape[0]):
                trajectory = Trajectory(
                    [
                        JointState(
                            name=robot.target_joint_names,
                            position=traj[j, t, ...],
                        )
                        for t in range(traj.shape[1])
                    ]
                )

                target_joint_indexes = scene.robot.joint_name_to_indexes(
                    robot.target_joint_names
                )

                _traj_qs = trajectory.get(robot.target_joint_names)

                contact_distances = []
                for qs in _traj_qs[1:-1]:
                    _dist = [c.contactDistance for c in get_colliding_points(qs)]
                    if len(_dist) > 0:
                        contact_distances.append(min(_dist))
                    else:
                        contact_distances.append(0)

                method_results[kernel_method.name].append(contact_distances)

    scene.clear()
    robot.destroy()

    return method_results


if __name__ == "__main__":
    args = parser.parse_args()

    project_path = get_project_root()

    # folder = Path(args.data_folder)
    # tag_name = folder.name.split("-")[1]  # + "_panda"
    # all_stats[tag_name] = get_stats(folder, NUM_TIMESTEP_PER_SAMPLE=10)

    cache_folder = Path("cached_pkl_stats")

    if cache_folder.exists():
        while True:
            ansewr = (
                input(
                    f"Cache folder {cache_folder} already exists.\nLoad existing results (Y) or delete existing and recompute (d)? [Y/d] "
                )
                .lower()
                .strip()
            )
            if ansewr in ("y", ""):
                break
            elif ansewr == "n":
                shutil.rmtree(cache_folder)
                break

    if not cache_folder.exists():
        cache_folder.mkdir(exist_ok=True)
        for folder in tqdm.tqdm(
            [_ for _ in Path(args.data_folder).glob("*") if _.is_dir()], desc="scenario"
        ):
            cache_fname = cache_folder / f"{folder.name}.pkl"
            if os.path.exists(cache_fname):
                continue
            tag_name = folder.name.split("-")[1]  # + "_panda"
            _stat = get_stats(folder, NUM_TIMESTEP_PER_SAMPLE=100)
            with open(cache_fname, "wb") as f:
                pickle.dump(dict(_stat), f)

    # load from disk
    all_stats = {}
    for pkl in Path(cache_folder).glob("*.pkl"):
        with open(pkl, "rb") as f:
            all_stats[pkl.name[:-4]] = pickle.load(f)

    def shorten_tag_name(name):
        token = "robot-"
        if name.startswith(token):
            name = name[len(token) :]
        token = "_panda"
        if name.endswith(token):
            name = name[: -len(token)]
        return name

    df_jsons = []
    for tag_name, method_results in all_stats.items():
        df_jsons.append({})
        df_json = df_jsons[-1]

        df_json["env"] = shorten_tag_name(tag_name)
        # print(f"{tag_name:<50}", " & ", end="")
        for opt_method, contact_distances in method_results.items():
            max_depths = [np.mean(depths) * 1000 for depths in contact_distances]
            colfree = [
                sum(1 for d in depths if d == 0) * 100 / len(depths)
                for depths in contact_distances
            ]
            df_json[f"{opt_method}__md_μ"] = -np.mean(max_depths)
            df_json[f"{opt_method}__md_σ"] = -np.std(max_depths)
            df_json[f"{opt_method}__cr_μ"] = -np.mean(colfree)
            df_json[f"{opt_method}__cr_σ"] = -np.std(colfree)

            # print(
            #     f"{:>5.2f} ({np.std(max_depths):.2f}) & {np.mean(colfree):.2f} ({np.std(colfree):.2f}) ",
            #     end="",
            # )
        # print()
    df = pd.DataFrame(df_jsons)
    print("\n=================================")
    print(df)
