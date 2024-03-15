import argparse
import sys
import time
from os import path
from pathlib import Path

import pybullet as p
import pybullet_tools.utils as pu
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from src.models.robot import robot_scene
from src.models.robot.robot_scene import Trajectory, JointState, PathRequest
from src.models.robot.robot_simulator import PandaRobot
from src.models.robot_learning import (
    continuous_occupancy_map,
    continuous_self_collision_pred,
)
from src.utils.helper import get_project_root

this_directory = Path(path.abspath(path.dirname(__file__)))
sys.path.insert(0, str(this_directory / ".."))

from examples.script_planning_robot import create_body_points

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", type=str)
parser.add_argument(
    "-d0", "--delay-between-interpolated-joint", default=0.001, type=float
)
parser.add_argument("-d1", "--delay-between-joint", default=0.001, type=float)
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


def compute_all_cost(
    traj, use_ee_for_traj_dist=False,
):
    """
        batch = x.shape[0]
    knots = torch.cat(
        (start_pose.repeat(batch, 1, 1), x, target_pose.repeat(batch, 1, 1)), 1
    )
    traj = create_spline_trajectory(knots, timesteps)
    """
    _original_shape = traj.shape
    # xs shape is [n_dof, batch * timesteps, 3]
    xs_all_joints = robot.qs_to_joints_xs(traj.reshape(-1, _original_shape[-1]))

    traj_of_end_effector = xs_all_joints[-1, ...]
    traj_of_end_effector = traj_of_end_effector.reshape(
        _original_shape[0], _original_shape[1], -1
    )

    # this collision prob is in the shape of [n_dof * n_pts x (batch x timesteps) x 1]
    body_points = create_body_points(xs_all_joints)
    collision_prob = occmap(body_points)
    # the following is now in the shape of [batch x timesteps]
    collision_prob = (
        collision_prob.sum(0)
        .reshape(_original_shape[0], _original_shape[1])
        .squeeze(-1)
    )
    # # we can then sums up the collision prob across timesteps
    # collision_prob = collision_prob.sum(1)

    self_collision_prob = self_collision_predictor(traj)
    # we can then sums up the collision prob across timesteps
    self_collision_prob = self_collision_prob  # .sum(1).squeeze(1)

    # compute piece-wise linear distance
    if use_ee_for_traj_dist:
        traj_dist = torch.linalg.norm(
            traj_of_end_effector[:, 1:, :] - traj_of_end_effector[:, :-1, :], dim=2
        ).sum(1)
    else:
        # xs shape is [n_joints, batch, timesteps, 3]
        _xs_all_joints = xs_all_joints.reshape(
            xs_all_joints.shape[0],
            _original_shape[0],
            _original_shape[1],
            xs_all_joints.shape[-1],
        )
        traj_dist = (
            torch.linalg.norm(
                _xs_all_joints[:, :, 1:, :] - _xs_all_joints[:, :, :-1, :], dim=3
            ).sum(0)
            # .sum(-1)
        )
    return dict(
        env_coll=collision_prob.squeeze().numpy(),
        self_coll=self_collision_prob.squeeze().numpy(),
        traj_dist=traj_dist.squeeze().numpy(),
    )


if __name__ == "__main__":
    args = parser.parse_args()

    project_path = get_project_root()

    folder = Path(args.data_folder)
    tag_name = folder.name.split("-")[1]  # + "_panda"

    robot = PandaRobot(
        device="cpu",
        # p_client=p.DIRECT,
        p_client=p.GUI,
        include_plane=False,
    )
    scene = robot_scene.RobotScene(robot=robot, tag_name=tag_name)
    scene.build_scene()
    print(f"Scene: {tag_name}\n")

    rand_q = robot.get_joints_limits()

    subfolders = list(folder.glob("*"))
    i = -1
    for subfolder in subfolders:
        if subfolder.is_dir():
            i += 1

            req_i = int(subfolder.name.split("-")[0])
            req = PathRequest.from_yaml(scene.request_paths[req_i])
            q_initial = torch.as_tensor(req.start_state.get(robot.target_joint_names))
            q_target = torch.as_tensor(req.target_state.get(robot.target_joint_names))

            gt_traj = robot_scene.Trajectory.from_yaml(scene.trajectory_paths[req_i])

            occmap = continuous_occupancy_map.load_trained_model(str(scene.weight_path))
            self_collision_predictor = continuous_self_collision_pred.load_trained_model(
                str(robot.self_collision_model_weight_path)
            )
            # occ_plot = UpdatableSequentialPlot(nrows=4)

            for kernel_method in subfolder.glob("*"):
                _name = f"{folder.name.split('-')[1]} [{kernel_method.name}]"
                print(_name)
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
                print(gt_traj)
                print(req)

                traj = create_spline_trajectory(knots, timesteps=100)

                for j in range(traj.shape[0] - 19):
                    tid = pu.add_text(
                        f"{_name}. Req: {i + 1}/{len(subfolders)}  n: {j + 1}/{traj.shape[0]}",
                        position=[-1, -1, 1.5],
                    )
                    #     occ_plot.clear()
                    #     occ_plot.figure.suptitle(f"Cost: {_name}", fontsize=20)
                    #     occ_plot.axs[0].set_title("Environment-collision Cost", fontsize=14)
                    #     occ_plot.axs[1].set_title("Self-collision Cost", fontsize=14)
                    #     occ_plot.axs[2].set_title("Piece-wise Traj len Cost", fontsize=14)
                    #     occ_plot.axs[3].set_title("ALL cumulative Cost", fontsize=14)
                    #     occ_plot.axs[3].set_xlabel("Timestep")
                    #     occ_plot.axs[0].set_xlim(0, traj.shape[1])
                    #
                    #     with torch.no_grad():
                    #         _cost_history = compute_all_cost(traj[j : j + 1, ...])
                    #         # pair-wise traj dist is one short.
                    #         _cost_history["traj_dist"] = np.array(
                    #             [0] + _cost_history["traj_dist"].tolist()
                    #         )
                    #         _cost_history_cumulative = dict(
                    #             env_coll=np.cumsum(_cost_history["env_coll"].copy()),
                    #             self_coll=np.cumsum(_cost_history["self_coll"].copy()),
                    #             traj_dist=np.cumsum(_cost_history["traj_dist"].copy()),
                    #         )
                    #
                    #     def plotting_callback(qs, i):
                    #         with torch.no_grad():
                    #             occ_plot.add_data(
                    #                 "cost: env-col",
                    #                 _cost_history["env_coll"][i],
                    #                 index=0,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.add_data(
                    #                 "cost: self-col",
                    #                 _cost_history["self_coll"][i],
                    #                 index=1,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.add_data(
                    #                 "cost: traj-dist",
                    #                 _cost_history["traj_dist"][i],
                    #                 index=2,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.add_data(
                    #                 "cost: env-col",
                    #                 _cost_history_cumulative["env_coll"][i],
                    #                 index=3,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.add_data(
                    #                 "cost: self-col",
                    #                 _cost_history_cumulative["self_coll"][i],
                    #                 index=3,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.add_data(
                    #                 "cost: traj-dist",
                    #                 _cost_history_cumulative["traj_dist"][i],
                    #                 index=3,
                    #                 auto_ylim=True,
                    #                 update=False,
                    #             )
                    #             occ_plot.update()

                    scene.play(
                        Trajectory(
                            [
                                JointState(
                                    name=robot.target_joint_names,
                                    position=traj[j, t, ...],
                                )
                                for t in range(traj.shape[1])
                            ]
                        ),
                        robot.target_joint_names,
                        interpolate_step=2,
                        delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                        delay_between_joint=args.delay_between_joint,
                        # callback=plotting_callback,
                    )
                    # occ_plot.clear()
                    # scene.play(
                    #     gt_traj,
                    #     robot.target_joint_names,
                    #     interpolate_step=5,
                    #     delay_between_interpolated_joint=args.delay_between_interpolated_joint,
                    #     delay_between_joint=args.delay_between_joint,
                    # )

                    time.sleep(args.delay_between_solution)
                    pu.remove_debug(tid)

                time.sleep(args.delay_between_scene)

    scene.clear()
    robot.destroy()
