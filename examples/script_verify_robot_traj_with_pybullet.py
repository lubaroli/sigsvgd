import glob
import time
from os import path
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_tools.utils as pu
import torch
import yaml
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import trange

this_directory = Path(path.abspath(path.dirname(__file__)))

device = "cpu"


def build_scene(fname):
    added_bodies = []

    with open(fname, "r") as stream:
        yamlobj = yaml.safe_load(stream)

    print(fname)
    if "fixed_frame_transforms" in yamlobj:
        p.resetBasePositionAndOrientation(
            robot.pyb_robot_id,
            yamlobj["fixed_frame_transforms"][0]["transform"]["translation"],
            yamlobj["fixed_frame_transforms"][0]["transform"]["rotation"],
        )

    for obj in yamlobj["world"]["collision_objects"]:
        # pprint(obj)

        if "primitives" in obj:
            assert len(obj["primitives"]) == 1

            # primitive objects
            _type = obj["primitives"][0]["type"]
            _ = obj["primitive_poses"][0]
            pose = _["position"], _["orientation"]
            dim = obj["primitives"][0]["dimensions"]
            if _type == "box":
                added_bodies.append(pu.create_box(*dim, pose=pose))
            elif _type == "cylinder":
                dim = obj["primitives"][0]["dimensions"]
                added_bodies.append(
                    pu.create_cylinder(radius=dim[1], height=dim[0], pose=pose)
                )
            else:
                raise NotImplementedError(_type)

        elif "meshes" in obj:
            assert len(obj["meshes"]) == 1

            # meshes
            _ = obj["mesh_poses"][0]
            pose = _["position"], _["orientation"]
            mesh = obj["meshes"][0]["vertices"], obj["meshes"][0]["triangles"]
            added_bodies.append(pu.create_mesh(mesh, pose=pose))

        else:
            raise NotImplementedError(str(obj))

    return added_bodies


def test_all_scene():
    yamls = list(glob.glob("robodata/*-scene0001.yaml"))
    for yaml_fname in yamls:
        added_bodies = build_scene(yaml_fname)
        time.sleep(1)
        while len(added_bodies) > 0:
            bid = added_bodies.pop(0)
            p.removeBody(bid)

    # test_all_scene()


#
# print(box)
# print(dir(box))


def create_spline_trajectory(knots, timesteps=100):
    t = torch.linspace(0, 1, timesteps).to(device)
    t_knots = torch.linspace(0, 1, knots.shape[-2]).to(device)
    coeffs = natural_cubic_spline_coeffs(t_knots, knots)
    spline = NaturalCubicSpline(coeffs)
    return spline.evaluate(t)


def get_collision_fn(
    body,
    joints,
    obstacles=None,
    attachments=None,
    self_collisions=True,
    disabled_collisions=None,
    custom_limits=None,
    use_aabb=False,
    cache=False,
    max_distance=pu.MAX_DISTANCE,
    check_joint_limits=False,
    **kwargs,
):
    if custom_limits is None:
        custom_limits = dict()
    if disabled_collisions is None:
        disabled_collisions = set()
    if attachments is None:
        attachments = list()
    if obstacles is None:
        obstacles = list()
    check_link_pairs = (
        pu.get_self_link_pairs(body, joints, disabled_collisions)
        if self_collisions
        else []
    )
    moving_links = frozenset(
        link for link in pu.get_moving_links(body, joints) if pu.can_collide(body, link)
    )  # TODO: propagate elsewhere
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [pu.CollisionPair(body, moving_links)] + list(
        map(pu.parse_body, attached_bodies)
    )
    get_obstacle_aabb = pu.cached_fn(
        pu.get_buffered_aabb, cache=cache, max_distance=max_distance / 2.0, **kwargs
    )
    if check_joint_limits:
        limits_fn = pu.get_limits_fn(body, joints, custom_limits=custom_limits)
    else:
        limits_fn = lambda *args: False

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True
        pu.set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        # wait_for_duration(1e-2)
        get_moving_aabb = pu.cached_fn(
            pu.get_buffered_aabb, cache=True, max_distance=max_distance / 2.0, **kwargs
        )

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            if (
                not use_aabb
                or pu.aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))
            ) and pu.pairwise_link_collision(
                body, link1, body, link2
            ):  # , **kwargs):
                return True

        for body1, body2 in pu.product(moving_bodies, obstacles):
            if (
                not use_aabb
                or pu.aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pu.pairwise_collision(body1, body2, **kwargs):
                return True
        return False

    return collision_fn


if __name__ == "__main__":
    target_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_joint8",
        "panda_hand_joint",
    ]
    robot = build_robot()
    robot.print_info()

    joint_indexes = robot.joint_name_to_indexes(target_joint_names)
    # joint_indexes = list(range(1, 9+1))
    print(joint_indexes)
    print(target_joint_names)
    # riestn
    added_bodies = []

    for robot_problem_dir in (this_directory / ".." / "data").glob("robot-*"):
        print(robot_problem_dir.name)

        while len(added_bodies) > 0:
            bid = added_bodies.pop(0)
            p.removeBody(bid)
        added_bodies = build_scene(
            this_directory
            / ".."
            / "robodata"
            / f"{robot_problem_dir.name.split('-')[1]}_panda-scene0001.yaml"
        )
        collision_func = get_collision_fn(
            robot.pyb_robot_id,
            joint_indexes,
            obstacles=added_bodies,
            self_collisions=False,
        )

        for problem_instance in robot_problem_dir.glob("*"):
            if not problem_instance.is_dir():
                continue
            for kernel_instance in problem_instance.glob("*"):
                print(kernel_instance)
                data = torch.load(kernel_instance / "data.pt", map_location="cpu")

                if kernel_instance.name == "pathsig":
                    # wu_data_dict, data_dict
                    trace = torch.cat([data[0]["trace"], data[1]["trace"]], axis=0)
                elif kernel_instance.name == "sgd":
                    trace = data["trace"]
                else:
                    raise NotImplementedError()

                # print(trace.shape)

                # for opt_idx in trange(traj.shape[0]):
                opt_idx = -1

                collision_counts = []
                for particle_idx in trange(traj.shape[1], desc="particle"):

                    collision_counts.append(0)
                    for time_idx in trange(
                        traj.shape[2], desc="time-step", leave=False
                    ):
                        joint = traj[opt_idx, particle_idx, time_idx, :].numpy()
                        # print(joint)
                        # robot.set_qs(joint, joint_indexes)
                        collision_counts[-1] += int(collision_func(joint, True))

                        # time.sleep(0.03)

                collision_counts = np.array(collision_counts) / traj.shape[2] * 100
                print(
                    f"{np.mean(collision_counts):.4f} +- {np.std(collision_counts):.4f} %"
                )
                print()
                print()
            #         break
            # break
