from functools import cached_property
from typing import Optional, List, Tuple, Union, Iterable

import pybullet as p
import pybullet_data
import pybullet_tools.utils as pu
import torch
from differentiable_robot_model import DifferentiableRobotModel

from src.utils.helper import get_project_root

from . import pybullet_collision_check

WorkSpaceType = torch.Tensor
ConfigurationSpaceType = torch.Tensor


class Robot:
    def __init__(
        self,
        urdf_path: str,
        target_link_names: List[str],
        target_joint_names: List[str],
        end_effector_link_name: str,
        start_position: List[float] = (0, 0, 0),
        start_orientation: List[float] = (0, 0, 0, 1),
        device=None,
        include_plane=True,
        p_client=p.DIRECT,
        has_gravity=False,
    ):
        self.device = device
        self.urdf_path = urdf_path
        self.target_link_names = target_link_names
        self.target_joint_names = target_joint_names

        assert len(start_position) == 3
        assert len(start_orientation) == 4

        ###############################################################
        # setup differentiable robot
        self.learnable_robot_model = DifferentiableRobotModel(
            urdf_path, name="my_robot", device=device
        ).eval()
        # setup pos and rot
        self.learnable_robot_model._bodies[0].pose._trans[:, ...] = torch.Tensor(
            start_position
        )
        self.learnable_robot_model._bodies[0].pose._rot[:, ...] = torch.Tensor(
            p.getMatrixFromQuaternion(start_orientation)
        ).reshape(3, 3)

        ###############################################################
        # setup pybullet
        self.physicsClient = p.connect(p_client)
        # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        if has_gravity:
            p.setGravity(0, 0, -9.81)
            # p.setGravity(0,0,-10)
        else:
            pu.disable_gravity()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        if include_plane:
            planeId = p.loadURDF("plane.urdf")
        # startOrientation = p.getQuaternionFromEualer([0, 0, 0])

        self.pyb_robot_id = p.loadURDF(
            urdf_path, start_position, start_orientation, useFixedBase=True
        )
        self.ee_joint_idx = pu.link_from_name(self.pyb_robot_id, end_effector_link_name)
        ###############################################################

        if not all(
            name in self.joints_names_to_index_mapping for name in target_joint_names
        ):
            raise ValueError(
                f"Not all target joint exists for the robot. "
                f"Given {target_joint_names}. "
                f"Contains {self.joints_names}."
            )

    @property
    def self_collision_model_weight_path(self):
        return (
            get_project_root()
            / "robodata"
            / f"selfCollisionModel_{self.__class__.__name__}.pkl"
        )

    @property
    def dof(self):
        return len(self.target_joint_names)

    def joint_name_to_indexes(self, joint_name: List[str]):
        return [
            pu.get_joint(self.pyb_robot_id, joint_name) for joint_name in joint_name
        ]

    def set_qs(
        self,
        qs: Union[ConfigurationSpaceType, Iterable[float]],
        joint_indexes: List[int] = None,
    ):
        if joint_indexes is None:
            joint_indexes = self.joint_name_to_indexes(self.target_joint_names)
        pu.set_joint_positions(self.pyb_robot_id, joint_indexes, qs)

    def ee_xs_to_qll_qs(
        self, xs: WorkSpaceType, reference_orientation: Optional[List[float]] = None
    ) -> ConfigurationSpaceType:
        """
        Given xs with size [b x 3], returns the corresponding configurations
        """
        assert xs.shape[-1] == 3

        qs = []
        for x in xs:
            qs.append(
                pu.inverse_kinematics_helper(
                    self.pyb_robot_id, self.ee_joint_idx, (x, reference_orientation)
                )
            )
        return qs

    def ee_xs_to_qs(
        self, xs: WorkSpaceType, reference_orientation: Optional[List[float]] = None
    ) -> ConfigurationSpaceType:
        qs = self.ee_xs_to_qll_qs(xs, reference_orientation)
        mapped_qs = []
        for _qs in qs:
            mapped_qs.append([])
            for idx in self.target_joint_indices:
                mapped_qs[-1].append(_qs[idx])
        return mapped_qs

    def qs_to_joints_xs(self, qs: ConfigurationSpaceType) -> WorkSpaceType:
        """
        Given batch of qs [b x d]
        Returns a batch of pos  [b x d x 3]
        """
        # ignore orientation
        return torch.stack([pose[0] for pose in self.qs_to_joints_pose(qs)])

    def qs_to_joints_pose(
        self, qs: ConfigurationSpaceType
    ) -> List[Tuple[torch.Tensor]]:
        """
        Given batch of qs
        Returns a list of pose (i.e., pos in R^3 and orientation in R^4)
        """
        # defaulting the non-used index as zero
        if len(self.joints_names) == qs.shape[-1]:
            mapped_qs = qs
        else:
            mapped_qs = torch.zeros(
                (*qs.shape[:-1], self.learnable_robot_model._n_dofs),
                device=qs.device,
                dtype=qs.dtype,
            )

            for idx in self.target_joint_indices:
                mapped_qs[..., idx] = qs[..., idx]

        joints_xs = self.learnable_robot_model.compute_forward_kinematics_all_links(
            mapped_qs
        )
        # base had been skipped
        return [joints_xs[name] for name in self.target_link_names]

    def print_info(self):
        print(pu.get_body_info(self.pyb_robot_id))

        for j_idx in range(pu.get_num_joints(self.pyb_robot_id)):
            print(f"===== Joint {j_idx} =====")
            print(pu.get_joint_info(self.pyb_robot_id, j_idx))

    def get_all_joints_limits(self, as_tensor=True):
        """Returns all available joints limits"""
        lowers = []
        uppers = []
        for limits in self.learnable_robot_model.get_joint_limits():
            lowers.append(limits["lower"])
            uppers.append(limits["upper"])
        if as_tensor:
            lowers, uppers = torch.Tensor(lowers), torch.Tensor(uppers)
        return lowers, uppers

    @cached_property
    def joints_names(self):
        names = []
        for i in range(pu.get_num_joints(self.pyb_robot_id)):
            names.append(pu.get_joint_name(self.pyb_robot_id, i))
        return names

    @cached_property
    def joints_names_to_index_mapping(self):
        return {name: i for i, name in enumerate(self.joints_names)}

    @cached_property
    def target_joint_indices(self):
        return [
            self.joints_names_to_index_mapping[name] for name in self.target_joint_names
        ]

    def get_joints_limits(self, as_tensor=True):
        """Returns all requested (target) joints limits"""
        all_joint_limits = self.get_all_joints_limits(as_tensor=False)

        lowers = []
        uppers = []
        for target_joint_name in self.target_joint_names:
            idx = self.joints_names.index(target_joint_name)
            lowers.append(all_joint_limits[0][idx])
            uppers.append(all_joint_limits[1][idx])
        if as_tensor:
            lowers, uppers = torch.Tensor(lowers), torch.Tensor(uppers)
        return lowers, uppers

    def get_colliding_points_functor(self, **kwargs):
        return self.__get_collision_base_functor(_return_closest_points=True, **kwargs)

    def get_collision_functor(self, **kwargs):
        return self.__get_collision_base_functor(_return_closest_points=False, **kwargs)

    def __get_collision_base_functor(
        self,
        _return_closest_points,
        obstacles: Optional[List[int]] = None,
        attachments=None,
        self_collisions: bool = True,
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

        joint_indexes = self.joint_name_to_indexes(self.target_joint_names)
        if not self_collisions:
            check_link_pairs = []
        else:
            check_link_pairs = pu.get_self_link_pairs(
                self.pyb_robot_id, joint_indexes, disabled_collisions
            )

        moving_links = frozenset(
            link
            for link in pu.get_moving_links(self.pyb_robot_id, joint_indexes)
            if pu.can_collide(self.pyb_robot_id, link)
        )  # TODO: propagate elsewhere
        attached_bodies = [attachment.child for attachment in attachments]
        moving_bodies = [pu.CollisionPair(self.pyb_robot_id, moving_links)] + list(
            map(pu.parse_body, attached_bodies)
        )
        get_obstacle_aabb = pu.cached_fn(
            pu.get_buffered_aabb, cache=cache, max_distance=max_distance / 2.0, **kwargs
        )
        if check_joint_limits:
            limits_fn = pu.get_limits_fn(
                self.pyb_robot_id, joint_indexes, custom_limits=custom_limits
            )
        else:
            limits_fn = lambda *args: False

        def check_aabb_overlap(aabb1, aabb2):
            if not use_aabb:
                # skip guarding against aabb
                return True
            return pu.aabb_overlap(aabb1, aabb2)

        kwargs_dict = dict(
            pyb_robot_id=self.pyb_robot_id,
            moving_bodies=moving_bodies,
            obstacles=obstacles,
            attachments=attachments,
            max_distance=max_distance,
            limits_fn=limits_fn,
            get_obstacle_aabb=get_obstacle_aabb,
            joint_indexes=joint_indexes,
            check_aabb_overlap=check_aabb_overlap,
            check_link_pairs=check_link_pairs,
            **kwargs,
        )
        if _return_closest_points:
            return pybullet_collision_check.get_colliding_points_functor(**kwargs_dict)
        else:
            return pybullet_collision_check.get_collision_functor(**kwargs_dict)

    def destroy(self):
        p.disconnect(self.physicsClient)

    # def get_joints_limits(self):
    #     lowers = []
    #     uppers = []
    #     for j_idx in range(pu.get_num_joints(self.pyb_robot_id)):
    #         limit = pu.get_joint_limits(self.pyb_robot_id, j_idx)
    #         lowers.append(limit[0])
    #         uppers.append(limit[1])
    #     return torch.Tensor(lowers),torch.Tensor(uppers)


class PandaRobot(Robot):
    def __init__(self, **kwargs):
        project_path = get_project_root()

        self.urdf_path = project_path.joinpath("robot_resources/panda/urdf/panda.urdf")
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
        super().__init__(
            urdf_path=str(self.urdf_path),
            target_link_names=target_link_names,
            target_joint_names=target_joint_names,
            end_effector_link_name="panda_hand",
            **kwargs,
        )
