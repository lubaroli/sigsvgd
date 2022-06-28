from typing import Optional, List, Tuple

import pybullet as p
import pybullet_data
import pybullet_tools
import pybullet_tools.utils
import torch

from differentiable_robot_model import DifferentiableRobotModel

WorkSpaceType = torch.Tensor
ConfigurationSpaceType = torch.Tensor


class Robot:
    def __init__(
        self,
        urdf_path: str,
        target_link_names: List[str],
        end_effector_link_name: str,
        start_position: List[float] = (0, 0, 0),
        start_orientation: List[float] = (0, 0, 0, 1),
        device=None,
    ):
        self.urdf_path = urdf_path
        self.target_link_names = target_link_names

        assert len(start_position) == 3
        assert len(start_orientation) == 4

        # TODO: set initial frame offset

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
        self.physicsClient = p.connect(p.DIRECT)
        # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        pybullet_tools.utils.disable_gravity()
        # p.setGravity(0,0,-10)
        p.setGravity(0, 0, -9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        planeId = p.loadURDF("plane.urdf")
        # startOrientation = p.getQuaternionFromEualer([0, 0, 0])

        self.pyb_robot_id = p.loadURDF(
            urdf_path, start_position, start_orientation, useFixedBase=True
        )
        self.ee_joint_idx = pybullet_tools.utils.link_from_name(
            self.pyb_robot_id, end_effector_link_name
        )
        ###############################################################

    def ee_xs_to_qs(
        self, xs: WorkSpaceType, reference_orientation: Optional[List[float]] = None
    ) -> ConfigurationSpaceType:
        """
        Given xs with size [b x 3], returns the corresponding configurations
        """
        assert xs.shape[-1] == 3

        qs = []
        for x in xs:
            qs.append(
                pybullet_tools.utils.inverse_kinematics_helper(
                    self.pyb_robot_id, self.ee_joint_idx, (x, reference_orientation)
                )
            )
        return qs

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
        joints_xs = self.learnable_robot_model.compute_forward_kinematics_all_links(qs)
        # base had been skipped
        return [joints_xs[name] for name in self.target_link_names]

    def print_info(self):
        print(pybullet_tools.utils.get_body_info(self.pyb_robot_id))

        for j_idx in range(pybullet_tools.utils.get_num_joints(self.pyb_robot_id)):
            print(f"===== Joint {j_idx} =====")
            print(pybullet_tools.utils.get_joint_info(self.pyb_robot_id, j_idx))
