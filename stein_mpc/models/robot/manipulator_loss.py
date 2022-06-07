import numpy as np
import torch as th

from stein_mpc.models.robot.differentiable_robot import DifferentiableRobot


class JacoJointsLoss:
    """
    Loss function of angular distance with manipulator joints
    """

    def __init__(
        self,
        target_q: th.Tensor,
        mul_factor_q: float = 100.0,
        mul_factor_qd: float = 1.0,
        mul_factor_qdd: float = 0.01,
    ):
        self.target_q = target_q
        self.mul_factor_q = mul_factor_q
        self.mul_factor_qd = mul_factor_qd
        self.mul_factor_qdd = mul_factor_qdd

        # range from -1 (exactly opposite) to 1 (exactly same)
        self.cos_sim = th.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _angular_distance(self, theta):
        """
        angluar dist = 1 - arccor( cos-sim( a, b ) ) / pi

        Range from 0 (opposite) to 1 (similar)
        """
        return th.arccos(self.cos_sim(theta, self.target_q)) / np.pi

    def inst_cost(self, states, controls=None, n_pol=1, debug=None):
        theta, theta_d = states.chunk(2, dim=1)

        control_cost = 0
        if controls is not None:
            control_cost = self.mul_factor_qdd * controls ** 2

        theta_cost = self.mul_factor_q * self._angular_distance(theta)
        vel_cost = self.mul_factor_qd * theta_d ** 2

        # print(theta_cost, vel_cost, control_cost)
        return theta_cost + (vel_cost + control_cost).mean(1)

    def term_cost(self, *args, **kwargs):
        return self.inst_cost(*args, **kwargs).squeeze()


class JacoEndEffectorLoss:
    """
    Loss function of MSE with manipulator end effector
    """

    def __init__(
        self,
        differentiable_robot: DifferentiableRobot,
        target_ee_name: str,
        target_ee_pos: th.Tensor,
    ):
        self.differentiable_robot = differentiable_robot
        self.target_ee_name = target_ee_name
        self.target_ee_pos = target_ee_pos

    def _ee_mse(self, theta):
        pos, rot = self.differentiable_robot.robot_model.compute_forward_kinematics(
            theta, self.target_ee_name
        )
        return ((pos - self.target_ee_pos) ** 2).mean(1)

    def inst_cost(self, states, controls=None, n_pol=1, debug=None):
        theta, theta_d = states.chunk(2, dim=1)

        return self._ee_mse(theta)

    def term_cost(self, *args, **kwargs):
        return self.inst_cost(*args, **kwargs).squeeze()


class JacoEndEffectorTerminateOnlyLoss(JacoEndEffectorLoss):
    """
    Similar to JacoEndEffectorLoss, but only apply cost at terminating state
    """

    def inst_cost(self, states, *args, **kwargs):
        return th.zeros([states.shape[0]])

    def term_cost(self, states, *args, **kwargs):
        theta, theta_d = states.chunk(2, dim=1)
        return self._ee_mse(theta)
