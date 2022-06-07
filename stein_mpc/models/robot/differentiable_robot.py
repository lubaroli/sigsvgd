import functools
from typing import Tuple

import torch as th
import wrapt
from differentiable_robot_model import DifferentiableRobotModel
from differentiable_robot_model import spatial_vector_algebra, utils as diff_robot_utils
from torch import nn

from stein_mpc.models.base import BaseModel
from stein_mpc.utils.spaces import Box


####################################################################
# Patch differentiable robot library to make it supports batch mass
####################################################################


@wrapt.patch_function_wrapper(
    spatial_vector_algebra.DifferentiableSpatialRigidBodyInertia, "multiply_motion_vec"
)
def multiply_motion_vec__with_batch_mass(wrapped, self, args, kwargs):
    (smv,) = args
    mass, com, inertia_mat = self._get_parameter_values()
    if len(mass.shape) < 2:
        # use the original function
        return wrapped(smv)

    mcom = com * mass
    com_skew_symm_mat = diff_robot_utils.vector3_to_skew_symm_matrix(com)
    assert len(mass.shape) == 2, mass.shape
    assert mass.shape[1] == 1, mass.shape
    inertia = inertia_mat + mass[..., None] * (
        com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
    )

    new_lin_force = mass * smv.lin - diff_robot_utils.cross_product(mcom, smv.ang)
    new_ang_force = (inertia @ smv.ang.unsqueeze(2)).squeeze(
        2
    ) + diff_robot_utils.cross_product(mcom, smv.lin)

    return spatial_vector_algebra.SpatialForceVec(new_lin_force, new_ang_force)


@wrapt.patch_function_wrapper(
    spatial_vector_algebra.DifferentiableSpatialRigidBodyInertia, "get_spatial_mat"
)
def get_spatial_mat__with_batch_mass(wrapped, self, args, kwargs):
    mass, com, inertia_mat = self._get_parameter_values()
    if len(mass.shape) < 2:
        # use the original function
        return wrapped()

    mcom = mass * com
    com_skew_symm_mat = diff_robot_utils.vector3_to_skew_symm_matrix(com)
    inertia = inertia_mat + mass[..., None] * (
        com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
    )
    batch_size = mass.shape[0]

    mat = th.zeros((batch_size, 6, 6), device=self._device)
    mat[:, :3, :3] = inertia
    mat[:, 3, 0] = 0
    mat[:, 3, 1] = mcom[0, 2]
    mat[:, 3, 2] = -mcom[0, 1]
    mat[:, 4, 0] = -mcom[0, 2]
    mat[:, 4, 1] = 0.0
    mat[:, 4, 2] = mcom[0, 0]
    mat[:, 5, 0] = mcom[0, 1]
    mat[:, 5, 1] = -mcom[0, 0]
    mat[:, 5, 2] = 0.0

    mat[:, 0, 3] = 0
    mat[:, 0, 4] = -mcom[0, 2]
    mat[:, 0, 5] = mcom[0, 1]
    mat[:, 1, 3] = mcom[0, 2]
    mat[:, 1, 4] = 0.0
    mat[:, 1, 5] = -mcom[0, 0]
    mat[:, 2, 3] = -mcom[0, 1]
    mat[:, 2, 4] = mcom[0, 0]
    mat[:, 2, 5] = 0.0

    mat[:, 3, 3] = mass.squeeze(1)
    mat[:, 4, 4] = mass.squeeze(1)
    mat[:, 5, 5] = mass.squeeze(1)

    class ReturnObjectWrapper:
        """
        This wrapper is to make sure that when the outer function calls for
        torch repeat function (because diff-robot lib does not expects batched mass), we
        will simply returns what we had already computed instead.
        Because we are already performing batched computation.
        i.e. the line
        ```
        body.IA = body.inertia.get_spatial_mat().repeat((batch_size, 1, 1))
        ```
        will returns the wrapped object instead.
        """

        def __init__(self, mat):
            self.mat = mat

        def repeat(self, new_shape):
            assert new_shape[0] == self.mat.shape[0]
            assert new_shape[1] == new_shape[2] == 1
            return self.mat

    return ReturnObjectWrapper(mat)


####################################################################
# Finish patching differentiable robot library.
####################################################################


class BatchedDifferentiableRobotParam(nn.Module):
    """
    This is a wrapper class to for differentiable robot's internal param,
    for allowing a batched parameters.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.params = None

    def set_params(self, params: th.Tensor):
        """
        Modify the contained parameters everytime the step function is called in the
        `DifferentiableRobot` model, such that we are always using the parameters
        that the user had specified.
        """
        self.params = params

    def forward(self):
        """
        Return the contained parameters.
        """
        return self.params


# interface to the differentiable robot forward dynamics
class DifferentiableRobot(BaseModel):
    """Model for robots within the differentiable robot framework

    For more information refer to the repo:
    https://github.com/facebookresearch/differentiable-robot-model
    """

    def __init__(
        self, urdf_path, link_param_mappings=None, device="cpu", **kwargs,
    ):
        """Constructor for DifferentiableRobot.

        :param urdf_path: urdf that defines the robot to use.
        :param link_param_mappings: A dictionary that maps a link parameters to some
            tensor values. For available link param, use the `print_learnable_params`
            function.
        :param device: Set the device for setting the inference device, defaults to cpu.
        :key dt: Duration of each discrete update in s. Defaults to 0.05
        :type dt: float
        :key uncertain_params: A tuple containing the uncertain parameters of
            the forward model. Is used as keys for assigning sampled parameters
            from the `params_dist` function. Defaults to None.
        :key params_dist: A distribution to sample parameters for the forward
            model.
        """
        self.device = device
        self.include_gravity = False
        self.include_damping = False
        learnable_robot_model = DifferentiableRobotModel(
            urdf_path, name="my_robot", device=self.device
        )
        self.robot_model = learnable_robot_model
        self._learnable_params = {}

        if link_param_mappings is not None:
            for k, v in link_param_mappings.items():
                assert type(v) is th.Tensor
                # check if given parameters exists in the robot
                link_name, param_name = self._check_given_param_is_learnable(k)
                self.modify_robot_model_link_param(
                    self.robot_model, link_name, param_name, v
                )

        super().__init__(params_dict=link_param_mappings, **kwargs)

        # make all uncertain_params learnable parameters
        if "uncertain_params" in kwargs:
            for k in kwargs["uncertain_params"]:
                link_name, param_name = self._check_given_param_is_learnable(k)
                self._learnable_params[k] = BatchedDifferentiableRobotParam(k)
                self.robot_model.make_link_param_learnable(
                    link_name, param_name, self._learnable_params[k],
                )

        ########################
        # define robot limits
        limits_per_joint = self.robot_model.get_joint_limits()

        # the followings are to clamp inputs
        joint_max_velocity = [joint["velocity"] for joint in limits_per_joint]
        joint_max_force = [joint["effort"] for joint in limits_per_joint]
        self._joint_max_velocity = th.tensor(joint_max_velocity, device=self.device)
        self._joint_max_force = th.tensor(joint_max_force, device=self.device)

        joint_lower_bounds = [joint["lower"] for joint in limits_per_joint]
        joint_upper_bounds = [joint["upper"] for joint in limits_per_joint]
        self._joint_limits_min = th.tensor(joint_lower_bounds, device=self.device)
        self._joint_limits_max = th.tensor(joint_upper_bounds, device=self.device)

        # setup box bounds
        self.__observation_space = Box(
            dim=self.dof * 2,
            low=th.hstack([self._joint_limits_min, self._joint_max_velocity]),
            high=th.hstack([self._joint_limits_max, self._joint_max_velocity]),
            dtype=th.float,
        )
        self.__action_space = Box(
            dim=self.dof,
            low=-self._joint_max_force,
            high=self._joint_max_force,
            dtype=th.float,
        )

    @functools.cached_property
    def dof(self):
        return self.robot_model._n_dofs

    @functools.cached_property
    def learnable_link_name(self):
        return set([b.name for b in self.robot_model._bodies])

    @functools.cached_property
    def learnable_param_types(self):
        return {"trans", "rot_angles", "joint_damping", "mass", "inertia_mat", "com"}

    def _check_given_param_is_learnable(self, name: str) -> Tuple[str, str]:
        parts = name.split("__")
        if len(parts) != 2:
            raise ValueError(
                "Parameter name must be in the format of "
                "'${LINK_NAME}__${PARAM_TYPE}', but was "
                f"'{name}'."
            )
        link_name, param_name = parts
        nl = "\n"
        if link_name not in self.learnable_link_name:
            raise ValueError(
                f"Given link '{link_name}' does not exists. Possible choice are:\n"
                f"{nl.join(f'- {name}' for name in sorted(self.learnable_link_name))}\n"
            )
        if param_name not in self.learnable_param_types:
            raise ValueError(
                f"Given param '{param_name}' does not exists. Possible choice are:\n"
                f"{nl.join(f'- {name}' for name in sorted(self.learnable_param_types))}"
                "\n"
            )
        return link_name, param_name

    def print_learnable_params(self):
        nl = "\n"
        print(
            "All parameters are named as '${LINK_NAME}__${PARAM_TYPE}'.\n"
            "Available LINK_NAME are:\n"
            f"{nl.join(f'- {name}' for name in self.learnable_link_name)}\n"
            "Available PARAM_TYPE are:\n"
            "> Link properties:\n"
            "  - trans          shape=[1, 3]\n"
            "  - rot_angles     shape=[1, 3]\n"
            "  - joint_damping  shape=[1]\n"
            "> Link inertia:\n"
            "  - mass           shape=[1]\n"
            "  - inertia_mat    shape=[3, 3]\n"
            "  - com            shape=[1, 3]\n\n"
            f"e.g. {list(self.learnable_link_name)[0]}__mass"
        )

    @staticmethod
    def modify_robot_model_link_param(
        robot_model, link_name: str, parameter_name: str, parametrization: th.Tensor
    ):
        parent_object = robot_model._get_parent_object_of_param(
            link_name, parameter_name
        )
        # change the parameters stored in the robot model (which were
        # initially read directly from the urdf file)
        parametrization = parametrization.to(robot_model._device)
        parent_object.__delattr__(parameter_name)
        setattr(parent_object, parameter_name, lambda: parametrization)

    def print_all_current_params(self):
        for b in self.robot_model._bodies:
            print(
                f"{b.name}\n"
                f"- trans: {b.trans()}\n"
                f"- rot_angles: {b.rot_angles()}\n"
                f"- joint_damping: {b.joint_damping()}\n"
                f"- mass: {b.inertia.mass()}\n"
                f"- inertia_mat: {b.inertia.inertia_mat()}\n"
                f"- com: {b.inertia.com()}"
            )

    @property
    def observation_space(self):
        """The Pendulum observation space.

        :return: A space with the Pendulum observation space.
        :rtype: Box
        """
        return self.__observation_space

    @property
    def action_space(self):
        """The Pendulum action space.

        :return: A space with the Pendulum action space.
        :rtype: Box
        """
        return self.__action_space

    def step(self, states, actions, params_dict=None):
        """Receives tensors of current states and actions and computes the
        states for the subsequent timestep. If sampled parameters are provided,
        these must be used, otherwise default model parameters are used.

        Must be bounded by observation and action spaces.

        The differentiable_robot environment performs the following simulation. A
        state is represents by the joint configurations (q) and joint speeds (qd). An
        action is defined as the force to be applied on each joint (qdd). The
        internal simulator will then computes the dynamics of the new joint
        accelerations (qdd), of which will be applied for the duration of dt.
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f/tau: forces to be applied [batch_size x n_dofs]
            qdd: computed by forward dynamics; accelerations that are the result of
                applying forces f in state q, qd

        :param states: A tensor containing the current states of one or multiple
            trajectories.
        :type states: th.Tensor
        :param actions: A tensor containing the next planned actions of one or
            multiple trajectories.
        :type actions: th.Tensor
        :param params_dict: A tensor containing samples for the uncertain
            system parameters. Note that the number of samples must be either 1
            or the number of trajectories. If 1, a single sample is used for all
            trajectories, otherwise use one sample per trajectory.
        :type params_dict: dict
        :returns: A tensor with the next states of one or multiple trajectories.
        :rtype: th.Tensor
        """

        self.robot_model = self.robot_model

        assert len(states.shape) == 2, (
            "states must of of shape [batch,state_dof]. " f"But was {states.shape}"
        )
        if params_dict is not None:
            for k in self.uncertain_params:
                assert k in params_dict, (
                    f"{k} was previously set as uncertain_params, but was not given in "
                    f"step function (got params_dic={params_dict})."
                )
                self._learnable_params[k].set_params(params_dict[k])

            # set q to require grad to correctly force output getting gradient
            states.requires_grad_(True)
        elif self.uncertain_params is not None:
            raise RuntimeError(
                f"This object was initialised with uncertain_params="
                f"{self.uncertain_params}, but no params_dict is given "
                f"in the step function."
            )

        dt = self.dt

        input_has_batch = True
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            input_has_batch = False

        q, qd = states.clone().chunk(2, dim=-1)
        tau = actions

        if len(tau.shape) == 1:
            tau = tau.reshape(1, -1)
        # batch actions across batch
        tau = tau.repeat(q.shape[0], 1)

        # enforce force limits
        tau = th.where(tau > self._joint_max_force, self._joint_max_force, tau,)
        tau = th.where(tau < -self._joint_max_force, -self._joint_max_force, tau,)

        # perform the actual step forward
        qdd = self.robot_model.compute_forward_dynamics(
            q=q,
            qd=qd,
            f=tau,
            include_gravity=self.include_gravity,
            use_damping=self.include_damping,
        )
        # qdd = self.robot_model.compute_forward_dynamics(
        #     q=q, qd=qd, f=tau, include_gravity=self.include_gravity, use_damping=True
        # )

        qd = qd + dt * qdd

        # enforce speed limits
        qd = th.where(qd > self._joint_max_velocity, self._joint_max_velocity, qd,)
        qd = th.where(qd < -self._joint_max_velocity, -self._joint_max_velocity, qd,)

        q = q + dt * qd

        # enforce joints limits
        q = th.where(q > self._joint_limits_max, self._joint_limits_max, q,)
        q = th.where(q < self._joint_limits_min, self._joint_limits_min, q,)

        result = th.cat((q, qd), dim=-1)
        if not input_has_batch:
            result = result.squeeze(0)

        return result
