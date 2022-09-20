from typing import Callable, Tuple

import torch
import torch.autograd as autograd
import torch.optim as optim

from . import SVGD
from ..kernels import BaseKernel, TrajectoryKernel, PathSigKernel
from sigkernel import SigKernel


class TrajectorySVGD(SVGD):
    """An implementation of Stein variational gradient descent that supports custom kernels.
    """

    def __init__(
        self,
        kernel: BaseKernel,
        log_p: Callable = None,
        log_prior: Callable = None,
        bw_scale: float = 1.0,
        gradient_mask=None,
        optimizer_class: optim.Optimizer = optim.Adam,
        **opt_args,
    ):
        super().__init__(
            kernel, log_p, log_prior, bw_scale, optimizer_class, **opt_args
        )
        self.gradient_mask = gradient_mask

    def _compute_kernel(self, X, **kwargs):
        if isinstance(self.kernel, TrajectoryKernel):
            # Selects only X, Y position starting on time t+1
            tau = kwargs["trajectories"][..., 1:, :2]
            if kwargs["sample_shape"]:
                tau = tau.mean(0)
            k_xx, grad_k = (0, 0)
            for i in range(tau.shape[-1]):
                # computing gradients w.r.t. policies is equal to computing the sum
                # of gradients w.r.t. sampled actions
                k_xx_i, grad_k_i = self.kernel(
                    tau[..., i], tau[..., i].detach(), X, compute_grad=True
                )
                k_xx = k_xx + k_xx_i
                grad_k = grad_k + grad_k_i
            k_xx = k_xx / tau.shape[-1]
            grad_k = grad_k.flatten(1) / tau.shape[-1]
        elif isinstance(self.kernel, PathSigKernel):
            # Selects only X, Y position starting on time t+1
            tau = kwargs["trajectories"][..., 1:, :2]
            if kwargs["sample_shape"]:
                tau = tau.mean(0)
            k_xx, grad_k = self.kernel(tau, tau, X, compute_grad=True)
            grad_k = grad_k.detach().flatten(1)
        elif isinstance(self.kernel, SigKernel):
            # Selects only X, Y position starting on time t+1
            tau = kwargs["trajectories"][..., 1:, :2]
            if kwargs["sample_shape"]:
                tau = tau.mean(0)
            k_xx = self.kernel.compute_Gram(
                tau.double(), tau.detach().double(), sym=False
            )
            k_xx = k_xx.float()
            # grad_k = torch.autograd.grad(k_xx.sum(), X)[0].flatten(1)
            grad_k = torch.autograd.grad(k_xx.sum(), kwargs["actions"])[0]
            if kwargs["sample_shape"]:
                grad_k = grad_k.mean(0).flatten(1)
            else:
                grad_k = grad_k.flatten(1)
        elif hasattr(self.kernel, "analytic_grad") and self.kernel.analytic_grad:
            # grad_k is batch x batch x dim
            k_xx, grad_k = self.kernel(X, X)
            grad_k = grad_k.sum(1)  # aggregates gradient wrt to first input
        else:
            X = X.detach().requires_grad_(True)
            k_xx = self.kernel(X, X.detach(), compute_grad=False)
            grad_k = autograd.grad(-k_xx.sum(), X)[0]
        return k_xx.detach(), grad_k.detach()

    def _velocity(
        self, X: torch.Tensor, grad_log_p: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        velocity, iter_dict = super()._velocity(X, grad_log_p, **kwargs)
        return velocity * self.gradient_mask, iter_dict
