from typing import Callable, Tuple

import torch
import torch.autograd as autograd
import torch.optim as optim

from . import SVGD
from ..kernels import BaseKernel, TrajectoryKernel, PathSigKernel


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

    def _compute_kernel(self, X, **kernel_args):
        if isinstance(self.kernel, TrajectoryKernel):
            # Selects only X, Y position starting on time t+1
            tau = kernel_args["trajectories"][..., 1:, :2]
            if kernel_args["sample_shape"]:
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
            tau = kernel_args["trajectories"][..., 1:, :2]
            if kernel_args["sample_shape"]:
                tau = tau.mean(0)
            k_xx, grad_k = self.kernel(tau, tau, X, compute_grad=True)
            grad_k = grad_k.detach().flatten(1)
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
        self, X: torch.Tensor, grad_log_p: torch.Tensor, **kernel_args
    ) -> Tuple[torch.Tensor, dict]:
        if self.log_p is None and grad_log_p is None:
            raise ValueError(
                "SVGD needs a function to evaluate the log probability of the target",
                "distribution or an estimate of the gradient for every particle.",
            )
        k_xx, grad_k = self._compute_kernel(X, **kernel_args)

        if grad_log_p is None:
            X = X.detach().requires_grad_(True)
            log_lik = self.log_p(X).sum()
            score_func = autograd.grad(log_lik, X)[0].flatten(1)
            X.detach_()
            loss = -log_lik.detach()
        else:
            score_func = grad_log_p
            loss = grad_log_p.norm()

        if self.log_prior is not None:
            X = X.detach().requires_grad_(True)
            log_prior_sum = self.log_prior(X).sum()
            log_prior_grad = torch.autograd.grad(log_prior_sum, X)[0].detach()
            score_func = score_func + log_prior_grad
            X.detach_()

        velocity = (k_xx @ score_func + grad_k) / X.shape[0]
        velocity = -velocity.reshape(X.shape) * self.gradient_mask

        iter_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": loss}
        iter_dict.update(kernel_args)
        iter_dict = {
            k: v.detach() if hasattr(v, "detach") else v for k, v in iter_dict.items()
        }
        return velocity, iter_dict
