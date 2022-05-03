from typing import Callable, Union

import torch

from ..utils.math import pw_dist_sq, scaled_pw_dist_sq
from . import BaseKernel

scalar_function = Callable[[torch.Tensor], float]
kernel_output = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class TrajectoryKernel(BaseKernel):
    def __init__(self, bandwidth_fn: scalar_function = None, **kwargs):
        r"""Computes a the covariance matrix based on the RBF (squared exponential)
        kernel between state-space projections of inputs `X` and `Y` with metric `M`:

            `k(\phi(X), \phi(Y)) = exp((-0.5 / h^2) * (\phi(X) - \phi(Y)) * M * (\phi(X) - \phi(Y))^T)`

        where `\phi(\cdot)` is a differentiable mapping from control actions to
        trajectory space.

        Args:
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.
        """
        super().__init__(bandwidth_fn, analytic_grad=False, **kwargs)

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_actions: torch.Tensor,
        h: float = None,
        compute_grad=True,
        **kwargs,
    ) -> kernel_output:
        """Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Args:
            X (torch.Tensor): Input data of shape [batch, dim].
            Y (torch.Tensor): Input data of shape [batch, dim].
            h (float): The kernel bandwidth. If None, use self.get_bandwidth to
                compute bandwidth from squared distances. Defaults to None.
            compute_grad (bool): If True, computes the first derivative of the kernel
                w.r.t. X. Defaults to True.

        Returns:
            If `compute_grad` is False returns the Kernel Gram Matrix (`K`). Else return
            a tuple of `K` and it's first derivative w.r.t. `X` (`d_K`). The shape of
            `K` is [batch, batch] and the shape of `d_K` is [batch, batch, dim].
        """
        assert X.shape == Y.shape, "X and Y must have the same dimensions."

        sq_dists = pw_dist_sq(X, Y)
        if h is None:
            h = self.get_bandwidth(sq_dists)
        else:
            h = float(h)

        gamma = -0.5 / h ** 2
        K = (gamma * sq_dists).exp()
        if compute_grad:
            d_K = torch.autograd.grad(K.sum(), X_actions, retain_graph=True)[0]
            return K, d_K
        else:
            return K
