from typing import Callable, Union, Tuple

import torch
import signatory
import sigkernel

from ..utils.math import pw_dist_sq
from . import BaseKernel, GaussianKernel

scalar_function = Callable[[torch.Tensor], float]
kernel_output = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class TrajectoryKernel(BaseKernel):
    def __init__(self, bandwidth_fn: scalar_function = None, **kwargs):
        r"""Computes the gram matrix based on the RBF (squared exponential) kernel
        between state-space projections of inputs `X` and `Y`:

        `k(\phi(X), \phi(Y)) = exp((-0.5 / h^2)`* ||\phi(X) - \phi(Y)||^2 `

        where `\phi(\cdot)` is a differentiable mapping from control actions to
        trajectory space.

        # Args:
        #     bandwidth_fn (scalar_function, optional): A function that receives the
        #         pairwise squared distances and computes a scalar kernel bandwidth.
        #         If None, the median heuristic is used. Defaults to None.
        # """
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
            If `compute_grad` is False returns the kernel Gram matrix (`K`). Else return
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


class PathSigKernel(BaseKernel):
    def __init__(
        self,
        bandwidth_fn: scalar_function = None,
        static_kernel: BaseKernel = GaussianKernel(),
        **kwargs,
    ):
        r"""Computes the gram matrix based on the RBF (squared exponential) kernel
        between Path Signatures inputs `X` and `Y`:

        `k(\phi(X), \phi(Y)) = exp((-0.5 / h^2)`* ||S(X, d) - S(Y, d)||^2`

        where `S(\cdot, d)` is the (differentiable) signature transform of depth d
        for a given path.

        Args:
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.
        """
        super().__init__(bandwidth_fn, analytic_grad=False, **kwargs)
        self.static_kernel = static_kernel

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        ref_vector: torch.Tensor = None,
        depth: int = 3,
        h: float = None,
        compute_grad=True,
        **kwargs,
    ) -> kernel_output:
        """Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Args:
            X (torch.Tensor): Input data of shape [batch, length, channels].
            Y (torch.Tensor): Input data of shape [batch, length, channels].
            h (float): The kernel bandwidth. If None, use self.get_bandwidth to
                compute bandwidth from squared distances. Defaults to None.
            compute_grad (bool): If True, computes the first derivative of the kernel
                w.r.t. X. Defaults to True.

        Returns:
            If `compute_grad` is False returns the kernel Gram matrix (`K`). Else return
            a tuple of `K` and it's first derivative w.r.t. `X` (`d_K`). The shape of
            `K` is [batch, batch] and the shape of `d_K` is [batch, batch, dim].
        """
        assert X.shape == Y.shape, "X and Y must have the same dimensions."
        X, Y = torch.atleast_3d((X, Y))
        if ref_vector is None and compute_grad is True:
            X.detach().requires_grad_(True)
            ref_vector = X
        X_sig = signatory.signature(X, depth, basepoint=True)
        Y_sig = signatory.signature(Y, depth, basepoint=True)
        # if self.kernel_type == "gaussian":
        #     xty = torch.matmul(X_sig, Y_sig.T)
        #     if h is None:
        #         h = self.get_bandwidth(xty)
        #     else:
        #         h = float(h)

        #     gamma = -0.5 / h ** 2
        #     K = (gamma * xty).exp()
        # else:  # cosine similarity
        #     Z = X_sig.norm(dim=1, keepdim=True) * Y_sig.norm(dim=1, keepdim=True).T
        #     K = torch.matmul(X_sig, Y_sig.T) / Z
        if compute_grad:
            K, d_K = self.static_kernel(X_sig, Y_sig, compute_grad=True)
            return K, d_K
        else:
            K = self.static_kernel(X_sig, Y_sig, compute_grad=False)
            return K


class BatchGaussianKernel(BaseKernel):
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, bandwidth_fn: scalar_function = None, **kwargs):
        super().__init__(bandwidth_fn, analytic_grad=False, **kwargs)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> kernel_output:
        return self.batch_kernel(X, Y, **kwargs)

    def batch_kernel(self, X, Y, h=None):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2.0 * torch.bmm(X, Y.permute(0, 2, 1))
        dist += torch.reshape(Xs, (A, M, 1)) + torch.reshape(Ys, (A, 1, N))
        if h is None:
            h = self.get_bandwidth(dist)
        else:
            h = float(h)
        return torch.exp(-dist / h)

    def Gram_matrix(self, X, Y, h=None):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        dist = -2.0 * torch.einsum("ipk,jqk->ijpq", X, Y)
        dist += torch.reshape(Xs, (A, 1, M, 1)) + torch.reshape(Ys, (1, B, 1, N))
        if h is None:
            h = self.get_bandwidth(dist)
        else:
            h = float(h)
        return torch.exp(-dist / h)


class SignatureKernel:
    def __init__(self, bandwidth_fn: scalar_function = None, depth: int = 3, **kwargs):
        static_kernel = BatchGaussianKernel(bandwidth_fn=bandwidth_fn)
        self.kernel = sigkernel.SigKernel(static_kernel, dyadic_order=depth)

    def __call__(self, X, Y, **kwargs):
        ctx = {"device": X.device, "dtype": X.dtype}
        k_xx = self.kernel.compute_Gram(X.double(), Y.double())
        return k_xx.to(**ctx)
