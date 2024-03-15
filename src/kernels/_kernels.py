from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple

import torch

from ..utils.math import bw_median, pw_dist_sq, scaled_pw_dist_sq

scalar_function = Callable[[torch.Tensor], float]
kernel_output = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class BaseKernel(ABC, torch.nn.Module):
    def __init__(
        self,
        bandwidth_fn: scalar_function = None,
        analytic_grad: bool = True,
        **kwargs,
    ):
        r"""Computes a the covariance matrix based on the RBF (squared exponential)
        kernel between inputs `X` and `Y`:

            `k(X, Y) = exp((-0.5 / h^2) * ||X - Y||^2)`

        Args:
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.
            analytic_grad (bool, optional): Whether or not the Kernel has an analytic
                gradient. Defaults to True.
        """
        super().__init__(**kwargs)
        self.analytic_grad = analytic_grad
        self.get_bandwidth = None
        if bandwidth_fn is None:
            self.get_bandwidth = bw_median
        elif callable(bandwidth_fn):
            self.get_bandwidth = bandwidth_fn
        else:
            raise ValueError(
                "Kernel bandwidth must be a callable scalar function, got "
                + f"{bandwidth_fn} instead.",
            )

    @abstractmethod
    def __call__(
        self, X: torch.Tensor, Y: torch.Tensor, compute_grad=True, **kwargs
    ) -> kernel_output:
        """Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Args:
            X (torch.Tensor): Data, of shape [batch, dim].
            Y (torch.Tensor): Data, of shape [batch, dim].
            compute_grad (bool): If True, computes the first derivative of the kernel
                w.r.t. X. Defaults to True.

        Returns:
            If `return_grad` is False returns the Kernel Gram Matrix (`K`). Else return
            a tuple of `K` and it's first derivative (`d_K`), wrt to `X`. The shape of
            `K` is [batch, batch] and the shape of `d_K` is [batch, batch, dim].
        """
        pass


class GaussianKernel(BaseKernel):
    def __init__(self, bandwidth_fn: scalar_function = None, **kwargs):
        r"""Computes the covariance matrix and its derivative based on the RBF (squared
        exponential) kernel between inputs `X` and `Y`:

            `k(X, Y) = exp((-0.5 / h^2) * ||X - Y||^2)`

        Args:
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.
        """
        super().__init__(bandwidth_fn, analytic_grad=True, **kwargs)

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        h: float = None,
        compute_grad=True,
        **kwargs,
    ) -> kernel_output:
        """Computes the covariance matrix and its derivative (cross-covariance) for a
        batch of inputs.

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
        X, Y = torch.atleast_2d((X, Y))
        X, Y = X.flatten(1), Y.flatten(1)  # enforces 2-D tensors
        sq_dists = pw_dist_sq(X, Y)
        if h is None:
            h = self.get_bandwidth(sq_dists)
        else:
            h = float(h)
        K = (-0.5 / h ** 2 * sq_dists).exp()
        if compute_grad:
            d_K = -(X.unsqueeze(1) - Y) / (h ** 2) * K.unsqueeze(-1)
            return K, d_K.sum(1)
        else:
            return K


class ScaledGaussianKernel(BaseKernel):
    def __init__(self, bandwidth_fn: scalar_function = None, **kwargs):
        r"""Computes a scaled and preconditioned covariance matrix and its derivative based
        on the Gaussian RBF kernel between a batch of inputs `X` and `Y` with metric `M`:

            `k(X, Y) = exp(-(0.5 / h^2) * (X - Y) * M * (X - Y)^T)`

        Args:
            bandwidth_fn (scalar_function, optional): A function that receives the
                scaled pairwise squared distances and computes a scalar kernel
                bandwidth. If None, the median heuristic is used. Defaults to None.

        Note:
            The scaled kernel is not left multiplied by M^(-1). This allows us to
            compute the resulting stein operator efficiently, using `solve()` at a
            later stage.
        """
        super().__init__(bandwidth_fn, analytic_grad=True, **kwargs)

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        M: torch.Tensor = None,
        h: float = None,
        compute_grad=True,
        **kwargs,
    ) -> kernel_output:
        """Computes the covariance matrix and its derivative (cross-covariance) for a
        batch of inputs.

        Args:
            X (torch.Tensor): Input data of shape [batch, dim].
            Y (torch.Tensor): Input data of shape [batch, dim].
            M (torch.Tensor): Metric matrix (e.g. Hessian), of shape [dim, dim].
                If None, an identity matrix is used (i.e. standard RBF kernel).
                Defaults to None.
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
        X, Y = torch.atleast_2d((X, Y))
        X, Y = X.flatten(1), Y.flatten(1)  # enforces 2-D tensors
        ctx = {"device": X.device, "dtype": X.dtype}
        if M is None:
            M = torch.eye(X.shape[-1], **ctx)
        else:
            assert M.shape == M.T.shape, "M must be a square matrix."
            M = (0.5 * (M + M.T)).to(**ctx)  # PSD stabilization

        sq_dists, sq_dists_grad = scaled_pw_dist_sq(X, Y, M, return_gradient=True)
        if h is None:
            h = self.get_bandwidth(sq_dists)
        else:
            h = float(h)

        gamma = -0.5 / h ** 2
        K = (gamma * sq_dists).exp()
        if compute_grad:
            d_K = -sq_dists_grad * K.unsqueeze(-1) / (h ** 2)
            return K, d_K.sum(1)
        else:
            return K


class IMQKernel(BaseKernel):
    def __init__(
        self, bandwidth_fn: callable = None, **kwargs,
    ):
        r"""Computes a scaled and preconditioned covariance matrix based on the Inverse
        Multiquadric RBF kernel between inputs `X` and `Y` with metric `M`:

            `k(X, Y) = (1 + h^(-2) * ||X - Y||^2)^(-1/2)`

        Args:
            alpha (float, optional): Similarity ratio, a positive number which defines
                the kernel amplitude. Defaults to 1.0.
            beta (float, optional): The order exponent, a negative number that defines
                the steepness of the kernel. Defaults to -0.5.
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.

        Note:
            The scaled kernel is not left multiplied by M^(-1). This allows us to
            compute the resulting stein operator efficiently, using `solve()` at a
            later stage.
        """
        super().__init__(bandwidth_fn, analytic_grad=True, **kwargs)

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        h: float = None,
        compute_grad: bool = True,
        **kwargs,
    ) -> kernel_output:
        assert X.shape == Y.shape, "X and Y must have the same dimensions."
        X, Y = torch.atleast_2d((X, Y))
        X, Y = X.flatten(1), Y.flatten(1)  # enforces 2-D tensors

        sq_dists = pw_dist_sq(X, Y)
        if h is None:
            h = self.get_bandwidth(sq_dists)
        else:
            h = float(h)

        denom = 1 + 0.5 * sq_dists / h ** 2
        K = denom ** -0.5
        if compute_grad:
            d_K = -0.5 * (denom.unsqueeze(-1) ** -1.5) * ((Y - X.unsqueeze(1)) / h ** 2)
            return K, d_K.sum(1)
        else:
            return K


class ScaledIMQKernel(BaseKernel):
    def __init__(
        self, bandwidth_fn: callable = None, **kwargs,
    ):
        r"""Computes a scaled and preconditioned covariance matrix based on the Inverse
        Multiquadric RBF kernel between inputs `X` and `Y` with metric `M`:

            `k(X, Y) = (1 + h^(-2) * (X - Y) * M * (X - Y)^T )^(-1/2)`

        Args:
            alpha (float, optional): Similarity ratio, a positive number which defines
                the kernel amplitude. Defaults to 1.0.
            beta (float, optional): The order exponent, a negative number that defines
                the steepness of the kernel. Defaults to -0.5.
            bandwidth_fn (scalar_function, optional): A function that receives the
                pairwise squared distances and computes a scalar kernel bandwidth.
                If None, the median heuristic is used. Defaults to None.

        Note:
            The scaled kernel is not left multiplied by M^(-1). This allows us to
            compute the resulting stein operator efficiently, using `solve()` at a
            later stage.
        """
        super().__init__(bandwidth_fn, analytic_grad=True, **kwargs)

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        M: torch.Tensor = None,
        h: float = None,
        compute_grad: bool = True,
        **kwargs,
    ):
        assert X.shape == Y.shape, "X and Y must have the same dimensions."
        X, Y = torch.atleast_2d((X, Y))
        X, Y = X.flatten(1), Y.flatten(1)  # enforces 2-D tensors
        ctx = {"device": X.device, "dtype": X.dtype}
        if M is None:
            M = torch.eye(X.shape[-1], **ctx)
        else:
            assert M.shape == M.T.shape, "M must be a square matrix."
            assert M.shape[-1] == X.shape[-1], "Matrix M must match last dim of inputs."
            M.to(**ctx)

        sq_dists, sq_dists_grad = scaled_pw_dist_sq(X, Y, M, return_gradient=True)
        if h is None:
            h = self.get_bandwidth(sq_dists)
        else:
            h = float(h)

        denom = 1 + 0.5 * sq_dists / h ** 2
        K = denom ** -0.5
        if compute_grad:
            d_K = -0.5 * (denom.unsqueeze(-1) ** -1.5) * (sq_dists_grad / h ** 2)
            return K, d_K.sum(1)
        else:
            return K
