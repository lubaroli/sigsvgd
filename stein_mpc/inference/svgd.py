from typing import Callable

import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import trange

from ..kernels import BaseKernel
from ..LBFGS import LBFGS


class SVGD:
    """
    An implementation of Stein variational gradient descent.

    Adapted from: https://github.com/activatedgeek/svgd
    """

    def __init__(
        self,
        kernel: BaseKernel,
        p_log_prob: Callable = None,
        bw_scale: float = 1.0,
        optimizer_class=optim.Adam,
        **opt_args,
    ):
        self.kernel = kernel
        self.log_p = p_log_prob
        self.bw_scale = bw_scale
        self.optimizer_class = optimizer_class
        self.opt_args = opt_args

    def _velocity(self, X: torch.Tensor, grad_log_p: torch.Tensor) -> torch.Tensor:
        if self.log_p is None and grad_log_p is None:
            raise ValueError(
                "SVGD needs a function to evaluate the log probability of the target",
                "distribution or an estimate of the gradient for every particle.",
            )
        # batch should be X.shape[0], remaining dims will be flattened
        shape = X.shape
        if grad_log_p is None:
            X = X.detach().requires_grad_(True)
            loss = self.log_p(X).sum()
            score_func = autograd.grad(loss, X)[0]
            X.detach_()
        else:
            score_func = grad_log_p
            loss = 0

        if hasattr(self.kernel, "analytic_grad"):
            if self.kernel.analytic_grad:
                k_xx, grad_k = self.kernel(X, X)
        else:
            X.detach_().requires_grad_(True)
            k_xx = self.kernel(X, X.detach())
            grad_k = autograd.grad(k_xx.sum(), X)[0]
            k_xx.detach_()

        velocity = (k_xx @ score_func + grad_k) / shape[0]
        print(f"Grad K norm: {grad_k.norm()}")
        print(f"Grad L norm: {score_func.norm()}")
        return -velocity, loss

    def step(
        self,
        X: torch.Tensor,
        grad_log_p: torch.Tensor = None,
        optimizer: optim.Optimizer = None,
    ):
        # if no optimizer given, redefine optimizer to include x as parameter, since x
        # could change between calls, n.b. this will reset momentum based optimizers
        if optimizer is None:
            optimizer = self.optimizer_class(params=[X], **self.opt_args)

        def closure():
            optimizer.zero_grad()
            X.grad, loss = self._velocity(X, grad_log_p)
            return loss

        if isinstance(optimizer, LBFGS):
            loss = closure()
            options = {"closure": closure, "current_loss": loss}
            optimizer.step(options)
        else:
            optimizer.step(closure)

    def optimize(
        self,
        particles: torch.Tensor,
        score_estimator: Callable = None,
        n_steps: int = 100,
        debug: bool = False,
    ) -> list:
        X = particles.detach()
        # defined here to include x as parameter, since x could change between calls
        optimizer = self.optimizer_class(params=[X], **self.opt_args)
        grad_log_p = None
        data = []
        if debug:
            iterator = trange(n_steps, position=0, leave=True)
        else:
            iterator = range(n_steps)

        X_seq = X.clone().unsqueeze(0)
        for _ in iterator:
            if score_estimator is not None:
                X.requires_grad_(True)
                grad_log_p, datum = score_estimator(X)
                data.append(datum)
                X.detach_()
            self.step(X, grad_log_p, optimizer)
            X_seq = torch.cat([X_seq, X.unsqueeze(0)], dim=0)
            if debug:
                iterator.set_postfix(loss=X.grad.detach().norm(), refresh=False)
        X.detach_()
        return X_seq, data


class ScaledSVGD(SVGD):
    """
    An implementation of Stein variational gradient descent.

    Adapted from: https://github.com/activatedgeek/svgd
    """

    def __init__(
        self,
        kernel: BaseKernel,
        p_log_prob: Callable = None,
        bw_scale: float = 1.0,
        optimizer_class=optim.Adam,
        **opt_args,
    ):
        super().__init__(kernel, p_log_prob, bw_scale, optimizer_class, **opt_args)

    def _velocity(self, X: torch.Tensor, grad_log_p: torch.Tensor) -> torch.Tensor:
        if self.log_p is None and grad_log_p is None:
            raise ValueError(
                "SVGD needs a function to evaluate the log probability of the target",
                "distribution or an estimate of the gradient for every particle.",
            )
        batch, dim = X.shape
        if grad_log_p is None:
            X.detach_().requires_grad_(True)
            loss = self.log_p(X).sum()
            score_func = autograd.grad(loss, X)[0]
            X.detach_()
        else:
            score_func = grad_log_p
            loss = 0

        metric = self._GaussNewtonHessian(-score_func).mean(dim=0)

        if hasattr(self.kernel, "analytic_grad"):
            if self.kernel.analytic_grad:
                k_xx, grad_k = self.kernel(X, X, M=metric)
        else:
            X.detach_().requires_grad_(True)
            k_xx = self.kernel(X, X.detach(), M=metric)
            grad_k = autograd.grad(k_xx.sum(), X)[0]
            k_xx.detach_()

        velocity = (k_xx @ score_func + grad_k) / batch
        print(f"Grad L norm: {score_func.norm()}")
        print(f"Grad K norm: {grad_k.norm()}")
        return -velocity @ metric, loss

    def _GaussNewtonHessian(self, score_function):
        return 2 * score_function[:, :, None] * score_function[:, None, :]
