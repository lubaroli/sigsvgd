from typing import Callable, Tuple

import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import trange

from ..kernels import BaseKernel
from ..LBFGS import LBFGS


class SVGD:
    """An implementation of Stein variational gradient descent.
    """

    def __init__(
        self,
        kernel: BaseKernel,
        log_p: Callable = None,
        log_prior: Callable = None,
        bw_scale: float = 1.0,
        optimizer_class: optim.Optimizer = optim.Adam,
        **opt_args,
    ):
        self.kernel = kernel
        self.log_p = log_p
        self.log_prior = log_prior
        self.bw_scale = bw_scale
        self.optimizer_class = optimizer_class
        self.opt_args = opt_args

    def _velocity(
        self, X: torch.Tensor, grad_log_p: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.log_p is None and grad_log_p is None:
            raise ValueError(
                "SVGD needs a function to evaluate the log probability of the target",
                "distribution or an estimate of the gradient for every particle.",
            )
        # batch should be X.shape[0], remaining dims will be flattened
        shape = X.shape
        X = X.flatten(1)
        if grad_log_p is None:
            X = X.detach().requires_grad_(True)
            log_lik = self.log_p(X.reshape(shape)).sum()
            score_func = autograd.grad(log_lik, X)[0]
            X.detach_()
            loss = -log_lik.detach()
        else:
            score_func = grad_log_p
            loss = grad_log_p.norm()

        if self.log_prior is not None:
            X = X.detach().requires_grad_(True)
            log_prior_sum = self.log_prior(X.reshape(shape)).sum()
            log_prior_grad = torch.autograd.grad(log_prior_sum, X)[0].detach()
            score_func = score_func + log_prior_grad
            X.detach_()

        if not ("k_xx" in kwargs and "grad_k" in kwargs):
            if hasattr(self.kernel, "analytic_grad") and self.kernel.analytic_grad:
                # grad_k is batch x batch x dim
                k_xx, grad_k = self.kernel(X, X)
                grad_k = grad_k.sum(1)  # aggregates gradient wrt to first input
            else:
                X = X.detach().requires_grad_(True)
                k_xx = self.kernel(X, X.detach(), compute_grad=False)
                grad_k = autograd.grad(-k_xx.sum(), X)[0]
                X.detach_()
                k_xx.detach_()
        else:
            k_xx = kwargs["k_xx"]
            grad_k = kwargs["grad_k"]

        velocity = (k_xx @ score_func + grad_k) / shape[0]
        return -velocity.reshape(shape), loss

    def step(
        self,
        X: torch.Tensor,
        grad_log_p: torch.Tensor = None,
        optimizer: optim.Optimizer = None,
        **kwargs,
    ):
        # if no optimizer given, redefine optimizer to include x as parameter, since x
        # could change between calls, n.b. this will reset momentum based optimizers
        if optimizer is None:
            # optimizer = self.optimizer_class(params=[X], **self.opt_args)
            grad, loss = self._velocity(X, grad_log_p, **kwargs)
            X = X + self.opt_args["lr"] * grad

        def closure():
            optimizer.zero_grad()
            X.grad, loss = self._velocity(X, grad_log_p, **kwargs)
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
        opt_state: dict = None,
        n_steps: int = 100,
        debug: bool = False,
        **kwargs,
    ) -> tuple:
        X = particles.detach()
        # defined here to include x as parameter, since x could change between calls
        optimizer = self.optimizer_class(params=[X], **self.opt_args)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        grad_log_p = None
        data_dict = {}
        if debug:
            # iterator = trange(n_steps, position=0, leave=True)
            iterator = trange(n_steps, leave=False, desc='SVGD')
        else:
            iterator = range(n_steps)

        X_seq = X.clone().unsqueeze(0)
        for i in iterator:
            if score_estimator is not None:
                X.requires_grad_(True)
                grad_log_p, datum = score_estimator(X)
                data_dict[i] = {**datum}
                kwargs.update(datum)
                X.detach_()
            self.step(X, grad_log_p, optimizer, **kwargs)
            X_seq = torch.cat([X_seq, X.unsqueeze(0)], dim=0)
            if debug:
                iterator.set_postfix(loss=X.grad.detach().norm(), refresh=False)
        data_dict["trace"] = X_seq
        return data_dict, optimizer.state_dict()


class ScaledSVGD(SVGD):
    """
    An implementation of Stein variational gradient descent.

    Adapted from: https://github.com/activatedgeek/svgd
    """

    def __init__(
        self,
        kernel: BaseKernel,
        log_p: Callable = None,
        log_prior: Callable = None,
        bw_scale: float = 1.0,
        optimizer_class: optim.Optimizer = optim.Adam,
        metric: str = "GaussNewton",
        precondition: bool = True,
        **opt_args,
    ):
        super().__init__(
            kernel, log_p, log_prior, bw_scale, optimizer_class, **opt_args
        )
        self.metric = metric
        self.precondition = precondition

    def _velocity(self, X: torch.Tensor, grad_log_p: torch.Tensor) -> torch.Tensor:
        if self.log_p is None and grad_log_p is None:
            raise ValueError(
                "SVGD needs a function to evaluate the log probability of the target",
                "distribution or an estimate of the gradient for every particle.",
            )
        # batch should be X.shape[0], remaining dims will be flattened
        shape = X.shape
        X = X.flatten(start_dim=1)
        if grad_log_p is None:
            X = X.detach().requires_grad_(True)
            log_lik = self.log_p(X.reshape(shape)).sum()
            score_func = autograd.grad(log_lik, X)[0].detach()
            X.detach_()
            loss = -log_lik.detach()
        else:
            score_func = grad_log_p  # expects the gradient to *minimize* the loss
            loss = score_func.norm()  # this should be the NLL

        if self.metric.lower() == "gaussnewton":
            # M = self._estimate_gn_hessian(score_func).mean(dim=0)
            M = self._psd_estimate_gn_hessian(score_func, eps=X.var())
        elif self.metric.lower() == "bfgs":
            assert isinstance(self.optimizer_class, torch.optim.LBFGS), (
                "Optimizer must be a instance of `torch.LBFGS` to use approximated "
                "LBFGS Hessian.",
            )
            pass
        elif self.metric.lower() == "fischer":
            raise NotImplementedError
        elif self.metric.lower() == "hessian":
            raise NotImplementedError
            # metric = self.model.Hessian_log_p(X).mean(axis=0)
        else:
            raise ValueError("Unrecognized metric type: {}.".format(self.metric))

        if hasattr(self.kernel, "analytic_grad") and self.kernel.analytic_grad:
            # grad_k is batch x batch x dim
            k_xx, grad_k = self.kernel(X, X, M=M)
            grad_k = grad_k.sum(1)  # aggregates gradient wrt to first input
        else:
            X = X.detach().requires_grad_(True)
            k_xx = self.kernel(X, X.detach(), M=M, compute_grad=False)
            grad_k = autograd.grad(-k_xx.sum(), X)[0]
            X.detach_()
            k_xx.detach_()

        if self.log_prior is not None:
            X = X.detach().requires_grad_(True)
            log_prior_sum = self.log_prior(X.reshape(shape)).sum()
            log_prior_grad = torch.autograd.grad(log_prior_sum, X)[0].detach()
            score_func = score_func + log_prior_grad
            X.detach_()

        velocity = (k_xx @ score_func + grad_k) / shape[0]
        if self.precondition:
            velocity = torch.linalg.solve(M, velocity.T).T
        return -velocity.reshape(shape), loss

    def _psd_estimate_gn_hessian(self, jacobian, eps=1e-3):
        """Computes a Gauss Newton approximation of the Hessian matrix, given the Jacobian.
        """
        avg_hess = torch.mean(2 * jacobian[:, :, None] * jacobian[:, None, :], dim=0)
        # eig, _ = torch.linalg.eig(avg_hess)
        # if eig.real.min() < 0:
        #     eps = -eig.real.min()
        # else:
        #     eps = 0
        psd_avg_hess = avg_hess + torch.eye(avg_hess.shape[-1]) * eps
        return psd_avg_hess

    def _estimate_gn_hessian(self, jacobian):
        return 2 * jacobian[:, :, None] * jacobian[:, None, :]
