from typing import Callable, Tuple

import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import trange

from ..kernels import BaseKernel


class SVGD:
    """An implementation of Stein variational gradient descent that supports custom kernels.
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

    def _compute_kernel(self, X, **kernel_args):
        if hasattr(self.kernel, "analytic_grad") and self.kernel.analytic_grad:
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

        iter_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": loss}
        iter_dict.update(kernel_args)
        iter_dict = {
            k: v.detach() if hasattr(v, "detach") else v for k, v in iter_dict.items()
        }
        return -velocity.reshape(X.shape), iter_dict

    def step(
        self,
        X: torch.Tensor,
        grad_log_p: torch.Tensor = None,
        optimizer: optim.Optimizer = None,
        **kernel_args,
    ):
        def closure():
            optimizer.zero_grad()
            X.grad, iter_dict = self._velocity(X, grad_log_p, **kernel_args)
            return iter_dict

        if isinstance(optimizer, torch.optim.Optimizer):
            iter_dict = optimizer.step(closure)
        else:  # manually compute SGD
            grad, iter_dict = self._velocity(X, grad_log_p, **kernel_args)
            X = X + self.opt_args["lr"] * grad
        return iter_dict

    def optimize(
        self,
        particles: torch.Tensor,
        score_estimator: Callable = None,
        opt_state: dict = None,
        n_steps: int = 100,
        debug: bool = False,
        **kernel_args,
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
                grad_log_p, score_dict = score_estimator(X)
                kernel_args.update(score_dict)
            data_dict[i] = {**self.step(X, grad_log_p, optimizer, **kernel_args)}
            X_seq = torch.cat([X_seq, X.detach().unsqueeze(0)], dim=0)
            if debug:
                iterator.set_postfix(loss=X.grad.detach().norm(), refresh=False)
        data_dict["trace"] = X_seq
        return data_dict, optimizer.state_dict()


class ScaledSVGD(SVGD):
    """
    An implementation of second-order (matrix) Stein variational gradient descent.
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
        if grad_log_p is None:
            X = X.detach().requires_grad_(True)
            log_lik = self.log_p(X).sum()
            score_func = autograd.grad(log_lik, X)[0].flatten(1)
            X.detach_()
            loss = -log_lik.detach()
        else:
            score_func = grad_log_p
            loss = grad_log_p.norm()

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
            log_prior_sum = self.log_prior(X).sum()
            log_prior_grad = torch.autograd.grad(log_prior_sum, X)[0].detach()
            score_func = score_func + log_prior_grad
            X.detach_()

        velocity = (k_xx @ score_func + grad_k) / X.shape[0]
        if self.precondition:
            velocity = torch.linalg.solve(M, velocity.T).T

        iter_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": loss}
        iter_dict = {
            k: v.detach() if hasattr(v, "detach") else v for k, v in iter_dict.items()
        }
        return -velocity.reshape(X.shape), iter_dict

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
