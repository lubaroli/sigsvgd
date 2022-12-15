import torch
from torch.autograd import grad as ag
from ..kernels import BaseKernel, SignatureKernel


class PlanningEstimator:
    def __init__(
        self, kernel, cost_fn, cost_fn_params, scheduler=None, ctx={"device": "cpu"},
    ):
        self.ctx = ctx
        self.kernel = kernel
        self.cost_fn = cost_fn
        self.cost_fn_params = cost_fn_params
        if not scheduler:
            self.scheduler = lambda: 1
        else:
            self.scheduler = scheduler

        if isinstance(self.kernel, SignatureKernel):
            self.score = self._pathsig_score
        elif isinstance(self.kernel, BaseKernel):
            if self.kernel.analytic_grad is True:
                self.score = self._svgd_score
            else:
                self.score = self._svgd_ag_score

    # need score estimator to include trajectory length regularization cost
    def sgd_score(self, x):
        cost, cost_dict = self.cost_fn(x, *self.cost_fn_params)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = torch.eye(x.shape[0], **self.ctx)
        grad_k = torch.zeros_like(grad_log_p, **self.ctx)
        score_dict = {"k_xx": k_xx, "grad_k": grad_k, "loss": cost, **cost_dict}
        return grad_log_p, score_dict

    def _svgd_score(self, x):
        cost, cost_dict = self.cost_fn(x, *self.cost_fn_params)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx, grad_k = self.kernel(x.flatten(1), x.flatten(1), compute_grad=True)
        grad_k = grad_k.sum(1)  # aggregates gradient wrt to first input
        score_dict = {
            "k_xx": k_xx,
            "grad_k": self.scheduler() * grad_k,
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict

    def _svgd_ag_score(self, x):
        cost, cost_dict = self.cost_fn(x, *self.cost_fn_params)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = self.kernel(x.flatten(1), x.flatten(1), compute_grad=False)
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx,
            "grad_k": self.scheduler() * grad_k,
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict

    def _pathsig_score(self, x):
        cost, cost_dict = self.cost_fn(x, *self.cost_fn_params)
        # considering the likelihood is exp(-cost)
        grad_log_p = ag(-cost.sum(), x, retain_graph=True)[0]
        k_xx = self.kernel(x, x)
        grad_k = ag(k_xx.sum(), x)[0]
        score_dict = {
            "k_xx": k_xx.detach(),
            "grad_k": self.scheduler() * grad_k.detach(),
            "loss": cost,
            **cost_dict,
        }
        return grad_log_p, score_dict
