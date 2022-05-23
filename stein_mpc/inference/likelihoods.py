from abc import ABC, abstractmethod

import torch
import torch.distributions as dist

from ..models.base import BaseModel


class GaussianLikelihood(ABC):
    def __init__(
        self,
        initial_obs: torch.Tensor,
        obs_std: float,
        model: BaseModel,
        log_space: bool = False,
    ):
        assert (
            initial_obs.ndim == 1
        ), "Gaussian likelihood needs a single dimensional loc tensor."
        self.dim = initial_obs.shape[0]
        self.sigma = obs_std
        # condition also sets loc
        self.density = self.condition(new_obs=initial_obs, action=None)
        self.model = model
        self.log_space = log_space

    def sample(self, theta: torch.Tensor):
        assert (
            self.past_action is not None
        ), "Previous action is None. Need at least one observation to start sampling."
        if self.log_space:
            params = theta.exp()
        else:
            params = theta
        if str(self.model.__class__) == "<class '__main__.SSModel'>":
            states = self.model.step(
                self.past_obs.view(1, -1), self.past_action.view(1, -1), params
            )
        else:
            params_dict = self.model.params_to_dict(params)
            states = self.past_obs.repeat(theta.shape[0], 1)
            states = self.model.step(states, self.past_action, params_dict)
        return states

    def log_prob(self, samples: torch.Tensor):
        return self.density.log_prob(samples).unsqueeze(-1)

    def condition(
        self,
        action: torch.Tensor,
        new_obs: torch.Tensor,
        covariance_matrix: torch.Tensor = None,
    ):
        try:
            self.past_obs = self.loc
        except AttributeError:
            self.past_obs = None
        self.loc = new_obs
        self.past_action = action
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        self.density = dist.MultivariateNormal(
            self.loc, self.sigma ** 2 * torch.eye(self.dim)
        )


class CostLikelihood(ABC):
    def __init__(self, alpha: float):
        self.alpha = alpha

    @abstractmethod
    def log_p(self, costs: torch.Tensor = None):
        pass


class ExponentiatedUtility(CostLikelihood):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(alpha, **kwargs)

    def log_p(self, costs: torch.Tensor):
        """Computes the un-normalized log-likelihood of a batch of cost samples with
        shape [[batch, ] event].
        """
        try:
            batch_shape = torch.atleast_2d(costs).shape[0]
        except AttributeError:
            batch_shape = costs.shape[0] if costs.dim() >= 2 else 1
        # if more than one cost sample, subtract min for numerical stability
        if batch_shape > 1:
            costs = costs - costs.min()
        score = (-1 / self.alpha) * costs
        return score
