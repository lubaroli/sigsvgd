from typing import Callable, Tuple

import torch
from gpytorch.priors import SmoothedBoxPrior
from torch import distributions as dist
from torch import optim

from ..controllers import BaseController
from ..inference import CostLikelihood, ExponentiatedUtility, ScaledSVGD, TrajectorySVGD
from ..kernels import BaseKernel, ScaledGaussianKernel
from ..models.base import BaseModel
from ..utils.math import grad_gmm_log_p, to_gmm
from ..utils.spaces import Box


class DuSt(BaseController):
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hz_len: int,
        n_pol: int,
        n_action_samples: int = 0,
        n_params_samples: int = 0,
        pol_mean: torch.Tensor = None,
        pol_cov: torch.Tensor = None,
        pol_hyper_prior: bool = True,
        stein_sampler: str = "SVGD",
        kernel: BaseKernel = ScaledGaussianKernel(),
        likelihood: CostLikelihood = None,
        temperature: float = 1.0,
        inst_cost_fn: Callable = None,
        term_cost_fn: Callable = None,
        params_log_space: bool = False,
        action_primitives: torch.Tensor = None,
        weighted_prior: bool = False,
        device: str = "cuda",
        roll_strategy: str = "repeat",
        optimizer_class: optim.Optimizer = optim.Adam,
        **opt_args,
    ):
        """Constructor class for Stein MPC controller.

        Args:
            observation_space (Box): A Box object defining the observation space.
            action_space (Box): A Box object defining the action space.
            hz_len (int): Number of time steps in the control horizon.
            n_pol (int): Number of controller policies (i.e. Stein particles).
            n_action_samples (int, optional): If not zero, defines the number of actions
              sampled from each policy and used to compute the likelihood gradient.
              Defaults to 0.
            n_params_samples (int, optional): If not zero, defines how many samples of
              model parameters are taken when evaluating rollouts. Defaults to 0.
            pol_mean (torch.Tensor, optional): The initial policies. If None, policies
              are randomly initialized from a Normal distribution with `pol_cov`
              covariance. Defaults to None.
            pol_cov (torch.Tensor, optional): The covariance matrix of each policy. If
              None an identity matrix is used. Defaults to None.
            pol_hyper_prior (bool, optional): If True, a Smoothed Box hyper-prior is
              used to regularize gradients and keep policies within the action space.
              Defaults to True.
            stein_sampler (str, optional): Defines the type of Stein variational method.
              Options are `SVGD` for first-order gradients, `ScaledSVGD` for
              metric-scaled kernels and `MatrixSVGD` for second-order Stein gradients.
              Defaults to "SVGD".
            kernel (BaseKernel, optional): The kernel used by the Stein sampler.
              Defaults to ScaledGaussianKernel().
            likelihood (CostLikelihood, optional): The likelihood function. If None, an
              Exponential Utility is used. Defaults to None.
            temperature (float, optional): Controller temperature parameter. Defaults
              to 1.0.
            inst_cost_fn (Callable, optional): The instantaneous cost function which
              receives states and actions as inputs. If None, a null function is used.
              At least one of the cost functions must be defined. Defaults to None.
            term_cost_fn (Callable, optional): The terminal cost function which receives
              states as input. If None, a null function is used. At least one of the
              cost functions must be defined. Defaults to None.
            params_log_space (bool, optional): If True, parameters are sampled in
              log-space to avoid negative values. Defaults to False.
            weighted_prior (bool, optional): If True, uses last step costs to weigh
              particles. Defaults to False.
            roll_strategy (str, optional): Defines the strategy to refill policies at
              each time-step transition. Choices are "repeat", "resample", "mean".
              Defaults to "repeat".
            optimizer_class (optim.Optimizer, optional): Type of optimizer to use.
              Defaults to optim.Adam.

        Raises:
            ValueError: Raises a ValueError if an invalid Stein Sampler is chosen.
        """
        if device == "cuda" and torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")
        super().__init__(
            observation_space, action_space, hz_len, inst_cost_fn, term_cost_fn,
        )
        # configure model params inference
        assert n_params_samples >= 0 and isinstance(
            n_params_samples, int
        ), "Number of model parameter samples must be non-negative integer."
        self.n_params_samples = n_params_samples
        if n_params_samples == 0:  # no inference, use model default params
            self.params_shape = torch.Size([])
        else:  # use Monte Carlo sampling to estimate params
            self.params_shape = torch.Size([n_params_samples])
        self._params_log_space = params_log_space

        # configure grad_log_p estimation
        assert n_action_samples >= 0 and isinstance(
            n_action_samples, int
        ), "Number of policy samples must be a non-negative integer."
        if n_action_samples == 0:  # use autograd
            self.action_shape = torch.Size([])
        else:  # use Monte Carlo sampling out of policies
            self.action_shape = torch.Size([n_action_samples])

        # initialize policies, sets attributes:
        # n_pol, n_prim, rollout_shape, policies_shape, n_rollouts, n_total_actions
        self._init_policies(10.0, n_pol, pol_mean, pol_cov, action_primitives)

        self.prior_weights = torch.ones(self.n_pol).to(self.dev)
        self.prior = to_gmm(self.pol_mean, self.prior_weights, self.pol_cov)
        hyper_prior = (
            SmoothedBoxPrior(self.min_a, self.max_a, 0.1).log_prob
            if pol_hyper_prior
            and not torch.isinf(self.min_a).any()
            and not torch.isinf(self.max_a).any()
            else None
        )

        # configure likelihood
        self.temp = temperature
        if likelihood is None:
            self.likelihood = ExponentiatedUtility(self.temp)
        else:
            self.likelihood = likelihood

        # configure stein sampler
        self.opt_state = None
        gradient_mask = torch.ones_like(self.pol_mean).to(self.dev)
        gradient_mask[: self.n_prim] = 0.0
        if stein_sampler == "SVGD":
            self.stein_sampler = TrajectorySVGD(
                kernel,
                log_prior=hyper_prior,
                optimizer_class=optimizer_class,
                gradient_mask=gradient_mask,
                **opt_args,
            )
        elif stein_sampler == "ScaledSVGD":
            self.stein_sampler = ScaledSVGD(
                kernel,
                log_prior=hyper_prior,
                precondition=False,
                optimizer_class=optimizer_class,
                **opt_args,
            )
        elif stein_sampler == "MatrixSVGD":
            self.stein_sampler = ScaledSVGD(
                kernel,
                log_prior=hyper_prior,
                precondition=True,
                optimizer_class=optimizer_class,
                **opt_args,
            )
        else:
            raise ValueError(
                "Invalid value for 'stein_sampler': {}.".format(stein_sampler)
            )

        # additional options
        self.w_prior = weighted_prior
        self.roll_strategy = roll_strategy

    def _init_policies(
        self, uniform_range, n_pol, pol_mean, pol_cov, action_primitives
    ):
        """Initializes the policies and sets useful attributes."""
        # as actions are sampled i.i.d., policy covariance is of shape [dim_a x dim_a]
        if pol_cov is None:
            self.pol_cov = torch.eye(self.dim_a).to(self.dev)
        else:
            self.pol_cov = pol_cov.to(self.dev)
        if pol_mean is None:
            self.pol_mean = (
                torch.empty(n_pol, self.hz_len, self.dim_a)
                .uniform_(
                    torch.max(self.min_a.max(), torch.tensor(-uniform_range)),
                    torch.min(self.max_a.min(), torch.tensor(uniform_range)),
                )
                .to(self.dev)
            )
        else:
            assert (
                pol_mean.shape == self.policies_shape
            ), "Initial policies shape mismatch."
            self.pol_mean = pol_mean.clone().to(self.dev)
        if action_primitives is not None:
            assert (
                action_primitives.shape[-2:] == (self.hz_len, self.dim_a)
                and action_primitives.ndim == 3
            )
            self.n_prim = action_primitives.shape[0]
            self.pol_mean = torch.cat(
                [action_primitives.to(self.dev), self.pol_mean], dim=0
            )
            # number of particles
            self.n_pol = self.pol_mean.shape[0]
        else:
            # number of particles
            self.n_pol = n_pol
            self.n_prim = 0

        # useful attributes
        self.rollout_shape = (
            self.params_shape + self.action_shape + torch.Size([self.n_pol])
        )
        self.policies_shape = torch.Size([self.n_pol, self.hz_len, self.dim_a])
        self.n_rollouts = self.rollout_shape.numel()
        self.n_total_actions = (self.action_shape + (self.n_pol,)).numel()

    def _compute_cost(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Estimate trajectories cost.

        Args:
            states (torch.Tensor): A tensor with the states of each trajectory.
            actions (torch.Tensor): A tensor with the sampled actions for each policy.

        Returns:
            torch.Tensor: The aggregated cost of each policy.
        """
        # Dims are n_params, n_actions, n_pol, hz_len, dim_s/dim_a
        x_vec = states[..., :-1, :].reshape(-1, self.dim_s)
        x_final = states[..., -1, :].reshape(-1, self.dim_s)
        a_vec = actions.reshape(-1, self.dim_a)
        inst_costs = self.inst_cost_fn(x_vec, a_vec, n_pol=self.n_pol)
        term_costs = self.term_cost_fn(x_final, n_pol=self.n_pol)

        # aggregate instant costs over control horizon
        inst_costs = inst_costs.reshape(self.rollout_shape + (self.hz_len,)).sum(-1)
        term_costs = term_costs.reshape(self.rollout_shape)
        state_cost = inst_costs + term_costs
        if self.params_shape:  # if not empty, average over params
            state_cost = (inst_costs + term_costs).mean(0)
        return state_cost

    def _rollout(
        self,
        init_state: torch.Tensor,
        base_actions: torch.Tensor,
        model: BaseModel,
        params_dist: dist.Distribution,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Generates rollouts based on initial state and policies. Model can be
        deterministic or contain uncertain params.

        Args:
            init_state (torch.Tensor): The system initial state.
            base_actions (torch.Tensor): A Tensor with the actions applied at each step.
            model (BaseModel): A model object which provides a `step` function to
              generate the system rollouts.
            params_dist (dist.Distribution): A distribution from which model parameters
              are sampled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: Returns the sequence of states,
              actions and parameters sampled for each trajectory.
        """
        # prepare tensors for batch rollout
        if self.params_shape:  # if not empty, sample params from `params_dist`
            base_params = params_dist.sample(self.params_shape)
            if not params_dist.event_shape:  # if event shape empty
                # ensures params is at least a column vector
                base_params = base_params.reshape(-1, 1)
            if self._params_log_space is True:
                base_params = base_params.exp()
            # use repeat_interleave to apply same params to all actions
            params = base_params.repeat_interleave(self.n_total_actions, dim=0)
            # create dict with `uncertain_params` as keys for `model.step()`
            params_dict = model.params_to_dict(params)
            # flatten policies into single dim and repeat for each sampled param
            actions = base_actions.flatten(end_dim=-3).tile(self.n_params_samples, 1, 1)
        else:  # use default `model` params
            params_dict = None
            actions = base_actions.flatten(end_dim=-3)
        # expand initial state to the total amount of rollouts
        states = init_state.expand(self.n_rollouts, 1, -1).clone()

        # generate rollouts
        for t in range(self.hz_len):
            states = torch.cat(
                [
                    states,
                    model.step(states[:, t], actions[:, t], params_dict).unsqueeze(1),
                ],
                dim=1,
            )

        # restore vectors dims, `n_params` is now first dimension
        states = states.reshape(self.rollout_shape + (self.hz_len + 1, self.dim_s))
        actions = actions.reshape(self.rollout_shape + (self.hz_len, self.dim_a))
        # TODO: see if can return just base_params_dict
        return states, actions, params_dict

    def _sample_actions(self, pol_mean: torch.Tensor) -> torch.Tensor:
        """Sample actions from Multivariate Normal policies while keeping the
        computational graph.

        Args:
            pol_mean (torch.Tensor): The policies (distribution mean) for each
              Multivariate Normal. Covariance is defined by `self.pol_cov`.

        Returns:
            torch.Tensor: Sampled actions.
        """
        pi = dist.MultivariateNormal(pol_mean, self.pol_cov)
        # Use rsample to preserve gradient
        actions = pi.rsample(self.action_shape)
        return actions

    def _get_costs(self, state, base_actions, model, params_dist):
        """Wrapper function to compute policies costs.
        """
        states, actions, params_dict = self._rollout(
            state, base_actions, model, params_dist
        )
        costs = self._compute_cost(states, actions)
        return costs, states, params_dict

    def _get_grad_log_p(
        self, costs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the gradient of the log-posterior distribution of policies.

        Args:
            costs (torch.Tensor): The costs of each sampled action sequence or policy.
            actions (torch.Tensor): The sampled action sequences or policies.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the gradient of the
              log-posterior and the negative log-likelihood loss.
        """
        # prior gradient (i.e. prior of policies w.r.t to previous step prior)
        with torch.no_grad():
            grad_pri = grad_gmm_log_p(self.prior, self.pol_mean)

        # likelihood gradient...
        log_lik = self.likelihood.log_p(costs)
        if self.action_shape:  # ... using monte-carlo sampling
            with torch.no_grad():
                # computes gradients of action samples w.r.t. their policies
                grad_log_pol = (actions - self.pol_mean) @ self.pol_cov.inverse()
                # elementwise product and aggregate over samples
                bc_dims = torch.Size([1]) * len(self.prior.event_shape)
                pol_weight = log_lik.reshape(log_lik.shape + bc_dims).softmax(dim=0)
                grad_lik = torch.sum(pol_weight * grad_log_pol, dim=0)
                # grad_pri = torch.sum(pol_weight * grad_pri, dim=0)
                loss = -log_lik.sum(0)
        else:  # ... using autograd
            grad_lik = torch.autograd.grad(log_lik.sum(), actions, retain_graph=True)[0]
            loss = -log_lik

        grad_log_p = grad_pri + grad_lik
        return grad_log_p, loss

    def _get_pol_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """Aggregates sampled costs and computes particles weights.

        Args:
            costs (torch.Tensor): A tensor with the forecasted cost of each policy. If
              actions sequences are being sampled, these will be averaged per policy.

        Returns:
            torch.Tensor: A tensor with the softmax weight of each policy.
        """
        log_lik = self.likelihood.log_p(costs)
        if self.action_shape:
            # elementwise product and aggregate over samples
            weights = log_lik.mean(0).softmax(0)
        else:
            weights = log_lik.softmax(0)
        return weights

    def _update_optimizer(self):
        """Updates the optimizer state after performing a shift operation on policies.
        Currently only implemented for the L-BFGS optimizer.
        """
        steps = -1 * self.dim_a
        if self.stein_sampler.optimizer_class == optim.LBFGS:
            tensors = ["d", "prev_flat_grad"]
            for v in tensors:
                new_v = self.opt_state["state"][0][v]
                new_v = new_v.roll(steps)
                new_v[steps:] = 0  # add zeros at the end of tensor
                self.opt_state["state"][0][v] = new_v
            lists = ["old_dirs", "old_stps"]
            for v in lists:
                new_v = self.opt_state["state"][0][v]
                if len(new_v) != 0:
                    if len(new_v) == 1:
                        new_v = new_v[0].unsqueeze(-1)
                    else:
                        new_v = torch.stack(new_v, dim=1)
                    new_v = new_v.roll(steps, dims=0)
                    new_v[steps:] = 0  # add zeros at the end of tensor
                    self.opt_state["state"][0][v] = [
                        new_v[:, i] for i in range(new_v.shape[-1])
                    ]

    def _update_prior(self, weights: torch.Tensor = None):
        """Updates the controller GMM prior over particles. If using weighted priors,
        assign current policy weights as the prior GMM weights.

        Args:
            weights (torch.Tensor, optional): Weights used by the prior if self.w_prior
                is True. Defaults to None.
        """
        if self.w_prior and weights is not None:
            self.prior_weights = weights
        else:
            self.prior_weights = torch.ones_like(self.prior_weights)
        self.prior = to_gmm(self.pol_mean, self.prior_weights, self.pol_cov)

    def roll(self, steps=-1, strategy="repeat"):
        # self.pol_mean.clamp_(self.min_a, self.max_a)
        # roll along time axis
        self.pol_mean = self.pol_mean.roll(steps, dims=-2)
        if strategy == "repeat":
            # repeat last action
            self.pol_mean[..., -1, :] = self.pol_mean[..., -2, :]
        elif strategy == "resample":
            # sample a new action from prior for last step
            self.pol_mean[..., -1, :] = self.prior.sample([self.n_particles])[
                ..., -1, :
            ]
        elif strategy == "mean":
            # last action will be the mean of each policy
            self.pol_mean[..., -1, :] = self.pol_mean.mean(dim=-2)
        else:
            raise ValueError("{} is an invalid roll strategy.".format(strategy))

    def forward(
        self,
        state: torch.Tensor,
        model: BaseModel,
        params_dist: dist.Distribution,
        steps: int = 5,
    ) -> Tuple[torch.Tensor, dict]:
        """Computes the next sequence of actions and updates the controller state.

        Note that in Stein MPC, each particle is a control policy over a fixed control
        horizon. The forward function will perform all the necessary steps to compute
        the gradients, update policies, and return the current best action plan. This
        requires performing the following steps:
        - If using Monte Carlo samples to compute the likelihood gradient, sample
          action sequences from current policies to evaluate costs. Otherwise, use
          policy means and autograd to compute likelihood gradients;
        - Computes rollouts and their costs using the forward model;
        - Computes the score function (gradient of the target posterior distribution);
        - Calls the Stein optimizer to update policies;
        - Selects the current best policy based on rollout costs;
        - Returns current best action sequence and an iteration dictionary;
        - Rolls policies forward and updates optimizer state accordingly.

        Args:
            state (torch.Tensor): The current state of the system.
            model (BaseModel): The forward model with a `step` function and able to
              receive a dictionary of samples for uncertain model parameters.
            params_dist (dist.Distribution): The distribution used to sample uncertain
              model parameters.
            steps (int, optional): Number of Stein optimization steps performed.
              Defaults to 5.

        Returns:
            Tuple[torch.Tensor, dict]: Returns the current best action sequence and a
              dictionary with iteration data.
        """ """        """
        state = torch.as_tensor(state, dtype=torch.float).to(self.dev)

        def score_estimator(policies: torch.Tensor) -> Tuple[torch.Tensor, dict]:
            """This function is called by the Stein optimizer to compute the gradient
            of the log-posterior and, optionally, of a custom kernel.

            Args:
                policies (torch.Tensor): The set of policy particles with zeroed
                  gradients.

            Returns:
                Tuple[torch.Tensor, dict]: A tuple with a flattened tensor containing
                  the gradient of the log-posterior and a dictionary with the iteration
                  data. The kernel gram-matrix and gradient can be provided in this
                  dictionary using the keys `k_xx` and `grad_k`, respectively.
            """
            # flat_pol gradients will be set by Stein Sampler
            score_dict = {}
            if self.action_shape:
                actions = self._sample_actions(policies)
            else:  # will use autograd on policies mean
                actions = policies
            costs, trajectories, params_dict = self._get_costs(
                state, actions, model, params_dist
            )
            grad_log_p, loss = self._get_grad_log_p(costs, actions)
            # sum gradient along actions samples and flatten for stein
            grad_log_p = grad_log_p.flatten(1)

            score_dict["actions"] = actions
            score_dict["costs"] = costs
            score_dict["loss"] = loss
            score_dict["trajectories"] = trajectories
            score_dict["model_params"] = params_dict
            score_dict["sample_shape"] = self.action_shape
            return grad_log_p, score_dict

        # stein particles have shape [batch, extra_dims] and flattens extra_dims.
        data_dict, self.opt_state = self.stein_sampler.optimize(
            self.pol_mean, score_estimator, self.opt_state, steps
        )
        # to compute the weights, we may either re-sample the likelihood to get the
        # expected cost of the new θ_i or re-use the costs computed during the
        # `optimize` step to save computation.
        pol_weights = self._get_pol_weights(data_dict[steps - 1]["costs"])

        # Pick best policy
        i_star = pol_weights.argmax()
        a_seq = self.pol_mean[i_star].detach().clone()

        # housekeeping for next step
        self.roll(strategy=self.roll_strategy)
        self._update_optimizer()
        self._update_prior(pol_weights)
        return a_seq, data_dict
