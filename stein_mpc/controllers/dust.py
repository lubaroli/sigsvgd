import torch
from torch import optim
import torch.distributions as dist

from ..controllers import BaseController
from ..inference.likelihoods import CostLikelihood, ExponentiatedUtility
from ..inference.svgd import SVGD, ScaledSVGD
from ..kernels import BaseKernel, ScaledGaussianKernel
from ..utils.math import grad_gmm_log_p, to_gmm
from gpytorch.priors import SmoothedBoxPrior

Empty = torch.Size([])


class DuSt(BaseController):
    def __init__(
        self,
        observation_space,
        action_space,
        hz_len,
        n_pol,
        n_pol_samples,
        n_params_samples=5,
        pol_mean=None,
        pol_cov=None,
        pol_hyper_prior=True,
        stein_sampler: str = "SVGD",
        kernel: BaseKernel = ScaledGaussianKernel(),
        likelihood: CostLikelihood = None,
        temperature=1.0,
        inst_cost_fn=None,
        term_cost_fn=None,
        params_sampling=True,
        params_log_space=False,
        weighted_prior: bool = False,
        autograd: bool = False,
        roll_strategy="repeat",
        optimizer_class=torch.optim.Adam,
        **opt_args,
    ):
        """Constructor for DuSt.

        :param observation_space: A Box object defining the action space.
        :type observation_space: gym.spaces.Space
        :param action_space: A Box object defining the action space.
        :type action_space: gym.spaces.Space
        :param hz_len: Number of time steps in the control horizon.
        :type hz_len: int
        :param temperature: Controller temperature parameter. Defaults to 1.0.
        :type temperature: float
        :param a_cov: covariance matrix of the actions multiplicative Gaussian
            noise. Effectively determines the amount of exploration
            of the system. If None, an appropriate identity matrix is used.
            Defaults to None.
        :type a_cov: torch.Tensor
        :key inst_cost_fn: A function that receives a trajectory and returns
            the instantaneous cost. Must be defined if no `term_cost_fn` is
            given. Defaults to None.
        :type kwargs: function
        :key term_cost_fn: A function that receives a state
            and returns its terminal cost. Must be defined if no
            `inst_cost_fn` is given. Defaults to None.
        :param params_sampling: Can be set to either 'none, 'single',
            'extended', or a Transformer object. If 'none', mean values of the
            parameter distribution are used if available. Otherwise, default
            model parameters are used. If 'single', one sample per rollout is
            taken and used for all `n_actions` trajectories. If 'extended',
            `n_actions` samples are taken per rollout, meaning each trajectory
            has their own sampled parameters. Finally, if a Transformer is
            provided it will be used *instead* of sampling parameters. Defaults
            to 'extended'.
        :type params_sampling: str or utils.utf.MerweScaledUTF
        :param init_actions:  A tensor of dimension `n_pol` x `hz_len` x
            `action_space.shape` containing the initial set of control actions.
            If None, the sequence is initialized to zeros. Defaults to None.
        :type init_actions: torch.Tensor

        .. note::
            * Actions will be clipped according to bounding action space regardless
              of the covariance set. Effectively `epsilon <= (max_a - min_a)`.
        """
        super().__init__(
            observation_space, action_space, hz_len, inst_cost_fn, term_cost_fn,
        )
        # initialize policies
        self.n_pol = n_pol
        self.pol_samples = n_pol_samples
        if pol_cov is None:
            self.pol_cov = torch.eye(self.dim_a)
        else:
            self.pol_cov = pol_cov
        if pol_mean is None:
            self.pol_mean = torch.zeros(self.n_pol, self.hz_len, self.dim_a)
        else:
            assert pol_mean.shape == (
                self.n_pol,
                self.hz_len,
                self.dim_a,
            ), "Initial policies shape mismatch."
            self.pol_mean = pol_mean.clone()
        self.pol_weights = torch.ones(self.n_pol)
        self.prior = to_gmm(self.pol_mean, self.pol_weights, self.pol_cov)
        hyper_prior = (
            SmoothedBoxPrior(self.min_a, self.max_a, 0.1).log_prob
            if pol_hyper_prior
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
        if stein_sampler == "SVGD":
            self.stein_sampler = SVGD(
                kernel,
                log_prior=hyper_prior,
                optimizer_class=optimizer_class,
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

        # configure params inference
        self._params_sampling = params_sampling
        self._params_log_space = params_log_space
        if (
            params_sampling is False
            or params_sampling is None
            or params_sampling == "none"
        ):
            self.params_samples = 1
            self._params_shape = None
        elif params_sampling is True:
            self.params_samples = n_params_samples
            self._params_shape = [self.params_samples]
        else:
            raise ValueError(
                "Invalid value for 'params_sampling': {}.".format(params_sampling)
            )
        # total amount of rollouts
        self.n_rollouts = self.params_samples * self.pol_samples * self.n_pol
        self.w_prior = weighted_prior
        self.autograd = autograd
        self.roll_strategy = roll_strategy

    def _compute_cost(self, states, actions, debug=False):
        """Estimate trajectories cost.

        :param states: A tensor with the states of each trajectory.
        :type states: torch.Tensor
        :param eps: A tensor with the difference of the current planned action
            sequence and the actions on each trajectory.
        :type eps: torch.Tensor
        :returns: A tensor with the costs for the given trajectories.
        :rtype: torch.Tensor
        """
        # Dims are n_params, n_actions, n_pol, hz_len, dim_s/dim_a
        x_vec = states[..., :-1, :].reshape(-1, self.dim_s)
        x_final = states[..., -1, :].reshape(-1, self.dim_s)
        a_vec = actions.reshape(-1, self.dim_a)
        inst_costs = self.inst_cost_fn(x_vec, a_vec, n_pol=self.n_pol, debug=debug)
        term_costs = self.term_cost_fn(x_final, n_pol=self.n_pol, debug=debug)

        inst_costs = inst_costs.view(
            self.params_samples, self.pol_samples, self.n_pol, self.hz_len
        )
        inst_costs = inst_costs.sum(dim=-1)  # sum over control horizon
        term_costs = term_costs.view(self.params_samples, self.pol_samples, self.n_pol)
        state_cost = (inst_costs + term_costs).mean(0)  # avg costs over params
        return state_cost

    def _rollout(self, init_state, actions, model, params_dist):
        """Perform rollouts based on current state and control plan.

        :param model: A model object which provides a `step` function to
            generate the system rollouts. If params_sampling is used, it must
            also implement a `sample_params` function for the parameters of the
            transition function.
        :type model: models.base.BaseModel
        :param state: The initial state of the system.
        :type state: torch.Tensor
        :param ext_actions: A matrix of shape `n_actions` x `n_pol` x `hz_len` x
            `dim_a` action sequences.
        :type actions: torch.Tensor
        :returns: A tuple of (actions, states, eps) for `n_actions` rollouts.
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # prepare tensors for batch rollout
        if self._params_shape is not None:  # sample params from `params_dist`
            event_shape = params_dist.event_shape
            dim_params = 1 if event_shape == Empty else event_shape[0]
            # `params` is a tensor of size `n_params` x `dim_params`
            params = params_dist.sample(self._params_shape)
            if self._params_log_space is True:
                params = params.exp()
            # repeat and reshape so each param is applied to all `n_actions`
            # (just using repeat would replicate the batch and wouldn't work)
            params = params.repeat(1, self.pol_samples * self.n_pol).reshape(
                -1, dim_params
            )
            # create dict with `uncertain_params` as keys for `model.step()`
            params_dict = model.params_to_dict(params)
        else:  # use default `model` params
            params_dict = None
        # flatten policies into single dim and repeat for each sampled param
        actions = actions.reshape(-1, self.hz_len, self.dim_a).repeat(
            self.params_samples, 1, 1
        )
        # expand initial state to the total amount of rollouts
        states = init_state.expand(self.n_rollouts, 1, -1).clone()

        # generate rollout
        for t in range(self.hz_len):
            states = torch.cat(
                [
                    states,
                    model.step(states[:, t], actions[:, t], params_dict).unsqueeze(1),
                ],
                dim=1,
            )

        # restore vectors dims, `n_params` is now first dimension
        states = states.reshape(
            -1, self.pol_samples, self.n_pol, self.hz_len + 1, self.dim_s
        )
        actions = actions.reshape(
            -1, self.pol_samples, self.n_pol, self.hz_len, self.dim_a
        )
        return states, params_dict

    def _sample_actions(self, pol_mean):
        # actions = self.prior.sample(sample_shape=[self.pol_samples])
        pi = dist.MultivariateNormal(pol_mean, self.pol_cov)
        # Use rsample to preserve gradient
        actions = pi.rsample([self.pol_samples])
        return actions

    def _get_costs(self, state, actions, model, params_dist):
        states, params_dict = self._rollout(state, actions, model, params_dist)
        costs = self._compute_cost(states, actions, debug=False)
        return costs, states, params_dict

    def _get_grad_log_p(self, costs, actions):
        # likelihood gradient...
        log_lik = self.likelihood.log_prob(costs)
        if self.autograd:  # ...using autograd
            grad_lik = torch.autograd.grad(log_lik.sum(), actions)[0]
            grad_lik = grad_lik.mean(0)  # average over samples
        else:  # ...likelihood trick (need softmax since we are using log_lik)
            bc_dims = torch.Size([1]) * len(self.prior.event_shape)
            pol_weight = log_lik.softmax(0).reshape(log_lik.shape + bc_dims)
            # computes gradients of action samples w.r.t. their policies
            grad_log_pol = (actions - self.pol_mean) @ self.pol_cov.inverse()
            grad_lik = torch.sum(pol_weight * grad_log_pol, dim=0)
        # prior gradient (i.e. prior of particles w.r.t to previous step prior)
        with torch.no_grad():
            grad_pri = grad_gmm_log_p(self.prior, actions).mean(0)
        grad_log_p = grad_pri + grad_lik
        return grad_log_p, log_lik

    def _get_pol_weights(self, costs, actions):
        """Aggregates sampled costs and computes particles weights.
        """
        log_l = self.likelihood.log_prob(costs)
        log_p = self.prior.log_prob(actions)
        log_w = (log_l + log_p).logsumexp(0)  # aggregate over samples
        log_w = log_w - log_w.logsumexp(0)  # normalize
        return log_w.exp()

    def _update_optimizer(self):
        """Updates the optimizer state after performing a shift operation on policies.
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

    def _update_prior(self, weights=None):
        if self.w_prior and weights is not None:
            self.pol_weights = weights
        else:
            self.pol_weights = torch.ones_like(self.pol_weights)
        self.prior = to_gmm(self.pol_mean, self.pol_weights, self.pol_cov)

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

    def forward(self, state, model, params_dist, steps=5):
        """Computes the next sequence of actions and updates the controller state.

            Called after the SVGD loop.
            1. Evaluate weights on updated stein particles : requires
            re-estimating the gradients on likelihood and prior for the new
             parameter values. Likelihood gradients therefore require additional
             sampling.
            2. Pick the best particle according to weights.
            3. Sets action sequence according to particle.
            4. Shift particles.
            5. Update prior using new weights.
       """
        state = torch.as_tensor(state, dtype=torch.float)

        def score_estimator(X):
            # X requires_grad will be set by Stein Sampler
            iter_dict = {}
            actions = self._sample_actions(X)
            costs, trajectories, params_dict = self._get_costs(
                state, actions, model, params_dist
            )
            grad_log_p, log_lik = self._get_grad_log_p(costs, actions)
            # sum gradient along actions samples and flatten for stein
            grad_log_p = grad_log_p.flatten(1)
            loss = -log_lik.sum(0)
            iter_dict["actions"] = actions.detach()
            iter_dict["costs"] = costs.detach()
            iter_dict["loss"] = loss.detach()
            iter_dict["trajectories"] = trajectories.detach()
            iter_dict["dyn_params"] = params_dict
            return grad_log_p, iter_dict

        # stein particles have shape [batch, extra_dims] and flattens extra_dims.
        trace, data_dict, self.opt_state = self.stein_sampler.optimize(
            self.pol_mean, score_estimator, self.opt_state, steps
        )
        # to compute the weights, we may either re-sample the likelihood to get the
        # expected cost of the new θ_i or re-use the costs computed during the
        # `optimize` step to save computation.
        pol_weights = self._get_pol_weights(
            data_dict[-1]["costs"], data_dict[-1]["actions"]
        )

        # Pick best particle
        i_star = pol_weights.argmax()
        a_seq = self.pol_mean[i_star].detach().clone()

        # roll thetas for next step
        self.roll(strategy=self.roll_strategy)
        self._update_optimizer()
        # and set the mixture of the new prior
        self._update_prior(pol_weights)
        return a_seq, data_dict
