import emcee
import matplotlib.pyplot as plt
import numpy as np
import torch
from stein_mpc.utils.helper import save_progress
from stein_mpc.inference.likelihoods import GaussianLikelihood
from stein_mpc.inference.mpf import MPF
from torch.distributions import MultivariateNormal


class SSModel:
    def __init__(self, state_dim: int, control_dim: int, parameter: torch.Tensor):
        self.state_matrix = torch.randn(state_dim, state_dim)
        self.control_matrix = torch.randn(state_dim, control_dim)
        self.parameter = parameter
        self.state_dim = state_dim

    def step(self, x: torch.Tensor, u: torch.Tensor, parameter: torch.Tensor):
        """
        :param x: NxD states matrix
        :param u: NxM controls matrix
        :param parameter: Nx1 or NxM
        :return: NxD resulting states matrix
        """
        if parameter.numel() == 1:
            return x @ self.state_matrix.t() + (u / parameter) @ self.control_matrix.t()
        raw_x_to_x = x @ self.state_matrix.t()
        x_to_x = raw_x_to_x.repeat(parameter.shape[0], 1, 1)
        u_to_x = (u @ self.control_matrix.t()).repeat(
            parameter.shape[0], 1, 1
        ) / parameter.view(parameter.shape[0], 1, 1)
        return x_to_x + u_to_x


class Posterior:
    def __init__(
        self,
        ssm: SSModel,
        param_dim: int,
        observation_noise_sd: float,
        prior_mean: float,
    ):
        self.ssm = ssm
        self.prior = MultivariateNormal(prior_mean, torch.eye(param_dim))
        self.obs_noise_sd = observation_noise_sd
        self._states_list = []
        self._next_states_list = []
        self._controls_list = []

    def update(
        self,
        prev_state: torch.Tensor,
        applied_control: torch.Tensor,
        next_states: torch.Tensor,
    ):
        """
        :param next_states:
        :param prev_state: D-array
        :param applied_control: M-array
        :return:
        """
        self._states_list += [prev_state]
        self._controls_list += [applied_control]
        self._next_states_list += [next_states]

    @property
    def states(self):
        return torch.stack(self._states_list)

    @property
    def controls(self):
        return torch.stack(self._controls_list)

    @property
    def next_states(self):
        return torch.stack(self._next_states_list)

    def log_prior(self, parameter: torch.Tensor):
        log_p = self.prior.log_prob(parameter)
        assert log_p.numel() == parameter.shape[0]
        return log_p

    def log_likelihood(
        self,
        parameter: torch.Tensor,
        previous_states: torch.Tensor,
        controls: torch.Tensor,
        resulting_state: torch.Tensor,
    ):
        prediction = self.ssm.step(previous_states, controls, parameter)
        log_l = (
            MultivariateNormal(
                resulting_state, scale_tril=torch.eye(self.ssm.state_dim)
            )
            .log_prob(prediction)
            .sum(dim=1)
        )
        assert log_l.numel() == parameter.shape[0]
        return log_l

    def log_prob_np(self, parameter: np.ndarray):
        return self.log_prob(
            torch.tensor(parameter, dtype=torch.get_default_dtype())
        ).numpy()

    def log_prob(self, parameter: torch.Tensor):
        states = self.states
        next_states = self.next_states
        controls = self.controls
        log_p = self.log_prior(parameter)
        log_l = self.log_likelihood(parameter, states, controls, next_states).view_as(
            log_p
        )
        return log_l + log_p


if __name__ == "__main__":
    s_dim = 2
    p_dim = 1
    c_dim = 1

    true_parameter = torch.rand(p_dim) + 10
    model = SSModel(s_dim, c_dim, true_parameter)
    obs_noise_sd = 0.1
    posterior = Posterior(model, p_dim, obs_noise_sd, 9 * torch.ones(p_dim))
    n_walkers = 4
    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=p_dim,
        vectorize=True,
        log_prob_fn=posterior.log_prob_np,
    )
    n_samples = 400
    n_burn = 200

    state = torch.randn(s_dim)

    likelihood = GaussianLikelihood(state, obs_noise_sd, model, log_space=False)

    mpf = MPF(
        posterior.prior.sample([n_samples]),
        likelihood,
        optimizer_class=torch.optim.Adam,
        lr=0.1,
        bw_scale=1.0,
    )
    save_path = "inference_tests"
    n_steps = 10
    ax1 = plt.subplot(211)
    plt.title("Step 0")
    ax1.hist(posterior.prior.sample([n_samples]).numpy(), density=True, bins=40)
    ax1.axvline(true_parameter.item(), ls="--", c="r")
    mpf_samples = mpf.prior.sample([n_samples]).reshape(-1).numpy()
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.hist(mpf_samples, density=True, bins=40)
    ax2.axvline(true_parameter.item(), ls="--", c="r")
    save_progress(save_path, fig=ax2, fig_name="{:3d}.jpg".format(0))
    plt.close()
    for t in range(n_steps):
        previous_state = state
        control = torch.randn(c_dim) * 100
        state = model.step(state.view(1, -1), control.view(1, -1), true_parameter).view(
            -1
        )
        posterior.update(previous_state, control, state)

        grads = mpf.optimize(control, state, bw=None, n_steps=100)
        sampler.run_mcmc(posterior.prior.sample([n_walkers]), n_samples + n_burn)
        mc_samples = sampler.get_chain(flat=True, discard=n_burn)
        bw = float(mpf.prior.component_distribution.variance[0, 0].sqrt())
        ax1 = plt.subplot(311)
        plt.title("Step {0}, bandwidth {1}".format(t + 1, bw))
        ax1.hist(mc_samples.reshape(-1), density=True, bins=40)
        ax1.axvline(true_parameter.item(), ls="--", c="r")
        mpf_samples = mpf.prior.sample([n_samples]).reshape(-1).numpy()
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.hist(mpf_samples, density=True, bins=40)
        ax2.axvline(true_parameter.item(), ls="--", c="r")
        ax3 = plt.subplot(313)
        ax3.plot(grads)
        save_progress(save_path, fig=ax2, fig_name="{:3d}.jpg".format(t + 1))
        plt.close()

    print("Done")
