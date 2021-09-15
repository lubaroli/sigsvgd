import matplotlib.pyplot as plt
import emcee
import seaborn as sns
import torch as th
import torch.distributions as dist
from stein_mpc.models.particle import Particle
from stein_mpc.utils.helper import save_progress
from stein_mpc.inference.likelihoods import GaussianLikelihood
from stein_mpc.inference.mpf import MPF
from tqdm import trange

th.manual_seed(0)


class Posterior:
    def __init__(self, prior, noise_sigma, model):
        self.densities = [prior]
        self.sigma = noise_sigma
        self.model = model

    def __call__(self, theta):
        return th.sum([p.log_prob(theta) for p in self.densities])

    def update(self, action, observation):
        new_lik = GaussianLikelihood(
            self.densities[-1].loc, self.sigma, self.model, log_space=True
        )
        new_lik.condition(action, observation)
        self.densities.append(new_lik)


if __name__ == "__main__":
    system_kwargs = {
        "dt": 0.015,
        "control_type": "acceleration",
        "init_state": th.zeros(4),
        "with_obstacle": False,
        "can_crash": False,
        "deterministic": True,
        "max_speed": None,
        "max_accel": 50,
        "map_cell_size": 0.1,
        "map_size": [10, 10],
        "map_type": "direct",
    }
    system = Particle(mass=1, **system_kwargs)
    model = Particle(uncertain_params=["mass"], **system_kwargs)
    steps = 200
    n_particles = 100
    prior = dist.MultivariateNormal(th.tensor([2.0]), th.eye(1) * 0.2)
    init_particles = prior.sample([n_particles]).clamp_(min=0.0)
    state = th.zeros(4)
    likelihood = GaussianLikelihood(state, 0.1, model, log_space=True)
    mpf = MPF(
        init_particles.log(),
        likelihood,
        optimizer_class=th.optim.SGD,
        lr=0.001,
        bw_scale=1.0,
    )
    mcmc_post = Posterior(prior, sigma, model)
    sampler = emcee.EnsembleSampler(
        4, 1, posterior, vectorize=True, args=[x, y, dx, dy]
    )
    save_path = "inference_tests"
    fig, axes = plt.subplots(2, sharex=True)
    sns.displot(
        mpf.prior.sample([200]).exp().numpy(), kind="kde", rug=True, ax=axes[0],
    )
    fig.set(xlim=(0, 4), ylim=(0, 10))
    plt.tight_layout()
    save_progress(save_path, fig=fig, fig_name="{:3d}.jpg".format(0))
    plt.close()
    for step in trange(steps):
        action = (th.rand(2) * 20) - 10  # action between -10 and 10
        new_state = system.step(state, action)
        mpf.optimize(action, new_state, bw=None, n_steps=100)

        state = new_state
        facet = sns.displot(
            mpf.prior.sample([200]).exp().numpy(), kind="kde", rug=True, ax=axes[0],
        )
        facet.set(xlim=(0, 4), ylim=(0, 10))
        # facet.set(xlim=(0, 30))
        plt.tight_layout()
        save_progress(save_path, fig=facet, fig_name="{:3d}.jpg".format(step + 1))
        plt.close()
