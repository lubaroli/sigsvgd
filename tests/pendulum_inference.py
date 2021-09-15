import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
import torch.distributions as dist
from stein_mpc.models.pendulum import PendulumModel
from stein_mpc.utils.helper import save_progress
from stein_mpc.inference.likelihoods import GaussianLikelihood
from stein_mpc.inference.mpf import MPF
from tqdm import trange

th.manual_seed(0)

if __name__ == "__main__":
    system = PendulumModel(mass=2)
    model = PendulumModel(uncertain_params=["mass"])
    steps = 200
    n_particles = 100
    init_particles = (
        dist.Normal(loc=th.tensor(1.0), scale=th.tensor(0.2))
        .sample([n_particles])
        .unsqueeze(-1)
    )
    init_particles.clamp_(min=0.0)
    likelihood = GaussianLikelihood(th.zeros(2,), 0.1, model, log_space=True)
    state = th.zeros(2)

    mpf = MPF(
        init_particles.log(),
        likelihood,
        optimizer_class=th.optim.SGD,
        lr=0.001,
        bw_scale=1.0,
    )
    save_path = "inference_tests"
    for step in trange(steps):
        facet = sns.displot(mpf.prior.sample([200]).exp().numpy(), kind="kde")
        # facet.set(xlim=(0, 4), ylim=(0, 8))
        # facet.set(xlim=(0, 30))
        plt.tight_layout()
        save_progress(save_path, fig=facet, fig_name="{:3d}.jpg".format(step))
        plt.close()
        action = (th.rand(1) / (0.5) ** 2) - 2  # action between -2 and 2
        new_state = system.step(state, action)
        mpf.optimize(action, new_state, bw=0.01, n_steps=100)
        state = new_state
