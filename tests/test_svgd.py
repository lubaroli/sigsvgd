import time

import torch
import torch.distributions as dist
from gpytorch.priors import SmoothedBoxPrior
from stein_mpc.inference.svgd import SVGD, ScaledSVGD
from stein_mpc.kernels import ScaledGaussianKernel
from stein_mpc.LBFGS import FullBatchLBFGS
from stein_mpc.utils.plots import create_2d_particles_movie
from stein_mpc.models.environment import star_gaussian

# torch.random.manual_seed(42)
# torch.set_default_dtype(torch.float64)
N_ITER = 200
BATCH = 100
DIM = 2
LR = 0.2


if __name__ == "__main__":
    X0 = torch.randn([BATCH, DIM])
    X0 = X0.uniform_(-8, 8)
    X = X0.clone()
    # kernel = ScaledGaussianKernel(bandwidth_fn=lambda *args: DIM ** 0.5)
    # kernel.analytic_grad = False
    kernel = ScaledGaussianKernel()

    optimizer = "adam"
    if optimizer.lower() == "adam":
        opt = torch.optim.Adam
        opt_kwargs = {
            "lr": LR,
        }
    elif optimizer.lower() == "adagrad":
        opt = torch.optim.Adagrad
        opt_kwargs = {
            "lr": LR,
        }
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD
        opt_kwargs = {
            "lr": LR,
        }
    elif optimizer.lower() == "fullbatch":
        # Params for FullBatchLBFGS
        opt = FullBatchLBFGS
        opt_kwargs = {
            "lr": LR,
            "line_search": "None",
        }
    elif optimizer.lower() == "lbfgs":
        # Params for LBFGS
        opt = torch.optim.LBFGS
        opt_kwargs = {
            "lr": LR,
            "max_iter": 1,
            "line_search_fn": None,
        }
    else:
        raise ValueError("Invalid optimizer: {}".format(optimizer))

    # Parameters for the test distribution
    w = torch.tensor([0.5, 0.5])
    mean = torch.tensor([[-3, -3], [3, 3]])
    var = torch.tensor([[1.5, 1.5], [1.5, 1.5]])
    mix = dist.Categorical(w)
    comp = dist.Independent(dist.Normal(mean, var), 1)
    lik = dist.MixtureSameFamily(mix, comp)
    pri = SmoothedBoxPrior(-1, 5, sigma=0.1)

    env = star_gaussian(50, 5)  # star gaussian mixture example

    log_p = lik.log_prob
    log_prior = pri.log_prob
    # log_prior = None

    stein_method = "scaled"
    if stein_method.lower() == "svgd":
        stein_sampler = SVGD(
            kernel, log_p, log_prior, optimizer_class=opt, **opt_kwargs
        )
    else:
        stein_sampler = ScaledSVGD(
            kernel,
            log_p,
            log_prior,
            optimizer_class=opt,
            precondition=True,
            **opt_kwargs,
        )

    start = time.process_time()
    X_all = stein_sampler.optimize(X, n_steps=N_ITER)[0]

    # uncomment this code to test resetting the optimizer at every timestep
    # X_all = []
    # for step in range(N_ITER):
    #     stein_sampler.step(X)
    #     X_all.append(X.clone().detach().numpy())

    print(f"Time taken: {time.process_time() - start:.2f} seconds.")
    print("Creating movie...")
    create_2d_particles_movie(
        X_all, log_p, n_iter=N_ITER, save_path="tests/results/stein_movie_2D.mp4"
    )
    print("Done.")
