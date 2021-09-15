import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.optim as optim
from stein_mpc.inference.svgd import ScaledSVGD
from stein_mpc.kernels import ScaledGaussianKernel
from stein_mpc.LBFGS import FullBatchLBFGS

torch.random.manual_seed(42)
N_ITER = 100
BATCH = 100
DIM = 2


def create_movie_2D(x_all, log_p):
    fig = plt.figure(figsize=(8, 8))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    ngrid = 100
    x = np.linspace(-10, 10, ngrid)
    y = np.linspace(-10, 10, ngrid)
    X, Y = torch.tensor(np.meshgrid(x, y))
    Z = torch.exp(log_p(torch.vstack((torch.flatten(X), torch.flatten(Y))).T)).reshape(
        ngrid, ngrid
    )

    ax.set_title(str(0) + "$ ^{th}$ iteration")
    (markers,) = ax.plot(x_all[0][:, 0], x_all[0][:, 1], "ro", markersize=5)
    plt.axis([-10, 10, -10, 10])
    plt.contourf(X, Y, Z.data.numpy(), 30)

    def _init():  # only required for blitting to give a clean slate.
        ax.set_title(str(0) + "$ ^{th}$ iteration")
        return (markers,)

    def _animate(i):
        if i <= N_ITER:
            ax.set_title(str(i) + "$ ^{th}$ iteration")
            markers.set_xdata(x_all[i][:, 0])  # update particles
            markers.set_ydata(x_all[i][:, 1])  # update particles
            yield markers

    ani = animation.FuncAnimation(
        fig, _animate, init_func=_init, interval=100, blit=True, save_count=N_ITER
    )
    ani.save("tests/stein_movie_2D.mp4")


def plot_graph_2d(x_all, i, log_p):
    n_grid = 100
    x = np.linspace(-10, 10, n_grid)
    y = np.linspace(-10, 10, n_grid)
    X, Y = torch.tensor(np.meshgrid(x, y))
    Z = torch.exp(log_p(torch.vstack((torch.flatten(X), torch.flatten(Y))).T)).reshape(
        n_grid, n_grid
    )

    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.contourf(X, Y, Z.data.numpy(), 30)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.plot(x_all[i][:, 0], x_all[i][:, 1], "ro", markersize=5)
    plt.show()


if __name__ == "__main__":
    X0 = torch.randn([BATCH, DIM])
    X = X0.clone()
    kernel = ScaledGaussianKernel(bandwidth_fn=lambda *args: 1.0)

    optimizer = "LBFGS"
    if optimizer == "Adam":
        # Params for Adam
        opt = optim.Adam
        opt_kwargs = {
            "lr": 0.1,
        }
    elif optimizer == "FullBatch":
        # Params for FullBatchLBFGS
        optimizer = FullBatchLBFGS
        optimizer_kwargs = {
            "lr": 1.0,
            "line_search": "None",
        }
    else:
        # Params for LBFGS
        opt = optim.LBFGS
        opt_kwargs = {
            "lr": 1.0,
            "max_iter": 1,
            "line_search_fn": None,
        }

    # Parameters for the test distribution
    w = torch.tensor([0.5, 0.5])
    mean = torch.tensor([[-3, -3], [3, 3]])
    var = torch.tensor([[1.5, 1.5], [1.5, 1.5]])
    mix = dist.Categorical(w)
    comp = dist.Independent(dist.Normal(mean, var), 1)

    log_p = dist.MixtureSameFamily(mix, comp).log_prob

    svgd = ScaledSVGD(
        kernel=kernel, p_log_prob=log_p, optimizer_class=opt, **opt_kwargs
    )

    start = time.process_time()
    X_all, _ = svgd.optimize(X, n_steps=N_ITER)

    print(time.process_time() - start)
    create_movie_2D(X_all.numpy(), log_p)
    print("Done.")
