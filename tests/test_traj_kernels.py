import torch
from stein_mpc.kernels import PathSigKernel


torch.random.manual_seed(0)
BATCH, HZ, DIM = [128, 25, 1]


def phi(X):
    return torch.cat((X.cos(), X.sin()), -1)


X = torch.randn(BATCH, HZ, DIM).requires_grad_(True)
Y = torch.randn(BATCH, HZ, DIM)
h = torch.tensor(2.0)


kernel = PathSigKernel()
K, dK = kernel(phi(X), phi(Y), X, depth=3, h=h.sqrt())
