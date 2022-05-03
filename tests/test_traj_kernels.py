import time

import torch
from stein_mpc.kernels import TrajectoryKernel


BATCH, DIM = [4, 2]
torch.set_default_dtype(torch.double)


def phi(X):
    return X.norm(dim=-1).view(-1, 1)


X = torch.randn(BATCH, DIM).requires_grad_(True)
Y = torch.randn(BATCH, DIM)
# Y = torch.tensor([[3, 4]], dtype=torch.double)
# X = Y.clone().requires_grad_(True)
M = torch.eye(DIM)
h = torch.tensor(2.0)

kernel = TrajectoryKernel()
K, dK = kernel(phi(X), phi(Y), X, h.sqrt())
