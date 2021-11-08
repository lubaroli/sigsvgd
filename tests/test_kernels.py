import time

import torch
import torch.autograd as autograd
from stein_mpc.kernels import (
    GaussianKernel,
    IMQKernel,
    ScaledGaussianKernel,
    ScaledIMQKernel,
)

from stein_mpc.kernels.ref_kernels import gaussian_kernel

BATCH, DIM = [4, 2]
torch.set_default_dtype(torch.double)


def get_runtime(kernel, X, Y, M, h, n_iter=100):
    start = time.time()
    M = torch.randn(DIM, DIM)
    M = M @ M.T
    h = 2.0
    for _ in range(n_iter):
        kernel(X, Y, M=M, h=h)
    end = time.time()
    return (end - start) / n_iter


X = torch.randn(BATCH, DIM).requires_grad_(True)
Y = torch.randn(BATCH, DIM)
# Y = torch.tensor([[3, 4]], dtype=torch.double)
# X = Y.clone().requires_grad_(True)
M = torch.eye(DIM)
h = torch.tensor(2.0)
# kernel = IMQKernel()
kernel = gaussian_kernel(M)
K_test, dK_test = kernel.calculate_kernel(X)
dK_test_ag = autograd.grad(K_test.sum(), X)[0]
X.detach_()
# kernel = ScaledIMQKernel()
kernel = ScaledGaussianKernel()
K, dK = kernel(X.requires_grad_(True), Y, M, h.sqrt())
dK_ag = autograd.grad(K.sum(), X)[0]
print(f"Normed diff: {(K - K_test).norm()}")
print(f"Normed grad diff: {(dK - dK_test).norm()}")

runtime = get_runtime(IMQKernel(), X, Y, M, h)
print(f"RBF kernel mean runtime: {runtime}")
runtime = get_runtime(ScaledIMQKernel(), X, Y, M, h)
print(f"Scaled RBF kernel mean runtime: {runtime}")
