import time

import torch
import torch.autograd as autograd

# from stein_mpc.kernels.ref_kernels import gaussian_kernel
# from gpytorch.kernels.rbf_kernel import RBFKernel
from stein_mpc.kernels import GaussianKernel, ScaledGaussianKernel


def get_runtime(kernel, X, Y, M, h, n_iter=100):
    start = time.time()
    M = torch.randn(DIM, DIM)
    M = M @ M.T
    h = 2.0
    for _ in range(n_iter):
        kernel(X, Y, M=M, h=h)
    end = time.time()
    return (end - start) / n_iter


def naive_rbf(X, Y, h):
    batch, _ = X.shape
    gamma = -0.5 / h ** 2
    diffs = X[:, None, :] - Y[None, :, :]
    K = torch.empty(batch, batch)
    dK = torch.empty_like(diffs)
    for i in range(batch):
        for j in range(batch):
            # using row vectors
            dot = diffs[i, j] @ diffs[i, j].T
            K[i, j] = (gamma * dot).exp()
            dK[i, j] = 2 * gamma * diffs[i, j] * K[i, j]
    return K, dK


BATCH, DIM = [4, 2]
torch.set_default_dtype(torch.double)

X = torch.randn(BATCH, DIM)
Y = torch.randn(BATCH, DIM)
Y = X.clone()
M = torch.eye(DIM)
h = torch.tensor(1.0)
# kernel = RBF(h.numpy())
# kernel.lengthscale = h
# kernel = gaussian_kernel(M)
# K_test, dK_test = kernel.calculate_kernel(X)
K_test, dK_test = naive_rbf(X, Y, h)
# dK_test_ag = autograd.grad(K_test.sum(), X)[0]
# X.detach_()
kernel = GaussianKernel()
K, dK = kernel(X.requires_grad_(True), Y, h)
dK_ag = autograd.grad(K.sum(), X)[0]
X.detach_()
print(f"Kernel all close: {torch.allclose(K, K_test)}")
print(f"Kernel grad all close: {torch.allclose(dK.sum(1), dK_ag)}")

runtime = get_runtime(GaussianKernel(), X, Y, M, h)
print(f"RBF kernel mean runtime: {runtime}")
runtime = get_runtime(ScaledGaussianKernel(), X, Y, M, h)
print(f"Scaled RBF kernel mean runtime: {runtime}")
