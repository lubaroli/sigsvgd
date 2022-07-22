import torch
import sigkernel
import matplotlib.pyplot as plt

from stein_mpc.kernels import GaussianKernel, PathSigKernel


# create two paths with different parameterization
def path_fn(t: torch.Tensor, freq=2.0, offset=0.0):
    PI = torch.asin(torch.tensor(1.0))
    return torch.sin(2 * PI * freq * (t + offset) / t.max())


# making the interval with same amount of points so we can compute the RBF kernel
int_1 = torch.arange(100)
int_2 = torch.arange(100) / 2
path_1 = path_fn(int_1)
path_2 = path_fn(int_2, offset=10)
path_3 = path_fn(int_2, offset=-10)

show_plot = False
if show_plot:
    plt.plot(int_1, path_1, label="Path 1")
    plt.plot(int_2, path_2, label="Path 2")
    plt.plot(int_2, path_3, label="Path 3")
    plt.legend()
    plt.show()

batched_paths = torch.cat([p.unsqueeze(0) for p in [path_1, path_2, path_3]], dim=0)

depth = 5
k_rbf = GaussianKernel()
k_myps = PathSigKernel()
static_kernel = sigkernel.RBFKernel(sigma=0.5)
k_ps = sigkernel.SigKernel(static_kernel, depth)
streams = batched_paths.unsqueeze(-1).double()

print("\nSimilarity using Gaussian Kernel:")
print(k_rbf(batched_paths, batched_paths, compute_grad=False))

print("\nSimilarity using Path Signature Kernel on coefficients:")
print(k_myps(streams, streams, ref_vector=path_1, depth=depth, compute_grad=False))

print("\nSimilarity using Path Signature Kernel:")
print(k_ps.compute_Gram(streams, streams))
