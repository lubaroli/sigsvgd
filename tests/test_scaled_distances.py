import time

import torch
from stein_mpc.utils.math import naive_scaled_pw_dist_sq, scaled_pw_dist_sq

x = torch.randn(100, 10)
y = torch.randn(100, 10)
metric = torch.eye(10)
start = time.time()
ans1 = naive_scaled_pw_dist_sq(x, y, metric)
end = time.time()
print(f"Naive scaled dist time: {end-start}")
start = time.time()
ans2 = scaled_pw_dist_sq(x, y, metric)
end = time.time()
print(f"Batched scaled dist time: {end-start}")
print(f"Normed diff: {(ans1 - ans2).norm()}")
