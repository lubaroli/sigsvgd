import torch
import matplotlib.pyplot as plt
from gpytorch.priors import SmoothedBoxPrior

b = torch.tensor([2.9671, 1.8326, 2.9671, 0.0873, 2.9671, 3.8223, 2.9671])
a = -b

f = SmoothedBoxPrior(a, b, 0.1)
x = torch.linspace(-5, 5, 100)
x = x[:, None].repeat(1, 7)

plt.plot(x, f.log_prob(x))
plt.show()
