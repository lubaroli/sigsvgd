"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Modified from kameleon_mcmc written (W) 2013 Heiko Strathmann and Dino Sejdinovic.
"""

from .distribution import Distribution
from torch import linspace, hstack, eye, sqrt, arange, zeros, tensor, randn, as_tensor
from torch.distributions import MultivariateNormal


class Banana(Distribution):
    """
    Banana distribution from Haario et al, 1999
    """

    def __init__(self, dimension=2, mean=0.0, bananicity=0.03, V=100.0):
        assert dimension >= 2
        Distribution.__init__(self, dimension)

        self.bananicity = tensor(bananicity)
        self.V = tensor(V)
        self.mean = tensor(mean)

    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "bananicity=" + str(self.bananicity)
        s += ", V=" + str(self.V)
        s += ", " + Distribution.__str__(self)
        s += "]"
        return s

    def sample(self, n=1):
        X = randn(list(n) + [2]) + self.mean
        X[:, 0] = sqrt(self.V) * X[:, 0]
        X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.V)
        if self.dimension > 2:
            X = hstack((X, randn(list(n) + [self.dimension - 2])))

        return X

    def log_prob(self, X):
        X = as_tensor(X)
        assert X.shape[-1] == self.dimension

        transformed = X.clone()
        transformed[..., 1] = X[..., 1] - self.bananicity * ((X[..., 0] ** 2) - self.V)
        transformed[..., 0] = X[..., 0] / sqrt(self.V)
        phi = MultivariateNormal(zeros([self.dimension]), eye(self.dimension))
        return phi.log_prob(transformed)

    def get_plotting_bounds(self):
        if self.bananicity == 0.03 and self.V == 100.0:
            return [(-20, 20), (-7, 12)]
        elif self.bananicity == 0.1 and self.V == 100.0:
            return [(-20, 20), (-5, 30)]
        else:
            return Distribution.get_plotting_bounds(self)

    def get_proposal_points(self, n):
        """
        Returns n points which lie on a uniform grid on the "center" of the banana
        """
        if self.dimension == 2:
            (xmin, xmax), _ = self.get_plotting_bounds()
            x1 = linspace(xmin, xmax, n)
            x2 = self.bananicity * (x1 ** 2 - self.V)
            return tensor([x1, x2]).T
        else:
            return Distribution.get_proposal_points(self, n)
