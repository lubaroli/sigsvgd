from __future__ import annotations

from typing import Union

import torch
import torch.distributions as dist
from scipy.stats import scoreatpercentile as p_score


def _select_sigma(x: torch.Tensor, percentile: int = 25):
    """
    Returns the smaller of std or normalized IQR of x over axis 0. Code originally from:
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py
    References
    ----------
    Silverman (1986) p.47
    """
    # normalize = norm.ppf(.75) - norm.ppf(.25)
    normalize = 1.349
    IQR = (p_score(x, 100 - percentile) - p_score(x, percentile)) / normalize
    std_dev = torch.std(x, axis=0)
    if IQR > 0 and IQR < std_dev.min():
        return IQR
    else:
        return std_dev


def bw_median(
    sq_dists: torch.Tensor, bw_scale: float = 1.0, tol: float = 1.0e-8
) -> torch.Tensor:
    h = torch.median(sq_dists)
    h = h / torch.tensor(sq_dists.shape[0] + 1.0).log()
    h = bw_scale * h.sqrt()
    return h.clamp_min_(tol)


def bw_silverman(sq_dists: torch.Tensor, bw_scale: float = 1.0) -> torch.Tensor:
    """
    Computes bandwidth according to `Silverman's Rule of Thumb`_.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth

    Returns
    -------
    bw : float
      :  The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::
       A = min(std(x), IQR/1.349)
       IQR = score_at_percentile(x, [75,25]))

    References
    ----------
    Silverman, B.W. (1986) `Density Estimation.`

    .. _Reference Code:
        https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py
    """
    A = _select_sigma(sq_dists)
    n = len(sq_dists)
    return bw_scale * (0.9 * A * n ** (-0.2))


def pw_dist_sq(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise squared distance matrix between batched vectors:
        `dist = ||mat1 - mat2||^2`

    Args:
        mat1 (torch.Tensor): The first [batch, dim] vector.
        mat2 (torch.Tensor): The second [batch, dim] vector.

    Returns:
        torch.Tensor: A [batch, batch] matrix with the squared distances between
        elements.
    """
    mat1_norm = mat1.pow(2).sum(dim=-1, keepdim=True)
    mat2_norm = mat2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        mat2_norm.transpose(-2, -1), mat1, mat2.transpose(-2, -1), alpha=-2
    ).add_(mat1_norm)
    return res.clamp(min=0)  # avoid negative distances due to num precision


def naive_scaled_pw_dist_sq(
    mat1: torch.Tensor, mat2: torch.Tensor, metric: torch.Tensor
) -> torch.Tensor:
    """Computes the pairwise distance matrix between batched row vectors.
        `dist = (mat1 - mat2)^T @ metric @ (mat1 - mat2)`

    Args:
        mat1 (torch.Tensor): The first [batch, dim] vector.
        mat2 (torch.Tensor): The second [batch, dim] vector.

    Returns:
        torch.Tensor: A [batch, batch] matrix with the squared distances between
        elements.
    """
    batch = mat1.shape[0]
    if mat2 is None:
        mat2 = mat1
    res = torch.empty([batch, batch])
    for i in range(batch):
        for j in range(batch):
            vec1 = mat1[i].unsqueeze(1)
            vec2 = mat2[j].unsqueeze(1)
            diff = vec1 - vec2
            res[i, j] = diff.T @ metric @ diff
    return res


def scaled_pw_dist_sq(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    metric: torch.Tensor,
    return_gradient: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Computes squared distance matrix between batched row vectors.
        `dist = (mat1 - mat2) @ metric @ (mat1 - mat2)^T`

    Args:
        mat1 (torch.Tensor): The first [batch, dim] matrix.
        mat2 (torch.Tensor): The second [batch, dim] matrix.
        metric (torch.Tensor): The metric matrix of size [dim, dim].
        return_gradient(bool): If True, returns the `mat1 @ metric` matrix
            multiplication.

    Returns:
        If `return_product` is False, returns a [batch, batch] matrix with the squared
        distances between elements. Otherwise, returns a tuple of squared_distances
        and the `mat1 @ metric` matrix multiplication.
    """
    # Use brodcasting to compute the cross-difference
    diff = mat1.unsqueeze(0) - mat2.unsqueeze(1)
    diff_M = diff @ metric
    res = (diff_M * diff).sum(axis=-1)
    if return_gradient:
        return res.clamp(min=0), diff_M
    else:
        return res.clamp(min=0)


def to_gmm(
    x: torch.Tensor, weights: torch.Tensor, covariance: torch.Tensor
) -> dist.mixture_same_family.MixtureSameFamily:
    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.MultivariateNormal(x.detach(), covariance), 1)
    return dist.mixture_same_family.MixtureSameFamily(mix, comp)


def grad_gmm_log_p(p: torch.distributions.MixtureSameFamily, samples: torch.tensor):
    mu = p.component_distribution.mean
    cov = p.component_distribution.variance
    grad = -(samples - mu) / cov
    return grad
