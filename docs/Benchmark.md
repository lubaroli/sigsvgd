Before, with `torch`:

```sh
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    40                                               @profile
    41                                               def phi(self, bw):
    42                                                   """
    43                                                       Uses manually-derived likelihood gradient.
    44                                                   """
    45      1328      13138.0      9.9      0.1          x = self.x.detach().clone().requires_grad_(True)
    46      1328    7821281.0   5889.5     35.5          grad_prior = th.autograd.grad(self.prior.log_prob(x).sum(), x)[0]
    47      1327    4789499.0   3609.3     21.7          obs = self.likelihood.sample(x)
    48      1327    4927073.0   3712.9     22.4          log_l = self.likelihood.log_prob(obs)
    49                                                   # Analytic gradient has 2 parts, d_log Normal and d_log Model
    50                                                   # grad_lik = (obs - x) / (self.likelihood.sigma ** 2)
    51      1327     813499.0    613.0      3.7          grad_lik = th.autograd.grad(log_l.sum(), x)[0]
    52      1327       9428.0      7.1      0.0          score_func = grad_lik + grad_prior
    53                                           
    54      1327    2909813.0   2192.8     13.2          k_xx = self.kernel(x.flatten(1, -1), x.detach().clone().flatten(1, -1), bw=bw)
    55      1327     666791.0    502.5      3.0          grad_k = th.autograd.grad(k_xx.sum(), x)[0]
    56                                           
    57      1327      71980.0     54.2      0.3          phi = grad_k + th.tensordot(k_xx.detach(), score_func, dims=1) / x.size(0)
    58      1327       1345.0      1.0      0.0          return phi
```


After, with `fast_gmm_diff`:

```sh
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    53                                               @profile
    54                                               def phi(self, bw):
    55                                                   """
    56                                                       Uses manually-derived likelihood gradient.
    57                                                   """
    58      1214      14554.0     12.0      0.1          x = self.x.detach().clone().requires_grad_(True)
    59                                           
    60      1214     108677.0     89.5      1.1          grad_prior = th.from_numpy(self.prior2.log_prob_grad(self.x.numpy()))
    61                                                   # grad_prior = th.autograd.grad(self.prior.log_prob(x).sum(), x)[0]
    62                                           
    63      1214    5322813.0   4384.5     52.4          obs = self.likelihood.sample(x)
    64                                           
    65                                                   # log_l = self.likelihood.log_prob(obs)
    66                                                   # grad_lik = th.autograd.grad(log_l.sum(), x)[0]
    67                                           
    68                                                   # The following implementation still needs to pass gradient through
    69                                                   # `self.likelihood.sample(x)`.
    70      1213     728271.0    600.4      7.2          grad_lik = self.likelihood.log_prob_grad(obs, x)
    71                                           
    72                                                   # Analytic gradient has 2 parts, d_log Normal and d_log Model
    73                                                   # grad_lik = (obs - x) / (self.likelihood.sigma ** 2)
    74      1213       9867.0      8.1      0.1          score_func = grad_lik + grad_prior
    75                                           
    76      1213    3162175.0   2606.9     31.1          k_xx = self.kernel(x.flatten(1, -1), x.detach().clone().flatten(1, -1), bw=bw)
    77      1213     737362.0    607.9      7.3          grad_k = th.autograd.grad(k_xx.sum(), x)[0]
    78                                           
    79      1213      79233.0     65.3      0.8          phi = grad_k + th.tensordot(k_xx.detach(), score_func, dims=1) / x.size(0)
    80      1213       1494.0      1.2      0.0          return phi
```
