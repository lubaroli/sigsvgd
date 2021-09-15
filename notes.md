1. If stein inference is initiated with a `model.log_p` likelihood distribution, is it updated if model parameters change?
2. In sequential stein updates during the same timestep, do we update the prior (`q`) after each intermediate mapping step? No
3. When shifting timesteps in MPC, how to keep Hessian approximation history in L-BFGS? Do a rank 1 approximation?
4. Should the median heuristic have the 0.5 factor? Why?
5. Could the norm of the bound on the stein velocity norm be used to dynamically adjust the step size?
6. Should we add a control cost term to SVMPC and SQNMPC rollouts?
7. In stein MPC, should the SVGD prior be a GMM with independent dimensions along the control horizon or a summation of Multivariate Gaussians?

Debug matrix_svgd

import torch

xg = torch.as_tensor(x).requires_grad_(True)
Fg = torch.zeros(n)
Jg = torch.zeros_like(xg)
for k in range(self.K):
	pdfi = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.as_tensor(self.mu[k,:]), covariance_matrix=torch.as_tensor(self.sigma[k,:,:])).log_prob(xg).exp() + 1e-20
	Fg += pdfi
	Jg += pdfi[:, None] * torch.matmul(torch.as_tensor(self.mu[k,:]) - xg, torch.as_tensor(self.inv_sigma[k,:,:]))
augrad = torch.autograd.grad((Jg/Fg[:, None]).sum(), xg)