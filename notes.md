# Questions
- When shifting time-steps in MPC, how to keep Hessian approximation history in L-BFGS? Do a rank 1 approximation?
- Should the median heuristic have the 0.5 factor? Why?
- Could the norm of the bound on the stein velocity norm be used to dynamically adjust the step size?
- In stein MPC, should the SVGD prior be a GMM with independent dimensions along the control horizon or a summation of Multivariate Gaussians?
- When computing the Exponentiated Likelihood, should we sum over all trajectories (i.e. compute a policy likelihood) or should we compute the likelihood of each action (i.e. each with a single sample) and then use a Softmax to aggregate each gradient?

# TODO
- Add MPF to current repo
- Add support for L-BFGS Hessian matrix approximation
- Add support to Fischer Matrix
- Add support to full-Hessian
- Figure out why L-BFGS with preconditioning is not working...

# IDEAS
1. Encode control trajectories with path signature as a sort of invariant parametrization, allowing for reduced dimensionality. Update: wouldn't invariance be a negative feature for trajectory variety? Similar trajectories at a different rotation/orientation are effectively different in state space.
2. Embed fixed particles in control trajectories to enforce diversity, gradients for those particles would be always zero. This seems to work well conceptually in 2D space (as control sequences are always relative to position), would it be valid in higher dimensions?
3. Use mappings (perhaps GPs) to project control trajectories to state-space. Compute dissimilarities between trajectories using Frechet (or path signatures?) and inverse-map to control space.
4. Is it possible to reduce the dimension of the kernel repulsive force by mapping control sequences to trajectories in state space, computing the Frechet distance, and reverting to control space? If so, we would a single repulsive force for all control actions?

# DOUBTS
- Is there a point in enforcing smooth control sequences (trajectories) if on the decision step the trajectory selected or sampled has no guarantee of being the same as in the previous time-step? In other words, since each decision step is independent of the previous ones, there will be discontinuities at each time-step.
- Should we recode particles as action sequences instead of policies? Or just keep the code as is using a single policy in the configuration?
- In Sasha's paper, the Monte-Carlo gradient of the log-likelihood is weighted by the softmax of the likelihood times the gradient of the log prior. However, on REINFORCE the weights seem to be just the likelihood of each sample (divided by the number of samples). Which is correct? Seems like REINFORCE would result in larger and flatter weights...
- What about likelihood gradients using autograd? For when multiple policies are used, should we just compute the gradient w.r.t. the policy mean? Or should we also sample and aggregate the gradients of the samples? If so, how to aggregate?
- When computing policy weights, should we take the prior into account? Also, when taking samples to compute gradients, should we average over the sampled costs or use the softmax (twice!)?