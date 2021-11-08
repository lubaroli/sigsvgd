import torch


class scaled_hessian_RBF2(torch.nn.Module):
    def __init__(self, sigma=None):
        super(scaled_hessian_RBF2, self).__init__()
        self.sigma = sigma

    def __call__(self, X, Y, metric=None):
        if torch.is_tensor(metric):
            self.sigma = metric
        else:
            self.sigma = torch.eye(X.shape[1])

        def compute_script(X, Y, sigma):
            K_XY = torch.zeros(X.size(0), Y.size(0))
            dK_XY = torch.zeros(X.shape)
            for i in range(X.shape[0]):
                sign_diff = Y[i, :] - X
                Msd = torch.matmul(sign_diff, sigma)
                K_XY[i, :] = torch.exp(-0.5 * torch.sum(sign_diff * Msd, 1))
                dK_XY[i] = K_XY[i, :].matmul(Msd) * 2
            return K_XY, dK_XY

        K_XY, dK_XY = compute_script(X, Y, self.sigma)
        return K_XY, dK_XY


class gaussian_kernel(object):
    def __init__(self, Q, adaptive=False, decay=False):
        self.Q = 0.5 * (Q + Q.T)
        self.d = Q.shape[0]
        self.adaptive = adaptive
        self.decay = decay

    def calculate_kernel(self, x):
        n, d = x.shape
        diff = x[:, None, :] - x[None, :, :]
        Qdiff = torch.matmul(diff, self.Q)
        if self.adaptive:
            h = torch.mean(
                torch.sum(diff * Qdiff, axis=-1)
            )  # faster calculation, for small number of particles should use median distant
            # h = np.median(np.sum(diff * Qdiff, axis = -1))
            if self.decay:
                h /= 10.0
            else:
                h /= 2.0
            h /= torch.log(n)
        else:
            h = self.d
        K = torch.exp(-torch.sum(Qdiff * diff, axis=-1) / (2.0 * h))
        gradK = -Qdiff * K[:, :, None] / h
        return K, gradK
