import torch


class SquareRootScheduler:
    r"""Decays a constant parameter by the square root every epoch. By default,
    increments the epoch each time it is called.

    $$\rho=\rho_0 (t+1)^{-\frac{1}{2}}$$

    Args:
        parameter (float): The base parameter.
    """

    def __init__(self, parameter):
        self.param = torch.as_tensor(parameter)
        self.last_epoch = 0

    def __call__(self, update_epoch=True):
        val = self.param * (self.last_epoch + 1) ** -0.5
        if update_epoch:
            self.last_epoch += 1
        return val


class FactorScheduler:
    r"""Multiplicative decay by a constant parameter by gamma every epoch until reaching
    a predefined minimum value. By default, increments the epoch each time it is called.

    $$\rho_{t+1} \leftarrow \max \left(\rho_{\min }, \rho_t+\gamma\right) $$

    Args:
        parameter (float): The base parameter.
        gamma (float): Multiplicative factor of decay.
        parameter_min (float): Minimum value of parameter. Default: 1e-7.
    """

    def __init__(self, parameter, gamma, parameter_min=1e-7):
        self.param = torch.as_tensor(parameter)
        self.gamma = gamma
        self.param_min = torch.as_tensor(parameter_min)
        self.last_epoch = 0

    def __call__(self, update_epoch=True):
        val = torch.max(self.param_min, self.param * self.gamma ** self.last_epoch)
        if update_epoch:
            self.last_epoch += 1
        return val


class CosineScheduler:
    r"""A cosine-like schedule proposed by Loshchilov and Hutter (2016). Has the
    following functional form for learning rates in the range $t \in [0 , T]$. After
    $T$, the target parameter is sustained. By default, increments the epoch each time
    it is called.

    $$\rho_t = \rho_T + \frac{\rho_0-\rho_T}{2} (1 + \cos(\pi t/T))$$

    Args:
        parameter (float): The base parameter.
        target_parameter (float): The desired final value for the parameter.
        final_epoch (int): The epoch in which target_parameter is achieved.
        warmup_steps (int): Epoch at which the parameter decay starts. Default: 0.
    """

    def __init__(self, parameter, target_paremeter, final_epoch, warmup_steps=0):
        self.param = torch.as_tensor(parameter)
        self.target = torch.as_tensor(target_paremeter)
        self.final_epoch = final_epoch
        self.warmup = warmup_steps
        self.last_epoch = 0
        self.pi = torch.acos(torch.tensor(-1.0))

    def __call__(self, update_epoch=True):
        if self.last_epoch <= self.warmup:
            val = self.param
        elif self.last_epoch <= self.final_epoch:
            val = self.target + (self.param - self.target) / 2 * (
                1
                + torch.cos(
                    self.pi * (self.last_epoch - self.warmup) / self.final_epoch
                )
            )
        else:
            val = self.target
        if update_epoch:
            self.last_epoch += 1
        return val
