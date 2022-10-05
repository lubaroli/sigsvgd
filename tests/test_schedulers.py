from stein_mpc.utils.scheduler import (
    CosineScheduler,
    FactorScheduler,
    SquareRootScheduler,
)

param = 1
steps = 500

CosSch = CosineScheduler(param, 0.1, 20, 5)
print("Cosine: ", [CosSch().item() for _ in range(steps)])

SqrSch = SquareRootScheduler(param)
print("Square Root: ", [SqrSch().item() for _ in range(steps)])

FacSch = FactorScheduler(param, 0.99, 0.0)
print("Factor: ", [FacSch().item() for _ in range(steps)])
