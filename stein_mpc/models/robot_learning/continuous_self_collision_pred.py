import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import optim, nn, utils

from stein_mpc.models.robot.robot_simulator import PandaRobot
from stein_mpc.utils.helper import get_project_root


class ContinuousSelfCollisionPredictor(nn.Module):
    def __init__(self, n_dimension, hidden_size=200, n_hidden_layers=5, device="cpu"):
        super().__init__()

        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(n_dimension, hidden_size))
                layers.append(nn.ReLU())
            elif i == n_hidden_layers - 1:
                layers.append(nn.Linear(hidden_size, 1))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, qs):
        # ASSUME inputs are in radians
        return self.layers(qs)
        # return self.layers(torch.cos(qs))


class ModelTrainer(pl.LightningModule):
    def __init__(
        self,
        n_dimension,
        class_weight=(1,),
        hidden_size=200,
        n_hidden_layers=5,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.net = ContinuousSelfCollisionPredictor(
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            n_dimension=n_dimension,
        )

        self.Loss = nn.BCEWithLogitsLoss(
            # pos_weight=torch.Tensor(class_weight)
        )
        self.n_dimension = n_dimension
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def _step(self, batch, batch_idx):
        x = batch[:, : self.n_dimension]
        y = batch[:, self.n_dimension : self.n_dimension + 1]

        # skip the last sigmoid and use BCEwithLogit
        pred = self.net.layers[:-1](x)
        loss = self.Loss(pred, y)
        return loss

    def training_step(self, *args):
        loss = self._step(*args)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def validation_step(self, *args):
        loss = self._step(*args)
        self.log("val_loss", loss, on_step=True)
        return loss


def train(data_fname, n_dimension, save_path=None):
    data = np.load(data_fname, allow_pickle=True).item()
    queried_qs = np.array(data["queried_qs"])
    queried_result = np.array(data["queried_result"])

    dataset = torch.Tensor(np.hstack([queried_qs, queried_result.reshape(-1, 1)]))
    # invert the probability
    # dataset[:, -1] = 1 - dataset[:, -1]

    n_total = int(dataset.shape[0])
    n_occupied = int(dataset[:, -1].sum())
    n_free = n_total - n_occupied
    print(f"free: {n_free}, occupied: {n_occupied}, ratio={n_free / n_occupied:.3f}")

    # the original data is too sharp which made it hard to descent... adding noise to
    # smooth it out.
    dataset[:, :] += torch.randn(dataset.shape) * 0.05

    class_weight = [n_free / n_occupied]

    model = ModelTrainer(n_dimension=n_dimension, class_weight=class_weight)
    trainer = pl.Trainer(
        #     limit_train_batches=100,
        #     max_epochs=10,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accelerator="gpu",
    )

    ##########################################
    train_set_size = int(len(dataset) * 0.9)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = utils.data.random_split(
        dataset, [train_set_size, valid_set_size], generator=seed
    )
    ##########################################
    batch_size = 1024
    train_set = utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12,
    )
    valid_set = utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=12,)
    trainer.fit(model, train_set, valid_set)
    if save_path is not None:
        trainer.save_checkpoint(save_path)
    return model


def load_trained_model(weight_fname, map_location="cpu"):
    try:
        # use the built in loading function
        return ModelTrainer.load_from_checkpoint(weight_fname).net

    except TypeError:
        # try to manually load the weight
        weights = torch.load(weight_fname, map_location=map_location)
        _inner_weights = {}
        _net_prefix_token = "net."
        for k, v in weights["state_dict"].items():
            if k.startswith(_net_prefix_token):
                _inner_weights[k[len(_net_prefix_token) :]] = v
        net = ContinuousSelfCollisionPredictor(
            n_dimension=weights["state_dict"]["net.layers.0.weight"].shape[1]
        )
        net.load_state_dict(_inner_weights)

        return net


if __name__ == "__main__":
    project_root = get_project_root()

    robot = PandaRobot()
    train(
        project_root / "robodata" / "panda_self_collision_dataset.npy",
        n_dimension=robot.dof,
        save_path=robot.self_collision_model_weight_path,
    )
