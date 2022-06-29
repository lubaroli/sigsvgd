import csv

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import optim, nn, utils


class ContinuousOccupancyMap(nn.Module):
    def __init__(self, hidden_size=200, n_hidden_layers=5, n_dimension=3):
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

    def forward(self, *args):
        return self.layers(*args)


class ModelTrainer(pl.LightningModule):
    def __init__(
        self,
        class_weight=(1,),
        hidden_size=200,
        n_hidden_layers=5,
        n_dimension=3,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.net = ContinuousOccupancyMap(
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            n_dimension=n_dimension,
        )

        self.Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weight))
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


def train(data_fname="obspts.csv"):
    with open(data_fname) as f:
        data = np.array([list(map(float, l)) for l in csv.reader(f)])

    dataset = torch.Tensor(data)
    # invert the probability
    dataset[:, -1] = 1 - dataset[:, -1]

    n_total = int(dataset.shape[0])
    n_occupied = int(dataset[:, -1].sum())
    n_free = n_total - n_occupied
    print(f"free: {n_free}, occupied: {n_occupied}, ratio={n_free / n_occupied:.3f}")

    class_weight = [n_free / n_occupied]

    model = ModelTrainer(class_weight)
    trainer = pl.Trainer(
        #     limit_train_batches=100,
        #     max_epochs=10,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accelerator="gpu",
    )

    ##########################################
    train_set_size = int(len(dataset) * 0.99)
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
    return model


def load_trained_model(weight_fname):
    return ModelTrainer.load_from_checkpoint(weight_fname).net


def visualise_model_pred(
    model,
    prob_threshold=0.1,
    min_x=-2,
    max_x=2,
    min_y=-2,
    max_y=2,
    min_z=-0.7,
    max_z=1.5,
    num_steps=100,
    with_random_gaussian_noise=None,
    marker_showscale=True,
):
    x_ = np.linspace(min_x, max_x, num=num_steps)
    y_ = np.linspace(min_y, max_y, num=num_steps)
    z_ = np.linspace(min_z, max_z, num=int(num_steps))
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    coords = np.c_[x.ravel(), y.ravel(), z.ravel()]

    query_pt = torch.Tensor(coords)
    if with_random_gaussian_noise is not None:
        query_pt = query_pt + torch.randn(coords.shape) * float(
            with_random_gaussian_noise
        )

    query_pt_pred = model(query_pt)

    import plotly.graph_objects as go

    prob = query_pt_pred[:, 0].detach().cpu()

    _criteria = prob > prob_threshold
    _query_pt = query_pt[_criteria]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=_query_pt[:, 0],
                y=_query_pt[:, 1],
                z=_query_pt[:, 2],
                marker_color=prob[_criteria],
                marker_showscale=marker_showscale,
                name="prob-map",
                mode="markers",
            )
        ]
    )
    return fig
