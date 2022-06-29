import csv
import os

import numpy as np
import plotly.graph_objects as go
import torch

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(f"{THIS_DIR}/../robodata/001_dataset.csv") as f:
    data = np.array([list(map(float, l)) for l in csv.reader(f)])

dataset = torch.Tensor(data)
# invert the probability, into having 1 means occupied and 0 being free
dataset[:, -1] = 1 - dataset[:, -1]

############################################################
# display occupied points

pts_to_display = dataset[dataset[:, -1] > 0]

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pts_to_display[:, 0],
            y=pts_to_display[:, 1],
            z=pts_to_display[:, 2],
            mode="markers",
        )
    ],
    layout_title="occupied point in dataset",
)
fig.show()

############################################################
# load NN model and display prob

from stein_mpc.models.ros_robot import continuous_occupancy_map

net = continuous_occupancy_map.load_trained_model(
    f"{THIS_DIR}/../robodata/001_continuous-occmap-weight.ckpt"
)

fig = continuous_occupancy_map.visualise_model_pred(net, prob_threshold=0.8)

fig.show()
