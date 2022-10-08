import argparse
import csv

import numpy as np
import plotly.graph_objects as go
import torch

from stein_mpc.models.robot import robot_scene

parser = argparse.ArgumentParser()
parser.add_argument("tagname", nargs="?", default=robot_scene.tag_names[0])
args = parser.parse_args()

scene = robot_scene.RobotScene(None, args.tagname)

with open(scene.dataset_path) as f:
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

from stein_mpc.models.robot_learning import continuous_occupancy_map

net = continuous_occupancy_map.load_trained_model(scene.weight_path)
fig = continuous_occupancy_map.visualise_model_pred(net, prob_threshold=0.95)

fig.show()
