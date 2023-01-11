from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from stein_mpc.models.robot import robot_scene
from stein_mpc.models.robot.robot_simulator import PandaRobot
from stein_mpc.models.robot_learning import (
    continuous_occupancy_map,
    continuous_self_collision_pred,
)


def create_body_points(x, n_pts=10):
    inter_pts = torch.arange(0, 1, 1 / n_pts)
    device = x.device
    body_points = x[:-1, None] + inter_pts[:, None, None].to(device) * x[1:, None]
    return body_points.flatten(0, 1)


def check_success(robot, occmap, sc_pred, qs, threshold=0.2):
    _original_shape = qs.shape
    # xs shape is [n_dof, batch * timesteps, 3]
    xs = robot.qs_to_joints_xs(qs.reshape(-1, _original_shape[-1]))
    ee_xs = xs[-1, ...]
    ee_xs = ee_xs.reshape(_original_shape[0], -1)
    # compute piece-wise linear distance
    ee_dist = torch.linalg.norm(ee_xs[1:, :] - ee_xs[:-1, :], dim=-1).sum()
    # collision prob is in the shape of [n_dof * (n_pts - 1) x batch * timesteps x 1]
    n_pts = 10
    body_points = create_body_points(xs, n_pts)
    collision_prob = occmap(body_points).squeeze(-1).max()

    # worst case collision prob
    collision_prob = collision_prob
    self_collision_prob = sc_pred(qs).squeeze(-1).max()
    if collision_prob <= threshold and self_collision_prob <= threshold:
        return True, ee_dist
    else:
        return False, ee_dist


def compile_results(path):
    res = {}
    device = "cpu"
    robot = PandaRobot(device=device)
    problem = path.name.split("-")[1]
    scene = robot_scene.RobotScene(robot, problem)
    try:
        occmap = continuous_occupancy_map.load_trained_model(scene.weight_path)
        occmap.to(device)
        sc_pred = continuous_self_collision_pred.load_trained_model(
            robot.self_collision_model_weight_path
        )
        sc_pred.to(device)
    except FileNotFoundError as e:
        print(
            f"\nERROR: File not found at {scene.weight_path}."
            f"\nHave you downloaded the weight file via running 'Make'?\n"
        )
        raise e
    for method in tqdm(["svgd", "pathsig", "sgd"]):
        res[method] = {}
        for folder in path.iterdir():
            if folder.joinpath(f"{method}/data.pt").exists():
                req = str(folder.name).split("-")[0]
                if req not in res[method].keys():
                    res[method][req] = {
                        "costs": torch.tensor([]),
                        "eps": 0,
                        "successes": 0,
                        "min_len": torch.tensor([]),
                    }
                with folder.joinpath(f"{method}/data.pt") as f:
                    data = torch.load(f, map_location=torch.device("cpu"))
                    last_key = list(data.keys())[-2]
                    # we missed the squeeze on self collision costs when summing on robot-final experiment
                    # so we need to recompute the loss
                    ep_cost = (
                        1 * data[last_key]["costs_col"]
                        + 10 * data[last_key]["costs_self_col"].squeeze()
                        + 2.5 * data[last_key]["costs_dist"]
                    )
                    best_qs = data[last_key]["trajectories"][ep_cost.argmin()]
                    res[method][req]["costs"] = torch.cat(
                        [res[method][req]["costs"], ep_cost.unsqueeze(0)]
                    )
                    res[method][req]["eps"] += 1
                    succ, min_len = check_success(robot, occmap, sc_pred, best_qs)
                    res[method][req]["successes"] += succ
                    res[method][req]["min_len"] = torch.cat(
                        [res[method][req]["min_len"], min_len.unsqueeze(0)]
                    )
    table = {}
    for m in res.keys():
        for v in ["Best", "Length", "NLL"]:
            table[(m, v)] = {}
            table[(m, v)][problem] = torch.tensor([])
        for r in res[m].keys():
            table[(m, "Best")][problem] = torch.cat(
                [
                    table[(m, "Best")][problem],
                    res[m][r]["costs"].min(-1)[0].unsqueeze(0),
                ]
            )
            table[(m, "Length")][problem] = torch.cat(
                [table[(m, "Length")][problem], res[m][r]["min_len"].unsqueeze(0)]
            )
            table[(m, "NLL")][problem] = torch.cat(
                [table[(m, "NLL")][problem], res[m][r]["costs"].sum(-1).unsqueeze(0)]
            )
    for k1 in table.keys():
        for k2 in table[k1].keys():
            val = table[k1][k2]
            table[k1][k2] = (
                val.mean().numpy().round(2),
                val.std(-1).mean().numpy().round(2),
            )
    return pd.DataFrame(table)


if __name__ == "__main__":
    path = Path("data/robot-final")
    for folder in path.iterdir():
        results = compile_results(folder)
        results.to_markdown(buf=folder / "results.md")
        results.style.to_latex(buf=folder / "results.tex")
