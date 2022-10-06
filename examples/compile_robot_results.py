from pathlib import Path

import pandas as pd
import torch


def compile_results(path):
    svgd_costs = torch.tensor([])
    svgd_episodes = 0
    ps_costs = torch.tensor([])
    ps_episodes = 0
    sgd_costs = torch.tensor([])
    sgd_episodes = 0

    for folder in path.iterdir():
        if folder.joinpath("svgd/data.pt").exists():
            with folder.joinpath("svgd/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                svgd_costs = torch.cat([svgd_costs, ep_cost.unsqueeze(0)])
                svgd_episodes = svgd_episodes + 1

        if folder.joinpath("pathsig/data.pt").exists():
            with folder.joinpath("pathsig/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                ps_costs = torch.cat([ps_costs, ep_cost.unsqueeze(0)])
                ps_episodes = ps_episodes + 1

        if folder.joinpath("sgd/data.pt").exists():
            with folder.joinpath("sgd/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                sgd_costs = torch.cat([sgd_costs, ep_cost.unsqueeze(0)])
                sgd_episodes = sgd_episodes + 1

    return {
        "SVGD": {
            "Cost - Min Avg": svgd_costs.min().mean().numpy().round(2),
            "Cost - Avg": svgd_costs.mean().numpy().round(2),
            "Cost - Std": svgd_costs.std().numpy().round(2),
            "Episodes": svgd_episodes,
        },
        "Path Signature": {
            "Cost - Min Avg": ps_costs.min().mean().numpy().round(2),
            "Cost - Avg": ps_costs.mean().numpy().round(2),
            "Cost - Std": ps_costs.std().numpy().round(2),
            "Episodes": ps_episodes,
        },
        "SGD": {
            "Cost - Min Avg": sgd_costs.min().mean().numpy().round(2),
            "Cost - Avg": sgd_costs.mean().numpy().round(2),
            "Cost - Std": sgd_costs.std().numpy().round(2),
            "Episodes": sgd_episodes,
        },
    }


if __name__ == "__main__":
    path = Path("data/local/robot-table_bars-20220929-223010")
    results = compile_results(path)
    df = pd.DataFrame(data=results)
    df.to_markdown(buf=path / "results.md")
