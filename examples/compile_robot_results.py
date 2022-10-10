from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch


def compile_results(path):
    res = {}
    for folder in tqdm(path.iterdir()):
        if folder.joinpath("svgd/data.pt").exists():
            req = str(folder.name).split("-")[0]
            if "SVGD_" + req not in res.keys():
                res["SVGD_" + req] = {
                    "costs": torch.tensor([]),
                    "eps": 0,
                }
            with folder.joinpath("svgd/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                res["SVGD_" + req]["costs"] = torch.cat(
                    [res["SVGD_" + req]["costs"], ep_cost.unsqueeze(0)]
                )
                res["SVGD_" + req]["eps"] += 1

        if folder.joinpath("pathsig/data.pt").exists():
            req = str(folder.name).split("-")[0]
            if "PathSig_" + req not in res.keys():
                res["PathSig_" + req] = {
                    "costs": torch.tensor([]),
                    "eps": 0,
                }
            with folder.joinpath("pathsig/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                res["PathSig_" + req]["costs"] = torch.cat(
                    [res["PathSig_" + req]["costs"], ep_cost.unsqueeze(0)]
                )
                res["PathSig_" + req]["eps"] += 1

        if folder.joinpath("sgd/data.pt").exists():
            req = str(folder.name).split("-")[0]
            if "SGD_" + req not in res.keys():
                res["SGD_" + req] = {
                    "costs": torch.tensor([]),
                    "eps": 0,
                }
            with folder.joinpath("sgd/data.pt") as f:
                data = torch.load(f)
                last_key = list(data.keys())[-2]
                ep_cost = data[last_key]["loss"].cpu()
                res["SGD_" + req]["costs"] = torch.cat(
                    [res["SGD_" + req]["costs"], ep_cost.unsqueeze(0)]
                )
                res["SGD_" + req]["eps"] += 1

    table = {}
    for k in res.keys():
        table[k] = {}
        table[k]["Min Avg Cost"] = res[k]["costs"].min().mean().numpy().round(2)
        table[k]["Cost Avg"] = res[k]["costs"].mean().numpy().round(2)
        table[k]["Cost Std"] = res[k]["costs"].std().numpy().round(2)
        table[k]["Episodes"] = res[k]["eps"]
    return table


if __name__ == "__main__":
    path = Path("data/local/robot-table_under_pick_panda-20221009-155150")
    results = compile_results(path)
    df = pd.DataFrame(data=results)
    df.to_markdown(buf=path / "results.md")
