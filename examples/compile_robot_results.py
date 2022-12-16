from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch


def compile_results(path):
    res = {}
    for folder in tqdm(path.iterdir()):
        for method in ["svgd", "pathsig", "sgd"]:
            if folder.joinpath(f"{method}/data.pt").exists():
                req = str(folder.name).split("-")[0]
                if f"{method}_" + req not in res.keys():
                    res[f"{method}_" + req] = {
                        "costs": torch.tensor([]),
                        "eps": 0,
                    }
                with folder.joinpath(f"{method}/data.pt") as f:
                    data = torch.load(f, map_location=torch.device("cpu"))
                    last_key = list(data.keys())[-2]
                    ep_cost = data[last_key]["loss"]
                    res[f"{method}_" + req]["costs"] = torch.cat(
                        [res[f"{method}_" + req]["costs"], ep_cost.unsqueeze(0)]
                    )
                    res[f"{method}_" + req]["eps"] += 1

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
