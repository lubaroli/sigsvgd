from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def compile_results(path):
    table = {
        ("svmp", "best"): {},
        ("svmp", "nll"): {},
        ("sgd", "best"): {},
        ("sgd", "nll"): {},
        ("pathsig", "best"): {},
        ("pathsig", "nll"): {},
    }
    for seed_folder in tqdm(path.iterdir()):
        if seed_folder.is_dir():
            for exp_folder in seed_folder.iterdir():
                exp = exp_folder.name
                for method in ["svmp", "sgd", "pathsig"]:
                    if exp_folder.joinpath(f"{method}_data.pt").exists():
                        with exp_folder.joinpath(f"{method}_data.pt") as f:
                            data = torch.load(f, map_location=torch.device("cpu"))
                            last_key = list(data.keys())[-2]
                            ep_cost = data[last_key]["loss"]
                            for result in ["best", "nll"]:
                                k = (method, result)
                                if k not in table.keys():
                                    table[k] = {}
                                if exp not in table[k].keys():
                                    table[k][exp] = torch.tensor([])
                            table[(method, "best")][exp] = torch.cat(
                                [
                                    table[(method, "best")][exp],
                                    ep_cost.min(dim=-1)[0].unsqueeze(0),
                                ]
                            )
                            table[(method, "nll")][exp] = torch.cat(
                                [
                                    table[(method, "nll")][exp],
                                    ep_cost.mean(dim=-1).unsqueeze(0),
                                ]
                            )

    for k1 in table.keys():
        for k2 in table[k1].keys():
            val = table[k1][k2]
            table[k1][k2] = (val.mean(), val.std())
    return pd.DataFrame(table)


if __name__ == "__main__":
    path = Path("data/local/path-plan-20221216-100623")
    results = compile_results(path)
    results.to_markdown(buf=path / "results.md")
    results.style.to_latex(buf=path / "results.tex")
