from pathlib import Path

import pandas as pd
import torch


def compile_results(path):
    folders = [p for p in path.iterdir()]
    ps_costs = torch.tensor([])
    ps_steps = torch.tensor([])
    sv_costs = torch.tensor([])
    sv_steps = torch.tensor([])

    for folder in folders:
        ps_data = torch.load(
            folder / "pathsig/data.pkl", map_location=torch.device("cpu")
        )
        ps_costs = torch.cat([ps_costs, ps_data["costs"].sum().unsqueeze(0)], 0)
        ps_steps = torch.cat([ps_steps, torch.tensor([len(ps_data["costs"])])], 0)

        sv_data = torch.load(
            folder / "svmpc/data.pkl", map_location=torch.device("cpu")
        )
        sv_costs = torch.cat([sv_costs, sv_data["costs"].sum().unsqueeze(0)], 0)
        sv_steps = torch.cat([sv_steps, torch.tensor([len(sv_data["costs"])])], 0)

    return {
        "Path Signature": {
            "Cost - Avg": ps_costs.mean().numpy().round(2),
            "Cost - Std": ps_costs.std().numpy().round(2),
            "Steps - Avg": ps_steps.mean().numpy().round(2),
            "Steps - Std": ps_steps.std().numpy().round(2),
        },
        "SVMPC": {
            "Cost - Avg": sv_costs.mean().numpy().round(2),
            "Cost - Std": sv_costs.std().numpy().round(2),
            "Steps - Avg": sv_steps.mean().numpy().round(2),
            "Steps - Std": sv_steps.std().numpy().round(2),
        },
    }

    ro = ps_res["rollouts"]
    steps, samples, policies, _, _ = ro.shape
    hd_dist = torch.zeros(steps, samples, samples, policies, policies)

    from hausdorff import hausdorff_distance

    for k in range(steps):
        for si in range(samples):
            for sj in range(samples):
                for pi in range(policies):
                    for pj in range(policies):
                        hd_dist[k, si, sj, pi, pj] = hausdorff_distance(
                            ro[k, si, pi, 1:, :2], ro[k, sj, pj, 1:, :2]
                        )


if __name__ == "__main__":
    path = Path("data/local/maze-20220912-190818")
    results = compile_results(path)
    df = pd.DataFrame(data=results)
    print(df.to_markdown())
