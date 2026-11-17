"""Round 13: tune the lookahead planner (rl/planner.py) on the validation weeks only.

    python -m rl.plan_search [--workers 5] [--out results/round13]

Grid: horizon x value weight, K and S fixed, on the 3 round 9 champion seeds and
both validation weeks. Writes planner_val.{csv,md}; the best setting (mean $/week
over seeds and weeks) is the one tested in rl.eval_round13.
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

FEATURES = "risk+travel+failures+memory"
SEEDS = (0, 1, 2)
HORIZONS = (180, 360, 720, 1440)
VALUE_WEIGHTS = (0.0, 1.0)
K, S = 8, 6


def champion_path(seed: int) -> str:
    return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}", "best_model.zip")


def job(args) -> dict:
    horizon, vw, seed, w = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN
    from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
    from rl.planner import run
    from rl.wrapper import VAL_WINDOWS, no_truck_rewards
    window = VAL_WINDOWS[w]
    m = DQN.load(champion_path(seed), device="cpu")
    r = run(m, window, k=K, s=S, horizon=horizon, value_weight=vw, seed=seed)
    gain = r["total_reward"] - float(no_truck_rewards(window, DEFAULT_INIT_OCCUPANCY).sum())
    return {"horizon": horizon, "value_weight": vw, "seed": seed, "week": w,
            "gain": gain, "changed": r["changed"], "decisions": r["decisions"],
            "bikes_moved": r["bikes_moved"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default=os.path.join("results", "round13"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = list(itertools.product(HORIZONS, VALUE_WEIGHTS, SEEDS, (0, 1)))
    jobs.sort(key=lambda j: -j[0])            # long horizons first
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "planner_val.csv"), index=False)
    summarize(a.out)


def summarize(out: str) -> None:
    from rl.inventory_search import md_table
    df = pd.read_csv(os.path.join(out, "planner_val.csv"))
    per_seed = df.groupby(["horizon", "value_weight", "seed"])["gain"].mean().unstack("seed")
    per_seed["mean"] = per_seed.mean(1)
    per_seed["changed_pct"] = (df.groupby(["horizon", "value_weight"])["changed"].sum()
                               / df.groupby(["horizon", "value_weight"])["decisions"].sum() * 100)
    best = per_seed["mean"].idxmax()
    with open(os.path.join(out, "planner_val.md"), "w", encoding="utf-8") as f:
        f.write(f"# Planner tuning on the validation weeks (K={K} orders, S={S} scenarios)\n\n"
                "Mean $/week vs no trucks over both validation weeks, per champion seed.\n\n")
        f.write(md_table(per_seed) + "\n\n")
        f.write(f"Best: horizon {best[0]} min, value weight {best[1]}\n")
    json.dump({"horizon": int(best[0]), "value_weight": float(best[1]), "k": K, "s": S},
              open(os.path.join(out, "planner_best.json"), "w"))
    print(per_seed.round(1))


if __name__ == "__main__":
    main()
