"""Round 7: is round 4's DQN (1 truck) result luck? Three training seeds per zone set.

    python -m rl.eval_round7 [--out results/round7]

Tests DQN with 1 truck on the Aug 1 -> Sep 20 test period (one continuous run, as
rl.evaluate), trained with seeds 0, 1, 2 on each zone set:
  - k-means zones (round 4 setup): seed 0 = models/round4/dqn_1trucks/;
  - flow zones w = 0.3 (round 5 setup): seed 0 = models/dqn_1trucks/;
  - seeds 1 and 2: models/round7/dqn_1trucks_<zones>_s<seed>/.
Writes rl_test_results.{md,csv}, round7.json (daily rewards, top orders) and
rl_test_daily.png to --out.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.eval_round6 import run, validation
from rl.evaluate import run_no_trucks
from rl.wrapper import TEST_PERIOD

ZONE_SETS = (("kmeans", "k-means (round 4)"), ("flow_w0.3", "flow w = 0.3 (round 5)"))
SEEDS = (0, 1, 2)
EARLY_DAYS = 14          # validation-sized stretch at the start of the test


def model_folder(zones: str, seed: int) -> str:
    if seed == 0:
        return os.path.join("models", "round4", "dqn_1trucks") if zones == "kmeans" \
            else os.path.join("models", "dqn_1trucks")
    return os.path.join("models", "round7", f"dqn_1trucks_{zones}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round7")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    rows = []
    for zones, label in ZONE_SETS:
        for seed in SEEDS:
            folder = model_folder(zones, seed)
            path = os.path.join(folder, "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {zones} seed {seed}: {path} not found")
                continue
            m = DQN.load(path, device="cpu")
            r = run(1, zones, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]))
            step, val = validation(folder, 1)
            diff = r["daily"] - nt_daily
            gain = float(diff.sum())
            rows.append({"zones": zones, "zones_label": label, "seed": seed,
                         "total_reward": r["total_reward"], "vs_no_trucks": gain,
                         "test_per_week": gain / n_days * 7,
                         "first_14_days": float(diff[:EARLY_DAYS].sum()),
                         "rest": float(diff[EARLY_DAYS:].sum()),
                         "days_ahead": int((diff > 0).sum()),
                         "val_per_week": val, "val_best_step": step,
                         **{c: r[c] for c in ("fulfilled_pct", "lost_requests",
                                              "blocked_returns", "truck_cost", "bikes_moved",
                                              "truck_km", "waits")},
                         "n_orders": sum(o[3] for o in r["orders"]),
                         "n_routes": len(r["orders"]), "top_orders": r["orders"][:6],
                         "daily_vs_no_trucks": diff.tolist()})
            print(f"  DQN 1 truck {zones} seed {seed}: {gain:+,.0f} "
                  f"(val {val:,.0f}/week at {step:,})", flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "rows": rows},
              open(os.path.join(out, "round7.json"), "w"), default=float)

    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 7: DQN (1 truck), 3 seeds per zone set, test {TEST_PERIOD[0]} to "
                f"{TEST_PERIOD[1]}\n\nSeed 0 rows are the round 4 (k-means) and round 5 "
                f"(w = 0.3) models. No trucks: ${nt_total:,.0f}.\n\n")
        cols = ["zones", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
                "days_ahead", "val_per_week", "val_best_step", "fulfilled_pct",
                "lost_requests", "blocked_returns", "truck_cost", "bikes_moved", "truck_km",
                "n_orders", "n_routes", "waits"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.2f}" if c == "fulfilled_pct" else f"{v:,.0f}"
                elif isinstance(v, int) and c != "seed":
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## Per zone set: mean of seeds (min to max)\n\n"
                "| zones | seeds | vs no trucks | first 14 days | rest |\n|---|---|---|---|---|\n")
        for zones, g in df.groupby("zones", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            f.write(f"| {zones} | {len(g)} | {s('vs_no_trucks')} | {s('first_14_days')} | "
                    f"{s('rest')} |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = {"kmeans": "tab:blue", "flow_w0.3": "tab:orange"}
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["zones"]],
                ls=("-", "--", ":")[r["seed"]], label=f"{r['zones_label']}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 7 test: DQN (1 truck), 3 training seeds per zone set")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round7.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
