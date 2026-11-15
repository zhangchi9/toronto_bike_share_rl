"""Round 9: DQN (1 truck) on the round 8 zones with extra input features, 3 seeds each.

    python -m rl.eval_round9 [--out results/round9]

Tests on the Aug 1 -> Sep 20 test period (one continuous run, as rl.evaluate), zones
flow_w0.3_k20_need throughout:
  - baseline (round 8 inputs): models/round8/dqn_1trucks_flow_w0.3_k20_need_s<seed>/;
  - each feature set in FEATURE_SETS (rl.features groups, built up one at a time):
    models/round9/dqn_1trucks_<set>_s<seed>/.
Writes rl_test_results.{md,csv}, round9.json and rl_test_daily.png to --out.
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

ZONES = "flow_w0.3_k20_need"
FEATURE_SETS = ("base", "risk", "risk+travel", "risk+travel+failures",
                "risk+travel+failures+memory")
SEEDS = (0, 1, 2)
EARLY_DAYS = 14


def model_folder(features: str, seed: int) -> str:
    if features == "base":
        return os.path.join("models", "round8", f"dqn_1trucks_{ZONES}_s{seed}")
    return os.path.join("models", "round9", f"dqn_1trucks_{features}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round9")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    rows = []
    for feats in FEATURE_SETS:
        for seed in SEEDS:
            folder = model_folder(feats, seed)
            path = os.path.join(folder, "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {feats} seed {seed}: {path} not found")
                continue
            m = DQN.load(path, device="cpu")
            r = run(1, ZONES, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]),
                    features=None if feats == "base" else feats)
            step, val = validation(folder, 1)
            diff = r["daily"] - nt_daily
            gain = float(diff.sum())
            rows.append({"features": feats, "seed": seed, "total_reward": r["total_reward"],
                         "vs_no_trucks": gain, "test_per_week": gain / n_days * 7,
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
            print(f"  DQN 1 truck {feats:28s} seed {seed}: {gain:+,.0f} "
                  f"(val {val:,.0f}/week at {step:,})", flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "rows": rows}, open(os.path.join(out, "round9.json"), "w"), default=float)

    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 9: DQN (1 truck), extra input features, zones {ZONES}, test "
                f"{TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\nbase = the round 8 models. "
                f"No trucks: ${nt_total:,.0f}.\n\n")
        cols = ["features", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
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
        f.write("\n## Per feature set: mean of seeds (min to max)\n\n"
                "| features | seeds | vs no trucks | validation $/week | first 14 days | rest |\n"
                "|---|---|---|---|---|---|\n")
        for feats, g in df.groupby("features", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            f.write(f"| {feats} | {len(g)} | {s('vs_no_trucks')} | {s('val_per_week')} | "
                    f"{s('first_14_days')} | {s('rest')} |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = dict(zip(FEATURE_SETS, plt.get_cmap("tab10").colors))
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["features"]],
                ls=("-", "--", ":")[r["seed"]], label=f"{r['features']}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 9 test: DQN (1 truck), extra features, 3 training seeds each")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round9.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
