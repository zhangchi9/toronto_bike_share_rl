"""Round 10, part 2: DQN retrained on the need_near/cap5 station rule, 3 seeds.

    python -m rl.eval_round10_retrain [--out results/round10/retrain]

Zones flow_w0.3_k20_need, all four feature groups, test Aug 1 -> Sep 20. Runs:
  - champion (round 9 models) under today's rule (most/cap5, built in);
  - champion, unchanged, under need_near/cap5 (round 10 part 1);
  - DQN retrained under need_near/cap5:
    models/round10/dqn_1trucks_need_near-cap5_s<seed>/.
Writes rl_test_results.{md,csv}, retrain.json and rl_test_daily.png to --out.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rl.eval_round6 import run, validation
from rl.evaluate import run_no_trucks
from rl.wrapper import TEST_PERIOD

ZONES = "flow_w0.3_k20_need"
FEATURES = "risk+travel+failures+memory"
RULE = "need_near/cap5"
SEEDS = (0, 1, 2)
EARLY_DAYS = 14
SETUPS = (  # (label, model folder pattern, stop rule)
    ("champion, today's rule",
     os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{{seed}}"), None),
    (f"champion, {RULE}",
     os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{{seed}}"), RULE),
    (f"retrained, {RULE}",
     os.path.join("models", "round10", "dqn_1trucks_need_near-cap5_s{seed}"), RULE),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round10", "retrain"))
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    rows = []
    for label, pattern, rule in SETUPS:
        for seed in SEEDS:
            folder = pattern.format(seed=seed)
            path = os.path.join(folder, "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {label} seed {seed}: {path} not found")
                continue
            m = DQN.load(path, device="cpu")
            r = run(1, ZONES, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]),
                    features=FEATURES, stop_rule=rule)
            step, val = validation(folder, 1)
            diff = r["daily"] - nt_daily
            gain = float(diff.sum())
            rows.append({"setup": label, "seed": seed, "total_reward": r["total_reward"],
                         "vs_no_trucks": gain, "test_per_week": gain / n_days * 7,
                         "first_14_days": float(diff[:EARLY_DAYS].sum()),
                         "rest": float(diff[EARLY_DAYS:].sum()),
                         "days_ahead": int((diff > 0).sum()),
                         "val_per_week_trained": val, "val_best_step": step,
                         **{c: r[c] for c in ("fulfilled_pct", "lost_requests",
                                              "blocked_returns", "truck_cost", "bikes_moved",
                                              "truck_km", "waits")},
                         "decisions": sum(o[3] for o in r["orders"]),
                         "n_routes": len(r["orders"]), "top_orders": r["orders"][:6],
                         "daily_vs_no_trucks": diff.tolist()})
            print(f"  {label:32s} seed {seed}: {gain:+,.0f} "
                  f"(val {val:,.0f}/week at {step:,})", flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "features": FEATURES, "rule": RULE, "rows": rows},
              open(os.path.join(out, "retrain.json"), "w"), default=float)

    b = df[df.setup == SETUPS[0][0]].set_index("seed").vs_no_trucks
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 10 part 2: DQN retrained on {RULE}, zones {ZONES}, features "
                f"{FEATURES}, test {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\n"
                f"No trucks: ${nt_total:,.0f}.\n\n")
        cols = ["setup", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
                "days_ahead", "val_per_week_trained", "val_best_step", "fulfilled_pct",
                "lost_requests", "blocked_returns", "truck_cost", "bikes_moved", "truck_km",
                "decisions", "n_routes", "waits"]
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
        f.write(f"\n## Per setup: mean of seeds (min to max); Welch t-test vs "
                f"\"{SETUPS[0][0]}\"\n\n| setup | vs no trucks | diff | p | validation $/week "
                "(own rule) | first 14 days | rest | truck km | bikes moved |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for label, g in df.groupby("setup", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            x = g.vs_no_trucks
            p = "" if label == SETUPS[0][0] or len(x) < 2 else \
                f"{stats.ttest_ind(x, b, equal_var=False).pvalue:.2f}"
            f.write(f"| {label} | {s('vs_no_trucks')} | {x.mean() - b.mean():+,.0f} | {p} | "
                    f"{s('val_per_week_trained')} | {s('first_14_days')} | {s('rest')} | "
                    f"{g.truck_km.mean():,.0f} | {g.bikes_moved.mean():,.0f} |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = dict(zip([s[0] for s in SETUPS], plt.get_cmap("tab10").colors))
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["setup"]],
                ls=("-", "--", ":")[r["seed"]], label=f"{r['setup']}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title(f"Round 10 test: DQN retrained on {RULE} vs the champion")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, retrain.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
