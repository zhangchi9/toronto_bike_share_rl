"""Round 10: station rules inside a zone, tested with the round 9 champion unchanged.

    python -m rl.eval_round10 [--out results/round10]

Zones flow_w0.3_k20_need. The champion (DQN, 1 truck, all four feature groups,
models/round9/dqn_1trucks_risk+travel+failures+memory_s<seed>/, 3 seeds) runs on
the Aug 1 -> Sep 20 test period under the default station rule (most/cap5, the
env's built-in round 3 rule) and under each finalist in results/round10/
finalists.json (picked by rl.stop_search without test data). The greedy
dispatcher runs under the same rules as a no-learning check. Writes
rl_test_results.{md,csv}, round10.json and rl_test_daily.png to --out.
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
from rl.eval_round8 import greedy_chooser
from rl.evaluate import run_no_trucks
from rl.stop_rules import DEFAULT_RULE
from rl.wrapper import TEST_PERIOD

ZONES = "flow_w0.3_k20_need"
FEATURES = "risk+travel+failures+memory"
SEEDS = (0, 1, 2)
EARLY_DAYS = 14


def champion_folder(seed: int) -> str:
    return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round10")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN
    finalists = json.load(open(os.path.join(out, "finalists.json")))["finalists"]
    rules = [DEFAULT_RULE] + finalists

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    rows = []

    def add(rule, policy, seed, r, extra=None):
        diff = r["daily"] - nt_daily
        gain = float(diff.sum())
        rows.append({"rule": rule, "policy": policy, "seed": seed,
                     "total_reward": r["total_reward"], "vs_no_trucks": gain,
                     "test_per_week": gain / n_days * 7,
                     "first_14_days": float(diff[:EARLY_DAYS].sum()),
                     "rest": float(diff[EARLY_DAYS:].sum()),
                     "days_ahead": int((diff > 0).sum()),
                     **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                          "truck_cost", "bikes_moved", "truck_km", "waits")},
                     "decisions": sum(o[3] for o in r["orders"]),
                     "n_routes": len(r["orders"]), **(extra or {}),
                     "top_orders": r["orders"][:6], "daily_vs_no_trucks": diff.tolist()})
        print(f"  {rule:18s} {policy:7s} seed {seed}: {gain:+,.0f}", flush=True)

    models = {s: DQN.load(os.path.join(champion_folder(s), "best_model.zip"), device="cpu")
              for s in SEEDS}
    for rule in rules:
        sr = None if rule == DEFAULT_RULE else rule        # default = env's built-in rule
        for seed in SEEDS:
            m = models[seed]
            r = run(1, ZONES, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]),
                    features=FEATURES, stop_rule=sr)
            step, val = validation(champion_folder(seed), 1)
            add(rule, "DQN", seed, r, {"val_per_week_trained": val, "val_best_step": step})
        add(rule, "Greedy", None, run(1, ZONES, greedy_chooser(ZONES), stop_rule=sr))

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "features": FEATURES, "rules": rules, "rows": rows},
              open(os.path.join(out, "round10.json"), "w"), default=float)

    dq = df[df.policy == "DQN"]
    b = dq[dq.rule == DEFAULT_RULE].vs_no_trucks
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 10: station rules, champion DQN (1 truck, {FEATURES}) unchanged, "
                f"zones {ZONES}, test {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\n"
                f"{DEFAULT_RULE} = the built-in round 3 rule. No trucks: ${nt_total:,.0f}.\n\n")
        cols = ["rule", "policy", "seed", "vs_no_trucks", "test_per_week", "first_14_days",
                "rest", "days_ahead", "fulfilled_pct", "lost_requests", "blocked_returns",
                "truck_cost", "bikes_moved", "truck_km", "decisions", "n_routes", "waits"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.2f}" if c == "fulfilled_pct" else f"{v:,.0f}"
                elif isinstance(v, int) and c != "seed":
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## DQN per rule: mean of seeds (min to max), Welch t-test vs "
                f"{DEFAULT_RULE}\n\n| rule | vs no trucks | diff | p | first 14 days | rest | "
                "bikes moved | truck km |\n|---|---|---|---|---|---|---|---|\n")
        for rule, g in dq.groupby("rule", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            x = g.vs_no_trucks
            p = "" if rule == DEFAULT_RULE else \
                f"{stats.ttest_ind(x, b, equal_var=False).pvalue:.2f}"
            f.write(f"| {rule} | {s('vs_no_trucks')} | {x.mean() - b.mean():+,.0f} | {p} | "
                    f"{s('first_14_days')} | {s('rest')} | {g.bikes_moved.mean():,.0f} | "
                    f"{g.truck_km.mean():,.0f} |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = dict(zip(rules, plt.get_cmap("tab10").colors))
    for r in rows:
        if r["policy"] != "DQN":
            continue
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["rule"]],
                ls=("-", "--", ":")[r["seed"]], label=f"{r['rule']}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 10 test: champion DQN (1 truck) under station rules, 3 seeds each")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round10.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
