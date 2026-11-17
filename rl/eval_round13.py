"""Round 13: two new ways to beat the 1-truck champion, on the test (Aug 1 -> Sep 20).

    python -m rl.eval_round13 [--out results/round13] [--workers 4]

  champion    the round 9 DQN (models/round9/...), seeds 0-2
  planner     idea 1, rl/planner.py on each champion seed; settings from
              planner_best.json (picked on the validation weeks by rl.plan_search)
  inventory   idea 2, rl/inventory.py; settings from inventory_best.json (picked on
              the validation weeks by rl.inventory_search). Deterministic: one run.
(Idea 3, an ensemble of DQNs, was dropped before testing.)

Each test run is saved to <out>/runs/<kind>_s<seed>.json and loaded from there when
present, so the planner's runs (hours each) are made once.
Writes rl_test_results.{md,csv}, round13.json and rl_test_daily.png to --out.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ZONES = "flow_w0.3_k20_need"
FEATURES = "risk+travel+failures+memory"
EARLY_DAYS = 14
LABEL = {"champion": "DQN champion (round 9)", "planner": "Idea 1: lookahead planner on the champion",
         "inventory": "Idea 2: inventory-model dispatcher"}
KINDS = ("champion", "planner", "inventory")
SEEDS = {"champion": (0, 1, 2), "planner": (0, 1, 2), "inventory": (0,)}


def folder(seed: int) -> str:
    return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}")


def _load(seed: int):
    from stable_baselines3 import DQN
    return DQN.load(os.path.join(folder(seed), "best_model.zip"), device="cpu")


def _best(out: str, name: str) -> dict:
    return json.load(open(os.path.join(out, name)))


def run_path(out: str, kind: str, seed: int) -> str:
    return os.path.join(out, "runs", f"{kind}_s{seed}.json")


def job(args) -> dict:
    kind, seed, out = args
    path = run_path(out, kind, seed)
    if os.path.exists(path):
        return json.load(open(path))
    import torch
    torch.set_num_threads(1)
    from rl.wrapper import TEST_PERIOD
    if kind == "champion":
        from rl.eval_round12 import run
        r = run([_load(seed)], None, ZONES, TEST_PERIOD)
    elif kind == "planner":
        from rl.planner import run
        b = _best(out, "planner_best.json")
        r = run(_load(seed), TEST_PERIOD, k=b["k"], s=b["s"], horizon=b["horizon"],
                value_weight=b["value_weight"], seed=seed)
    else:
        from rl.inventory import run
        b = _best(out, "inventory_best.json")
        r = run(TEST_PERIOD, horizon=b["horizon"], theta=b["theta"])
    r = {"kind": kind, "seed": seed, **r, "daily": np.asarray(r["daily"]).tolist()}
    if kind != "champion":                     # champion runs are quick: always re-run
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(r, open(path, "w"), default=float)
    return r


def validation(kind, seed, out):
    """$/week on the validation weeks (Jul 18-31) and where it comes from."""
    if kind == "champion":
        from rl.eval_round6 import validation as npz_val
        st, v = npz_val(folder(seed), 1)
        return v, f"best checkpoint {st:,}"
    if kind == "planner":
        b = _best(out, "planner_best.json")
        df = pd.read_csv(os.path.join(out, "planner_val.csv"))
        g = df[(df.horizon == b["horizon"]) & (df.value_weight == b["value_weight"]) & (df.seed == seed)]
        return float(g.gain.mean()), f"H {b['horizon']} min, value weight {b['value_weight']:g}"
    b = _best(out, "inventory_best.json")
    df = pd.read_csv(os.path.join(out, "inventory_val.csv"))
    g = df[(df.horizon == b["horizon"]) & (df.theta == b["theta"])]
    return float(g.gain.mean()), f"H {b['horizon']} min, theta {b['theta']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round13"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    from scipy import stats

    from rl.evaluate import run_no_trucks
    from rl.wrapper import TEST_PERIOD
    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)

    jobs = [(k, s, out) for k in ("planner", "inventory", "champion") for s in SEEDS[k]]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(job, jobs):
            res[(r["kind"], r["seed"])] = r
            print(f"  done {r['kind']} {r['seed']}", flush=True)

    rows = []
    for kind in KINDS:
        for s in SEEDS[kind]:
            r = res[(kind, s)]
            diff = np.asarray(r["daily"]) - nt_daily
            val, how = validation(kind, s, out)
            orders = r.get("orders")
            rows.append({"kind": kind, "seed": s, "total_reward": r["total_reward"],
                         "vs_no_trucks": float(diff.sum()),
                         "test_per_week": float(diff.sum()) / n_days * 7,
                         "first_14_days": float(diff[:EARLY_DAYS].sum()),
                         "rest": float(diff[EARLY_DAYS:].sum()),
                         "days_ahead": int((diff > 0).sum()), "val_per_week": val, "val_from": how,
                         **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                              "truck_cost", "bikes_moved", "truck_km")},
                         "decisions": r.get("decisions", sum(n for o in orders for *_, n in o)
                                            + r["waits"][0] if orders else None),
                         "waits": r["waits"][0],
                         "changed_pct": (100 * r["changed"] / r["decisions"]
                                         if "changed" in r else None),
                         "daily_vs_no_trucks": diff.tolist()})
    df = pd.DataFrame(rows)
    df.drop(columns=["daily_vs_no_trucks"]).to_csv(os.path.join(out, "rl_test_results.csv"),
                                                   index=False)
    champ_d = {r["seed"]: np.asarray(r["daily_vs_no_trucks"]) for r in rows if r["kind"] == "champion"}
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "features": FEATURES, "planner": _best(out, "planner_best.json"),
               "inventory": _best(out, "inventory_best.json"), "rows": rows},
              open(os.path.join(out, "round13.json"), "w"), default=float)

    champ = df[df.kind == "champion"].vs_no_trucks
    pl = df[df.kind == "planner"].set_index("seed").vs_no_trucks
    ch = df[df.kind == "champion"].set_index("seed").vs_no_trucks
    inv = df[df.kind == "inventory"].vs_no_trucks.item()
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 13: two new ways to beat the 1-truck champion, test {TEST_PERIOD[0]} to "
                f"{TEST_PERIOD[1]}\n\nNo trucks: ${nt_total:,.0f}. All settings were picked on the "
                "validation weeks (Jul 18-31); each test run was made once.\n\n")
        cols = ["kind", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
                "days_ahead", "val_per_week", "val_from", "fulfilled_pct", "lost_requests",
                "blocked_returns", "truck_cost", "bikes_moved", "truck_km", "decisions", "waits",
                "changed_pct"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if v is None:
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.2f}" if c in ("fulfilled_pct", "changed_pct") else f"{v:,.0f}"
                elif isinstance(v, int) and c != "seed":
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## Mean of runs (min to max); Welch t-test vs the champion\n\n"
                "| run | n | vs no trucks | diff | p | validation $/week | first 14 days | rest | "
                "fulfilled % | blocked returns | truck cost | truck km | bikes moved |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for kind in KINDS:
            g = df[df.kind == kind]

            def s(c):
                v = g[c]
                return f"{v.mean():,.0f}" if len(v) == 1 else \
                    f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            x = g.vs_no_trucks
            if kind == "champion":
                p = ""
            elif len(x) > 1:
                p = f"{stats.ttest_ind(x, champ, equal_var=False).pvalue:.2f}"
            else:
                p = f"{stats.ttest_1samp(champ, x.item()).pvalue:.2f} (1 run)"
            f.write(f"| {LABEL[kind]} | {len(x)} | {s('vs_no_trucks')} | "
                    f"{x.mean() - champ.mean():+,.0f} | {p} | {s('val_per_week')} | "
                    f"{s('first_14_days')} | {s('rest')} | {g.fulfilled_pct.mean():.2f} | "
                    f"{g.blocked_returns.mean():,.0f} | {g.truck_cost.mean():,.0f} | "
                    f"{g.truck_km.mean():,.0f} | {g.bikes_moved.mean():,.0f} |\n")
        d = pl - ch
        f.write("\n## Planner vs the same champion seed (paired)\n\n| seed | champion | planner | "
                "diff | days the planner beat the champion |\n|---|---|---|---|---|\n")
        pl_d = {r["seed"]: np.asarray(r["daily_vs_no_trucks"]) for r in rows if r["kind"] == "planner"}
        for s_ in d.index:
            f.write(f"| {s_} | {ch[s_]:,.0f} | {pl[s_]:,.0f} | {d[s_]:+,.0f} | "
                    f"{int((pl_d[s_] > champ_d[s_]).sum())} of {n_days} |\n")
        f.write(f"\nMean paired diff {d.mean():+,.0f}; paired t-test p "
                f"{stats.ttest_rel(pl, ch).pvalue:.2f}\n")
        f.write(f"\nPlanner vs the inventory dispatcher: {pl.mean() - inv:+,.0f} "
                f"(one-sample t-test p {stats.ttest_1samp(pl, inv).pvalue:.2f})\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = {"champion": "tab:blue", "planner": "tab:red", "inventory": "tab:green"}
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["kind"]],
                ls=("-", "--", ":")[r["seed"]], lw=1.3,
                label=LABEL[r["kind"]] + ("" if r["kind"] == "inventory" else f", seed {r['seed']}"))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 13 test: lookahead planner and inventory dispatcher vs the champion (1 truck)")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round13.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
