"""Round 12: divide and conquer, 2 trucks on 2 territories of 20 zones each.

    python -m rl.eval_round12 [--out results/round12] [--workers 12]

Zones flow_w0.3_k40_need (40 mainland zones + island) split by rl.territory into
territory A (west) and B (east). Each territory has its own DQN, trained alone with
1 truck that only orders moves between its 20 zones (all four feature groups,
today's station rule, 1M decisions, best checkpoint on the validation weeks):
    models/round12/dqn_1trucks_territory{A,B}_s<seed>
On the test (Aug 1 -> Sep 20, one continuous run) this runs:
  champion   the round 9 DQN, 1 truck, 20 zones (models/round9/...), 3 seeds
  pair       2 trucks: truck 0 driven by territory A's DQN, truck 1 by B's (seed s with s)
  A alone, B alone   1 truck on one territory, the other territory untouched
and, for reference only, each pair on the validation weeks (Jul 18-31).
Writes rl_test_results.{md,csv}, round12.json and rl_test_daily.png to --out.
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

CHAMP_ZONES = "flow_w0.3_k20_need"
SCHEME = "flow_w0.3_k40_need"
FEATURES = "risk+travel+failures+memory"
SEEDS = (0, 1, 2)
EARLY_DAYS = 14
LABEL = {"champion": "DQN champion (1 truck, 20 zones)", "pair": "Territory pair (2 trucks)",
         "A": "Territory A alone (1 truck)", "B": "Territory B alone (1 truck)"}


def folder(kind: str, seed: int) -> str:
    if kind == "champion":
        return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}")
    return os.path.join("models", "round12", f"dqn_1trucks_territory{kind}_s{seed}")


def _load(kind, seed):
    from stable_baselines3 import DQN
    return DQN.load(os.path.join(folder(kind, seed), "best_model.zip"), device="cpu")


def run(models: list, territories, scheme: str, window) -> dict:
    """One continuous run; truck k is driven by models[k] (eval_round6.run's numbers,
    plus orders and waits per truck)."""
    from rl.wrapper import RebalanceEnv
    env = RebalanceEnv(number_trucks=len(models), windows=[window], record_minutes=True,
                       zone_scheme=scheme, features=FEATURES, territories=territories)
    obs, _ = env.reset(seed=0)
    zof = env.zones.zone_of_station
    orders = [Counter() for _ in models]
    waits = [0] * len(models)
    done = False
    while not done:
        k = env.queue[0]
        a = int(models[k].predict(obs, deterministic=True)[0])
        row = env.fb.decode(a)
        if row.any():
            orders[k][(int(zof[row[0] - 1]), int(zof[row[1]]), int(row[2]))] += 1
        else:
            waits[k] += 1
        obs, _, done, _, _ = env.step(a)
    T = env.env.totals
    return {"total_reward": float(np.sum(env.minute_rewards)),
            "daily": np.asarray(env.minute_rewards).reshape(-1, 1440).sum(1),
            "fulfilled_pct": 100 * T["trips_started"] / (T["trips_started"] + T["lost_trips"]),
            "lost_requests": int(T["lost_trips"]), "blocked_returns": int(T["overflow_returns"]),
            "truck_cost": T["labor_cost"] + T["fuel_cost"] + T["fixed_cost"],
            "bikes_moved": int(T["bikes_moved"]), "truck_km": float(T["truck_km"]),
            "waits": waits, "orders": [[[*key, n] for key, n in o.most_common()] for o in orders]}


def job(args) -> dict:
    kind, seed, period = args
    import torch
    torch.set_num_threads(1)
    from rl.territory import load_territories
    from rl.wrapper import TEST_PERIOD, VAL_WINDOWS
    A, B = load_territories(SCHEME)
    if kind == "champion":
        models, terr, scheme = [_load("champion", seed)], None, CHAMP_ZONES
    elif kind == "pair":
        models, terr, scheme = [_load("A", seed), _load("B", seed)], [A, B], SCHEME
    else:
        models, terr, scheme = [_load(kind, seed)], [{"A": A, "B": B}[kind]], SCHEME
    windows = [TEST_PERIOD] if period == "test" else VAL_WINDOWS
    rs = [run(models, terr, scheme, w) for w in windows]
    return {"kind": kind, "seed": seed, "period": period, "runs": rs}


def val_from_npz(kind, seed):
    from rl.eval_round6 import validation
    return validation(folder(kind, seed), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round12"))
    ap.add_argument("--workers", type=int, default=12)
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    workers = ap.parse_args().workers
    from scipy import stats

    from rl.evaluate import run_no_trucks
    from rl.wrapper import TEST_PERIOD, VAL_WINDOWS, no_truck_rewards
    from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    val_base = [float(no_truck_rewards(w, DEFAULT_INIT_OCCUPANCY).sum()) for w in VAL_WINDOWS]

    jobs = [(k, s, "test") for k in ("champion", "pair", "A", "B") for s in SEEDS]
    jobs += [("pair", s, "val") for s in SEEDS]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(job, jobs):
            res[(r["kind"], r["seed"], r["period"])] = r["runs"]
            print(f"  done {r['kind']} seed {r['seed']} {r['period']}", flush=True)

    rows = []
    for kind in ("champion", "pair", "A", "B"):
        for s in SEEDS:
            r = res[(kind, s, "test")][0]
            diff = r["daily"] - nt_daily
            if kind == "pair":   # $/week on the validation weeks, gain vs no trucks
                v = res[("pair", s, "val")]
                val, step = float(np.mean([x["total_reward"] - b for x, b in zip(v, val_base)])), None
                step = "A {:,} / B {:,}".format(val_from_npz("A", s)[0], val_from_npz("B", s)[0])
            else:
                st, val = val_from_npz(kind, s)
                step = f"{st:,}"
            rows.append({"kind": kind, "seed": s, "total_reward": r["total_reward"],
                         "vs_no_trucks": float(diff.sum()),
                         "test_per_week": float(diff.sum()) / n_days * 7,
                         "first_14_days": float(diff[:EARLY_DAYS].sum()),
                         "rest": float(diff[EARLY_DAYS:].sum()),
                         "days_ahead": int((diff > 0).sum()), "val_per_week": val,
                         "val_best_step": step,
                         **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                              "truck_cost", "bikes_moved", "truck_km")},
                         "decisions": sum(n for o in r["orders"] for *_, n in o),
                         "waits": " / ".join(f"{w:,}" for w in r["waits"]),
                         "n_routes": " / ".join(str(len(o)) for o in r["orders"]),
                         "top_orders": [o[:6] for o in r["orders"]],
                         "daily_vs_no_trucks": diff.tolist()})
    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    from rl.territory import load_territories
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "scheme": SCHEME,
               "territories": [t.tolist() for t in load_territories(SCHEME)],
               "features": FEATURES, "rows": rows},
              open(os.path.join(out, "round12.json"), "w"), default=float)

    champ = df[df.kind == "champion"].vs_no_trucks
    ab = (df[df.kind == "A"].vs_no_trucks.to_numpy() + df[df.kind == "B"].vs_no_trucks.to_numpy())
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 12: 2 trucks, 2 territories of 20 zones ({SCHEME}), one DQN each, "
                f"test {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\nNo trucks: ${nt_total:,.0f}. "
                "Validation $/week: champion and territories alone from their training "
                "evaluations (best checkpoint); pairs from a run of both trucks on the "
                "validation weeks (reference only, not used to pick anything).\n\n")
        cols = ["kind", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
                "days_ahead", "val_per_week", "val_best_step", "fulfilled_pct", "lost_requests",
                "blocked_returns", "truck_cost", "bikes_moved", "truck_km", "decisions",
                "n_routes", "waits"]
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
        f.write("\n## Mean of seeds (min to max); Welch t-test vs the champion\n\n"
                "| run | vs no trucks | diff | p | validation $/week | first 14 days | rest | "
                "fulfilled % | truck cost | truck km | bikes moved |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for kind, g in df.groupby("kind", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            x = g.vs_no_trucks
            p = "" if kind == "champion" else \
                f"{stats.ttest_ind(x, champ, equal_var=False).pvalue:.2f}"
            f.write(f"| {LABEL[kind]} | {s('vs_no_trucks')} | {x.mean() - champ.mean():+,.0f} | "
                    f"{p} | {s('val_per_week')} | {s('first_14_days')} | {s('rest')} | "
                    f"{g.fulfilled_pct.mean():.2f} | {g.truck_cost.mean():,.0f} | "
                    f"{g.truck_km.mean():,.0f} | {g.bikes_moved.mean():,.0f} |\n")
        pair = df[df.kind == "pair"].vs_no_trucks.to_numpy()
        f.write("\n## Do the two territories add up?\n\n| seed | A alone | B alone | A + B | pair | "
                "pair - (A + B) |\n|---|---|---|---|---|---|\n")
        for i, s_ in enumerate(SEEDS):
            a_, b_ = df[(df.kind == "A") & (df.seed == s_)].vs_no_trucks.item(), \
                df[(df.kind == "B") & (df.seed == s_)].vs_no_trucks.item()
            f.write(f"| {s_} | {a_:,.0f} | {b_:,.0f} | {ab[i]:,.0f} | {pair[i]:,.0f} | "
                    f"{pair[i] - ab[i]:+,.0f} |\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = {"champion": "tab:blue", "pair": "tab:red", "A": "tab:green", "B": "tab:orange"}
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["kind"]],
                ls=("-", "--", ":")[r["seed"]], lw=1.2, label=f"{LABEL[r['kind']]}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 12 test: territory DQNs (2 trucks) vs the 1-truck champion")
    ax.legend(fontsize=6.5, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round12.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
