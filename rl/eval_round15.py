"""Round 15: a coordinated, whole-city 2-truck DQN, on the test (Aug 1 -> Sep 20).

    python -m rl.eval_round15 [--out results/round15] [--workers 3]

  coordinated   one DQN (models/round15/dqn_2trucks_s<seed>) driving 2 trucks over the
                whole city (no territory split), with partner-order visibility
                (rl/features.py, rl/wrapper.py); 3 seeds.

The 1-truck champion/planner (results/round13/round13.json) and round 12's independent
territory pair (results/round12/round12.json) are read from disk as reference, not re-run.
Writes rl_test_results.{md,csv}, round15.json and rl_test_daily.png to --out.
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
SEEDS = (0, 1, 2)


def folder(seed: int) -> str:
    return os.path.join("models", "round15", f"dqn_2trucks_s{seed}")


def _load(seed: int):
    from stable_baselines3 import DQN
    return DQN.load(os.path.join(folder(seed), "best_model.zip"), device="cpu")


def job(seed: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from rl.eval_round12 import run
    from rl.wrapper import TEST_PERIOD
    m = _load(seed)              # one model drives both trucks, same as in training
    r = run([m, m], None, ZONES, TEST_PERIOD)
    return {"seed": seed, **r, "daily": np.asarray(r["daily"]).tolist()}


def validation(seed: int) -> tuple[float, str]:
    """S1 aged-starts $/week of the checkpoint training kept (rl.validation.picked)."""
    from rl.validation import picked
    p = picked(folder(seed))
    if p is None:
        raise RuntimeError(f"{folder(seed)} has no best_pick.json; "
                           "was it trained with --validation aged (the default)?")
    steps, s1 = p
    return s1, f"best checkpoint {steps:,} (S1 aged starts)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round15"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    from scipy import stats

    from rl.evaluate import run_no_trucks
    from rl.wrapper import TEST_PERIOD
    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = {}
        for r in pool.map(job, SEEDS):
            res[r["seed"]] = r
            print(f"  done coordinated seed {r['seed']}", flush=True)

    rows = []
    for s in SEEDS:
        r = res[s]
        diff = np.asarray(r["daily"]) - nt_daily
        val, how = validation(s)
        rows.append({"seed": s, "total_reward": r["total_reward"],
                     "vs_no_trucks": float(diff.sum()),
                     "test_per_week": float(diff.sum()) / n_days * 7,
                     "first_14_days": float(diff[:EARLY_DAYS].sum()),
                     "rest": float(diff[EARLY_DAYS:].sum()),
                     "days_ahead": int((diff > 0).sum()), "val_per_week": val, "val_from": how,
                     **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                          "truck_cost", "bikes_moved", "truck_km")},
                     "waits": " / ".join(f"{w:,}" for w in r["waits"]),
                     "n_routes": " / ".join(str(len(o)) for o in r["orders"]),
                     "daily_vs_no_trucks": diff.tolist()})
    df = pd.DataFrame(rows)
    df.drop(columns=["daily_vs_no_trucks"]).to_csv(os.path.join(out, "rl_test_results.csv"),
                                                   index=False)

    # Reference numbers already on file, read, not re-run.
    champ = json.load(open(os.path.join("results", "round13", "round13.json")))
    planner_rows = [r for r in champ["rows"] if r["kind"] == "planner"]
    dqn_rows = [r for r in champ["rows"] if r["kind"] == "champion"]
    r12 = json.load(open(os.path.join("results", "round12", "round12.json")))
    pair_rows = [r for r in r12["rows"] if r["kind"] == "pair"]
    planner_gain = [r["vs_no_trucks"] for r in planner_rows]
    dqn_gain = [r["vs_no_trucks"] for r in dqn_rows]
    pair_gain = [r["vs_no_trucks"] for r in pair_rows]

    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "features": FEATURES, "rows": rows,
               "reference": {"planner_1truck": planner_gain, "dqn_1truck": dqn_gain,
                             "independent_pair_round12": pair_gain}},
              open(os.path.join(out, "round15.json"), "w"), default=float)

    coord = df.vs_no_trucks
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 15: a coordinated, whole-city 2-truck DQN, test {TEST_PERIOD[0]} to "
                f"{TEST_PERIOD[1]}\n\nNo trucks: ${nt_total:,.0f}. Training and checkpoint "
                "picking used only round 14's S1 aged-starts validation; this test run was "
                "made once.\n\n")
        cols = ["seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest", "days_ahead",
                "val_per_week", "val_from", "fulfilled_pct", "lost_requests", "blocked_returns",
                "truck_cost", "bikes_moved", "truck_km", "waits", "n_routes"]
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

        f.write("\n## Mean vs no trucks (min to max); Welch t-test vs the 1-truck planner "
                "champion\n\n| run | n | vs no trucks | diff vs planner | p |\n|---|---|---|---|---|\n")
        for label, vals in (("Coordinated 2-truck DQN (round 15)", coord.tolist()),
                            ("1-truck lookahead planner (champion)", planner_gain),
                            ("1-truck DQN alone (round 9)", dqn_gain),
                            ("Independent territory pair (round 12)", pair_gain)):
            v = np.asarray(vals, float)
            rng = f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            diff = v.mean() - np.mean(planner_gain)
            p = "" if label.startswith("1-truck lookahead") else \
                f"{stats.ttest_ind(v, planner_gain, equal_var=False).pvalue:.2f}"
            f.write(f"| {label} | {len(v)} | {rng} | {diff:+,.0f} | {p} |\n")
        f.write(f"\nCoordinated pair vs round 12's independent pair: "
                f"{coord.mean() - np.mean(pair_gain):+,.0f} "
                f"(Welch p {stats.ttest_ind(coord, pair_gain, equal_var=False).pvalue:.2f})\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color="tab:red",
                ls=("-", "--", ":")[r["seed"]], lw=1.4,
                label=f"Coordinated 2-truck DQN, seed {r['seed']}")
    for r in planner_rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color="tab:blue",
                ls=("-", "--", ":")[r["seed"]], lw=0.9, alpha=0.8,
                label=f"1-truck planner (champion), seed {r['seed']}")
    for r in pair_rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color="tab:gray",
                ls=("-", "--", ":")[r["seed"]], lw=0.7, alpha=0.6,
                label="Independent territory pair (round 12)" if r["seed"] == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 15 test: coordinated 2-truck DQN vs the champion and round 12's pair")
    ax.legend(fontsize=6.5, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round15.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
