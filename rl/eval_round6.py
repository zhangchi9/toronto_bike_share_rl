"""Round 6: how much does the road / demand weight of the zones matter?

    python -m rl.eval_round6 [--out results/round6]

Tests PPO (2 trucks) and DQN (1 truck) trained on each flow zone set, on the
Aug 1 -> Sep 20 test period (one continuous run, as rl.evaluate):
  - w = 0.3 (30% road, 70% demand): the round 5 models, models/<algo>_<k>trucks/;
  - w = 0.5 and w = 0.7: models/round6/<algo>_<k>trucks_w<w>/.
Also runs a truck placing random orders on each zone set, with 3 seeds per fleet
size, to show how much the zones alone help. Writes rl_test_results.{md,csv},
round6.json (daily rewards, order counts) and rl_test_daily.png to --out.
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.evaluate import run_no_trucks
from rl.wrapper import TEST_PERIOD, RebalanceEnv

WEIGHTS = (0.3, 0.5, 0.7)
AGENTS = (("ppo", 2), ("dqn", 1))
RANDOM_SEEDS = (0, 1, 2)


def model_folder(algo: str, k: int, w: float) -> str:
    if w == 0.3:
        return os.path.join("models", f"{algo}_{k}trucks")              # round 5
    return os.path.join("models", "round6", f"{algo}_{k}trucks_w{w:g}")


def run(trucks: int, scheme: str, choose, features=None, stop_rule=None, window=None) -> dict:
    """One test run; `choose(obs, env) -> discrete action`. Also counts orders."""
    env = RebalanceEnv(number_trucks=trucks, windows=[window or TEST_PERIOD],
                       record_minutes=True, zone_scheme=scheme, features=features,
                       stop_rule=stop_rule)
    obs, _ = env.reset(seed=0)
    done, orders, waits = False, Counter(), 0
    zof = env.zones.zone_of_station
    while not done:
        a = choose(obs, env)
        row = env.fb.decode(a)
        if row.any():
            orders[(int(zof[row[0] - 1]), int(zof[row[1]]), int(row[2]))] += 1
        else:
            waits += 1
        obs, _, done, _, _ = env.step(a)
    T = env.env.totals
    return {"total_reward": float(np.sum(env.minute_rewards)),
            "daily": np.asarray(env.minute_rewards).reshape(-1, 1440).sum(1),
            "fulfilled_pct": 100 * T["trips_started"] / (T["trips_started"] + T["lost_trips"]),
            "lost_requests": int(T["lost_trips"]), "blocked_returns": int(T["overflow_returns"]),
            "truck_cost": T["labor_cost"] + T["fuel_cost"] + T["fixed_cost"],
            "bikes_moved": int(T["bikes_moved"]), "truck_km": float(T["truck_km"]),
            "waits": waits, "orders": [[*key, n] for key, n in orders.most_common()]}


def validation(folder: str, k: int) -> tuple[int, float]:
    d = np.load(os.path.join(folder, "evaluations.npz"))
    val = d["results"].mean(1) * 10 - 7 * k * 260       # $ vs no trucks per week
    i = int(val.argmax())
    return int(d["timesteps"][i]), float(val[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round6")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN, PPO

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], base["daily_reward"]
    n_days = len(nt_daily)
    rows, curves = [], {}

    def add(policy, k, w, r, extra=None):
        gain = r["total_reward"] - nt_total
        rows.append({"policy": policy, "trucks": k, "zones_w": w,
                     "total_reward": r["total_reward"], "vs_no_trucks": gain,
                     "test_per_week": gain / n_days * 7,
                     "days_ahead": int((r["daily"] > nt_daily).sum()),
                     **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                          "truck_cost", "bikes_moved", "truck_km", "waits")},
                     "n_orders": sum(o[3] for o in r["orders"]), "n_routes": len(r["orders"]),
                     **(extra or {}), "top_orders": r["orders"][:8]})
        print(f"  {policy:7s} {k} truck(s) w={w}: {gain:+,.0f}", flush=True)
        return gain

    for w in WEIGHTS:
        scheme = f"flow_w{w:g}"
        for algo, k in AGENTS:
            folder = model_folder(algo, k, w)
            path = os.path.join(folder, "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {algo} {k} w={w}: {path} not found")
                continue
            m = {"ppo": PPO, "dqn": DQN}[algo].load(path, device="cpu")
            r = run(k, scheme, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]))
            step, val = validation(folder, k)
            add(algo.upper(), k, w, r, {"val_best_step": step, "val_per_week": val})
            curves[f"{algo.upper()} ({k} truck{'s' * (k > 1)}), w={w:g}"] = r["daily"] - nt_daily
        for k in (1, 2):
            for seed in RANDOM_SEEDS:
                rng = np.random.default_rng(seed)
                r = run(k, scheme, lambda o, e, rng=rng: int(rng.integers(e.action_space.n)))
                add(f"Random (seed {seed})", k, w, r)

    df = pd.DataFrame(rows)
    df.drop(columns="top_orders").to_csv(os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "rows": rows},
              open(os.path.join(out, "round6.json"), "w"), default=float)

    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 6: zone weight w, test {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\n"
                "w = share of road drive time in the zone distance (the rest is the demand "
                "profile). w = 0.3 rows are the round 5 models. No trucks: "
                f"${nt_total:,.0f}.\n\n")
        cols = ["policy", "trucks", "zones_w", "vs_no_trucks", "test_per_week", "days_ahead",
                "val_per_week", "val_best_step", "fulfilled_pct", "lost_requests",
                "blocked_returns", "truck_cost", "bikes_moved", "truck_km", "n_orders",
                "n_routes", "waits"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float) and c not in ("zones_w",):
                    v = f"{v:.2f}" if c == "fulfilled_pct" else f"{v:,.0f}"
                elif isinstance(v, int) and c not in ("trucks",):
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## Random orders, mean of 3 seeds (min to max)\n\n"
                "| trucks | zones_w | vs no trucks |\n|---|---|---|\n")
        rnd = df[df.policy.str.startswith("Random")]
        for (k, w), g in rnd.groupby(["trucks", "zones_w"]):
            v = g.vs_no_trucks
            f.write(f"| {k} | {w} | {v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f}) |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    for name, d in curves.items():
        ax.plot(days, np.cumsum(d), label=name, ls="-" if name.startswith("PPO") else "--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 6 test: PPO (2 trucks) and DQN (1 truck) by zone weight w")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round6.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
