"""Test every policy on one continuous run, Aug 1 -> Sep 20, 2026.

    python -m rl.evaluate [--out results/round3]

Writes <out>/rl_test_results.{md,csv} and <out>/rl_test_daily.png (test
curve + validation learning curves). Rewards are the env's true $ reward.
The RL / random policies run with multi-stop missions; the fixed shuttle keeps
its single pickup / drop-off station rule.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY
from rl import zones as zz
from rl.train import model_dir
from rl.wrapper import TEST_PERIOD, RebalanceEnv, make_base_env

ALGOS = ("ppo", "a2c", "dqn")
FLEETS = (1, 2)
OUT = "results"
_COST = ("labor_cost", "fuel_cost", "fixed_cost")


def _summary(name, trucks, totals, minute_rewards, secs):
    req = totals["trips_started"] + totals["lost_trips"]
    return {
        "policy": name, "trucks": trucks,
        "total_reward": float(np.sum(minute_rewards)),
        "fulfilled_pct": 100 * totals["trips_started"] / req,
        "lost_requests": int(totals["lost_trips"]),
        "blocked_returns": int(totals["overflow_returns"]),
        "labor_cost": totals["labor_cost"], "fuel_cost": totals["fuel_cost"],
        "fixed_cost": totals["fixed_cost"],
        "bikes_moved": int(totals["bikes_moved"]), "truck_km": totals["truck_km"],
        "daily_reward": np.asarray(minute_rewards).reshape(-1, MINUTES_PER_DAY).sum(1),
        "seconds": secs,
    }


def run_no_trucks():
    t0 = time.time()
    env = make_base_env(TEST_PERIOD, 0, DEFAULT_INIT_OCCUPANCY)
    env.reset(seed=0)
    noop = np.zeros((0, 3), dtype=np.int64)
    rewards = [env.step(noop)[1] for _ in range(env.total_minutes)]
    return _summary("No trucks (benchmark)", 0, env.totals, rewards, time.time() - t0)


def run_wrapped(name, trucks, choose, multi_stop=True, zone_scheme=None, forecast=True):
    """`choose(obs, env) -> discrete action` for each decision."""
    t0 = time.time()
    env = RebalanceEnv(number_trucks=trucks, windows=[TEST_PERIOD], record_minutes=True,
                       multi_stop=multi_stop, zone_scheme=zone_scheme, forecast=forecast)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(choose(obs, env))
    return _summary(name, trucks, env.env.totals, env.minute_rewards, time.time() - t0)


def policies_to_run():
    rng = np.random.default_rng(0)
    yield "No trucks (benchmark)", 0, None
    from rl.policy import FixShuttlePolicy
    shuttle = FixShuttlePolicy.action()
    for k in FLEETS:
        yield "Hired, never used", k, (lambda obs, env: 0)
        yield "Random", k, (lambda obs, env: int(rng.integers(env.action_space.n)))
        yield "fix_shuttle_policy", k, (lambda obs, env: shuttle)
        for algo in ALGOS:
            path = os.path.join(model_dir(algo, k), "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {algo.upper()} ({k} trucks): {path} not found")
                continue
            from stable_baselines3 import A2C, DQN, PPO
            model = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo].load(path, device="cpu")
            yield algo.upper(), k, (lambda obs, env, m=model: int(m.predict(obs, deterministic=True)[0]))


def learning_curves(ax):
    for k in FLEETS:
        for algo in ALGOS:
            f = os.path.join(model_dir(algo, k), "evaluations.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            # Shaped validation return (x10 = $ vs no-truck, fleet cost added back);
            # subtract 7 days of fleet cost to get the true $ difference per window.
            delta = d["results"].mean(1) * 10 - 7 * k * 260
            ax.plot(d["timesteps"] / 1e6, delta, label=f"{algo.upper()} ({k} truck{'s' * (k > 1)})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("training decisions (millions)")
    ax.set_ylabel("$ vs no trucks per 7-day validation window")
    ax.set_title("Validation (Jul 18-31) during training")
    ax.legend(fontsize=8)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="output folder (default: results)")
    OUT = ap.parse_args().out
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, k, choose in policies_to_run():
        shuttle = name == "fix_shuttle_policy"     # single-stop rule on the k-means zones
        r = run_no_trucks() if choose is None else run_wrapped(
            name, k, choose, multi_stop=not shuttle,
            zone_scheme="kmeans" if shuttle else zz.ROUND5_SCHEME, forecast=not shuttle)
        rows.append(r)
        print(f"  {name:22s} trucks={k}  reward ${r['total_reward']:,.0f}  ({r['seconds']:.0f}s)")

    base = rows[0]
    for r in rows:
        r["delta_vs_no_trucks"] = r["total_reward"] - base["total_reward"]
        r["delta_per_day"] = r["delta_vs_no_trucks"] / len(base["daily_reward"])
        r["truck_cost"] = sum(r[c] for c in _COST)

    cols = ["policy", "trucks", "total_reward", "delta_vs_no_trucks", "delta_per_day",
            "fulfilled_pct", "lost_requests", "blocked_returns", "truck_cost",
            "labor_cost", "fuel_cost", "fixed_cost", "bikes_moved", "truck_km"]
    df = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    df.to_csv(os.path.join(OUT, "rl_test_results.csv"), index=False)

    fmt = df.copy()
    for c in ("total_reward", "delta_vs_no_trucks", "delta_per_day", "truck_cost",
              "labor_cost", "fuel_cost", "fixed_cost"):
        fmt[c] = fmt[c].map(lambda v: f"${v:,.0f}")
    fmt["fulfilled_pct"] = fmt["fulfilled_pct"].map(lambda v: f"{v:.2f}%")
    fmt["truck_km"] = fmt["truck_km"].map(lambda v: f"{v:,.0f}")
    for c in ("lost_requests", "blocked_returns", "bikes_moved"):
        fmt[c] = fmt[c].map(lambda v: f"{v:,}")
    with open(os.path.join(OUT, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# RL rebalancing: test results, {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\n")
        f.write(f"One continuous {len(base['daily_reward'])}-day run, init_occupancy="
                f"{DEFAULT_INIT_OCCUPANCY}, bikes carry over between days. Trained on "
                "2026-05-01 to 2026-07-17, checkpoint chosen on 2026-07-18 to 07-31. "
                "Reward = env's true $ reward. Each truck costs $260/day "
                "(fixed + shift) whether used or not.\n\n")
        f.write("| " + " | ".join(fmt.columns) + " |\n")
        f.write("|" + "---|" * len(fmt.columns) + "\n")
        for row in fmt.itertuples(index=False):
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    for r in rows[1:]:
        ax[0].plot(days, np.cumsum(r["daily_reward"] - base["daily_reward"]),
                   label=f"{r['policy']} ({r['trucks']} truck{'s' * (r['trucks'] > 1)})",
                   ls="--" if r["policy"] in ("Hired, never used", "Random", "fix_shuttle_policy") else "-")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("cumulative $ vs no trucks")
    ax[0].set_title("Test: cumulative reward difference vs no-truck benchmark")
    ax[0].legend(fontsize=8)
    fig.autofmt_xdate()
    learning_curves(ax[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rl_test_daily.png"), dpi=120)
    print(f"wrote {OUT}/rl_test_results.md, .csv, rl_test_daily.png")


if __name__ == "__main__":
    main()
