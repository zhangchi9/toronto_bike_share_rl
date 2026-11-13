"""Does a second truck help if it simply copies the best 1-truck agent?

    python -m rl.eval_duplicated [--out results/duplicated_truck]

Runs, on the Aug 1 -> Sep 20 test period (one continuous run):
  - DQN (1 truck), round 3 model, minute by minute (should match rl.evaluate);
  - DQN duplicated (2 trucks): the same model, its orders run by both trucks
    together, truck 0 on the zone's station ranks 1, 3, 5, ... and truck 1 on
    ranks 2, 4, 6, ..., each collecting the ordered amount (env paired_trucks);
  - the same, each truck collecting half the order (same bikes per order as 1 truck);
  - DQN (2 trucks), the round 3 model trained with 2 trucks, for reference.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY
from rl.evaluate import _summary, run_no_trucks, run_wrapped
from rl.policy import POLICIES, DuplicatedTruckPolicy
from rl.wrapper import TEST_PERIOD, make_base_env


def run_policy(name, cls):
    """Run a dashboard policy class over the whole test period, minute by minute."""
    t0 = time.time()
    kw = cls.env_kwargs()
    env = make_base_env(TEST_PERIOD, cls.number_trucks, DEFAULT_INIT_OCCUPANCY,
                        paired_trucks=kw.get("paired_trucks", False))
    obs, info = env.reset(seed=0)
    policy = cls()
    policy.reset(env)
    rewards = np.zeros(env.total_minutes)
    orders = one_waiting = 0
    for m in range(env.total_minutes):
        idle = [not tk.busy(env.t) for tk in env.trucks]
        action = policy.act(obs, info)
        orders += sum(bool(action[k].any()) and idle[k]
                      for k in range(len(idle) if not env.paired_trucks else 1))
        obs, rewards[m], _, _, info = env.step(action)
        busy = [tk.busy(env.t) for tk in env.trucks]
        one_waiting += len(busy) == 2 and busy[0] != busy[1]
    out = _summary(name, cls.number_trucks, env.totals, rewards, time.time() - t0)
    out["orders"] = orders
    out["truck_minutes_waiting_for_partner"] = one_waiting
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/duplicated_truck")
    out_dir = ap.parse_args().out
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import DQN
    base = run_no_trucks()
    half = type("DuplicatedHalf", (DuplicatedTruckPolicy,), {"qty_divisor": 2, "_model": None})
    dqn2 = DQN.load("models/dqn_2trucks/best_model.zip", device="cpu")
    rows = [
        base,
        run_policy("DQN (1 truck)", POLICIES["dqn_1trucks"]),
        run_policy("DQN duplicated, full order each", DuplicatedTruckPolicy),
        run_policy("DQN duplicated, half order each", half),
        run_wrapped("DQN (2 trucks, trained)", 2,
                    lambda obs, env: int(dqn2.predict(obs, deterministic=True)[0])),
    ]
    for r in rows[1:]:
        print(f"  {r['policy']:34} ${r['total_reward']:,.0f}  ({r['seconds']:.0f}s)")

    b_daily = base["daily_reward"]
    table = []
    for r in rows:
        d = r["daily_reward"] - b_daily
        table.append({
            "policy": r["policy"], "trucks": r["trucks"],
            "total_reward": r["total_reward"],
            "vs_no_trucks": r["total_reward"] - base["total_reward"],
            "days_ahead": int((d > 0).sum()) if r is not base else None,
            "fulfilled_pct": r["fulfilled_pct"], "lost_requests": r["lost_requests"],
            "blocked_returns": r["blocked_returns"],
            "truck_cost": r["labor_cost"] + r["fuel_cost"] + r["fixed_cost"],
            "bikes_moved": r["bikes_moved"], "truck_km": r["truck_km"],
            "orders": r.get("orders"),
            "minutes_one_truck_waits": r.get("truck_minutes_waiting_for_partner"),
        })
    df = pd.DataFrame(table)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    def fmt(v, money=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        return f"${v:,.0f}" if money else (f"{v:,.2f}" if isinstance(v, float) and v < 100 else f"{v:,.0f}")

    lines = [f"# Duplicated DQN truck: test results, {TEST_PERIOD[0]} to {TEST_PERIOD[1]}", "",
             "One continuous 51-day run. Rewards are the env's true $ reward; every truck "
             f"costs ${260 * 1}/day (fixed + shift) plus night labor and fuel.", "",
             "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    money = {"total_reward", "vs_no_trucks", "truck_cost"}
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], c in money) if c not in ("policy", "trucks")
                                       else str(row[c]) for c in df.columns) + " |")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
