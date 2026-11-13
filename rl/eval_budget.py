"""Does a bigger training budget help DQN (1 truck)? Test the 1M and 2M runs.

    python -m rl.eval_budget [--out results/round4_dqn_2M]

Runs on the Aug 1 -> Sep 20 test period (one continuous run, as rl.evaluate):
  - DQN 1M: models/dqn_1trucks/best_model.zip (round 4, best checkpoint at 1M);
  - DQN 2M, best checkpoint: models/dqn_1trucks_2M/best_model.zip;
  - DQN 2M, final weights:   models/dqn_1trucks_2M/final_model.zip.
The two runs share the seed and the exploration schedule (epsilon decays over the
first 200k decisions), so the 2M run is the 1M run trained for longer.
"""
import argparse
import os

import numpy as np
import pandas as pd

from rl.evaluate import run_no_trucks, run_wrapped

RUNS = [("DQN, 1M decisions (round 4)", "models/dqn_1trucks"),
        ("DQN, 2M decisions", "models/dqn_1trucks_2M")]


def validation_curve(folder):
    d = np.load(os.path.join(folder, "evaluations.npz"))
    return d["timesteps"], d["results"].mean(1) * 10 - 7 * 260     # $/week vs no trucks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round4_dqn_2M")
    out_dir = ap.parse_args().out
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3 import DQN
    base = run_no_trucks()
    rows = []
    for label, folder in RUNS:
        steps, val = validation_curve(folder)
        files = [("best checkpoint", "best_model.zip")]
        if "2M" in folder:
            files.append(("final weights", "final_model.zip"))
        for kind, f in files:
            model = DQN.load(os.path.join(folder, f), device="cpu")
            r = run_wrapped(f"{label}, {kind}", 1,
                            lambda obs, env: int(model.predict(obs, deterministic=True)[0]))
            d = r["daily_reward"] - base["daily_reward"]
            i = int(val.argmax()) if kind == "best checkpoint" else len(val) - 1
            rows.append({
                "policy": r["policy"], "checkpoint_step": int(steps[i]),
                "validation_per_week": float(val[i]),
                "total_reward": r["total_reward"],
                "vs_no_trucks": r["total_reward"] - base["total_reward"],
                "test_per_week": (r["total_reward"] - base["total_reward"]) / len(d) * 7,
                "days_ahead": int((d > 0).sum()),
                "fulfilled_pct": r["fulfilled_pct"], "lost_requests": r["lost_requests"],
                "blocked_returns": r["blocked_returns"],
                "truck_cost": r["labor_cost"] + r["fuel_cost"] + r["fixed_cost"],
                "bikes_moved": r["bikes_moved"], "truck_km": r["truck_km"],
            })
            print(f"  {r['policy']:45} {rows[-1]['vs_no_trucks']:+,.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    curves = []
    for label, folder in RUNS:
        steps, val = validation_curve(folder)
        curves.append(pd.DataFrame({"run": label, "step": steps, "validation_per_week": val}))
    pd.concat(curves).to_csv(os.path.join(out_dir, "validation_curves.csv"), index=False)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
