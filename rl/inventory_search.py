"""Round 13: tune the inventory-model dispatcher (rl/inventory.py) on the validation weeks.

    python -m rl.inventory_search [--workers 4] [--out results/round13] [--horizons 2160 2880]

Grid: horizon x theta. Horizons already in inventory_val.csv are not run again, so the
grid can be extended. Writes inventory_val.{csv,md} and inventory_best.json (the best
mean $/week over both validation weeks), which rl.eval_round13 tests.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

HORIZONS = (240, 480, 720, 1440, 2160, 2880)
THETAS = (0.0, 0.25, 0.5, 1.0)


def job(horizon) -> list[dict]:
    from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
    from rl.inventory import run
    from rl.wrapper import VAL_WINDOWS, no_truck_rewards
    rows = []
    for theta in THETAS:                       # one process per horizon: the cost table is cached
        for w, window in enumerate(VAL_WINDOWS):
            r = run(window, horizon=horizon, theta=theta)
            gain = r["total_reward"] - float(no_truck_rewards(window, DEFAULT_INIT_OCCUPANCY).sum())
            rows.append({"horizon": horizon, "theta": theta, "week": w, "gain": gain,
                         "bikes_moved": r["bikes_moved"], "truck_km": r["truck_km"]})
            print(rows[-1], flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=os.path.join("results", "round13"))
    p.add_argument("--horizons", type=int, nargs="*", default=list(HORIZONS))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "inventory_val.csv")
    old = pd.read_csv(path) if os.path.exists(path) else None
    done = set(old.horizon) if old is not None else set()
    todo = [h for h in a.horizons if h not in done]
    rows = []
    if todo:
        with ProcessPoolExecutor(min(a.workers, len(todo))) as ex:
            rows = [r for rs in ex.map(job, todo) for r in rs]
    df = pd.concat([d for d in (old, pd.DataFrame(rows)) if d is not None and len(d)],
                   ignore_index=True)
    df.to_csv(path, index=False)
    summarize(a.out)


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate package."""
    df = df.reset_index()
    fmt = lambda v: (f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}") if isinstance(v, float) \
        else str(v)   # noqa: E731
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summarize(out: str) -> None:
    df = pd.read_csv(os.path.join(out, "inventory_val.csv"))
    tab = df.groupby(["horizon", "theta"])[["gain", "bikes_moved"]].mean()
    best = tab["gain"].idxmax()
    with open(os.path.join(out, "inventory_val.md"), "w", encoding="utf-8") as f:
        f.write("# Inventory dispatcher tuning on the validation weeks\n\n"
                "Mean $/week vs no trucks and bikes moved per week, over both weeks.\n\n")
        f.write(md_table(tab) + "\n\n")
        f.write(f"Best: horizon {best[0]} min, theta {best[1]}\n")
    json.dump({"horizon": int(best[0]), "theta": float(best[1])},
              open(os.path.join(out, "inventory_best.json"), "w"))
    print(tab.round(1))


if __name__ == "__main__":
    main()
