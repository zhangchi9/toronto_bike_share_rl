"""Round 8: DQN (1 truck) on zone sets tuned for it, 3 training seeds each.

    python -m rl.eval_round8 [--out results/round8]

Tests on the Aug 1 -> Sep 20 test period (one continuous run, as rl.evaluate):
  - baseline flow_w0.3 (rounds 5 and 7): seed 0 = models/dqn_1trucks/,
    seeds 1-2 = models/round7/dqn_1trucks_flow_w0.3_s<seed>/;
  - each finalist in results/round8/finalists.json (picked by rl.zone_search on the
    training period): models/round8/dqn_1trucks_<scheme>_s<seed>/.
For every zone set it also runs, with no learning: random orders (3 seeds), the
greedy dispatcher and the best fixed route from the screen. Writes
rl_test_results.{md,csv}, round8.json and rl_test_daily.png to --out.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.eval_round6 import run, validation
from rl.evaluate import run_no_trucks
from rl.wrapper import TEST_PERIOD

BASELINE = "flow_w0.3"
SEEDS = (0, 1, 2)
EARLY_DAYS = 14


def model_folder(scheme: str, seed: int) -> str:
    if scheme == BASELINE:
        return os.path.join("models", "dqn_1trucks") if seed == 0 else \
            os.path.join("models", "round7", f"dqn_1trucks_{scheme}_s{seed}")
    return os.path.join("models", "round8", f"dqn_1trucks_{scheme}_s{seed}")


def greedy_chooser(scheme: str):
    """The screen's greedy dispatcher as a run() chooser, on the test calendar."""
    from rl import forecast as fc
    from rl import zones as zz
    from rl.features import encode_action, truck_zones
    from rl.wrapper import static_data
    from rl.zone_search import LOOKAHEAD_H, TARGET_FILL
    s = static_data()
    cap = s["stations"]["capacity"].to_numpy(float)
    z = zz.load_zones(scheme)
    zof, Z = z.zone_of_station, z.n_zones
    tz = truck_zones(z, s["ids"])
    docks = np.bincount(zof, weights=cap, minlength=Z)[tz]
    P = zz.demand_profiles(s["ids"], cap)
    flow = np.stack([np.bincount(zof, weights=P[:, h] * cap, minlength=Z)
                     for h in range(48)], 1)[tz]
    days = pd.date_range(*TEST_PERIOD)
    off = np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))

    def choose(obs, env):
        t = env.env.t
        d, h = t // 1440, (t % 1440) // 60
        hours = [(h + i) % 24 + 24 * off[min(d + (h + i) // 24, len(off) - 1)]
                 for i in range(LOOKAHEAD_H)]
        bikes = np.bincount(zof, weights=env.env.bikes, minlength=Z)[tz]
        surplus = bikes + flow[:, hours].sum(1) - TARGET_FILL * docks
        i, j = int(surplus.argmax()), int(surplus.argmin())
        room = min(surplus[i], -surplus[j])
        qty = next((q for q in (40, 20, 10) if q <= room), 0)
        return encode_action(i, j, qty, len(tz)) if qty and i != j else 0
    return choose


def route_chooser(scheme: str, src: int, dst: int, qty: int):
    from rl import zones as zz
    from rl.features import encode_action, truck_zones
    from rl.wrapper import static_data
    tz = list(truck_zones(zz.load_zones(scheme), static_data()["ids"]))
    a = encode_action(tz.index(src), tz.index(dst), qty, len(tz))
    return lambda obs, env: a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/round8")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    from stable_baselines3 import DQN
    finalists = json.load(open(os.path.join(out, "finalists.json")))["finalists"]
    screen = pd.read_csv(os.path.join(out, "screen.csv")).set_index("scheme")

    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)
    rows = []

    def add(scheme, policy, seed, r, extra=None):
        diff = r["daily"] - nt_daily
        gain = float(diff.sum())
        rows.append({"zones": scheme, "policy": policy, "seed": seed,
                     "total_reward": r["total_reward"], "vs_no_trucks": gain,
                     "test_per_week": gain / n_days * 7,
                     "first_14_days": float(diff[:EARLY_DAYS].sum()),
                     "rest": float(diff[EARLY_DAYS:].sum()),
                     "days_ahead": int((diff > 0).sum()),
                     **{c: r[c] for c in ("fulfilled_pct", "lost_requests", "blocked_returns",
                                          "truck_cost", "bikes_moved", "truck_km", "waits")},
                     "n_orders": sum(o[3] for o in r["orders"]),
                     "n_routes": len(r["orders"]), **(extra or {}),
                     "top_orders": r["orders"][:6], "daily_vs_no_trucks": diff.tolist()})
        print(f"  {scheme:22s} {policy:12s} seed {seed}: {gain:+,.0f}", flush=True)

    for scheme in [BASELINE] + finalists:
        for seed in SEEDS:
            folder = model_folder(scheme, seed)
            path = os.path.join(folder, "best_model.zip")
            if not os.path.exists(path):
                print(f"  skip {scheme} seed {seed}: {path} not found")
                continue
            m = DQN.load(path, device="cpu")
            r = run(1, scheme, lambda o, e, m=m: int(m.predict(o, deterministic=True)[0]))
            step, val = validation(folder, 1)
            add(scheme, "DQN", seed, r, {"val_per_week": val, "val_best_step": step})
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            add(scheme, "Random", seed,
                run(1, scheme, lambda o, e, rng=rng: int(rng.integers(e.action_space.n))))
        add(scheme, "Greedy", None, run(1, scheme, greedy_chooser(scheme)))
        src, dst_q = screen.loc[scheme, "best_route"].split("->")
        dst, qty = dst_q.split(" x")
        add(scheme, "Best route", None,
            run(1, scheme, route_chooser(scheme, int(src), int(dst), int(qty))),
            {"route": screen.loc[scheme, "best_route"]})

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(),
               "finalists": finalists, "rows": rows},
              open(os.path.join(out, "round8.json"), "w"), default=float)

    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 8: DQN (1 truck) on tuned zones, test {TEST_PERIOD[0]} to "
                f"{TEST_PERIOD[1]}\n\nBaseline {BASELINE}: seeds 0-2 from rounds 5 and 7. "
                f"No trucks: ${nt_total:,.0f}.\n\n")
        cols = ["zones", "policy", "seed", "vs_no_trucks", "test_per_week", "first_14_days",
                "rest", "days_ahead", "val_per_week", "val_best_step", "fulfilled_pct",
                "lost_requests", "blocked_returns", "truck_cost", "bikes_moved", "truck_km",
                "n_orders", "n_routes", "waits"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.2f}" if c == "fulfilled_pct" else f"{v:,.0f}"
                elif isinstance(v, int) and c != "seed":
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## Per zone set and policy: mean (min to max)\n\n"
                "| zones | policy | runs | vs no trucks | first 14 days | rest |\n"
                "|---|---|---|---|---|---|\n")
        for (zones, policy), g in df.groupby(["zones", "policy"], sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            f.write(f"| {zones} | {policy} | {len(g)} | {s('vs_no_trucks')} | "
                    f"{s('first_14_days')} | {s('rest')} |\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    dq = [r for r in rows if r["policy"] == "DQN"]
    schemes = list(dict.fromkeys(r["zones"] for r in dq))
    colors = dict(zip(schemes, plt.get_cmap("tab10").colors))
    for r in dq:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["zones"]],
                ls=("-", "--", ":")[r["seed"]], label=f"{r['zones']}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 8 test: DQN (1 truck), 3 training seeds per zone set")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round8.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
