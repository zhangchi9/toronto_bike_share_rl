"""Round 8: screen candidate zone sets for DQN (1 truck) without RL training.

    python -m rl.zone_search [--out results/round8] [--workers 14]

Candidates: flow zones with w = 0.3 and k in {12, 16, 20, 25, 30} mainland zones,
balanced by docks or by docks + rebalancing need (rl.zones.build_flow_zones), plus
reference sets with known DQN results (k-means, w = 0.5, w = 0.7).

Each zone set is scored by two no-learning policies, run on the whole training
period (May 1 - Jul 17, one continuous run, bikes carry over). The test period is
never touched.
  - best fixed route: the truck repeats one zone-to-zone order (20 or 40 bikes)
    every time it is idle. Candidate sources are the 8 zones with the most blocked
    returns, destinations the 8 with the most lost requests, in a no-truck run.
    The good DQN runs are close to this: one route for most of their orders.
  - greedy: at each idle moment, move bikes from the zone with the largest expected
    surplus in 2 h (fill + average hourly net flow) to the one with the largest
    expected shortage.
Writes screen.md / screen.csv, routes.csv (every route's gain) and screen_maps.png.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from rl import zones as zz

W = 0.3
KS = (12, 16, 20, 25, 30)
REFERENCES = ("kmeans", "flow_w0.5", "flow_w0.7")
QTYS = (20, 40)
N_ENDS = 8                     # candidate sources / destinations per zone set
TARGET_FILL = 0.5              # greedy: aim for half-full zones
LOOKAHEAD_H = 2


def candidates() -> list[str]:
    return [zz.flow_scheme(W, k, b) for k in KS for b in zz.BALANCES]


def route_ends(scheme: str):
    """(truck zone ids, sources, destinations) from the no-truck training run."""
    from rl.features import truck_zones
    from rl.wrapper import static_data
    s = static_data()
    z = zz.load_zones(scheme)
    tz = truck_zones(z, s["ids"])
    with np.load(os.path.join(zz.ZONES_DIR, zz.NEED_FILE)) as f:
        lost, blocked = f["lost"], f["blocked"]
    zl = np.bincount(z.zone_of_station, weights=lost, minlength=z.n_zones)[tz]
    zb = np.bincount(z.zone_of_station, weights=blocked, minlength=z.n_zones)[tz]
    src = tz[np.argsort(-zb)[:N_ENDS]]
    dst = tz[np.argsort(-zl)[:N_ENDS]]
    return tz, src, dst


def _env(scheme, stop_rule=None):
    from rl.wrapper import TRAIN_RANGE, RebalanceEnv
    env = RebalanceEnv(number_trucks=1, windows=[TRAIN_RANGE], zone_scheme=scheme,
                       forecast=False, stop_rule=stop_rule)
    env.reset(seed=0)
    return env


def run_route(args) -> dict:
    """Gain over no trucks of repeating one order over the training period.
    args = (scheme, src, dst, qty[, stop_rule])."""
    scheme, src, dst, qty, *rule = args
    from rl.features import encode_action, truck_zones
    from rl.wrapper import static_data
    env = _env(scheme, *rule)
    tz = list(truck_zones(env.zones, static_data()["ids"]))
    a = encode_action(tz.index(src), tz.index(dst), qty, len(tz))
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(a)
    T = env.env.totals
    return {"scheme": scheme, "src": int(src), "dst": int(dst), "qty": qty,
            "gain": float(info["true_total"] - env.base.sum()),
            "bikes_moved": int(T["bikes_moved"])}


def run_greedy(scheme, stop_rule=None) -> dict:
    """Gain over no trucks of the surplus -> shortage greedy dispatcher."""
    from rl import forecast as fc
    from rl.features import encode_action, truck_zones
    from rl.wrapper import TRAIN_RANGE, static_data
    s = static_data()
    cap = s["stations"]["capacity"].to_numpy(float)
    env = _env(scheme, stop_rule)
    zof, Z = env.zones.zone_of_station, env.zones.n_zones
    tz = truck_zones(env.zones, s["ids"])
    docks = np.bincount(zof, weights=cap, minlength=Z)[tz]
    P = zz.demand_profiles(s["ids"], cap)                      # net flow per dock / h
    flow = np.stack([np.bincount(zof, weights=P[:, h] * cap, minlength=Z)
                     for h in range(48)], 1)[tz]               # (truck zones, 48) bikes/h
    days = pd.date_range(*TRAIN_RANGE)
    off = np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
    n = len(tz)
    done, info = False, {}
    while not done:
        t = env.env.t
        d, h = t // 1440, (t % 1440) // 60
        hours = [(h + i) % 24 + 24 * off[min(d + (h + i) // 24, len(off) - 1)]
                 for i in range(LOOKAHEAD_H)]
        bikes = np.bincount(zof, weights=env.env.bikes, minlength=Z)[tz]
        surplus = bikes + flow[:, hours].sum(1) - TARGET_FILL * docks
        i, j = int(surplus.argmax()), int(surplus.argmin())
        room = min(surplus[i], -surplus[j])
        qty = next((q for q in (40, 20, 10) if q <= room), 0)
        a = encode_action(i, j, qty, n) if qty and i != j else 0
        _, _, done, _, info = env.step(a)
    T = env.env.totals
    return {"scheme": scheme, "greedy_gain": float(info["true_total"] - env.base.sum()),
            "greedy_bikes_moved": int(T["bikes_moved"])}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bikeshare_env.env import ISLAND_STATIONS
    from rl.wrapper import static_data
    from rl.zones_compare import draw, scheme_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round8"))
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zz.station_need()
    schemes = candidates() + list(REFERENCES)
    for sc in candidates():                      # build serially before the workers load
        if sc.startswith("flow_w"):
            w, k, b = zz.parse_flow_scheme(sc)
            if not os.path.exists(os.path.join(zz.ZONES_DIR, f"{sc}.npz")):
                zz.build_flow_zones(w, k=k, balance=b)
    jobs = [(sc, int(a), int(b), q) for sc in schemes
            for a in route_ends(sc)[1] for b in route_ends(sc)[2] if a != b for q in QTYS]
    print(f"{len(schemes)} zone sets, {len(jobs)} route runs + {len(schemes)} greedy runs",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        greedy = pool.map(run_greedy, schemes)
        routes = []
        for i, r in enumerate(pool.map(run_route, jobs, chunksize=4)):
            routes.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} routes", flush=True)
        greedy = list(greedy)
    rt = pd.DataFrame(routes)
    rt.to_csv(os.path.join(args.out, "routes.csv"), index=False)

    s = static_data()
    st, ids = s["stations"], s["ids"]
    cap = st["capacity"].to_numpy(float)
    T = s["time"].astype(float)
    T = (T + T.T) / 2
    main_idx = np.flatnonzero(~np.isin(ids, ISLAND_STATIONS))
    P = zz.demand_profiles(ids, cap)
    rows = []
    for sc, g in zip(schemes, greedy):
        z = zz.load_zones(sc)
        m = scheme_metrics(sc, z, T, P, cap, main_idx)
        r = rt[rt.scheme == sc].sort_values("gain", ascending=False)
        best = r.iloc[0]
        rows.append({**m, "reference": sc in REFERENCES,
                     "best_route": f"{best.src}->{best.dst} x{best.qty}",
                     "best_route_gain": best.gain,
                     "top3_route_mean": r.gain.head(3).mean(),
                     "median_route_gain": r.gain.median(),
                     "greedy_gain": g["greedy_gain"],
                     "proxy": max(best.gain, g["greedy_gain"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "screen.csv"), index=False)

    with open(os.path.join(args.out, "screen.md"), "w", encoding="utf-8") as f:
        f.write("# Round 8 zone screen (training period May 1 - Jul 17, no RL)\n\n"
                "Gains are $ over no trucks for one truck over the 78-day training run. "
                "Proxy = max(best fixed route, greedy).\n\n")
        cols = ["scheme", "zones", "docks_min", "docks_max", "road_min_within_zone",
                "profile_corr_within_zone", "best_route", "best_route_gain",
                "top3_route_mean", "median_route_gain", "greedy_gain", "proxy"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in df.sort_values("proxy", ascending=False).iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.2f}" if c in ("road_min_within_zone",
                                            "profile_corr_within_zone") else f"{v:,.0f}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")

    lat, lon = st["lat"].to_numpy(), st["lon"].to_numpy()
    fig, axes = plt.subplots(2, 5, figsize=(25, 11))
    for ax, sc in zip(axes.ravel(), candidates()):
        z = zz.load_zones(sc)
        draw(ax, z, lat, lon, sc, island_zone=zz.island_zone(z))
        ax.set_xlim(-79.50, -79.28), ax.set_ylim(43.62, 43.72)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "screen_maps.png"), dpi=90)
    print(df.sort_values("proxy", ascending=False)[
        ["scheme", "best_route", "best_route_gain", "top3_route_mean", "greedy_gain"]]
        .to_string(index=False))


if __name__ == "__main__":
    main()
