"""Round 12: divide and conquer. Split a fine zone set into one territory per truck.

    python -m rl.territory [--zones flow_w0.3_k40_need]

The 40 mainland zones of flow_w0.3_k40_need (built like round 8's 20 zones: 30% road
time / 70% demand profile, balanced on half docks + half rebalancing need) are split
into 2 territories of 20 zones each. Each truck has its own DQN that only orders
moves between its territory's zones.

Split: the 2 groups of 20 that minimise the total drive time between zones in the
same territory (zone-to-zone time = mean road minutes between their truck stations).
Start from the best balanced 2-medoid split (every pair of medoids, each zone to the
nearer one, 20 each), then swap zones between the territories while that helps.
Training data only: the zones use training-period demand and need; the split uses
road times. Saved to models/zones/<scheme>_territories.json.

A truck's view of the zones (territory_view): its 20 zones keep their stations, and
every other station (the other territory and the island) forms one "outside" zone,
so the view has 21 zones, the same shape as the champion's 20 + island.
"""
import argparse
import json
import os

import numpy as np

from rl import zones as zz

SCHEME = "flow_w0.3_k40_need"
N_TERRITORIES = 2


def zone_drive_time(zones: zz.Zones, T: np.ndarray, truck_ok: np.ndarray,
                    zone_ids: np.ndarray) -> np.ndarray:
    """(n, n) mean road minutes between the truck stations of each pair of zones."""
    T = (T + T.T) / 2
    st = [idx[truck_ok[idx]] for idx in (zones.stations_in_zone[z] for z in zone_ids)]
    n = len(zone_ids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = T[np.ix_(st[i], st[j])].mean()
    return D


def _cost(D: np.ndarray, in_a: np.ndarray) -> float:
    a, b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
    return float(D[np.ix_(a, a)].sum() + D[np.ix_(b, b)].sum()) / 2


def split_two(D: np.ndarray) -> np.ndarray:
    """(n,) bool, True = territory A; exactly n/2 zones each."""
    n = len(D)
    half = n // 2
    best, best_in = np.inf, None
    for a in range(n):                       # balanced 2-medoid start
        for b in range(a + 1, n):
            order = np.argsort(D[:, a] - D[:, b], kind="stable")
            in_a = np.zeros(n, bool)
            in_a[order[:half]] = True
            c = _cost(D, in_a)
            if c < best:
                best, best_in = c, in_a
    in_a = best_in.copy()
    while True:                              # best single swap while it helps
        a_idx, b_idx = np.flatnonzero(in_a), np.flatnonzero(~in_a)
        gain, pick = 0.0, None
        for i in a_idx:
            for j in b_idx:
                trial = in_a.copy()
                trial[i], trial[j] = False, True
                g = best - _cost(D, trial)
                if g > gain + 1e-9:
                    gain, pick = g, (i, j)
        if pick is None:
            break
        in_a[pick[0]], in_a[pick[1]] = False, True
        best -= gain
    return in_a


def _path(scheme: str) -> str:
    return os.path.join(zz.ZONES_DIR, f"{scheme}_territories.json")


def build_territories(scheme: str = SCHEME, verbose: bool = True) -> list[np.ndarray]:
    from rl.features import truck_zones
    from rl.wrapper import static_data
    from bikeshare_env.env import ISLAND_STATIONS
    s = static_data()
    zones = zz.load_zones(scheme)
    ids = s["ids"]
    tz = truck_zones(zones, ids)
    if len(tz) % 2:
        raise ValueError(f"{scheme}: {len(tz)} truck zones can't be split in two halves")
    truck_ok = ~np.isin(ids, ISLAND_STATIONS)
    D = zone_drive_time(zones, s["time"].astype(float), truck_ok, tz)
    in_a = split_two(D)
    terr = [tz[in_a], tz[~in_a]]
    if 0 not in terr[0]:                     # territory A holds zone 0 (the westmost)
        terr = terr[::-1]
    cap = s["stations"]["capacity"].to_numpy(float)
    need = zz.station_need(verbose=False)
    info = []
    for k, t in enumerate(terr):
        idx = np.concatenate([zones.stations_in_zone[z] for z in t])
        i_t = np.flatnonzero(np.isin(tz, t))
        info.append({"territory": "AB"[k], "zones": [int(z) for z in t],
                     "stations": int(len(idx)), "docks": int(cap[idx].sum()),
                     "need": int(need[idx].sum()),
                     "mean_zone_to_zone_min": round(float(D[np.ix_(i_t, i_t)][
                         ~np.eye(len(i_t), dtype=bool)].mean()), 2)})
    os.makedirs(zz.ZONES_DIR, exist_ok=True)
    with open(_path(scheme), "w") as f:
        json.dump({"scheme": scheme, "territories": info}, f, indent=1)
    if verbose:
        for d in info:
            print(f"  territory {d['territory']}: {len(d['zones'])} zones, {d['stations']} stations, "
                  f"{d['docks']:,} docks, need {d['need']:,}, "
                  f"zone-to-zone {d['mean_zone_to_zone_min']} min")
    return terr


def load_territories(scheme: str = SCHEME) -> list[np.ndarray]:
    """The territories' zone ids in `scheme` (built on first use)."""
    p = _path(scheme)
    if not os.path.exists(p):
        return build_territories(scheme, verbose=False)
    with open(p) as f:
        return [np.array(d["zones"], dtype=np.int64) for d in json.load(f)["territories"]]


def territory_view(zones: zz.Zones, terr) -> zz.Zones:
    """The territory's zones as 0..n-1 (same stations) plus zone n = everything else."""
    terr = np.asarray(terr, dtype=np.int64)
    zone_of = np.full(len(zones.zone_of_station), len(terr), dtype=np.int64)
    for i, z in enumerate(terr):
        zone_of[zones.stations_in_zone[z]] = i
    return zz._zones(zone_of, len(terr) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default=SCHEME)
    build_territories(ap.parse_args().zones)


if __name__ == "__main__":
    main()
