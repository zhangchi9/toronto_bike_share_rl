"""Precomputed replays for the dashboard: one continuous run per policy, Aug 1 - Sep 30.

    python -m dashboard_data            # build every missing / out-of-date replay
    python -m dashboard_data --rebuild  # rebuild all of them

Each policy is simulated once over the whole replay period: stations are filled to
the default occupancy at 00:00 on Aug 1 and bikes carry over between days. The
dashboard never re-simulates: any date range the user picks is a slice of this run.
The browser is sent one day at a time (`day_chunk`), since a whole range would be
far too large.

Files: data/replay/<policy>.npz (one per policy) and data/replay/_trips.npz (the
trip log, the same for every policy). A replay is rebuilt automatically when its
fingerprint (format, start occupancy, period, model file size and time) changes,
e.g. after a model is retrained. Changes to the env itself are not detected: run
with --rebuild after editing bikeshare_env.
"""
import argparse
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY
from bikeshare_env.policies import POLICIES

try:   # registers the trained RL models as policies
    import rl.policy  # noqa: F401
except ImportError as e:
    print(f"RL policies not loaded: {e}")

REPLAY_START = "2026-08-01"
REPLAY_END = "2026-09-30"
REPLAY_DIR = os.path.join("data", "replay")
FORMAT_VERSION = 1
_PER_MINUTE = ("reward", "fulfilled", "lost", "active", "blocked", "pending")
_CUMULATIVE = ("reward", "fulfilled", "lost", "blocked")      # shown as running totals
_TRUCK = ("truck_status", "truck_load", "truck_station", "truck_moved")


def n_days() -> int:
    return (pd.Timestamp(REPLAY_END) - pd.Timestamp(REPLAY_START)).days + 1


def day_date(d: int) -> pd.Timestamp:
    return pd.Timestamp(REPLAY_START) + pd.Timedelta(days=int(d))


def _path(policy_key: str) -> str:
    return os.path.join(REPLAY_DIR, f"{policy_key}.npz")


def fingerprint(policy_key: str) -> dict:
    """What a replay depends on; a stored replay with a different fingerprint is rebuilt."""
    fp = {"format": FORMAT_VERSION, "policy": policy_key, "start": REPLAY_START,
          "end": REPLAY_END, "init_occupancy": DEFAULT_INIT_OCCUPANCY, "model": None}
    model = getattr(POLICIES[policy_key], "path", "")
    if model and os.path.exists(model):
        st = os.stat(model)
        fp["model"] = {"path": model.replace("\\", "/"), "size": st.st_size,
                       "mtime": int(st.st_mtime)}
    return fp


def _stored_fingerprint(policy_key: str) -> dict | None:
    try:
        with np.load(_path(policy_key)) as z:
            return json.loads(str(z["fingerprint"]))
    except (OSError, KeyError, ValueError):
        return None


def is_current(policy_key: str) -> bool:
    return _stored_fingerprint(policy_key) == fingerprint(policy_key)


# ── build ────────────────────────────────────────────────────────────────────
def build(policy_key: str, verbose: bool = True) -> str:
    """Simulate the whole replay period for one policy and save it."""
    from bikeshare_env.rollout import simulate_period
    import time
    t0 = time.time()
    r = simulate_period(REPLAY_START, REPLAY_END, policy_key, DEFAULT_INIT_OCCUPANCY)
    os.makedirs(REPLAY_DIR, exist_ok=True)
    trips_path = os.path.join(REPLAY_DIR, "_trips.npz")
    t = r.trips
    if not os.path.exists(trips_path):
        casual = t.casual if t.casual is not None else np.zeros(len(t), np.int8)
        _save(trips_path, start_min=t.start_min.astype(np.int32),
              end_min=t.end_min.astype(np.int32), start_idx=t.start_idx.astype(np.int32),
              end_idx=t.end_idx.astype(np.int32), casual=casual.astype(np.int8),
              period=np.array(f"{REPLAY_START}|{REPLAY_END}"))
    _save(_path(policy_key),
          fingerprint=np.array(json.dumps(fingerprint(policy_key))),
          totals=np.array(json.dumps({k: float(v) for k, v in r.totals.items()})),
          bikes=r.bikes, trip_started=r.trip_started, truck_legs=r.truck_legs,
          **{k: getattr(r, k) for k in _PER_MINUTE + _TRUCK})
    if verbose:
        days = r.reward.reshape(-1, MINUTES_PER_DAY).sum(1)
        print(f"  {policy_key:24} ${days.sum():,.0f} Aug 1-Sep 30 "
              f"(Aug 1-Sep 20: ${days[:51].sum():,.0f})  {time.time() - t0:.0f}s", flush=True)
    return policy_key


def _save(path, **arrays):
    tmp = path[:-4] + ".tmp.npz"                 # write, then rename: never a half file
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ── load ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_trips() -> dict:
    with np.load(os.path.join(REPLAY_DIR, "_trips.npz")) as z:
        return {k: z[k] for k in ("start_min", "end_min", "start_idx", "end_idx", "casual")}


@lru_cache(maxsize=3)
def load(policy_key: str) -> dict:
    """A policy's replay, built first if missing or out of date (a few minutes for RL)."""
    if not is_current(policy_key):
        build(policy_key)
    with np.load(_path(policy_key)) as z:
        rep = {k: z[k] for k in z.files if k not in ("fingerprint", "totals")}
        rep["totals"] = json.loads(str(z["totals"]))
    trips = load_trips()
    if len(trips["start_min"]) != len(rep["trip_started"]):
        raise ValueError(f"{REPLAY_DIR}/_trips.npz does not match {policy_key}; "
                         "rebuild with python -m dashboard_data --rebuild")
    # Per-day sums of the running-total metrics, for range offsets.
    rep["day_sums"] = {k: rep[k].reshape(-1, MINUTES_PER_DAY).sum(1) for k in _CUMULATIVE}
    return rep


def requests_in(d0: int, d1: int) -> int:
    """Bike requests (served or lost) starting on days d0..d1."""
    sm = load_trips()["start_min"]
    return int(np.searchsorted(sm, (d1 + 1) * MINUTES_PER_DAY)
               - np.searchsorted(sm, d0 * MINUTES_PER_DAY))


# ── one day for the browser ──────────────────────────────────────────────────
@lru_cache(maxsize=16)
def day_chunk(policy_key: str, d: int, range_start: int, with_routes: bool = True) -> dict:
    """Day `d` of the replay (0 = Aug 1), serialized for the browser. Minutes are
    relative to 00:00 that day; running totals count from 00:00 on day `range_start`."""
    import base64
    rep, trips = load(policy_key), load_trips()
    a, b = d * MINUTES_PER_DAY, (d + 1) * MINUTES_PER_DAY
    rows = slice(a, b)

    # Rides drawn on the map: served requests in progress at some minute of the day.
    sm, em = trips["start_min"], trips["end_min"]
    hi = np.searchsorted(sm, b)
    on = np.flatnonzero((em[:hi] > a) & rep["trip_started"][:hi])
    out = {
        "key": f"{policy_key}|{range_start}|{d}",
        "day": d - range_start,                    # day within the picked range
        "date": f"{day_date(d):%b} {day_date(d).day}",
        "start": str(day_date(range_start).date()),   # first picked day (slider minute 0)
        "bikes": base64.b64encode(np.ascontiguousarray(rep["bikes"][rows]).tobytes()).decode("ascii"),
        # Unrounded: rounding each minute to cents biases a day's running total by dollars.
        "reward": rep["reward"][rows].tolist(),
        **{k: rep[k][rows].tolist() for k in _PER_MINUTE if k != "reward"},
        "cum0": {k: float(rep["day_sums"][k][range_start:d].sum()) for k in _CUMULATIVE},
        "trips": {
            "sm": (sm[on] - a).tolist(), "em": (em[on] - a).tolist(),
            "s": trips["start_idx"][on].tolist(), "e": trips["end_idx"][on].tolist(),
            "casual": trips["casual"][on].tolist(), "ok": [1] * len(on),
        },
        "trucks": _trucks(rep, a, b, range_start, with_routes),
    }
    return out


def _trucks(rep, a, b, range_start, with_routes) -> dict | None:
    K = rep["truck_status"].shape[1]
    if K == 0:
        return None
    legs = []
    router = None
    if with_routes:
        from road_routes import get_router
        router = get_router()
    for k, frm, to, depart, arrive, load in rep["truck_legs"].tolist():
        if depart >= b or arrive <= a:
            continue
        if router is not None:
            lat, lon = router.route(frm, to)
            lat, lon = list(lat), list(lon)
        else:
            lat = lon = []
        legs.append({"k": k, "depart": depart - a, "arrive": arrive - a, "load": load,
                     "to": to, "lat": lat, "lon": lon})
    moved = rep["truck_moved"]
    base = int(moved[range_start * MINUTES_PER_DAY - 1]) if range_start > 0 else 0
    rows = slice(a, b)
    return {"n": K, "legs": legs,
            "status": rep["truck_status"][rows].T.tolist(),
            "load": rep["truck_load"][rows].T.tolist(),
            "station": rep["truck_station"][rows].T.tolist(),
            "moved": (moved[rows] - base).tolist()}


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("policies", nargs="*", help="policy keys (default: all)")
    args = ap.parse_args()
    keys = args.policies or list(POLICIES)
    todo = [k for k in keys if args.rebuild or not is_current(k)]
    print(f"replay {REPLAY_START} to {REPLAY_END}: {len(keys) - len(todo)} up to date, "
          f"building {len(todo)}", flush=True)
    if not todo:
        return
    # The trip log is written by the first build; do one first so workers don't race on it.
    if not os.path.exists(os.path.join(REPLAY_DIR, "_trips.npz")):
        build(todo.pop(0))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(build, todo))
    print("done")


if __name__ == "__main__":
    main()
