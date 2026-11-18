"""Build a static copy of the dashboard (no server) for GitHub Pages.

    python -m deploy.build_static          # -> deploy/static_site/

The page (deploy/static/: index.html, app.js, style.css) replays the same saved runs
as the Dash app (dashboard_data), with the same map drawing code, but everything is
read from files:

    data/meta.json.gz          policies, dates, per-day totals (for running totals)
    data/stations.json.gz      station positions, names, docks (the replays' order)
    data/zones.json.gz         the zone overlay
    data/routes.json.gz        road route of every truck leg, "frm,to" -> lats, lons
    data/trips/<d>.bin.gz      rides in progress on day d (the same for every policy)
    data/<policy>/<d>.bin.gz   one policy's day d: bikes per station, per-minute
                               metrics, which rides were served, trucks

Day d counts from Aug 1 (d = 0). A .bin.gz file is gzip of: uint32 header length,
a JSON header {"info": ..., "arrays": {name: [numpy dtype, byte offset, size]}}, then
the arrays (offsets from the end of the header, every array 8-byte aligned).

Every replay must be built and up to date (python -m dashboard_data) unless
--allow-missing is given. See deploy/README.md for publishing the folder.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "static")
OUT = os.path.join(HERE, "static_site")
M = 1440                                    # minutes per day
_CUMULATIVE = ("reward", "fulfilled", "lost", "blocked")
_COUNTS = ("fulfilled", "lost", "active", "blocked", "pending")


def _gz(path: str, raw: bytes) -> None:
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9,
                                              mtime=0) as g:   # same bytes on every build
        g.write(raw)


def write_json(path: str, obj) -> None:
    _gz(path, json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def write_bin(path: str, info: dict, arrays: dict) -> None:
    meta, parts, off = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        b = a.tobytes()
        pad = -len(b) % 8
        meta[name] = [a.dtype.str, off, int(a.size)]
        parts.append(b + b"\0" * pad)
        off += len(b) + pad
    head = json.dumps({"info": info, "arrays": meta}, separators=(",", ":")).encode()
    head += b" " * (-(len(head) + 4) % 8)          # arrays start 8-byte aligned
    _gz(path, struct.pack("<I", len(head)) + head + b"".join(parts))


def _small_int(a: np.ndarray) -> np.ndarray:
    """The smallest of int16 / int32 that holds `a`."""
    return a.astype(np.int16 if a.size == 0 or (a.min() >= -32768 and a.max() <= 32767)
                    else np.int32)


def trips_day(trips: dict, d: int) -> np.ndarray:
    """Indices of the rides in progress at some minute of day d (as dashboard_data)."""
    a, b = d * M, (d + 1) * M
    hi = np.searchsorted(trips["start_min"], b)
    return np.flatnonzero(trips["end_min"][:hi] > a)


def policy_day(rep: dict, on: np.ndarray, d: int) -> tuple[dict, dict]:
    a, b = d * M, (d + 1) * M
    bikes = rep["bikes"][a:b]
    changed = bikes[1:] != bikes[:-1]               # station changes from minute m-1 to m
    count = np.zeros(M, np.uint16)
    count[1:] = changed.sum(1)
    mins, st = np.nonzero(changed)
    arrays = {
        "bikes0": bikes[0].astype(np.uint8),
        "chg_count": count,
        "chg_station": st.astype(np.uint16),
        "chg_value": bikes[1:][mins, st].astype(np.uint8),
        "reward": rep["reward"][a:b].astype(np.float64),
        **{k: rep[k][a:b].astype(np.int32) for k in _COUNTS},
        "trip_ok": rep["trip_started"][on].astype(np.uint8),
    }
    K = rep["truck_status"].shape[1]
    info = {"day": d, "n_trucks": int(K)}
    if K:
        legs = rep["truck_legs"]
        legs = legs[(legs[:, 3] < b) & (legs[:, 4] > a)].astype(np.int64)
        legs[:, 3:5] -= a                            # depart / arrive: minutes from 00:00
        arrays.update({
            "truck_status": rep["truck_status"][a:b].T.astype(np.int8),     # [k, minute]
            "truck_load": rep["truck_load"][a:b].T.astype(np.int16),
            "truck_station": rep["truck_station"][a:b].T.astype(np.uint16),
            "truck_moved": rep["truck_moved"][a:b].astype(np.int32),        # since Aug 1
            "legs": legs.astype(np.int32),           # k, frm, to, depart, arrive, load
        })
    return info, arrays


def _commit() -> str:
    try:
        # e.g. "2988576", or "2988576-dirty" when built from uncommitted changes
        return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--allow-missing", action="store_true",
                    help="leave out replays that aren't built yet (for a trial build)")
    args = ap.parse_args()

    import dashboard_data as dd
    import rl.policy as rp
    from data_loader import load_stations
    if dd.VIEW_ONLY:
        raise SystemExit("unset DASHBOARD_VIEW_ONLY: the build reads the full project")
    keys = ["no_rebalancing", *rp.DASHBOARD_POLICIES]
    missing = [k for k in keys if k not in dd.POLICIES or not dd.is_current(k)]
    if missing and not args.allow_missing:
        raise SystemExit(f"replays not built or out of date: {missing}; run "
                         "python -m dashboard_data (or pass --allow-missing for a trial)")
    keys = [k for k in keys if k not in missing]

    out = args.out
    if os.path.exists(out):             # start clean, but keep a git checkout of the site
        for name in os.listdir(out):
            if name != ".git":
                p = os.path.join(out, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    data = os.path.join(out, "data")
    os.makedirs(os.path.join(data, "trips"))

    # The page, the map's truck icons and Plotly (the copy the plotly package ships).
    stamp = _commit()
    for name in os.listdir(SRC):
        shutil.copy2(os.path.join(SRC, name), out)
    shutil.copytree(os.path.join("assets", "truck_sprite"),
                    os.path.join(out, "assets", "truck_sprite"))
    import plotly
    shutil.copy2(os.path.join(os.path.dirname(plotly.__file__), "package_data",
                              "plotly.min.js"), out)
    index = os.path.join(out, "index.html")     # a file's URL changes when its content
    with open(index, encoding="utf-8") as f:    # does: browsers never keep a stale copy
        html = f.read()
    for name in ("app.js", "style.css", "plotly.min.js"):
        with open(os.path.join(out, name), "rb") as f:
            h = hashlib.sha1(f.read()).hexdigest()[:10]
        html = html.replace(f'"{name}"', f'"{name}?v={h}"')
    with open(index, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    open(os.path.join(out, ".nojekyll"), "w").close()     # serve files as they are

    sdf = load_stations()
    write_json(os.path.join(data, "stations.json.gz"),
               {"lat": sdf["lat"].round(6).tolist(), "lon": sdf["lon"].round(6).tolist(),
                "name": sdf["name"].tolist(), "cap": sdf["capacity"].astype(int).tolist()})
    import app                                  # the zone overlay, as the full app draws it
    write_json(os.path.join(data, "zones.json.gz"), app.zone_store())
    print("road routes of every truck leg ...", flush=True)
    routes = {k: [np.round(v, 5).tolist() for v in ll] for k, ll in dd.leg_routes(keys).items()}
    write_json(os.path.join(data, "routes.json.gz"), routes)

    trips, n_days = dd.load_trips(), dd.n_days()
    on_by_day = [trips_day(trips, d) for d in range(n_days)]
    for d, on in enumerate(on_by_day):
        a = d * M
        write_bin(os.path.join(data, "trips", f"{d}.bin.gz"), {"day": d}, {
            "sm": _small_int(trips["start_min"][on].astype(np.int64) - a),
            "em": _small_int(trips["end_min"][on].astype(np.int64) - a),
            "s": trips["start_idx"][on].astype(np.uint16),
            "e": trips["end_idx"][on].astype(np.uint16),
            "casual": trips["casual"][on].astype(np.uint8)})
    sm = trips["start_min"]
    requests = np.diff(np.searchsorted(sm, np.arange(n_days + 1) * M)).tolist()

    meta_pol = {}
    for key in keys:
        rep = dd.load(key)
        os.makedirs(os.path.join(data, key))
        for d, on in enumerate(on_by_day):
            info, arrays = policy_day(rep, on, d)
            write_bin(os.path.join(data, key, f"{d}.bin.gz"), info, arrays)
        moved = rep["truck_moved"]
        meta_pol[key] = {
            "label": dd.POLICIES[key].label,
            "n_trucks": int(rep["truck_status"].shape[1]),
            "day_sums": {k: rep["day_sums"][k].tolist() for k in _CUMULATIVE},
            # bikes moved by trucks since Aug 1, at the end of each day
            "moved_end": (moved.reshape(-1, M)[:, -1].tolist() if len(moved) else []),
        }
        print(f"  {key}: {n_days} days", flush=True)
    write_json(os.path.join(data, "meta.json.gz"), {
        "start": dd.REPLAY_START, "end": dd.REPLAY_END, "n_days": n_days,
        "n_stations": len(sdf), "requests": requests, "order": keys,
        "policies": meta_pol, "commit": stamp})

    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs
               if ".git" not in os.path.relpath(r, out).split(os.sep))
    print(f"wrote {out}: {len(keys)} solutions ({', '.join(keys)}), {size / 1e6:.0f} MB")
    if missing:
        print(f"left out (not built yet): {missing}")


if __name__ == "__main__":
    main()
