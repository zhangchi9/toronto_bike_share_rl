"""Zone demand forecast (round 4): rentals and returns per zone, next 60 / 180 min.

    python -m rl.forecast            # build models/forecast/zone_demand.npz + report

What is forecast: for every zone and every 15-minute slot start `m`, the number of
trips that start (rentals) and end (returns) in that zone during [m, m + H), for
H = 60 and 180 minutes. Trips are the 2026 trip log the env replays, so these are
the requests the env will see (including ones lost at empty stations).

Model: one sklearn HistGradientBoostingRegressor (Poisson loss) per target, fitted
on the training period only (May 1 - Jul 17). Inputs use only trips before `m`:
zone, slot of day, day of week, weekend/holiday flag, the zone's rentals and
returns in the last 60 min and last 24 h, the same H-minute window 1 and 7 days
earlier, and citywide rentals in the last 60 min, now and 7 days earlier (an
activity / weather proxy).

Leakage: forecasts for slots inside the training period are cross-fitted (each
training week is predicted by a model fitted on the other weeks), so the agents
never train on in-sample forecasts. Validation and test slots use the model
fitted on the whole training period.

The table covers every slot from Jan 1 to Sep 30 (index = minutes since Jan 1 // 15)
so any simulation window can look its forecast up (FeatureBuilder does).
"""
import json
import os

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
SLOT_MIN = 15
HORIZONS = (60, 180)
TARGETS = [(kind, h) for h in HORIZONS for kind in ("rentals", "returns")]  # 4 blocks
DATA_START = pd.Timestamp("2026-01-01")
DATA_END = pd.Timestamp("2026-09-30")
TRAIN_RANGE = ("2026-05-01", "2026-07-17")
VAL_RANGE = ("2026-07-18", "2026-07-31")
TEST_RANGE = ("2026-08-01", "2026-09-20")
# Ontario statutory holidays in the data period (Good Friday included).
HOLIDAYS = ("2026-01-01", "2026-02-16", "2026-04-03", "2026-05-18", "2026-07-01",
            "2026-08-03", "2026-09-07")
FORECAST_DIR = os.path.join("models", "forecast")
FORECAST_PATH = os.path.join(FORECAST_DIR, "zone_demand.npz")   # round 4 (k-means zones)


def table_path(zone_of_station: np.ndarray) -> str:
    """One table per zone layout: the round-4 file if it matches, else a file named
    after the layout (number of zones + a hash of the station -> zone map)."""
    import hashlib
    if os.path.exists(FORECAST_PATH):
        with np.load(FORECAST_PATH) as d:
            if np.array_equal(d["zone_of_station"], zone_of_station):
                return FORECAST_PATH
    z = np.asarray(zone_of_station, dtype=np.int64)
    tag = hashlib.sha1(z.tobytes()).hexdigest()[:8]
    return os.path.join(FORECAST_DIR, f"zone_demand_{z.max() + 1}z_{tag}.npz")
FEATURE_NAMES = ("zone", "slot_of_day", "day_of_week", "off_day",
                 "rentals_last60", "returns_last60", "rentals_last24h", "returns_last24h",
                 "rentals_same_window_1d", "returns_same_window_1d",
                 "rentals_same_window_7d", "returns_same_window_7d",
                 "city_rentals_last60", "city_rentals_last60_7d")


# ── data ─────────────────────────────────────────────────────────────────────
def minute_counts(zone_of_station: np.ndarray, station_ids: np.ndarray):
    """Cumulative per-minute trip starts / ends per zone, Jan 1 - Sep 30.
    Returns (C_start, C_end), each (n_minutes + 1, Z): C[m] = count before minute m."""
    from bikeshare_env.trips import load_period_trips
    trips = load_period_trips(str(DATA_START.date()), str(DATA_END.date()), station_ids)
    n_min = ((DATA_END - DATA_START).days + 1) * MINUTES_PER_DAY
    Z = int(zone_of_station.max()) + 1
    out = []
    for when, where in ((trips.start_min, trips.start_idx), (trips.end_min, trips.end_idx)):
        keep = when < n_min
        counts = np.zeros((n_min, Z), dtype=np.int32)
        np.add.at(counts, (when[keep], zone_of_station[where[keep]]), 1)
        out.append(np.vstack([np.zeros((1, Z), np.int64), np.cumsum(counts, axis=0)]))
    return out[0], out[1]


def _window(C: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Counts in [a, b) per origin and zone; NaN where the window starts before Jan 1."""
    n = C.shape[0] - 1
    ok = a >= 0
    aa, bb = np.clip(a, 0, n), np.clip(b, 0, n)
    out = (C[bb] - C[aa]).astype(np.float64)
    out[~ok] = np.nan
    return out


def features(C_start, C_end, origins: np.ndarray, H: int) -> np.ndarray:
    """Model inputs for each (origin minute, zone), using only trips before the origin.
    Rows are origin-major: row = i * Z + z. Shape (len(origins) * Z, len(FEATURE_NAMES))."""
    Z = C_start.shape[1]
    m = np.asarray(origins, dtype=np.int64)
    dates = DATA_START + pd.to_timedelta(m // MINUTES_PER_DAY, unit="D")
    dow = dates.dayofweek.to_numpy()
    off = ((dow >= 5) | np.isin(dates.strftime("%Y-%m-%d"), HOLIDAYS)).astype(float)
    col = lambda v: np.repeat(np.asarray(v, dtype=np.float64), Z)       # per origin
    W = lambda C, a, b: _window(C, a, b).reshape(-1)                     # per origin x zone
    city_s = C_start.sum(1, keepdims=True)
    D, WK = MINUTES_PER_DAY, 7 * MINUTES_PER_DAY
    cols = [
        np.tile(np.arange(Z, dtype=np.float64), len(m)),
        col((m % D) // SLOT_MIN), col(dow), col(off),
        W(C_start, m - 60, m), W(C_end, m - 60, m),
        W(C_start, m - D, m), W(C_end, m - D, m),
        W(C_start, m - D, m - D + H), W(C_end, m - D, m - D + H),
        W(C_start, m - WK, m - WK + H), W(C_end, m - WK, m - WK + H),
        col(_window(city_s, m - 60, m)[:, 0]), col(_window(city_s, m - WK - 60, m - WK)[:, 0]),
    ]
    return np.column_stack(cols)


def targets(C, origins: np.ndarray, H: int) -> np.ndarray:
    m = np.asarray(origins, dtype=np.int64)
    return _window(C, m, m + H).reshape(-1)


def _minute(day: str) -> int:
    return (pd.Timestamp(day) - DATA_START).days * MINUTES_PER_DAY


def origins_in(first_day: str, last_day: str, H: int = 0) -> np.ndarray:
    """Slot starts on [first_day, last_day] whose H-minute window ends by last_day's end."""
    end = _minute(last_day) + MINUTES_PER_DAY
    return np.arange(_minute(first_day), end - H + 1, SLOT_MIN)


# ── model ────────────────────────────────────────────────────────────────────
def new_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=[0],
        random_state=0)


def build(zone_of_station: np.ndarray, station_ids: np.ndarray, path: str = FORECAST_PATH,
          verbose: bool = True) -> dict:
    """Fit the forecast on the training period, write the slot table, return metrics."""
    C_start, C_end = minute_counts(zone_of_station, station_ids)
    Z = C_start.shape[1]
    n_slots = (C_start.shape[0] - 1) // SLOT_MIN
    all_origins = np.arange(n_slots) * SLOT_MIN
    train_lo, train_hi = _minute(TRAIN_RANGE[0]), _minute(TRAIN_RANGE[1]) + MINUTES_PER_DAY
    in_train = (all_origins >= train_lo) & (all_origins < train_hi)
    week_of = (all_origins - train_lo) // (7 * MINUTES_PER_DAY)     # cross-fit folds

    table = np.zeros((n_slots, Z, len(TARGETS)), dtype=np.float32)
    metrics = {}
    for j, (kind, H) in enumerate(TARGETS):
        C = C_start if kind == "rentals" else C_end
        # Training rows: windows that end inside the training period.
        tr = all_origins[in_train & (all_origins + H <= train_hi)]
        X_tr, y_tr = features(C_start, C_end, tr, H), targets(C, tr, H)
        w_tr = np.repeat(week_of[tr // SLOT_MIN], Z)

        full = new_model().fit(X_tr, y_tr)
        table[:, :, j] = full.predict(features(C_start, C_end, all_origins, H)).reshape(n_slots, Z)

        # Cross-fitted forecasts for every slot inside the training period.
        for wk in np.unique(w_tr):
            m_ = new_model().fit(X_tr[w_tr != wk], y_tr[w_tr != wk])
            slots = np.flatnonzero(in_train & (week_of == wk))
            table[slots, :, j] = m_.predict(
                features(C_start, C_end, all_origins[slots], H)).reshape(len(slots), Z)

        metrics[f"{kind}_{H}"] = evaluate(table[:, :, j], C_start, C_end, kind, H)
        if verbose:
            v, t = metrics[f"{kind}_{H}"]["validation"], metrics[f"{kind}_{H}"]["test"]
            print(f"  {kind:7} next {H:3} min: WAPE val {v['model']:.3f} "
                  f"(profile {v['profile']:.3f}, last week {v['last_week']:.3f}), "
                  f"test {t['model']:.3f}")

    table = np.maximum(table, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, table=table, slot_min=SLOT_MIN,
                        data_start=str(DATA_START.date()),
                        targets=np.array([f"{k}_{h}" for k, h in TARGETS]),
                        zone_of_station=zone_of_station)
    metrics_path = (os.path.join(os.path.dirname(path), "metrics.json") if path == FORECAST_PATH
                    else path[:-4] + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def evaluate(pred: np.ndarray, C_start, C_end, kind: str, H: int) -> dict:
    """WAPE (sum |error| / sum actual) of the model and two baselines, per period:
    'profile' = training-period mean for the zone, slot of day and weekday/off-day;
    'last_week' = the same window 7 days earlier."""
    C = C_start if kind == "rentals" else C_end
    Z = C.shape[1]
    tr = origins_in(*TRAIN_RANGE, H)
    Xtr = features(C_start, C_end, tr, H)
    prof = pd.DataFrame({"z": Xtr[:, 0], "s": Xtr[:, 1], "o": Xtr[:, 3],
                         "y": targets(C, tr, H)}).groupby(["z", "s", "o"])["y"].mean()
    out = {}
    for name, rng in (("validation", VAL_RANGE), ("test", TEST_RANGE)):
        o = origins_in(*rng, H)
        X, y = features(C_start, C_end, o, H), targets(C, o, H)
        p = pred[o // SLOT_MIN].reshape(-1)
        base = prof.reindex(pd.MultiIndex.from_arrays([X[:, 0], X[:, 1], X[:, 3]])).to_numpy()
        lw = X[:, FEATURE_NAMES.index(f"{kind}_same_window_7d")]
        wape = lambda q: float(np.abs(y - q).sum() / y.sum())
        out[name] = {"model": wape(p), "profile": wape(base), "last_week": wape(lw),
                     "mean_actual": float(y.mean()), "mean_forecast": float(p.mean())}
    return out


# ── lookup used by the RL features ───────────────────────────────────────────
_cache = {}


def load_table(zone_of_station: np.ndarray, station_ids: np.ndarray) -> np.ndarray:
    """(n_slots, Z, 4) forecast table for this zone layout; built on first use."""
    path = table_path(zone_of_station)
    if path not in _cache:
        if not os.path.exists(path):
            build(zone_of_station, station_ids, path=path)
        d = np.load(path)
        if not np.array_equal(d["zone_of_station"], zone_of_station):
            raise ValueError(f"{path} was built for other zones; delete it to rebuild")
        _cache[path] = d["table"]
    return _cache[path]


def slot_index(start_date: pd.Timestamp, t: int) -> int:
    """Forecast slot for env minute t of a simulation starting at start_date."""
    return ((pd.Timestamp(start_date) - DATA_START).days * MINUTES_PER_DAY + int(t)) // SLOT_MIN


if __name__ == "__main__":
    from rl.wrapper import static_data
    s = static_data()
    path = table_path(s["zones"].zone_of_station)
    print("building", path)
    build(s["zones"].zone_of_station, s["ids"], path=path)
