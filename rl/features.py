"""Compact observation + zone-level action decoding for BikeShareEnv.

Shared by the training wrapper (rl/wrapper.py) and the dashboard adapter
(rl/policy.py) so a trained model sees exactly the same inputs in both.

Action (Discrete(1 + Z*(Z-1)*len(QTY_LEVELS)), Z = zones trucks can reach):
    0                   wait
    1 + i               i -> (source zone, destination zone, qty level)
Observation (float32): per zone [fill, share empty, share full, 60-min net change],
(round 4, forecast=True) per zone forecast [rentals, returns] in the next 60 and
180 min per dock (rl/forecast.py), time of day (sin, cos), day of week one-hot,
the asked truck (zone one-hot, load), then every other truck (zone one-hot, busy,
minutes left).
Round 9 adds optional feature groups, appended in FEATURE_GROUPS order, per zone:
  risk      share of stations expected to be empty / full in 1 h and 2 h (bikes now +
            the station's average hourly net flow in the training period), and the
            bikes a truck could pick up / docks it could fill at <= 5 per station
  travel    road minutes from the asked truck to the zone's stations (mean)
  failures  lost requests and blocked returns in the last 60 min (log scale)
  memory    minutes since the asked truck last picked up from / delivered to the zone
"""
from collections import deque

import numpy as np
import pandas as pd

from bikeshare_env.env import MINUTES_PER_DAY, BikeShareEnv
from rl import forecast as fc
from rl.zones import Zones

QTY_LEVELS = (10, 20, 40)
TRUCK_CAPACITY = 40          # bikes per truck for the RL / zone policies
MAX_BIKES_PER_STOP = 5       # multi-stop missions (round 3): bikes per station visit
HISTORY_MIN = 60
NET_CHANGE_SCALE = 10.0

FEATURE_GROUPS = ("risk", "travel", "failures", "memory")
GROUP_SIZE = {"risk": 6, "travel": 1, "failures": 2, "memory": 2}   # values per zone
RISK_HOURS = 2                # risk: look 1 and 2 hours ahead
TRAVEL_SCALE = 60.0           # travel: minutes, 1.0 = an hour away
FAIL_LOG_MAX = np.log1p(100)  # failures: log1p(count) / log1p(100), 1.0 = 100 an hour
MEMORY_MIN = 360              # memory: 1.0 = 6 h ago or never
# Round 9 champion inputs (DQN, 1 truck, zones flow_w0.3_k20_need): all four groups.
CHAMPION_FEATURES = "+".join(FEATURE_GROUPS)


def parse_features(spec) -> tuple[str, ...]:
    """None / "" / "none" -> (); "risk+travel" or ["travel", "risk"] -> canonical order."""
    if spec is None or (isinstance(spec, str) and spec in ("", "none", "base")):
        return ()
    groups = spec.replace(",", "+").split("+") if isinstance(spec, str) else list(spec)
    bad = set(groups) - set(FEATURE_GROUPS)
    if bad:
        raise ValueError(f"unknown feature groups {sorted(bad)}; choose from {FEATURE_GROUPS}")
    return tuple(g for g in FEATURE_GROUPS if g in groups)


def features_name(spec) -> str:
    return "+".join(parse_features(spec)) or "base"


_cache: dict = {}


def _profiles_bikes(station_ids: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """(N, 48) average net bikes per hour per station (training period), cached."""
    key = ("P", station_ids.tobytes(), np.asarray(capacity).tobytes())
    if key not in _cache:
        from rl.zones import demand_profiles
        cap = np.asarray(capacity, float)
        _cache[key] = demand_profiles(station_ids, cap) * cap[:, None]
    return _cache[key]


def _zone_travel(env: BikeShareEnv, zones: Zones, truck_stations) -> np.ndarray:
    """(N, Z) mean road minutes from each station to the zone's truck stations."""
    key = ("T", zones.zone_of_station.tobytes(), env.station_ids.tobytes())
    if key not in _cache:
        T = env.time_min
        out = np.ones((len(T), zones.n_zones)) * 1e3
        for z, st in enumerate(truck_stations):
            if len(st):
                out[:, z] = T[:, st].mean(1)
        _cache[key] = out
    return _cache[key]


def n_actions(n_zones: int) -> int:
    """Wait + every (source, destination, load) over `n_zones` truck-reachable zones."""
    return 1 + n_zones * (n_zones - 1) * len(QTY_LEVELS)


def truck_zones(zones: Zones, station_ids: np.ndarray) -> np.ndarray:
    """Zones with at least one station trucks can serve, in zone order. Actions are
    over these only: an island-only zone (round 5) would just add dead actions."""
    from bikeshare_env.env import ISLAND_STATIONS
    ok = ~np.isin(station_ids, ISLAND_STATIONS)
    return np.array([z for z, idx in enumerate(zones.stations_in_zone) if ok[idx].any()])


def encode_action(src_zone: int, dst_zone: int, qty: int, n_zones: int) -> int:
    """Inverse of FeatureBuilder.decode's index mapping (src != dst)."""
    pair = src_zone * (n_zones - 1) + (dst_zone if dst_zone < src_zone else dst_zone - 1)
    return 1 + pair * len(QTY_LEVELS) + QTY_LEVELS.index(qty)


def obs_size(n_zones: int, n_trucks: int, forecast: bool = True, features=()) -> int:
    n_fc = len(fc.TARGETS) * n_zones if forecast else 0
    n_extra = n_zones * sum(GROUP_SIZE[g] for g in parse_features(features))
    return (4 * n_zones + n_fc + 2 + 7 + (n_zones + 1) + (n_trucks - 1) * (n_zones + 2)
            + n_extra)


class FeatureBuilder:
    """Holds the zone history for one env episode; builds obs and decodes actions."""

    def __init__(self, env: BikeShareEnv, zones: Zones, forecast: bool = True, features=()):
        self.env = env
        self.z = zones
        self.forecast = forecast
        self.features = parse_features(features)
        # Demand forecast table (slot, zone, target), looked up by absolute time.
        self.fc_table = fc.load_table(zones.zone_of_station, env.station_ids) if forecast else None
        self.fc_slot0 = fc.slot_index(env.start_date, 0) * fc.SLOT_MIN   # env minute 0, absolute
        Z = zones.n_zones
        self.zone_cap = np.bincount(zones.zone_of_station, weights=env.capacity, minlength=Z)
        self.n_in_zone = np.bincount(zones.zone_of_station, minlength=Z).astype(float)
        # Truck-usable stations per zone (no ferry-only stations); action zone index ->
        # zone id (all zones under k-means; every zone but the island in round 5).
        self.truck_stations = [s[~env.truck_blocked[s]] for s in zones.stations_in_zone]
        self.action_zones = np.array([z for z in range(Z) if len(self.truck_stations[z])])
        self.history = deque(maxlen=HISTORY_MIN + 1)
        if "risk" in self.features:
            self.flow = _profiles_bikes(env.station_ids, env.capacity)
            self.truck_ok = (~env.truck_blocked).astype(float)
            n_days = env.total_minutes // MINUTES_PER_DAY + 2
            days = pd.date_range(env.start_date, periods=n_days)
            self.off_day = np.asarray((days.dayofweek >= 5)
                                      | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
        if "travel" in self.features:
            self.travel = _zone_travel(env, zones, self.truck_stations)
        self.failures = deque(maxlen=HISTORY_MIN)

    def reset(self) -> None:
        self.history.clear()
        self.failures.clear()
        self._trip_ptr = self.env.trip_ptr
        K, Z = max(1, len(self.env.trucks)), self.z.n_zones
        self.last_pick = np.full((K, Z), -np.inf)
        self.last_drop = np.full((K, Z), -np.inf)
        self.record()

    def record(self) -> None:
        """Call once per simulated minute (after the minute) to track net change."""
        self.history.append(self._zone_bikes())
        if "failures" in self.features:
            env, zof, Z = self.env, self.z.zone_of_station, self.z.n_zones
            i0, i1 = self._trip_ptr, env.trip_ptr
            lost = env.trips.start_idx[i0:i1][~env.trip_started[i0:i1]]
            self._trip_ptr = i1
            # Returns that found their station full in the minute just simulated wait
            # in env.overflow under (that minute + overflow delay).
            blocked = [st for st, _ in env.overflow.get(env.t - 1 + env.overflow_delay_min, ())]
            self.failures.append(np.stack([
                np.bincount(zof[lost], minlength=Z),
                np.bincount(zof[np.asarray(blocked, dtype=np.int64)], minlength=Z)]))

    def note(self, truck_k: int, row: np.ndarray) -> None:
        """Remember an order given to truck k (decoded row) for the memory features."""
        if row.any():
            zof, t = self.z.zone_of_station, self.env.t
            self.last_pick[truck_k, zof[row[0] - 1]] = t
            self.last_drop[truck_k, zof[row[1]]] = t

    def _zone_bikes(self) -> np.ndarray:
        return np.bincount(self.z.zone_of_station, weights=self.env.bikes,
                           minlength=self.z.n_zones)

    def obs(self, truck_k: int) -> np.ndarray:
        env, Z, zof = self.env, self.z.n_zones, self.z.zone_of_station
        bikes = self._zone_bikes()
        fill = bikes / np.maximum(self.zone_cap, 1)
        empty = np.bincount(zof, weights=(env.bikes == 0), minlength=Z) / self.n_in_zone
        full = np.bincount(zof, weights=(env.bikes >= env.capacity), minlength=Z) / self.n_in_zone
        net = (bikes - self.history[0]) / np.maximum(self.zone_cap, 1) * NET_CHANGE_SCALE

        t = min(env.t, env.total_minutes - 1)
        mod = t % MINUTES_PER_DAY
        ang = 2 * np.pi * mod / MINUTES_PER_DAY
        dow = np.zeros(7)
        dow[(env.start_date.dayofweek + t // MINUTES_PER_DAY) % 7] = 1

        parts = [fill, empty, full, np.clip(net, -1, 1)]
        if self.forecast:          # forecast trips per dock, one block per target
            f = self.fc_table[(self.fc_slot0 + t) // fc.SLOT_MIN] / np.maximum(self.zone_cap, 1)[:, None]
            parts += list(np.clip(f, 0, 1).T)
        parts += [[np.sin(ang), np.cos(ang)], dow]
        me = env.trucks[truck_k]
        parts += [np.eye(Z)[zof[me.location]], [me.load / env.truck_capacity]]
        for j, tk in enumerate(env.trucks):
            if j == truck_k:
                continue
            left = min(1.0, max(0, tk.busy_until - env.t) / 60.0)
            parts += [np.eye(Z)[zof[tk.location]], [float(tk.busy(env.t)), left]]
        if "risk" in self.features:
            parts += self._risk(t)
        if "travel" in self.features:
            parts += [np.clip(self.travel[me.location] / TRAVEL_SCALE, 0, 1)]
        if "failures" in self.features:
            f = np.sum(self.failures, axis=0) if self.failures else np.zeros((2, Z))
            parts += list(np.clip(np.log1p(f) / FAIL_LOG_MAX, 0, 1))
        if "memory" in self.features:
            for last in (self.last_pick[truck_k], self.last_drop[truck_k]):
                parts += [np.clip((env.t - last) / MEMORY_MIN, 0, 1)]
        return np.concatenate(parts).astype(np.float32)

    def _risk(self, t: int) -> list[np.ndarray]:
        env, Z, zof = self.env, self.z.n_zones, self.z.zone_of_station
        bikes, cap = env.bikes.astype(float), env.capacity.astype(float)
        d, h = t // MINUTES_PER_DAY, (t % MINUTES_PER_DAY) // 60
        proj, out = bikes.copy(), []
        empty, full = [], []
        for i in range(RISK_HOURS):
            hh = h + i
            col = hh % 24 + 24 * int(self.off_day[min(d + hh // 24, len(self.off_day) - 1)])
            proj = proj + self.flow[:, col]
            empty.append(np.bincount(zof, weights=proj <= 0, minlength=Z) / self.n_in_zone)
            full.append(np.bincount(zof, weights=proj >= cap, minlength=Z) / self.n_in_zone)
        stop = MAX_BIKES_PER_STOP
        pick = np.bincount(zof, weights=np.minimum(bikes, stop) * self.truck_ok, minlength=Z)
        drop = np.bincount(zof, weights=np.minimum(cap - bikes, stop) * self.truck_ok,
                           minlength=Z)
        out += empty + full
        out += [np.clip(pick / TRUCK_CAPACITY, 0, 1), np.clip(drop / TRUCK_CAPACITY, 0, 1)]
        return out

    def decode(self, action: int) -> np.ndarray:
        """Discrete action -> env action row [pickup + 1, dropoff, qty] (zeros = wait)."""
        a = int(action)
        if a <= 0:
            return np.zeros(3, dtype=np.int64)
        Z, nq = len(self.action_zones), len(QTY_LEVELS)
        a -= 1
        q, pair = QTY_LEVELS[a % nq], a // nq
        src, dj = divmod(pair, Z - 1)
        dst = dj if dj < src else dj + 1
        src, dst = int(self.action_zones[src]), int(self.action_zones[dst])

        env = self.env
        cand = self.truck_stations[src]
        if not len(cand) or env.bikes[cand].max() == 0:
            return np.zeros(3, dtype=np.int64)
        pickup = int(cand[np.argmax(env.bikes[cand])])

        cand = self.truck_stations[dst]
        free = env.capacity[cand] - env.bikes[cand]
        cand = cand[free > 0]
        if not len(cand):
            return np.zeros(3, dtype=np.int64)
        if getattr(env, "multi_stop", False):     # most free docks (multi-stop rule)
            dropoff = int(cand[np.argmax(env.capacity[cand] - env.bikes[cand])])
        else:                                     # lowest fill ratio (single stop)
            dropoff = int(cand[np.argmin(env.bikes[cand] / env.capacity[cand])])
        return np.array([pickup + 1, dropoff, q], dtype=np.int64)
