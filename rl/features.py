"""Compact observation + zone-level action decoding for BikeShareEnv.

Shared by the training wrapper (rl/wrapper.py) and the dashboard adapter
(rl/policy.py) so a trained model sees exactly the same inputs in both.

Action (Discrete(1 + Z*(Z-1)*len(QTY_LEVELS))):
    0                   wait
    1 + i               i -> (source zone, destination zone, qty level)
Observation (float32): per zone [fill, share empty, share full, 60-min net change],
time of day (sin, cos), day of week one-hot, the asked truck (zone one-hot, load),
then every other truck (zone one-hot, busy, minutes left).
"""
from collections import deque

import numpy as np

from bikeshare_env.env import MINUTES_PER_DAY, BikeShareEnv
from rl.zones import Zones

QTY_LEVELS = (5, 10, 20)
HISTORY_MIN = 60
NET_CHANGE_SCALE = 10.0


def n_actions(n_zones: int) -> int:
    return 1 + n_zones * (n_zones - 1) * len(QTY_LEVELS)


def encode_action(src_zone: int, dst_zone: int, qty: int, n_zones: int) -> int:
    """Inverse of FeatureBuilder.decode's index mapping (src != dst)."""
    pair = src_zone * (n_zones - 1) + (dst_zone if dst_zone < src_zone else dst_zone - 1)
    return 1 + pair * len(QTY_LEVELS) + QTY_LEVELS.index(qty)


def obs_size(n_zones: int, n_trucks: int) -> int:
    return 4 * n_zones + 2 + 7 + (n_zones + 1) + (n_trucks - 1) * (n_zones + 2)


class FeatureBuilder:
    """Holds the zone history for one env episode; builds obs and decodes actions."""

    def __init__(self, env: BikeShareEnv, zones: Zones):
        self.env = env
        self.z = zones
        Z = zones.n_zones
        self.zone_cap = np.bincount(zones.zone_of_station, weights=env.capacity, minlength=Z)
        self.n_in_zone = np.bincount(zones.zone_of_station, minlength=Z).astype(float)
        # Truck-usable stations per zone (no ferry-only stations).
        self.truck_stations = [s[~env.truck_blocked[s]] for s in zones.stations_in_zone]
        self.history = deque(maxlen=HISTORY_MIN + 1)

    def reset(self) -> None:
        self.history.clear()
        self.record()

    def record(self) -> None:
        """Call once per simulated minute (after the minute) to track net change."""
        self.history.append(self._zone_bikes())

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

        parts = [fill, empty, full, np.clip(net, -1, 1), [np.sin(ang), np.cos(ang)], dow]
        me = env.trucks[truck_k]
        parts += [np.eye(Z)[zof[me.location]], [me.load / env.truck_capacity]]
        for j, tk in enumerate(env.trucks):
            if j == truck_k:
                continue
            left = min(1.0, max(0, tk.busy_until - env.t) / 60.0)
            parts += [np.eye(Z)[zof[tk.location]], [float(tk.busy(env.t)), left]]
        return np.concatenate(parts).astype(np.float32)

    def decode(self, action: int) -> np.ndarray:
        """Discrete action -> env action row [pickup + 1, dropoff, qty] (zeros = wait)."""
        a = int(action)
        if a <= 0:
            return np.zeros(3, dtype=np.int64)
        Z, nq = self.z.n_zones, len(QTY_LEVELS)
        a -= 1
        q, pair = QTY_LEVELS[a % nq], a // nq
        src, dj = divmod(pair, Z - 1)
        dst = dj if dj < src else dj + 1

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
        dropoff = int(cand[np.argmin(env.bikes[cand] / env.capacity[cand])])
        return np.array([pickup + 1, dropoff, q], dtype=np.int64)
