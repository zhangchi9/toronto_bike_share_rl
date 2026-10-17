"""Bike Share Toronto rebalancing environment.

Customer demand is replayed minute by minute from the real 2026 trip log; the
policy dispatches trucks to move bikes between stations.

Reward per step = unlock fees + return fees + overflow return fees
                  - lost-trip penalties - labor - fuel - truck fixed cost.
"""
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from bikeshare_env.road_network import haversine_km, load_road_matrices
from bikeshare_env.trips import TripEvents, load_period_trips

MINUTES_PER_DAY = 1440
DATA_FIRST_DATE = pd.Timestamp("2026-01-01")
DATA_LAST_DATE = pd.Timestamp("2026-09-30")
TORONTO_CENTER = (43.655, -79.38)
# ceil(capacity * 0.35) per station = 7,824 bikes (37.7% of 20,780 docks), matching
# live GBFS at midnight (~7.9k rentable of ~8.8k docked) and ~6-10k bikes/day in
# the summer 2026 trip data. 0.5 would start with 10,856 bikes, ~25% too many.
DEFAULT_INIT_OCCUPANCY = 0.35
# Toronto Island stations: ferry-only, no road access for trucks.
ISLAND_STATIONS = (8027, 8028, 8029, 8030, 8070, 8228, 8229)

_COST_KEYS = ("unlock_revenue", "return_revenue", "overflow_revenue", "lost_penalty",
              "labor_cost", "fuel_cost", "fixed_cost")
_COUNT_KEYS = ("trips_started", "lost_trips", "returns_docked", "overflow_returns",
               "overflow_placed", "bikes_moved", "truck_km")


@dataclass
class _Truck:
    location: int
    load: int = 0
    busy_until: int = 0                       # minute the current op finishes
    ops: deque = field(default_factory=deque) # remaining ops of the mission
    pending_qty: int = 0                      # requested qty for the load op
    last_paid_day: int = -1                   # day the shift + fixed cost was paid

    def busy(self, t: int) -> bool:
        return self.busy_until > t or bool(self.ops)


class BikeShareEnv(gym.Env):
    """See module docstring. All constructor args can be passed via gym.make."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        init_occupancy: float = DEFAULT_INIT_OCCUPANCY,
        simulation_period: tuple[str, str] = ("2026-06-15", "2026-06-15"),
        step_minutes: int = 1,
        # pricing
        unlock_fee: float = 1.0,
        per_min_fee: float = 0.15,
        overflow_per_min_fee: float = 0.10,
        overflow_delay_min: int = 15,
        lost_trip_penalty: float = 0.5,
        # fleet / costs
        n_trucks: int = 2,
        truck_capacity: int = 20,
        load_min_per_bike: float = 1.0,
        depot_station: int | None = None,
        truck_blocked_stations: tuple[int, ...] = ISLAND_STATIONS,
        shift: tuple[int, int] = (9, 17),
        day_hours: tuple[int, int] = (9, 17),
        day_rate: float = 20.0,
        night_rate: float = 40.0,
        truck_fixed_cost: float = 100.0,
        fuel_cost_per_km: float = 0.3,
        # injected data (tests); loaded from disk when None
        stations: pd.DataFrame | None = None,
        trips: TripEvents | None = None,
        road_dist_km: np.ndarray | None = None,
        road_time_min: np.ndarray | None = None,
    ):
        super().__init__()
        if not 0.0 <= init_occupancy <= 1.0:
            raise ValueError("init_occupancy must be in [0, 1]")
        start, end = (pd.Timestamp(d) for d in simulation_period)
        if end < start:
            raise ValueError("simulation_period end is before start")
        if trips is None and (start < DATA_FIRST_DATE or end > DATA_LAST_DATE):
            raise ValueError(f"simulation_period must be within "
                             f"{DATA_FIRST_DATE.date()}..{DATA_LAST_DATE.date()}")

        self.init_occupancy = init_occupancy
        self.start_date = start
        self.n_days = (end - start).days + 1
        self.total_minutes = self.n_days * MINUTES_PER_DAY
        self.step_minutes = int(step_minutes)

        self.unlock_fee = unlock_fee
        self.per_min_fee = per_min_fee
        self.overflow_per_min_fee = overflow_per_min_fee
        self.overflow_delay_min = int(overflow_delay_min)
        self.lost_trip_penalty = lost_trip_penalty

        self.n_trucks = int(n_trucks)
        self.truck_capacity = int(truck_capacity)
        self.load_min_per_bike = load_min_per_bike
        self.shift = shift
        self.day_hours = day_hours
        self.day_rate = day_rate
        self.night_rate = night_rate
        self.truck_fixed_cost = truck_fixed_cost
        self.fuel_cost_per_km = fuel_cost_per_km

        # ── static data ──────────────────────────────────────────────────────
        if stations is None:
            from data_loader import load_stations
            stations = load_stations()
        self.stations = stations.reset_index(drop=True)
        self.station_ids = self.stations["station_id"].to_numpy(dtype=np.int64)
        self.capacity = self.stations["capacity"].to_numpy(dtype=np.int64)
        self.n_stations = len(self.stations)

        if road_dist_km is None or road_time_min is None:
            road_dist_km, road_time_min = load_road_matrices(self.station_ids)
        self.dist_km = np.asarray(road_dist_km, dtype=np.float64)
        self.time_min = np.asarray(road_time_min, dtype=np.float64)
        # Stations ordered by road distance from each station (itself first).
        self.nearest = np.argsort(self.dist_km, axis=1, kind="stable").astype(np.int32)

        self.trips = trips if trips is not None else load_period_trips(
            str(start.date()), str(end.date()), self.station_ids)

        if depot_station is None:
            d = haversine_km(np.r_[TORONTO_CENTER[0], self.stations["lat"]],
                             np.r_[TORONTO_CENTER[1], self.stations["lon"]])[0, 1:]
            self.depot_idx = int(np.argmin(d))
        else:
            self.depot_idx = int(np.flatnonzero(self.station_ids == depot_station)[0])

        # Missions touching these stations are no-ops; leftovers never go there.
        self.truck_blocked = np.isin(self.station_ids, truck_blocked_stations)

        self.shift_cost = sum(self._hourly_rate(h) for h in range(*shift))

        # ── spaces ───────────────────────────────────────────────────────────
        n, k = self.n_stations, self.n_trucks
        cap_max = float(self.capacity.max())
        self.action_space = spaces.MultiDiscrete(
            np.array([[n + 1, n, self.truck_capacity + 1]] * k, dtype=np.int64))
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "bikes": spaces.Box(0, cap_max, (n,), f32),
            "free_docks": spaces.Box(0, cap_max, (n,), f32),
            "minute_of_day": spaces.Box(0, MINUTES_PER_DAY - 1, (1,), f32),
            "day_of_week": spaces.Box(0, 6, (1,), f32),
            "truck_location": spaces.Box(0, n - 1, (k,), f32),
            "truck_busy": spaces.Box(0, 1, (k,), f32),
            "truck_minutes_left": spaces.Box(0, MINUTES_PER_DAY, (k,), f32),
            "truck_load": spaces.Box(0, self.truck_capacity, (k,), f32),
        })

    # ── helpers ──────────────────────────────────────────────────────────────
    def _hourly_rate(self, hour: int) -> float:
        lo, hi = self.day_hours
        return self.day_rate if lo <= hour % 24 < hi else self.night_rate

    def _in_shift(self, minute_of_day: int) -> bool:
        return self.shift[0] * 60 <= minute_of_day < self.shift[1] * 60

    def _init_bikes(self) -> np.ndarray:
        bikes = np.ceil(self.capacity * self.init_occupancy - 1e-9).astype(np.int64)
        return np.minimum(bikes, self.capacity)

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.bikes = self._init_bikes()
        self.trip_ptr = 0
        self.trip_started = np.zeros(len(self.trips), dtype=bool)  # per trip row
        self.returns = defaultdict(list)    # minute -> [(station, rent_min)]
        self.overflow = defaultdict(list)   # release minute -> [(station, rent_min)]
        self.trucks = [_Truck(location=self.depot_idx) for _ in range(self.n_trucks)]
        self.totals = dict.fromkeys(_COST_KEYS + _COUNT_KEYS, 0.0)
        return self._obs(), self._info(dict.fromkeys(_COST_KEYS + _COUNT_KEYS, 0.0))

    def step(self, action):
        acc = dict.fromkeys(_COST_KEYS + _COUNT_KEYS, 0.0)
        self._dispatch(np.asarray(action).reshape(self.n_trucks, 3), acc)
        for _ in range(self.step_minutes):
            if self.t >= self.total_minutes:
                break
            self._simulate_minute(acc)
            self.t += 1

        reward = (acc["unlock_revenue"] + acc["return_revenue"] + acc["overflow_revenue"]
                  - acc["lost_penalty"] - acc["labor_cost"] - acc["fuel_cost"]
                  - acc["fixed_cost"])
        for key, v in acc.items():
            self.totals[key] += v
        terminated = self.t >= self.total_minutes
        return self._obs(), float(reward), terminated, False, self._info(acc)

    # ── simulation ───────────────────────────────────────────────────────────
    def _simulate_minute(self, acc: dict) -> None:
        t = self.t
        self._advance_trucks(acc)

        # Returns due now: dock if there's room, otherwise hold for overflow delay.
        for st, rent in self.returns.pop(t, ()):
            if self.bikes[st] < self.capacity[st]:
                self.bikes[st] += 1
                acc["return_revenue"] += self.per_min_fee * rent
                acc["returns_docked"] += 1
            else:
                self.overflow[t + self.overflow_delay_min].append((st, rent))
                acc["overflow_returns"] += 1

        # Overflowed bikes reappear at the nearest station with a free dock.
        for st, rent in self.overflow.pop(t, ()):
            dest = self._nearest_free(st)
            if dest is None:                       # whole network full: retry
                self.overflow[t + 1].append((st, rent))
                continue
            self.bikes[dest] += 1
            acc["overflow_revenue"] += self.overflow_per_min_fee * rent
            acc["overflow_placed"] += 1

        # Rentals starting now.
        tr, i, n = self.trips, self.trip_ptr, len(self.trips)
        while i < n and tr.start_min[i] <= t:
            st = tr.start_idx[i]
            if self.bikes[st] > 0:
                self.bikes[st] -= 1
                self.trip_started[i] = True
                acc["unlock_revenue"] += self.unlock_fee
                acc["trips_started"] += 1
                self.returns[max(int(tr.end_min[i]), t + 1)].append(
                    (int(tr.end_idx[i]), int(tr.rent_min[i])))
            else:
                acc["lost_penalty"] += self.lost_trip_penalty
                acc["lost_trips"] += 1
            i += 1
        self.trip_ptr = i

        # Labor outside the paid shift, per busy truck-minute.
        mod = t % MINUTES_PER_DAY
        if not self._in_shift(mod):
            rate = self._hourly_rate(mod // 60) / 60.0
            acc["labor_cost"] += rate * sum(tk.busy(t) for tk in self.trucks)

    def _nearest_free(self, st: int, trucks_only: bool = False) -> int | None:
        order = self.nearest[st]
        free = self.bikes[order] < self.capacity[order]
        if trucks_only:
            free &= ~self.truck_blocked[order]
        j = np.argmax(free)
        return int(self.nearest[st, j]) if free[j] else None

    # ── trucks ───────────────────────────────────────────────────────────────
    def _dispatch(self, action: np.ndarray, acc: dict) -> None:
        """Start missions for idle trucks. Busy trucks ignore their row."""
        for tk, (pickup, dropoff, qty) in zip(self.trucks, action):
            pickup, dropoff, qty = int(pickup), int(dropoff), int(qty)
            if tk.busy(self.t) or pickup == 0 or qty == 0:
                continue
            a, b = pickup - 1, dropoff
            if a == b or self.truck_blocked[a] or self.truck_blocked[b]:
                continue
            day = self.t // MINUTES_PER_DAY
            if tk.last_paid_day != day:
                tk.last_paid_day = day
                acc["fixed_cost"] += self.truck_fixed_cost
                acc["labor_cost"] += self.shift_cost
            tk.pending_qty = min(qty, self.truck_capacity)
            tk.ops = deque([("drive", a), ("load", a), ("drive", b), ("unload", b)])
            self._run_ops(tk, acc)

    def _advance_trucks(self, acc: dict) -> None:
        for tk in self.trucks:
            if tk.ops and tk.busy_until <= self.t:
                self._run_ops(tk, acc)

    def _run_ops(self, tk: _Truck, acc: dict) -> None:
        """Execute ops until one takes time (or the mission is done)."""
        t = self.t
        while tk.ops and tk.busy_until <= t:
            kind, st = tk.ops.popleft()
            if kind == "drive":
                km = self.dist_km[tk.location, st]
                acc["fuel_cost"] += self.fuel_cost_per_km * km
                acc["truck_km"] += km
                dur = math.ceil(self.time_min[tk.location, st]) if st != tk.location else 0
                tk.location = st
                tk.busy_until = t + dur
            elif kind == "load":
                n = int(min(tk.pending_qty, self.bikes[st], self.truck_capacity - tk.load))
                self.bikes[st] -= n
                tk.load += n
                tk.busy_until = t + math.ceil(n * self.load_min_per_bike)
            elif kind == "unload":
                n = int(min(tk.load, self.capacity[st] - self.bikes[st]))
                self.bikes[st] += n
                tk.load -= n
                acc["bikes_moved"] += n
                tk.busy_until = t + math.ceil(n * self.load_min_per_bike)
                if tk.load > 0:
                    # Station filled up: drop the rest at the next-nearest free dock.
                    dest = self._nearest_free(st, trucks_only=True)
                    if dest is not None:
                        tk.ops.extend([("drive", dest), ("unload", dest)])

    # ── observation / info ───────────────────────────────────────────────────
    def _obs(self) -> dict:
        t = min(self.t, self.total_minutes - 1)
        dow = (self.start_date + pd.Timedelta(days=t // MINUTES_PER_DAY)).dayofweek
        f32 = np.float32
        return {
            "bikes": self.bikes.astype(f32),
            "free_docks": (self.capacity - self.bikes).astype(f32),
            "minute_of_day": np.array([t % MINUTES_PER_DAY], f32),
            "day_of_week": np.array([dow], f32),
            "truck_location": np.array([tk.location for tk in self.trucks], f32),
            "truck_busy": np.array([tk.busy(self.t) for tk in self.trucks], f32),
            "truck_minutes_left": np.array(
                [min(MINUTES_PER_DAY, max(0, tk.busy_until - self.t))
                 for tk in self.trucks], f32),
            "truck_load": np.array([tk.load for tk in self.trucks], f32),
        }

    def _info(self, step_acc: dict) -> dict:
        return {
            **step_acc,
            "minute": self.t,
            "totals": dict(self.totals),
            "idle_trucks": np.array([not tk.busy(self.t) for tk in self.trucks]),
        }
