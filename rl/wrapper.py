"""RebalanceEnv: the RL-facing view of BikeShareEnv.

- One decision per idle truck per simulated minute; busy trucks are never asked.
  Several idle trucks in the same minute are asked one at a time, then the minute
  is simulated with all their orders. While every truck is busy the wrapper just
  advances time. Round 15: a truck asked after another one already decided this
  same minute sees that order in its own observation (rl/features.py's `pending`),
  even though it isn't applied to the env until every idle truck has answered.
- Action: Discrete zone-to-zone moves (see rl/features.py). Obs: compact Box.
- Training reward = (env reward - no-truck reward over the same minutes, with the
  fleet's unavoidable daily fixed + shift cost added back) / reward_scale.
  The underlying env and its true reward are unchanged (self.env, self.true_total).
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY, BikeShareEnv
from bikeshare_env.road_network import load_road_matrices
from bikeshare_env.trips import load_period_trips
from data_loader import load_stations
from rl import zones as zz
from rl.stop_rules import parse_rule
from rl.features import (MAX_BIKES_PER_STOP, TRUCK_CAPACITY, FeatureBuilder, n_actions,
                         parse_features,
                         obs_size, truck_zones)

# Data split. Validation (checkpoint selection) is the end of the training months;
# the test period is never used during training.
TRAIN_RANGE = ("2026-05-01", "2026-07-17")
VAL_WINDOWS = [("2026-07-18", "2026-07-24"), ("2026-07-25", "2026-07-31")]
TEST_PERIOD = ("2026-08-01", "2026-09-20")
EPISODE_DAYS = 7

_static = {}            # per-process cache: stations, road matrices, zones
_baseline_cache = {}    # (window, occupancy) -> per-minute no-truck reward


def static_data() -> dict:
    """Stations, road matrices and the RL zones (zones.ZONE_SCHEME), cached."""
    if not _static:
        stations = load_stations()
        ids = stations["station_id"].to_numpy(dtype=np.int64)
        dist, time = load_road_matrices(ids)
        _static.update(stations=stations, ids=ids, dist=dist, time=time)
        _static["zones"] = zz.load_zones()      # may read the entries above
    return _static


def multi_stop_kwargs(zone_scheme: str | None = None, stop_rule=None) -> dict:
    """Env settings for multi-stop missions over the RL zones (round 3 rule), with
    an optional round 10 station rule (rl.stop_rules; None = the round 3 rule)."""
    return {"station_zone": zz.load_zones(zone_scheme).zone_of_station,
            "max_bikes_per_stop": MAX_BIKES_PER_STOP, "stop_rule": parse_rule(stop_rule)}


def make_base_env(window, number_trucks, init_occupancy, trips=None,
                  multi_stop: bool = True, paired_trucks: bool = False,
                  zone_scheme: str | None = None, stop_rule=None) -> BikeShareEnv:
    s = static_data()
    if trips is None:
        trips = load_period_trips(window[0], window[1], s["ids"])
    return BikeShareEnv(init_occupancy=init_occupancy, simulation_period=window,
                        number_trucks=number_trucks, skip_when_all_busy=False,
                        truck_capacity=TRUCK_CAPACITY,
                        **(multi_stop_kwargs(zone_scheme, stop_rule) if multi_stop else {}),
                        paired_trucks=paired_trucks,
                        stations=s["stations"], trips=trips,
                        road_dist_km=s["dist"], road_time_min=s["time"])


def no_truck_rewards(window, init_occupancy, trips=None) -> np.ndarray:
    """Per-minute env reward of the no-truck benchmark over `window` (cached)."""
    key = (tuple(window), init_occupancy)
    if key not in _baseline_cache:
        env = make_base_env(window, 0, init_occupancy, trips)
        env.reset(seed=0)
        noop = np.zeros((0, 3), dtype=np.int64)
        out = np.zeros(env.total_minutes)
        for m in range(env.total_minutes):
            _, out[m], *_ = env.step(noop)
        _baseline_cache[key] = out
    return _baseline_cache[key]


class RebalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, number_trucks: int = 1, windows=None, sample_range=TRAIN_RANGE,
                 episode_days: int = EPISODE_DAYS,
                 init_occupancy: float = DEFAULT_INIT_OCCUPANCY,
                 reward_scale: float = 10.0, zone_scheme: str | None = None,
                 record_minutes: bool = False, multi_stop: bool = True,
                 forecast: bool = True, features=None, stop_rule=None, territories=None):
        """`windows`: fixed list of (start, end) episodes, used in order (eval/test).
        Otherwise each episode is a random `episode_days` window in `sample_range`.
        `territories` (round 12): one array of zone ids per truck. Truck k then only
        orders moves between its territory's zones and sees rl.territory's view of
        them (its zones + one outside zone, no other trucks), so a model trained on a
        territory with 1 truck runs unchanged next to the other territory's truck."""
        super().__init__()
        if number_trucks < 1:
            raise ValueError("RebalanceEnv needs at least one truck")
        self.K = int(number_trucks)
        self.windows = list(windows) if windows else None
        self.sample_range = sample_range
        self.episode_days = int(episode_days)
        self.init_occupancy = init_occupancy
        self.reward_scale = reward_scale
        self.record_minutes = record_minutes   # keep each minute's true reward (eval)
        self.multi_stop = multi_stop           # round-3 multi-stop missions (5 bikes/stop)
        self.forecast = forecast               # round-4 demand forecast features
        self.features = parse_features(features)   # round-9 extra feature groups
        self.stop_rule = parse_rule(stop_rule)      # round-10 station rule (None = round 3)
        self.zone_scheme = zone_scheme or zz.ZONE_SCHEME
        self.zones = zz.load_zones(self.zone_scheme)
        self._next_window = 0
        self.territories = None
        if territories is not None:
            self.territories = [np.asarray(t, dtype=np.int64) for t in territories]
            if len(self.territories) != self.K or len({len(t) for t in self.territories}) != 1:
                raise ValueError("need one territory per truck, all the same size")
            from rl.territory import territory_view
            self.views = [territory_view(self.zones, t) for t in self.territories]
            n = len(self.territories[0])
            self.action_space = spaces.Discrete(n_actions(n))
            self.observation_space = spaces.Box(
                -1.0, 1.0, (obs_size(n + 1, 1, forecast, self.features, others=False),),
                np.float32)
            self.env = None
            return

        # Actions cover the zones trucks can reach (not the island); obs covers all.
        n_truck_zones = len(truck_zones(self.zones, static_data()["ids"]))
        self.action_space = spaces.Discrete(n_actions(n_truck_zones))
        self.observation_space = spaces.Box(
            -1.0, 1.0, (obs_size(self.zones.n_zones, self.K, forecast, self.features),),
            np.float32)
        self.env: BikeShareEnv | None = None

    # ── episode setup ────────────────────────────────────────────────────────
    def _pick_window(self):
        if self.windows:
            w = self.windows[self._next_window % len(self.windows)]
            self._next_window += 1
            return w
        lo, hi = (pd.Timestamp(d) for d in self.sample_range)
        n_starts = (hi - lo).days - self.episode_days + 2
        start = lo + pd.Timedelta(days=int(self.np_random.integers(n_starts)))
        end = start + pd.Timedelta(days=self.episode_days - 1)
        return (str(start.date()), str(end.date()))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.window = self._pick_window()
        trips = load_period_trips(*self.window, static_data()["ids"])
        self.env = make_base_env(self.window, self.K, self.init_occupancy, trips,
                                 multi_stop=self.multi_stop, zone_scheme=self.zone_scheme,
                                 stop_rule=self.stop_rule)
        self.env.reset(seed=0)
        self.base = no_truck_rewards(self.window, self.init_occupancy, trips)
        self.fleet_daily = self.K * (self.env.truck_fixed_cost + self.env.shift_cost)
        if self.territories is None:
            self.fbs = [FeatureBuilder(self.env, self.zones, forecast=self.forecast,
                                       features=self.features)]
        else:
            n = len(self.territories[0])
            self.fbs = [FeatureBuilder(self.env, v, forecast=self.forecast,
                                       features=self.features, action_zone_ids=np.arange(n),
                                       others=False, forecast_zones=self.zones)
                        for v in self.views]
        for fb in self.fbs:
            fb.reset()
        self.true_total = 0.0
        self.minute_rewards: list[float] = []
        self.queue: list[int] = []
        self._advance()            # trucks start idle, so this asks right away
        return self._obs(self.queue[0]), self._info()

    @property
    def fb(self) -> FeatureBuilder:
        """The feature builder of the truck being asked (the only one without territories)."""
        return self._fb(self.queue[0] if self.queue else 0)

    def _fb(self, k: int) -> FeatureBuilder:
        return self.fbs[k if self.territories is not None else 0]

    def _obs(self, k: int) -> np.ndarray:
        """Round 15: `self.pending` carries any order already decided this minute by
        another idle truck but not yet applied to the env (see `step`), so a truck
        asked after it sees that live decision, not the other truck's stale pre-decision
        state. `self.pending` is zero rows for any truck not yet decided / busy / not
        part of this minute's idle set, so this has no effect for K=1 or outside a
        same-minute multi-truck decision."""
        return self._fb(k).obs(k, pending=self.pending)

    # ── stepping ─────────────────────────────────────────────────────────────
    def _sim_minute(self, action: np.ndarray) -> float:
        m = self.env.t
        _, r, *_ = self.env.step(action)
        self.true_total += r
        if self.record_minutes:
            self.minute_rewards.append(r)
        for fb in self.fbs:
            fb.record()
        shaped = r - self.base[m]
        if m % MINUTES_PER_DAY == 0:
            shaped += self.fleet_daily     # unavoidable: same for every policy
        return shaped / self.reward_scale

    def _advance(self) -> float:
        """Simulate until some truck is idle (filling self.queue) or the episode ends."""
        acc = 0.0
        noop = np.zeros((self.K, 3), dtype=np.int64)
        while self.env.t < self.env.total_minutes:
            idle = [k for k, tk in enumerate(self.env.trucks) if not tk.busy(self.env.t)]
            if idle:
                self.queue = idle
                self.pending = np.zeros((self.K, 3), dtype=np.int64)
                return acc
            acc += self._sim_minute(noop)
        self.queue = []
        return acc

    def step(self, action):
        k = self.queue.pop(0)
        fb = self._fb(k)
        self.pending[k] = fb.decode(action)
        fb.note(k, self.pending[k])
        if self.queue:                         # another idle truck, same minute
            return self._obs(self.queue[0]), 0.0, False, False, self._info()
        reward = self._sim_minute(self.pending)
        reward += self._advance()
        done = not self.queue
        obs = self._obs(self.queue[0] if self.queue else 0)
        return obs, float(reward), done, False, self._info()

    def action_masks(self) -> np.ndarray:
        """Valid actions now (sb3_contrib MaskablePPO calls this; round 11)."""
        return self.fb.action_mask()

    def _info(self) -> dict:
        return {"minute": self.env.t, "window": self.window,
                "true_total": self.true_total,
                "asking_truck": self.queue[0] if self.queue else None}
