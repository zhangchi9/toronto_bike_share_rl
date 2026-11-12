"""RebalanceEnv: the RL-facing view of BikeShareEnv.

- One decision per idle truck per simulated minute; busy trucks are never asked.
  Several idle trucks in the same minute are asked one at a time, then the minute
  is simulated with all their orders. While every truck is busy the wrapper just
  advances time.
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
from rl.features import MAX_BIKES_PER_STOP, TRUCK_CAPACITY, FeatureBuilder, n_actions, obs_size
from rl.zones import N_ZONES, make_zones

# Data split. Validation (checkpoint selection) is the end of the training months;
# the test period is never used during training.
TRAIN_RANGE = ("2026-05-01", "2026-07-17")
VAL_WINDOWS = [("2026-07-18", "2026-07-24"), ("2026-07-25", "2026-07-31")]
TEST_PERIOD = ("2026-08-01", "2026-09-20")
EPISODE_DAYS = 7

_static = {}            # per-process cache: stations, road matrices, zones
_baseline_cache = {}    # (window, occupancy) -> per-minute no-truck reward


def static_data(n_zones: int = N_ZONES) -> dict:
    if n_zones not in _static:
        stations = load_stations()
        ids = stations["station_id"].to_numpy(dtype=np.int64)
        dist, time = load_road_matrices(ids)
        _static[n_zones] = {
            "stations": stations, "ids": ids, "dist": dist, "time": time,
            "zones": make_zones(stations["lat"].to_numpy(), stations["lon"].to_numpy(), n_zones),
        }
    return _static[n_zones]


def multi_stop_kwargs() -> dict:
    """Env settings for multi-stop missions over the RL zones (round 3 rule)."""
    return {"station_zone": static_data()["zones"].zone_of_station,
            "max_bikes_per_stop": MAX_BIKES_PER_STOP}


def make_base_env(window, number_trucks, init_occupancy, trips=None,
                  multi_stop: bool = True) -> BikeShareEnv:
    s = static_data()
    if trips is None:
        trips = load_period_trips(window[0], window[1], s["ids"])
    return BikeShareEnv(init_occupancy=init_occupancy, simulation_period=window,
                        number_trucks=number_trucks, skip_when_all_busy=False,
                        truck_capacity=TRUCK_CAPACITY,
                        **(multi_stop_kwargs() if multi_stop else {}),
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
                 reward_scale: float = 10.0, n_zones: int = N_ZONES,
                 record_minutes: bool = False, multi_stop: bool = True):
        """`windows`: fixed list of (start, end) episodes, used in order (eval/test).
        Otherwise each episode is a random `episode_days` window in `sample_range`."""
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
        self.zones = static_data(n_zones)["zones"]
        self._next_window = 0

        self.action_space = spaces.Discrete(n_actions(n_zones))
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_size(n_zones, self.K),), np.float32)
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
                                 multi_stop=self.multi_stop)
        self.env.reset(seed=0)
        self.base = no_truck_rewards(self.window, self.init_occupancy, trips)
        self.fleet_daily = self.K * (self.env.truck_fixed_cost + self.env.shift_cost)
        self.fb = FeatureBuilder(self.env, self.zones)
        self.fb.reset()
        self.true_total = 0.0
        self.minute_rewards: list[float] = []
        self.queue: list[int] = []
        self._advance()            # trucks start idle, so this asks right away
        return self.fb.obs(self.queue[0]), self._info()

    # ── stepping ─────────────────────────────────────────────────────────────
    def _sim_minute(self, action: np.ndarray) -> float:
        m = self.env.t
        _, r, *_ = self.env.step(action)
        self.true_total += r
        if self.record_minutes:
            self.minute_rewards.append(r)
        self.fb.record()
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
        self.pending[k] = self.fb.decode(action)
        if self.queue:                         # another idle truck, same minute
            return self.fb.obs(self.queue[0]), 0.0, False, False, self._info()
        reward = self._sim_minute(self.pending)
        reward += self._advance()
        done = not self.queue
        obs = self.fb.obs(self.queue[0] if self.queue else 0)
        return obs, float(reward), done, False, self._info()

    def _info(self) -> dict:
        return {"minute": self.env.t, "window": self.window,
                "true_total": self.true_total,
                "asking_truck": self.queue[0] if self.queue else None}
