"""Run a policy through a simulated period (one or more days) and record per-minute state."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import bikeshare_env  # noqa: F401  (registers the env)
from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, BikeShareEnv
from bikeshare_env.policies import POLICIES
from bikeshare_env.trips import TripEvents


@dataclass
class DayRollout:
    """Index m of every per-minute array = state after minute m of the period was simulated
    (T = days * 1440 minutes; minute 0 = 00:00 on the first day)."""
    bikes: np.ndarray          # (T, N) uint8 bikes docked per station
    reward: np.ndarray         # (T,) reward earned in that minute
    fulfilled: np.ndarray      # (T,) requests served (rentals started)
    lost: np.ndarray           # (T,) requests lost (empty station)
    active: np.ndarray         # (T,) bikes out on a ride after that minute
    blocked: np.ndarray        # (T,) returns refused (end station full)
    pending: np.ndarray        # (T,) refused bikes still held (15-min delay)
    trips: TripEvents
    trip_started: np.ndarray   # (n_trips,) bool, request was served
    totals: dict
    # Trucks (empty when the policy has none):
    truck_legs: np.ndarray     # (n_legs, 6) int: truck, from, to, depart, arrive, load
    truck_status: np.ndarray   # (T, K) int8: 0 idle, 1 driving, 2 loading/unloading
    truck_load: np.ndarray     # (T, K) bikes on each truck after that minute
    truck_station: np.ndarray  # (T, K) station the truck is at / driving to
    truck_moved: np.ndarray    # (T,) cumulative bikes moved by trucks


def simulate_day(date: str, policy_key: str, init_occupancy: float = DEFAULT_INIT_OCCUPANCY,
                 **env_kwargs) -> DayRollout:
    return simulate_period(date, date, policy_key, init_occupancy, **env_kwargs)


def simulate_period(start: str, end: str, policy_key: str,
                    init_occupancy: float = DEFAULT_INIT_OCCUPANCY, **env_kwargs) -> DayRollout:
    """One continuous run from `start` 00:00 to `end` 23:59: stations are filled to
    `init_occupancy` once, at the start, and bikes carry over between days."""
    policy = POLICIES[policy_key]()
    env_kwargs.setdefault("number_trucks", policy.number_trucks)
    for key, value in policy.env_kwargs().items():
        env_kwargs.setdefault(key, value)
    # One step per minute so every minute is recorded, even while trucks are busy.
    env = gym.make("BikeShareToronto-v0", init_occupancy=init_occupancy,
                   simulation_period=(start, end), skip_when_all_busy=False,
                   **env_kwargs)
    u: BikeShareEnv = env.unwrapped
    obs, info = env.reset(seed=0)
    policy.reset(u)

    T, N = u.total_minutes, u.n_stations
    bikes = np.zeros((T, N), dtype=np.uint8)
    reward = np.zeros(T)
    fulfilled = np.zeros(T, dtype=np.int64)
    lost = np.zeros(T, dtype=np.int64)
    active = np.zeros(T, dtype=np.int64)
    blocked = np.zeros(T, dtype=np.int64)
    pending = np.zeros(T, dtype=np.int64)
    riding = held = 0
    K = u.number_trucks
    legs = []
    truck_status = np.zeros((T, K), dtype=np.int8)
    truck_load = np.zeros((T, K), dtype=np.int16)
    truck_station = np.zeros((T, K), dtype=np.int32)
    truck_moved = np.zeros(T, dtype=np.int64)
    prev_loc = [tk.location for tk in u.trucks]
    last_leg = [None] * K
    moved = 0
    for m in range(T):
        obs, r, done, _, info = env.step(policy.act(obs, info))
        # A drive op sets the truck's location to its destination and busy_until to
        # the arrival minute when it starts, so a location change = a new leg.
        for k, tk in enumerate(u.trucks):
            if tk.location != prev_loc[k]:
                last_leg[k] = (k, prev_loc[k], tk.location, m, tk.busy_until, tk.load)
                legs.append(last_leg[k])
                prev_loc[k] = tk.location
            leg = last_leg[k]
            if leg is not None and leg[3] <= m < leg[4]:
                truck_status[m, k] = 1
            elif tk.busy_until > m or tk.ops:
                truck_status[m, k] = 2
            truck_load[m, k] = tk.load
            truck_station[m, k] = tk.location
        moved += info["bikes_moved"]
        truck_moved[m] = moved
        bikes[m] = u.bikes
        reward[m] = r
        fulfilled[m] = info["trips_started"]
        lost[m] = info["lost_trips"]
        # A ride ends at the return attempt, whether it docks or overflows.
        riding += info["trips_started"] - info["returns_docked"] - info["overflow_returns"]
        active[m] = riding
        blocked[m] = info["overflow_returns"]
        held += info["overflow_returns"] - info["overflow_placed"]
        pending[m] = held
        if done:
            break
    return DayRollout(bikes, reward, fulfilled, lost, active, blocked, pending,
                      u.trips, u.trip_started.copy(), dict(info["totals"]),
                      np.array(legs, dtype=np.int64).reshape(-1, 6),
                      truck_status, truck_load, truck_station, truck_moved)
