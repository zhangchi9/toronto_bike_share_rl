"""Run a policy through one simulated day and record per-minute state/metrics."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import bikeshare_env  # noqa: F401  (registers the env)
from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY, BikeShareEnv
from bikeshare_env.policies import POLICIES
from bikeshare_env.trips import TripEvents


@dataclass
class DayRollout:
    """Index m of every per-minute array = state after minute m was simulated."""
    bikes: np.ndarray          # (1440, N) uint8 bikes docked per station
    reward: np.ndarray         # (1440,) reward earned in that minute
    fulfilled: np.ndarray      # (1440,) requests served (rentals started)
    lost: np.ndarray           # (1440,) requests lost (empty station)
    active: np.ndarray         # (1440,) bikes out on a ride after that minute
    blocked: np.ndarray        # (1440,) returns refused (end station full)
    pending: np.ndarray        # (1440,) refused bikes still held (15-min delay)
    trips: TripEvents
    trip_started: np.ndarray   # (n_trips,) bool, request was served
    totals: dict


def simulate_day(date: str, policy_key: str, init_occupancy: float = DEFAULT_INIT_OCCUPANCY,
                 **env_kwargs) -> DayRollout:
    env = gym.make("BikeShareToronto-v0", init_occupancy=init_occupancy,
                   simulation_period=(date, date), **env_kwargs)
    u: BikeShareEnv = env.unwrapped
    obs, info = env.reset(seed=0)
    policy = POLICIES[policy_key]()
    policy.reset(u)

    T, N = MINUTES_PER_DAY, u.n_stations
    bikes = np.zeros((T, N), dtype=np.uint8)
    reward = np.zeros(T)
    fulfilled = np.zeros(T, dtype=np.int64)
    lost = np.zeros(T, dtype=np.int64)
    active = np.zeros(T, dtype=np.int64)
    blocked = np.zeros(T, dtype=np.int64)
    pending = np.zeros(T, dtype=np.int64)
    riding = held = 0
    for m in range(T):
        obs, r, done, _, info = env.step(policy.act(obs, info))
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
                      u.trips, u.trip_started.copy(), dict(info["totals"]))
