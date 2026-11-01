"""Unit tests on tiny synthetic worlds (no disk data needed)."""
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from bikeshare_env.env import BikeShareEnv
from bikeshare_env.trips import TripEvents

# 4 stations on a line: road distance = 1 km per index gap, 2 min per km.
POS = np.arange(4)
DIST = np.abs(POS[:, None] - POS[None, :]).astype(float)
TIME = 2 * DIST


def make_env(capacity=(10, 10, 10, 10), trips=(), occupancy=0.5, **kw):
    stations = pd.DataFrame({
        "station_id": [100, 101, 102, 103],
        "lat": [43.65, 43.66, 43.67, 43.68], "lon": [-79.38] * 4,
        "capacity": list(capacity),
    })
    cols = ["start_min", "end_min", "start_idx", "end_idx", "rent_min"]
    events = TripEvents.from_frame(pd.DataFrame(list(trips), columns=cols))
    kw.setdefault("depot_station", 100)
    kw.setdefault("skip_when_all_busy", False)   # most tests step minute by minute
    kw.setdefault("number_trucks", 0)            # no daily truck costs unless a test wants them
    env = BikeShareEnv(init_occupancy=occupancy, simulation_period=("2026-06-15", "2026-06-16"),
                       stations=stations, trips=events,
                       road_dist_km=DIST, road_time_min=TIME, **kw)
    env.reset(seed=0)
    return env


def noop(env):
    return np.zeros((env.number_trucks, 3), dtype=np.int64)


def run(env, minutes, action_at=None):
    """Step `minutes` times; action_at maps minute -> action. Returns rewards."""
    rewards = []
    for m in range(minutes):
        a = (action_at or {}).get(m, noop(env))
        _, r, *_ = env.step(a)
        rewards.append(r)
    return np.array(rewards)


def test_init_occupancy_rounds_up():
    env = make_env(capacity=(10, 5, 3, 1), occupancy=0.5)
    assert env.bikes.tolist() == [5, 3, 2, 1]
    assert make_env(occupancy=0.0).bikes.sum() == 0


def test_rent_and_return_rewards():
    env = make_env(trips=[(0, 5, 0, 1, 5)])
    r = run(env, 6)
    assert r[0] == pytest.approx(1.0)             # unlock fee at rental
    assert r[5] == pytest.approx(0.15 * 5)        # per-minute fee at return
    assert r[1:5].sum() == 0
    assert env.bikes.tolist() == [4, 6, 5, 5]


def test_lost_trip_penalty_when_station_empty():
    env = make_env(trips=[(0, 5, 0, 1, 5)], occupancy=0.0)
    _, r, _, _, info = env.step(noop(env))
    assert r == pytest.approx(-0.5)
    assert info["lost_trips"] == 1
    assert run(env, 10).sum() == 0                # no return for a lost trip


def test_full_station_overflow_goes_to_nearest_free_after_delay():
    # Station 1 is full; station 0 frees a dock when the trip starts.
    env = make_env(capacity=(2, 1, 1, 5), trips=[(0, 3, 0, 1, 3)], occupancy=1.0)
    r = run(env, 25)
    assert r[3] == 0                              # no reward at the failed return
    assert r[3 + 15] == pytest.approx(0.10 * 3)   # 15-min delay not billed
    assert r.sum() == pytest.approx(1.0 + 0.3)
    assert env.bikes.tolist() == [2, 1, 1, 5]     # placed back at station 0
    assert env.totals["overflow_returns"] == 1


def test_truck_mission_timing_costs_and_non_cancelable():
    # Depot = station 0 (empty dock space). Pick 3 at station 1, drop at 3.
    env = make_env(number_trucks=1, shift=(9, 17))
    action = np.array([[1 + 1, 3, 3]])
    _, r0, _, _, info = env.step(action)
    assert not info["idle_trucks"][0]
    assert info["fixed_cost"] == 100
    assert info["labor_cost"] == pytest.approx(160 + 40 / 60)   # shift + 1 busy night min

    # A new order for the busy truck is ignored.
    env.step(np.array([[0 + 1, 2, 5]]))
    tk = env.trucks[0]
    assert [op for op in tk.ops] == [("load", 1), ("drive", 3), ("unload", 3)]

    run(env, 20)
    # drive 0->1: 2 min, load 3: 3 min, drive 1->3: 4 min, unload 3: 3 min = 12 busy min
    assert env.bikes.tolist() == [5, 2, 5, 8]
    assert env.trucks[0].location == 3 and env.trucks[0].load == 0
    assert env.totals["fuel_cost"] == pytest.approx(0.3 * (1 + 2))
    assert env.totals["labor_cost"] == pytest.approx(160 + 12 * 40 / 60)
    assert env.totals["bikes_moved"] == 3


def test_load_limited_by_bikes_and_leftover_dropped_at_nearest_free():
    # Station 1 has 4 bikes; station 2 has 1 free dock -> 3 leftover go to the
    # nearest free station from 2 (stations 1 and 3 tie at 1 km; lower index wins).
    env = make_env(capacity=(10, 8, 6, 10), number_trucks=1)
    env.bikes[:] = [0, 4, 5, 0]
    env.step(np.array([[1 + 1, 2, 10]]))
    run(env, 40)
    assert env.bikes.tolist() == [0, 3, 6, 0]
    assert env.trucks[0].location == 1
    assert env.totals["truck_km"] == pytest.approx(1 + 1 + 1)


def test_whole_fleet_hired_every_day_even_if_unused():
    env = make_env(number_trucks=2)
    _, r, _, _, info = env.step(noop(env))        # 00:00 of day 1
    assert info["fixed_cost"] == 2 * 100 and info["labor_cost"] == 2 * 160
    assert r == pytest.approx(-2 * (100 + 160))
    # Dispatching later the same day adds no fixed or shift cost.
    env.step(np.array([[1 + 1, 2, 1], [0, 0, 0]]))
    run(env, 1440 - 2)                            # finish day 1
    assert env.totals["fixed_cost"] == 200
    run(env, 1440)                                # day 2: charged again, unused
    assert env.totals["fixed_cost"] == 400
    # 2 trucks x 2 days of shifts + 6 busy night minutes (drive 2, load 1, drive 2, unload 1)
    assert env.totals["labor_cost"] == pytest.approx(4 * 160 + 6 * 40 / 60)


def test_zero_trucks_means_no_truck_costs():
    env = make_env(number_trucks=0)
    assert env.action_space.shape == (0, 3)
    r = run(env, 1440)
    assert env.totals["fixed_cost"] == 0 and env.totals["labor_cost"] == 0
    assert r.sum() == 0


def test_truck_missions_to_blocked_stations_are_noops():
    env = make_env(number_trucks=1, truck_blocked_stations=(103,))
    _, _, _, _, info = env.step(np.array([[1 + 1, 3, 2]]))
    assert info["idle_trucks"][0] and not env.trucks[0].ops and info["fuel_cost"] == 0


def test_step_skips_minutes_while_all_trucks_busy():
    # Mission: drive 0->1 2 min, load 3 3 min, drive 1->3 4 min, unload 3 3 min = 12 min.
    env = make_env(number_trucks=1, trips=[(5, 7, 2, 0, 2)], skip_when_all_busy=True)
    _, r, _, _, info = env.step(np.array([[1 + 1, 3, 3]]))
    assert info["minutes_elapsed"] == 12 and env.t == 12
    assert info["idle_trucks"][0]
    # The rental at minute 5 and its return at 7 happened inside the skipped minutes.
    assert info["trips_started"] == 1 and info["returns_docked"] == 1
    assert r == pytest.approx(1.0 + 0.15 * 2 - 100 - 160 - 12 * 40 / 60 - 0.3 * 3)
    # With an idle truck, a step is one minute again.
    _, _, _, _, info = env.step(noop(env))
    assert info["minutes_elapsed"] == 1


def test_episode_length_and_termination():
    env = make_env()
    done, steps = False, 0
    while not done:
        *_, done, _, _ = env.step(noop(env))
        steps += 1
    assert steps == 2 * 1440


@pytest.mark.parametrize("number_trucks", [0, 2])
def test_gymnasium_env_checker(number_trucks):
    check_env(make_env(trips=[(0, 5, 0, 1, 5)], number_trucks=number_trucks),
              skip_render_check=True)
