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


# ── multi-stop missions (round 3 rule) ──────────────────────────────────────────
# 6 stations on a line, zone 0 = stations 0-2, zone 1 = stations 3-5.
POS6 = np.arange(6)
DIST6 = np.abs(POS6[:, None] - POS6[None, :]).astype(float)


def make_multi_env(bikes, capacity=(10,) * 6, max_per_stop=5, stop_rule=None):
    stations = pd.DataFrame({
        "station_id": list(range(200, 206)),
        "lat": [43.65 + 0.01 * i for i in range(6)], "lon": [-79.38] * 6,
        "capacity": list(capacity),
    })
    events = TripEvents.from_frame(pd.DataFrame(
        [], columns=["start_min", "end_min", "start_idx", "end_idx", "rent_min"]))
    env = BikeShareEnv(init_occupancy=0.0, simulation_period=("2026-06-15", "2026-06-15"),
                       stations=stations, trips=events, road_dist_km=DIST6,
                       road_time_min=2 * DIST6, depot_station=200, number_trucks=1,
                       skip_when_all_busy=False, truck_capacity=40,
                       station_zone=np.array([0, 0, 0, 1, 1, 1]),
                       max_bikes_per_stop=max_per_stop, stop_rule=stop_rule)
    env.reset(seed=0)
    env.bikes[:] = bikes
    return env


def run_mission(env, pickup, dropoff, qty, minutes=200):
    """Dispatch one mission; return the stations visited (in order) and per-stop changes."""
    visits, prev = [], env.trucks[0].location
    before = env.bikes.copy()
    env.step(np.array([[pickup + 1, dropoff, qty]]))
    for _ in range(minutes):
        tk = env.trucks[0]
        if tk.location != prev:
            visits.append(tk.location)
            prev = tk.location
        if not tk.busy(env.t):
            break
        env.step(np.zeros((1, 3), dtype=np.int64))
    return visits, env.bikes - before


def test_multi_stop_picks_at_most_5_per_station_in_most_bikes_order():
    env = make_multi_env([8, 6, 3, 0, 1, 2])
    visits, delta = run_mission(env, pickup=0, dropoff=3, qty=12)
    # Pickup: 5 at 0, 5 at 1 (next most bikes), 2 at 2 (ordered amount reached).
    # Drop-off: 5 at 3, then most free docks: 4 (9 free) gets 5, then 5 gets 2.
    assert visits == [1, 2, 3, 4, 5]
    assert delta.tolist() == [-5, -5, -2, 5, 5, 2]
    assert env.totals["bikes_moved"] == 12 and env.trucks[0].load == 0
    assert np.abs(delta).max() <= 5


def test_multi_stop_stops_when_zone_runs_dry():
    env = make_multi_env([2, 1, 0, 0, 0, 0])
    _, delta = run_mission(env, pickup=0, dropoff=3, qty=40)
    assert delta.tolist() == [-2, -1, 0, 3, 0, 0]       # 3 collected, all fit at the first drop
    assert env.totals["bikes_moved"] == 3


def test_multi_stop_collecting_nothing_skips_the_drive_to_dropoff():
    env = make_multi_env([0, 0, 0, 0, 0, 0])
    visits, delta = run_mission(env, pickup=1, dropoff=4, qty=10)
    assert visits == [1]                                 # never drove to zone 1
    assert not delta.any() and env.totals["truck_km"] == pytest.approx(1.0)


def test_multi_stop_full_dropoff_zone_falls_back_to_nearest_free():
    env = make_multi_env([8, 6, 3, 1, 1, 1], capacity=(10, 10, 10, 2, 2, 2))
    _, delta = run_mission(env, pickup=0, dropoff=3, qty=10)
    # 1 free dock at each of 3, 4, 5; the other 7 go to the nearest free dock (station 2).
    assert delta.tolist() == [-5, -5, 7, 1, 1, 1]
    assert env.trucks[0].load == 0 and env.totals["bikes_moved"] == 10


def _rule(name):
    from rl.stop_rules import parse_rule
    return parse_rule(name)


@pytest.mark.parametrize("bikes,qty", [([8, 6, 3, 0, 1, 2], 12), ([2, 1, 0, 0, 0, 0], 40),
                                       ([8, 6, 3, 1, 1, 1], 10)])
def test_stop_rule_most_cap5_is_the_builtin_rule(bikes, qty):
    a, b = make_multi_env(bikes), make_multi_env(bikes, stop_rule=_rule("most/cap5"))
    va, da = run_mission(a, pickup=0, dropoff=3, qty=qty)
    vb, db = run_mission(b, pickup=0, dropoff=3, qty=qty)
    assert va == vb and da.tolist() == db.tolist() and a.totals == b.totals


def test_stop_rule_nearest_goes_to_the_nearest_station_next():
    env = make_multi_env([8, 2, 6, 0, 0, 0], stop_rule=_rule("nearest/cap5"))
    visits, delta = run_mission(env, pickup=2, dropoff=5, qty=12)
    # First stop by most bikes (0, where the truck is), then nearest: 1 before 2.
    assert visits[:2] == [1, 2] and delta[:3].tolist() == [-5, -2, -5]
    assert delta[3:].tolist() == [5, 5, 2]                  # 3 (most free), then 4, 5


def test_stop_rule_fill_picks_the_fullest_station_by_ratio():
    env = make_multi_env([10, 9, 2, 0, 0, 0], capacity=(20, 10, 10, 10, 10, 10),
                         stop_rule=_rule("fill/cap5"))
    _, delta = run_mission(env, pickup=0, dropoff=3, qty=5)
    assert delta[:3].tolist() == [0, -5, 0]                 # 9/10 beats 10/20


def test_stop_rule_target_moves_stations_toward_half_full():
    env = make_multi_env([8, 6, 3, 0, 4, 6], stop_rule=_rule("most/target"))
    _, delta = run_mission(env, pickup=0, dropoff=3, qty=40)
    # Pickup only above 5 of 10 docks: 3 from 0, 1 from 1; drop only up to 5.
    assert delta.tolist() == [-3, -1, 0, 4, 0, 0]
    assert ((env.bikes[:3] >= 5) | (delta[:3] == 0)).all()


def test_stop_rule_target_skips_a_zone_with_nothing_above_half():
    env = make_multi_env([5, 4, 3, 0, 0, 0], stop_rule=_rule("most/target"))
    visits, delta = run_mission(env, pickup=0, dropoff=3, qty=20)
    assert visits == [] and not delta.any() and env.totals["truck_km"] == 0


def test_stop_rule_cap10_moves_at_most_10_per_station():
    env = make_multi_env([12, 6, 3, 0, 0, 0], stop_rule=_rule("most/cap10"))
    _, delta = run_mission(env, pickup=0, dropoff=3, qty=20)
    assert delta.tolist() == [-10, -6, -3, 10, 9, 0]
    assert np.abs(delta).max() <= 10 and env.trucks[0].load == 0


def test_stop_rule_never_moves_more_than_ordered():
    for name in ("most/cap10", "need/cap5", "fill/target", "need_near/target"):
        env = make_multi_env([10, 10, 10, 0, 0, 0], stop_rule=_rule(name))
        _, delta = run_mission(env, pickup=0, dropoff=3, qty=7)
        assert -delta[:3].sum() <= 7 and delta.sum() == 0, name


def test_stop_rule_not_allowed_with_paired_trucks():
    kw = dict(init_occupancy=0.0, simulation_period=("2026-06-15", "2026-06-15"),
              stations=pd.DataFrame({"station_id": [1, 2], "lat": [43.6, 43.7],
                                     "lon": [-79.4, -79.4], "capacity": [10, 10]}),
              trips=TripEvents.from_frame(pd.DataFrame(
                  [], columns=["start_min", "end_min", "start_idx", "end_idx", "rent_min"])),
              road_dist_km=np.ones((2, 2)), road_time_min=np.ones((2, 2)),
              station_zone=np.array([0, 1]), max_bikes_per_stop=5, paired_trucks=True,
              number_trucks=2, stop_rule=_rule("need/cap5"))
    with pytest.raises(ValueError):
        BikeShareEnv(**kw)


def test_multi_stop_needs_both_settings():
    env = make_env(number_trucks=1, station_zone=np.zeros(4, dtype=int))
    assert not env.multi_stop


# Paired trucks: 8 stations on a line, zone 0 = stations 0-3, zone 1 = stations 4-7.
POS8 = np.arange(8)
DIST8 = np.abs(POS8[:, None] - POS8[None, :]).astype(float)


def make_pair_env(bikes, capacity=(10,) * 8):
    stations = pd.DataFrame({
        "station_id": list(range(300, 308)),
        "lat": [43.65 + 0.01 * i for i in range(8)], "lon": [-79.38] * 8,
        "capacity": list(capacity),
    })
    events = TripEvents.from_frame(pd.DataFrame(
        [], columns=["start_min", "end_min", "start_idx", "end_idx", "rent_min"]))
    env = BikeShareEnv(init_occupancy=0.0, simulation_period=("2026-06-15", "2026-06-15"),
                       stations=stations, trips=events, road_dist_km=DIST8,
                       road_time_min=2 * DIST8, depot_station=300, number_trucks=2,
                       skip_when_all_busy=False, truck_capacity=40,
                       station_zone=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                       max_bikes_per_stop=5, paired_trucks=True)
    env.reset(seed=0)
    env.bikes[:] = bikes
    return env


def run_pair_mission(env, pickup, dropoff, qty, minutes=300):
    """One paired order; per truck, the stations where its load changed and by how much."""
    stops = [[], []]
    before = env.bikes.copy()
    loads = [tk.load for tk in env.trucks]
    env.step(np.array([[pickup + 1, dropoff, qty], [0, 0, 0]]))
    for _ in range(minutes):
        for k, tk in enumerate(env.trucks):
            if tk.load != loads[k]:
                stops[k].append((tk.location, tk.load - loads[k]))
                loads[k] = tk.load
        if not any(tk.busy(env.t) for tk in env.trucks):
            break
        env.step(np.zeros((2, 3), dtype=np.int64))
    return stops, env.bikes - before


def test_paired_trucks_split_odd_and_even_ranks():
    # Pickup ranks (pickup station 0 first, then most bikes): 0, 2, 1, 3.
    # Drop-off ranks (station 4 first, then most free docks): 4, 5, 6, 7.
    env = make_pair_env([9, 7, 8, 3, 0, 2, 4, 6])
    stops, delta = run_pair_mission(env, pickup=0, dropoff=4, qty=10)
    assert stops[0] == [(0, 5), (1, 5), (4, -5), (6, -5)]      # ranks 1, 3
    assert stops[1] == [(2, 5), (3, 3), (5, -5), (7, -3)]      # ranks 2, 4 (zone ran dry)
    assert delta.tolist() == [-5, -5, -5, -3, 5, 5, 5, 3]
    assert env.totals["bikes_moved"] == 18
    assert all(abs(n) <= 5 for s in stops for _, n in s)


def test_paired_second_truck_stays_idle_without_a_station():
    env = make_pair_env([6, 0, 0, 0, 0, 0, 0, 0])
    stops, delta = run_pair_mission(env, pickup=0, dropoff=4, qty=10)
    assert stops[0] == [(0, 5), (4, -5)] and stops[1] == []
    assert delta.tolist() == [-5, 0, 0, 0, 5, 0, 0, 0]
    assert env.trucks[1].location == 0 and env.totals["truck_km"] == pytest.approx(4.0)


def test_paired_orders_wait_for_both_trucks_and_read_row_0_only():
    env = make_pair_env([9, 7, 8, 3, 0, 2, 4, 6])
    env.step(np.array([[0, 0, 0], [1, 4, 10]]))                 # row 1 is ignored
    assert not any(tk.busy(env.t) for tk in env.trucks)
    env.step(np.array([[1, 4, 10], [0, 0, 0]]))
    assert all(tk.busy(env.t) for tk in env.trucks)
    while env.trucks[0].busy(env.t) and env.trucks[1].busy(env.t):
        env.step(np.zeros((2, 3), dtype=np.int64))
    env.step(np.array([[4, 6, 10], [0, 0, 0]]))                 # one truck still busy: ignored
    while any(tk.busy(env.t) for tk in env.trucks):
        env.step(np.zeros((2, 3), dtype=np.int64))
    # Only the first order ran (same result as test_paired_trucks_split_odd_and_even_ranks).
    assert env.bikes.tolist() == [4, 2, 3, 0, 5, 7, 9, 9]


def test_paired_trucks_need_multi_stop_and_two_trucks():
    stations = pd.DataFrame({"station_id": [1, 2], "lat": [43.6, 43.7], "lon": [-79.4] * 2,
                             "capacity": [10, 10]})
    events = TripEvents.from_frame(pd.DataFrame(
        [], columns=["start_min", "end_min", "start_idx", "end_idx", "rent_min"]))
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    kw = dict(stations=stations, trips=events, road_dist_km=d, road_time_min=d,
              simulation_period=("2026-06-15", "2026-06-15"), paired_trucks=True)
    with pytest.raises(ValueError):
        BikeShareEnv(number_trucks=2, **kw)                     # no multi-stop
    with pytest.raises(ValueError):
        BikeShareEnv(number_trucks=1, station_zone=np.array([0, 0]), max_bikes_per_stop=5, **kw)
