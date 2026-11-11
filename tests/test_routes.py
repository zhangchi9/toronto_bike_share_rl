"""Truck trip recording (rollout) and road route geometry (dashboard)."""
import os

import numpy as np
import pytest

import rl.policy  # noqa: F401  (registers fix_shuttle_policy)
from bikeshare_env.rollout import simulate_day

DAY = "2026-07-18"


@pytest.fixture(scope="module")
def shuttle_day():
    return simulate_day(DAY, "fix_shuttle_policy")


def test_truck_legs_match_status_and_stations(shuttle_day):
    r = shuttle_day
    legs, status = r.truck_legs, r.truck_status[:, 0]
    assert len(legs) > 0 and (legs[:, 3] < legs[:, 4]).all()       # depart < arrive
    driving = np.zeros(1440, dtype=bool)
    for k, a, b, dep, arr, load in legs:
        assert a != b
        driving[dep:min(arr, 1440)] = True
        assert r.truck_station[dep, k] == b                          # location = destination
    assert ((status == 1) == driving).all()                          # status 1 exactly on legs
    # Consecutive legs chain: each trip starts where the previous one ended.
    assert (legs[1:, 1] == legs[:-1, 2]).all()
    assert r.totals["truck_km"] > 0 and r.truck_moved[-1] == r.totals["bikes_moved"]


def test_no_truck_rollout_has_empty_truck_fields():
    r = simulate_day(DAY, "no_rebalancing")
    assert r.truck_legs.shape == (0, 6) and r.truck_status.shape == (1440, 0)
    assert r.truck_moved[-1] == 0


@pytest.mark.skipif(not os.path.exists("data/toronto_drive.graphml"),
                    reason="road graph not built (python build_road_distances.py)")
def test_road_routes_follow_roads_between_the_stations(shuttle_day):
    from bikeshare_env.road_network import haversine_km
    from road_routes import get_router
    R = get_router()
    for k, a, b, *_ in shuttle_day.truck_legs[:20]:
        lat, lon = map(np.array, R.route(a, b))
        assert (lat[0], lon[0]) == (round(R.st_lat[a], 5), round(R.st_lon[a], 5))
        assert (lat[-1], lon[-1]) == (round(R.st_lat[b], 5), round(R.st_lon[b], 5))
        path_km = sum(haversine_km(lat[i:i + 2], lon[i:i + 2])[0, 1] for i in range(len(lat) - 1))
        straight = haversine_km(lat[[0, -1]], lon[[0, -1]])[0, 1]
        assert len(lat) > 2 and path_km >= straight * 0.999          # bends along streets
