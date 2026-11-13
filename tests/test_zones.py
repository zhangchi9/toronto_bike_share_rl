"""Zone schemes: k-means (rounds 1-4) and the dock-balanced road + demand flow zones."""
import numpy as np
import pytest

from bikeshare_env.env import ISLAND_STATIONS
from rl import zones as zz
from rl.wrapper import static_data


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """Flow zones built into a temp folder (w = 0.7)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(zz, "ZONES_DIR", str(tmp_path_factory.mktemp("zones")))
    z = zz.build_flow_zones(0.7, verbose=False)
    yield z
    mp.undo()


def test_island_is_its_own_zone_and_has_no_truck_stations(flow):
    ids = static_data()["ids"]
    island = np.isin(ids, ISLAND_STATIONS)
    assert flow.n_zones == zz.N_ZONES + 1 == 21
    assert island.sum() == 7
    assert np.all(flow.zone_of_station[island] == 20)
    assert not np.any(flow.zone_of_station[~island] == 20)


def test_every_mainland_zone_is_dock_balanced(flow):
    s = static_data()
    cap = s["stations"]["capacity"].to_numpy(float)
    main = ~np.isin(s["ids"], ISLAND_STATIONS)
    docks = np.bincount(flow.zone_of_station[main], weights=cap[main], minlength=20)
    assert set(np.unique(flow.zone_of_station[main])) == set(range(20))
    mean, slack = cap[main].sum() / 20, cap.max()
    assert docks.min() >= (1 - zz.DOCK_BAND) * mean - slack
    assert docks.max() <= (1 + zz.DOCK_BAND) * mean + slack
    assert sum(len(m) for m in flow.stations_in_zone) == len(cap)


def test_flow_zones_are_deterministic_and_round_trip(flow, monkeypatch, tmp_path):
    monkeypatch.setattr(zz, "ZONES_DIR", str(tmp_path))
    again = zz.build_flow_zones(0.7, verbose=False)
    assert np.array_equal(again.zone_of_station, flow.zone_of_station)
    loaded = zz.load_zones(zz.flow_scheme(0.7))
    assert np.array_equal(loaded.zone_of_station, flow.zone_of_station)


def test_round5_scheme_is_used_and_kmeans_still_reproduces():
    s = static_data()
    st = s["stations"]
    assert zz.ZONE_SCHEME == "flow_w0.3"
    assert s["zones"].n_zones == 21
    assert np.array_equal(s["zones"].zone_of_station, zz.load_zones().zone_of_station)
    km = zz.make_zones(st["lat"].to_numpy(), st["lon"].to_numpy())   # rounds 1-4, shuttle
    assert np.array_equal(zz.load_zones("kmeans").zone_of_station, km.zone_of_station)
