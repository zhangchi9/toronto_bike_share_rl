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


def test_default_scheme_is_round8_and_kmeans_still_reproduces():
    s = static_data()
    st = s["stations"]
    assert zz.ZONE_SCHEME == "flow_w0.3_k20_need"
    assert zz.ROUND5_SCHEME == "flow_w0.3"
    assert s["zones"].n_zones == 21
    assert np.array_equal(s["zones"].zone_of_station, zz.load_zones().zone_of_station)
    km = zz.make_zones(st["lat"].to_numpy(), st["lon"].to_numpy())   # rounds 1-4, shuttle
    assert np.array_equal(zz.load_zones("kmeans").zone_of_station, km.zone_of_station)


# ── round 8: zone count and need-balanced zones ──────────────────────────────
@pytest.fixture(scope="module")
def zones_dir(tmp_path_factory):
    """A temp zone folder holding a copy of the station-need cache (if built)."""
    import os
    import shutil
    d = tmp_path_factory.mktemp("zones8")
    src = os.path.join(zz.ZONES_DIR, zz.NEED_FILE)
    if os.path.exists(src):
        shutil.copy(src, d / zz.NEED_FILE)
    mp = pytest.MonkeyPatch()
    mp.setattr(zz, "ZONES_DIR", str(d))
    yield d
    mp.undo()


@pytest.mark.parametrize("k,balance", [(12, "docks"), (30, "need")])
def test_k_zones_island_last_and_weight_balanced(zones_dir, k, balance):
    s = static_data()
    ids, cap = s["ids"], s["stations"]["capacity"].to_numpy(float)
    z = zz.build_flow_zones(0.3, verbose=False, k=k, balance=balance)
    island = np.isin(ids, ISLAND_STATIONS)
    assert z.n_zones == k + 1
    assert np.all(z.zone_of_station[island] == k)
    assert set(np.unique(z.zone_of_station[~island])) == set(range(k))
    assert zz.island_zone(z) == k
    weight = zz.zone_weights(balance, cap)[~island]
    tot = np.bincount(z.zone_of_station[~island], weights=weight, minlength=k)
    mean, slack = weight.sum() / k, weight.max()
    assert tot.min() >= (1 - zz.DOCK_BAND) * mean - slack
    assert tot.max() <= (1 + zz.DOCK_BAND) * mean + slack


def test_scheme_names_round_trip(zones_dir):
    assert zz.flow_scheme(0.3) == "flow_w0.3"
    assert zz.flow_scheme(0.3, 30, "need") == "flow_w0.3_k30_need"
    assert zz.parse_flow_scheme("flow_w0.3_k30_need") == (0.3, 30, "need")
    assert zz.parse_flow_scheme("flow_w0.5") == (0.5, 20, "docks")
    with pytest.raises(ValueError):
        zz.parse_flow_scheme("flow_w0.3_k30_bogus")
    built = zz.build_flow_zones(0.3, verbose=False, k=12, balance="docks")
    zz._loaded.pop("flow_w0.3_k12_docks", None)
    loaded = zz.load_zones("flow_w0.3_k12_docks")
    assert loaded.n_zones == 13
    assert np.array_equal(loaded.zone_of_station, built.zone_of_station)


def test_need_counts_lost_and_blocked(zones_dir):
    need = zz.station_need(verbose=False)
    assert need.shape == (len(static_data()["ids"]),)
    assert need.min() >= 0 and need.sum() > 0


def test_env_sizes_follow_zone_count(zones_dir):
    from rl.features import n_actions, obs_size
    from rl.wrapper import RebalanceEnv
    zz.build_flow_zones(0.3, verbose=False, k=12, balance="docks")
    env = RebalanceEnv(number_trucks=1, zone_scheme="flow_w0.3_k12_docks", forecast=False)
    assert env.action_space.n == n_actions(12) == 397
    assert env.observation_space.shape == (obs_size(13, 1, forecast=False),)
