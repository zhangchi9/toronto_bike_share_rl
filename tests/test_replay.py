"""Multi-day rollouts and the dashboard's precomputed replay (dashboard_data)."""
import base64

import numpy as np
import pytest

import dashboard_data as dd
from bikeshare_env.rollout import simulate_day, simulate_period

D = 1440


@pytest.fixture(scope="module")
def two_days():
    return simulate_period("2026-08-01", "2026-08-02", "no_rebalancing")


def test_period_first_day_matches_single_day(two_days):
    one = simulate_day("2026-08-01", "no_rebalancing")
    assert two_days.bikes.shape[0] == 2 * D
    assert np.array_equal(two_days.bikes[:D], one.bikes)
    assert np.allclose(two_days.reward[:D], one.reward)
    assert np.array_equal(two_days.fulfilled[:D], one.fulfilled)


def test_period_carries_bikes_over_midnight(two_days):
    fresh = simulate_day("2026-08-02", "no_rebalancing")     # restarts from 35% fill
    b = two_days.bikes.astype(int)
    one_minute = np.abs(b[D] - b[D - 1]).sum()
    vs_fresh = np.abs(b[D] - fresh.bikes[0].astype(int)).sum()
    assert one_minute < 50 < vs_fresh                    # day 2 continues from day 1's end


@pytest.fixture
def tiny_replay(tmp_path, monkeypatch):
    """A 2-day replay cache in tmp_path."""
    monkeypatch.setattr(dd, "REPLAY_START", "2026-08-01")
    monkeypatch.setattr(dd, "REPLAY_END", "2026-08-02")
    monkeypatch.setattr(dd, "REPLAY_DIR", str(tmp_path))
    for f in (dd.load, dd.load_trips, dd.day_chunk):
        f.cache_clear()
    dd.build("no_rebalancing", verbose=False)
    yield dd.load("no_rebalancing")
    for f in (dd.load, dd.load_trips, dd.day_chunk):
        f.cache_clear()


def test_day_chunk_slices_the_run_and_offsets_running_totals(tiny_replay, two_days):
    rep = tiny_replay
    assert np.array_equal(rep["bikes"], two_days.bikes)          # the replay is the 2-day run
    c = dd.day_chunk("no_rebalancing", 1, 0, with_routes=False)
    bikes = np.frombuffer(base64.b64decode(c["bikes"]), np.uint8).reshape(D, -1)
    assert np.array_equal(bikes, rep["bikes"][D:2 * D])
    assert c["day"] == 1 and c["date"] == "Aug 2"
    assert c["cum0"]["reward"] == pytest.approx(rep["reward"][:D].sum())
    assert c["cum0"]["fulfilled"] == rep["fulfilled"][:D].sum()
    c1 = dd.day_chunk("no_rebalancing", 1, 1, with_routes=False)    # range starts on day 1
    assert c1["day"] == 0 and all(v == 0 for v in c1["cum0"].values())
    assert dd.requests_in(0, 1) == len(dd.load_trips()["start_min"])


def test_rides_over_midnight_show_on_both_days(tiny_replay):
    tr = dd.load_trips()
    ok = tiny_replay["trip_started"]
    i = np.flatnonzero(ok & (tr["start_min"] < D) & (tr["end_min"] > D + 5))[0]
    day0 = dd.day_chunk("no_rebalancing", 0, 0, with_routes=False)["trips"]
    day1 = dd.day_chunk("no_rebalancing", 1, 0, with_routes=False)["trips"]
    s, e = int(tr["start_min"][i]), int(tr["end_min"][i])
    assert (s, e) in set(zip(day0["sm"], day0["em"]))
    assert (s - D, e - D) in set(zip(day1["sm"], day1["em"]))      # started "yesterday"
    assert all(x == 1 for x in day1["ok"])                         # only served rides


def test_out_of_date_replay_is_detected(tiny_replay, monkeypatch):
    assert dd.is_current("no_rebalancing")
    monkeypatch.setattr(dd, "FORMAT_VERSION", dd.FORMAT_VERSION + 1)
    assert not dd.is_current("no_rebalancing")
