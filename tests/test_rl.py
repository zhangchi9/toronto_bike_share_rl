"""RL wrapper tests on real data (one short validation window)."""
import os

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl.features import QTY_LEVELS, n_actions, obs_size
from rl.wrapper import MINUTES_PER_DAY, RebalanceEnv

WINDOW = [("2026-07-18", "2026-07-18")]   # one validation day


@pytest.fixture(params=[1, 2])
def env(request):
    e = RebalanceEnv(number_trucks=request.param, windows=WINDOW, record_minutes=True)
    e.reset(seed=0)
    return e


def test_spaces_sizes(env):
    assert QTY_LEVELS == (10, 20, 40) and env.env.truck_capacity == 40
    # Round 5 zones: 20 truck zones (actions) + the island zone (observed only).
    assert env.zones.n_zones == 21 and len(env.fb.action_zones) == 20
    assert env.action_space.n == n_actions(20) == 1 + 20 * 19 * len(QTY_LEVELS) == 1141
    # Round 15: each other truck's block grew from n_zones+2 to 2*n_zones+3 (destination
    # zone one-hot + load), K=1 unaffected (no other-truck block at all).
    assert env.observation_space.shape == (obs_size(21, env.K),) == ((199,) if env.K == 1 else (244,))
    assert obs_size(20, env.K) == (190 if env.K == 1 else 233)                  # round 4
    assert obs_size(20, env.K, forecast=False) == (110 if env.K == 1 else 153)   # rounds 1-3


def test_island_zone_is_never_an_action(env):
    island = env.zones.n_zones - 1
    assert island not in env.fb.action_zones
    zof = env.zones.zone_of_station
    for a in range(1, env.action_space.n):
        row = env.fb.decode(a)
        if row.any():
            assert zof[row[0] - 1] != island and zof[row[1]] != island


def test_wait_only_training_reward_is_zero_and_true_reward_reconstructs(env):
    total, done = 0.0, False
    while not done:
        obs, r, done, _, info = env.step(0)
        total += r
        assert env.observation_space.contains(obs)
    assert total == pytest.approx(0.0, abs=1e-6)
    # True reward = no-truck reward minus the idle fleet's daily cost.
    assert info["true_total"] == pytest.approx(env.base.sum() - env.fleet_daily)
    assert sum(env.minute_rewards) == pytest.approx(info["true_total"])


def test_decode_gives_valid_env_action_or_wait(env):
    u, rng = env.env, np.random.default_rng(0)
    for a in rng.integers(0, env.action_space.n, 500):
        row = env.fb.decode(a)
        if not row.any():
            continue
        pickup, dropoff, qty = row
        assert 1 <= pickup <= u.n_stations and 0 <= dropoff < u.n_stations
        assert qty in QTY_LEVELS and pickup - 1 != dropoff
        assert not u.truck_blocked[pickup - 1] and not u.truck_blocked[dropoff]
        assert u.bikes[pickup - 1] > 0 and u.bikes[dropoff] < u.capacity[dropoff]


def test_idle_trucks_asked_every_minute_busy_never(env):
    u, rng = env.env, np.random.default_rng(1)
    idle_minutes = {k: [] for k in range(env.K)}     # truck idle at the start of minute t
    real_step = u.step

    def recording_step(action):
        for k, tk in enumerate(u.trucks):
            if not tk.busy(u.t):
                idle_minutes[k].append(u.t)
        return real_step(action)
    u.step = recording_step

    asked = {k: [] for k in range(env.K)}
    asked[env.queue[0]].append(u.t)
    done = False
    while not done:
        a = int(rng.integers(1, env.action_space.n)) if rng.random() < 0.02 else 0
        _, _, done, _, _ = env.step(a)
        if not done:
            k = env.queue[0]
            assert not u.trucks[k].busy(u.t)          # never ask a busy truck
            asked[k].append(u.t)
    assert u.t == MINUTES_PER_DAY
    assert u.totals["bikes_moved"] > 0                # trucks really were dispatched
    for k in range(env.K):                            # asked exactly when idle, once per minute
        assert asked[k] == idle_minutes[k]


def test_gymnasium_env_checker():
    check_env(RebalanceEnv(number_trucks=2, windows=WINDOW), skip_render_check=True)


def test_dashboard_adapter_matches_wrapper(monkeypatch):
    """RLPolicy (per-minute dashboard path) takes the same decisions as the wrapper."""
    import rl.policy as rp

    class Scripted:                      # deterministic stand-in for a trained model
        def __init__(self):
            self.calls = 0

        def predict(self, obs, deterministic=True):
            self.calls += 1
            return (7 if self.calls % 97 == 0 else 0), None

    from bikeshare_env.rollout import simulate_day
    cls = type("ScriptedPolicy", (rp.RLPolicy,), {"label": "t", "algo": "ppo", "path": "",
                                                  "number_trucks": 1})
    cls._model = Scripted()
    monkeypatch.setitem(rp.POLICIES, "scripted", cls)
    r = simulate_day("2026-07-18", "scripted")

    env = RebalanceEnv(number_trucks=1, windows=WINDOW, zone_scheme=cls.zone_scheme)
    env.reset(seed=0)
    model, done = Scripted(), False
    while not done:
        a, _ = model.predict(env.fb.obs(env.queue[0]))
        _, _, done, _, info = env.step(a)
    assert r.reward.sum() == pytest.approx(info["true_total"])
    assert r.totals["bikes_moved"] == env.env.totals["bikes_moved"] > 0


def test_encode_decode_roundtrip():
    from rl.features import encode_action
    seen = set()
    for src in range(20):
        for dst in range(20):
            if src == dst:
                continue
            for q in QTY_LEVELS:
                a = encode_action(src, dst, q, 20)
                assert 1 <= a < 1141
                seen.add(a)
    assert len(seen) == 1140                     # every move action, exactly once


def test_fix_shuttle_policy_same_on_dashboard_and_in_eval():
    from bikeshare_env.rollout import simulate_day
    from rl.policy import FixShuttlePolicy

    r = simulate_day("2026-07-18", "fix_shuttle_policy")
    env = RebalanceEnv(number_trucks=1, windows=WINDOW, multi_stop=False,   # single-stop rule
                       zone_scheme="kmeans", forecast=False)                   # rounds 1-4 zones
    env.reset(seed=0)
    zof, a, done = env.zones.zone_of_station, FixShuttlePolicy.action(), False
    while not done:
        row = env.fb.decode(a)
        if row.any():                            # always downtown (12) -> zone 11
            assert (zof[row[0] - 1], zof[row[1]], row[2]) == (12, 11, 10)
        _, _, done, _, info = env.step(a)
    assert r.reward.sum() == pytest.approx(info["true_total"])
    assert env.env.totals["bikes_moved"] > 0


def test_multi_stop_is_on_for_agents_and_first_dropoff_has_most_free_docks(env):
    from rl.features import MAX_BIKES_PER_STOP
    u = env.env
    assert u.multi_stop and u.max_bikes_per_stop == MAX_BIKES_PER_STOP == 5
    rng = np.random.default_rng(2)
    for a in rng.integers(1, env.action_space.n, 300):
        row = env.fb.decode(a)
        if not row.any():
            continue
        zone = env.zones.zone_of_station[row[1]]
        members = env.fb.truck_stations[zone]
        assert u.capacity[row[1]] - u.bikes[row[1]] == (u.capacity[members] - u.bikes[members]).max()


def test_agent_missions_move_at_most_5_bikes_per_stop(env):
    u = env.env
    a = next(a for a in range(1, env.action_space.n)
             if env.fb.decode(a).any() and env.fb.decode(a)[2] == 40)
    k = env.queue[0]
    loads, real_step = [u.trucks[k].load], u.step

    def recording_step(action):            # the wrapper simulates whole missions inside step()
        out = real_step(action)
        loads.append(u.trucks[k].load)
        return out
    u.step = recording_step
    env.step(a)                            # with 2 trucks the order waits for the other one
    for _ in range(3000):
        if len(loads) > 1 and not u.trucks[k].busy(u.t):
            break
        env.step(0)
    assert u.trucks[k].load == 0 and len(loads) > 1
    assert np.abs(np.diff(loads)).max() <= 5            # each stop handles at most 5 bikes
    assert max(loads) > 5                               # so it took several pickup stops


def test_duplicated_dqn_runs_both_trucks_on_every_order(monkeypatch):
    import rl.policy as rp
    from bikeshare_env.rollout import simulate_day
    if not os.path.exists(rp.DuplicatedTruckPolicy.path):
        pytest.skip("models/dqn_1trucks/best_model.zip not trained")
    monkeypatch.setitem(rp.POLICIES, "dqn_duplicated_2trucks", rp.DuplicatedTruckPolicy)
    r = simulate_day("2026-07-18", "dqn_duplicated_2trucks")
    assert r.totals["bikes_moved"] > 0
    assert set(r.truck_legs[:, 0].tolist()) == {0, 1}      # both trucks drove


def test_forecast_block_is_the_table_entry_for_the_current_slot(env):
    from rl import forecast as fc
    fb, u, Z = env.fb, env.env, env.zones.n_zones
    for _ in range(40):                          # move into the day a bit
        env.step(0)
    obs = fb.obs(env.queue[0])
    slot = ((u.start_date - fc.DATA_START).days * 1440 + u.t) // fc.SLOT_MIN
    want = np.clip(fb.fc_table[slot] / fb.zone_cap[:, None], 0, 1).T.reshape(-1)
    assert np.allclose(obs[4 * Z: 8 * Z], want)
    assert u.start_date == fc.DATA_START + __import__("pandas").Timedelta(days=slot * 15 // 1440)


def test_forecast_features_use_only_trips_before_the_origin():
    from rl import forecast as fc
    rng = np.random.default_rng(0)
    n_min, Z = 20 * 1440, 3
    counts = rng.poisson(0.3, (n_min, Z))
    cum = lambda c: np.vstack([np.zeros((1, Z), np.int64), np.cumsum(c, axis=0)])
    origins = np.array([9 * 1440 + 600, 12 * 1440])
    for H in fc.HORIZONS:
        X = fc.features(cum(counts), cum(counts[::-1]), origins, H)
        for m in origins:                        # change every trip at or after m
            later = counts.copy()
            later[m:] = rng.poisson(5.0, later[m:].shape)
            X2 = fc.features(cum(later), cum(counts[::-1]), [m], H)
            row = np.flatnonzero(origins == m)[0]
            assert np.array_equal(X[row * Z:(row + 1) * Z], X2)
        y = fc.targets(cum(counts), origins, H)  # target = the next H minutes
        assert y[0] == counts[origins[0]:origins[0] + H, 0].sum()


# ── round 9: extra feature groups ────────────────────────────────────────────
ALL_FEATURES = "risk+travel+failures+memory"


def test_feature_groups_parse_and_size():
    from rl.features import FEATURE_GROUPS, features_name, parse_features
    assert parse_features(None) == parse_features("base") == ()
    assert parse_features("memory,risk") == ("risk", "memory")
    assert features_name(["travel", "risk"]) == "risk+travel"
    with pytest.raises(ValueError):
        parse_features("risk+weather")
    base = obs_size(21, 1)
    assert obs_size(21, 1, features="risk") == base + 6 * 21
    assert obs_size(21, 1, features=ALL_FEATURES) == base + 11 * 21
    assert FEATURE_GROUPS == ("risk", "travel", "failures", "memory")


def test_champion_features_are_all_four_groups():
    from rl.features import CHAMPION_FEATURES, FEATURE_GROUPS, parse_features
    assert parse_features(CHAMPION_FEATURES) == FEATURE_GROUPS


def test_feature_obs_extends_base_obs_and_stays_in_bounds():
    base = RebalanceEnv(number_trucks=1, windows=WINDOW)
    full = RebalanceEnv(number_trucks=1, windows=WINDOW, features=ALL_FEATURES)
    ob, _ = base.reset(seed=0)
    of, _ = full.reset(seed=0)
    assert of.shape == full.observation_space.shape
    rng = np.random.default_rng(1)
    done = False
    while not done:
        assert np.array_equal(of[:len(ob)], ob)          # base inputs unchanged, extras last
        assert full.observation_space.contains(of)
        a = int(rng.integers(base.action_space.n)) if rng.random() < 0.05 else 0
        ob, _, done, *_ = base.step(a)
        of, _, done_f, *_ = full.step(a)
        assert done == done_f


def test_failure_features_count_every_lost_request_and_blocked_return():
    env = RebalanceEnv(number_trucks=1, windows=WINDOW, features="failures")
    env.reset(seed=0)
    fb, seen = env.fb, []
    record = fb.record
    fb.record = lambda: (record(), seen.append(fb.failures[-1]))
    done = False
    while not done:
        _, _, done, *_ = env.step(0)
    tot = np.sum(seen, axis=0)
    assert tot[0].sum() == env.env.totals["lost_trips"] > 0
    assert tot[1].sum() == env.env.totals["overflow_returns"] > 0


def test_memory_features_remember_the_last_order():
    from rl.features import MEMORY_MIN, encode_action, truck_zones
    from rl.wrapper import static_data
    env = RebalanceEnv(number_trucks=1, windows=WINDOW, features="memory")
    obs, _ = env.reset(seed=0)
    Z = env.zones.n_zones
    assert np.all(obs[-2 * Z:] == 1)                        # never visited
    tz = truck_zones(env.zones, static_data()["ids"])
    src, dst = int(tz[10]), int(tz[5])
    row = env.fb.decode(encode_action(10, 5, 20, len(tz)))
    t0 = env.env.t
    obs, *_ = env.step(encode_action(10, 5, 20, len(tz)))
    zof = env.zones.zone_of_station
    assert zof[row[0] - 1] == src and zof[row[1]] == dst
    pick, drop = obs[-2 * Z:-Z], obs[-Z:]
    assert pick[src] == pytest.approx(min(1, (env.env.t - t0) / MEMORY_MIN))
    assert drop[dst] == pytest.approx(min(1, (env.env.t - t0) / MEMORY_MIN))
    assert pick[src] < 1 and np.sum(pick < 1) == 1 and np.sum(drop < 1) == 1


def test_feature_obs_same_on_dashboard_and_in_wrapper(monkeypatch):
    """With every feature group, the dashboard path builds the same inputs."""
    import rl.policy as rp
    from bikeshare_env.rollout import simulate_day

    class Scripted:
        def __init__(self):
            self.seen = []

        def predict(self, obs, deterministic=True):
            self.seen.append(np.array(obs))
            return (7 if len(self.seen) % 97 == 0 else 0), None

    cls = type("ScriptedFeat", (rp.RLPolicy,), {"label": "t", "algo": "dqn", "path": "",
                                                "number_trucks": 1, "zone_scheme": None,
                                                "features": ("risk", "travel", "failures",
                                                             "memory")})
    cls._model = Scripted()
    monkeypatch.setitem(rp.POLICIES, "scripted_feat", cls)
    simulate_day("2026-07-18", "scripted_feat")

    env = RebalanceEnv(number_trucks=1, windows=WINDOW, features=ALL_FEATURES)
    obs, _ = env.reset(seed=0)
    model, done = Scripted(), False
    while not done:
        a, _ = model.predict(obs)
        obs, _, done, *_ = env.step(a)
    assert len(model.seen) == len(cls._model.seen)
    assert all(np.array_equal(a, b) for a, b in zip(model.seen, cls._model.seen))


def test_stop_rules_parse():
    from rl.stop_rules import RULES, parse_rule, rule_name
    assert parse_rule(None) is None and parse_rule("none") is None
    for r in RULES:
        assert parse_rule(r).name == r == parse_rule(rule_name(r)).name
    assert rule_name("need_near/target") == "need_near-target"
    for bad in ("need", "most/cap7", "best/cap5"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_need_rule_projects_with_training_period_profiles(env):
    from rl.features import _profiles_bikes
    from rl.stop_rules import parse_rule
    from rl.zones import demand_profiles
    e = env.env
    rule = parse_rule("need/cap5")
    cand = np.arange(10)
    P = demand_profiles(e.station_ids, e.capacity.astype(float)) * e.capacity[:, None]
    assert np.allclose(_profiles_bikes(e.station_ids, e.capacity), P)
    h = (e.t % 1440) // 60
    off = int(e.start_date.dayofweek >= 5)             # the WINDOW has no holidays
    want = e.bikes[cand] + P[cand][:, [h + 24 * off, (h + 1) % 24 + 24 * off]].sum(1)
    assert np.allclose(rule.projected(e, cand), want)


def test_stop_rule_same_on_dashboard_and_in_wrapper(monkeypatch):
    """A non-default station rule gives the same run on the dashboard path."""
    import rl.policy as rp
    from bikeshare_env.rollout import simulate_day

    class Scripted:
        def __init__(self):
            self.seen = []

        def predict(self, obs, deterministic=True):
            self.seen.append(np.array(obs))
            return (7 if len(self.seen) % 97 == 0 else 0), None

    cls = type("ScriptedRule", (rp.RLPolicy,), {"label": "t", "algo": "dqn", "path": "",
                                                "number_trucks": 1, "zone_scheme": None,
                                                "stop_rule": "need_near/target"})
    cls._model = Scripted()
    monkeypatch.setitem(rp.POLICIES, "scripted_rule", cls)
    simulate_day("2026-07-18", "scripted_rule")

    env = RebalanceEnv(number_trucks=1, windows=WINDOW, stop_rule="need_near/target")
    obs, _ = env.reset(seed=0)
    model, done = Scripted(), False
    while not done:
        a, _ = model.predict(obs)
        obs, _, done, *_ = env.step(a)
    assert len(model.seen) == len(cls._model.seen)
    assert all(np.array_equal(a, b) for a, b in zip(model.seen, cls._model.seen))


# ── round 11: other algorithms ───────────────────────────────────────────────
def test_action_mask_allows_only_orders_the_zones_can_fill(env):
    from rl.features import MAX_BIKES_PER_STOP, encode_action
    rng = np.random.default_rng(0)
    fb, e = env.fb, env.env
    for _ in range(5):
        m = env.action_masks()
        assert m.shape == (env.action_space.n,) and m.dtype == bool and m[0]
        Z = len(fb.action_zones)
        room = lambda z, pick: np.minimum(            # noqa: E731
            e.bikes[fb.truck_stations[z]] if pick else
            e.capacity[fb.truck_stations[z]] - e.bikes[fb.truck_stations[z]],
            MAX_BIKES_PER_STOP).sum()
        for i, j in rng.integers(0, Z, (40, 2)):
            if i == j:
                continue
            for q in QTY_LEVELS:
                a = encode_action(i, j, q, Z)
                zi, zj = fb.action_zones[i], fb.action_zones[j]
                assert m[a] == (room(zi, True) >= q and room(zj, False) >= q)
        for a in range(1, env.action_space.n):          # decode's waits are masked
            if not fb.decode(a).any():
                assert not m[a]
        env.step(int(rng.choice(np.flatnonzero(m))))


def test_masks_reach_maskable_ppo_through_monitor_and_vec_env():
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    e = RebalanceEnv(number_trucks=1, windows=WINDOW)
    venv = DummyVecEnv([lambda: Monitor(e)])
    venv.reset()
    assert (get_action_masks(venv)[0] == e.action_masks()).all()


def test_prioritized_replay_samples_by_priority():
    from rl.rainbow import PrioritizedReplay
    b = PrioritizedReplay(100, 2)
    for i in range(50):
        b.add(np.zeros(2), i, 0.0, np.zeros(2), 0.9)
    b.update(np.arange(50), np.r_[np.full(49, 1e-3), 100.0])      # index 49 dominates
    idx, w = b.sample(64, 0.4, np.random.default_rng(0))
    assert (idx < 50).all() and (idx == 49).mean() > 0.9
    assert np.isclose(w.max(), 1.0) and (w[idx == 49] < 1).all()  # IS weights shrink it
    assert np.isclose(b.tree[1], b.tree[b.tree_n:b.tree_n + 50].sum())


def test_rainbow_n_step_returns_and_save_load(tmp_path):
    from collections import deque
    from rl.rainbow import RainbowDQN
    env = RebalanceEnv(number_trucks=1, windows=WINDOW)
    r = RainbowDQN(env, gamma=0.5, n_step=3, buffer_size=16)
    o = np.zeros(env.observation_space.shape, np.float32)
    nq = deque((o + k, k, 1.0, o + k + 1, False) for k in range(3))
    r._store(nq, flush=False)                                      # one full 3-step window
    assert r.buffer.size == 1 and np.isclose(r.buffer.ret[0], 1 + 0.5 + 0.25)
    assert np.isclose(r.buffer.disc[0], 0.125) and r.buffer.next_obs[0][0] == 3
    nq = deque([(o, 0, 2.0, o + 1, False), (o + 1, 1, 1.0, o + 2, True)])
    r._store(nq, flush=True)                                       # episode end: no bootstrap
    assert r.buffer.size == 3 and r.buffer.disc[1] == 0 and np.isclose(r.buffer.ret[1], 2.5)
    obs, _ = env.reset(seed=0)
    r.save(str(tmp_path / "m.pt"))
    loaded = RainbowDQN.load(str(tmp_path / "m.pt"))
    assert loaded.predict(obs)[0] == r.predict(obs)[0]


def test_train_builds_every_round_11_algorithm():
    from rl.train import ALGOS, build_model
    assert set(ALGOS) >= {"ppo", "a2c", "dqn", "qrdqn", "maskppo", "rainbow"}
    for algo in ("qrdqn", "rainbow"):
        m = build_model(algo, 1, 0, 1, 1000, features="risk+travel+failures+memory")
        assert type(m).__name__ in ("QRDQN", "RainbowDQN")


# ── round 12: territories ────────────────────────────────────────────────────
def test_territory_split_is_balanced_and_disjoint():
    from rl import zones as zz
    from rl.territory import SCHEME, load_territories, split_two, territory_view
    A, B = load_territories(SCHEME)
    z = zz.load_zones(SCHEME)
    assert z.n_zones == 41 and len(A) == len(B) == 20
    assert not set(A) & set(B) and set(A) | set(B) == set(range(40))   # island (40) in neither
    v = territory_view(z, A)
    assert v.n_zones == 21
    for i, zid in enumerate(A):          # same stations, renumbered; the rest is "outside"
        assert np.array_equal(v.stations_in_zone[i], z.stations_in_zone[zid])
    assert len(v.stations_in_zone[20]) == len(z.zone_of_station) - sum(
        len(z.stations_in_zone[q]) for q in A)
    D = np.array([[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 1], [9, 9, 1, 0]], float)
    in_a = split_two(D)
    assert in_a.sum() == 2 and in_a[0] == in_a[1] and in_a[2] == in_a[3] and in_a[0] != in_a[2]


def test_territory_trucks_only_order_their_zones_and_obs_ignores_the_other_truck():
    from rl.features import CHAMPION_FEATURES
    from rl.territory import SCHEME, load_territories
    A, B = load_territories(SCHEME)
    one = RebalanceEnv(1, windows=WINDOW, zone_scheme=SCHEME, features=CHAMPION_FEATURES,
                       territories=[A])
    two = RebalanceEnv(2, windows=WINDOW, zone_scheme=SCHEME, features=CHAMPION_FEATURES,
                       territories=[A, B])
    assert one.action_space == two.action_space and one.action_space.n == n_actions(20)
    assert one.observation_space == two.observation_space        # same model runs in both
    assert one.observation_space.shape == (obs_size(21, 1, True, CHAMPION_FEATURES),)
    o1, _ = one.reset(seed=0)
    o2, _ = two.reset(seed=0)
    assert np.array_equal(o1, o2)          # truck 0 sees the same inputs either way
    zof = two.zones.zone_of_station
    rng = np.random.default_rng(0)
    done = False
    while not done:                        # one day
        k = two.queue[0]
        a = int(rng.integers(two.action_space.n))
        row = two.fb.decode(a)
        if row.any():
            terr = set((A, B)[k].tolist())
            assert zof[row[0] - 1] in terr and zof[row[1]] in terr
        obs, _, done, *_ = two.step(a)
        assert two.observation_space.contains(obs)
    # The forecast of the view sums the 40-zone forecast: territory zones as they are.
    fb = two.fbs[1]
    from rl import forecast as fc
    base = fc.load_table(two.zones.zone_of_station, two.env.station_ids)
    assert np.allclose(fb.fc_table[:, :20], base[:, B])
    assert np.allclose(fb.fc_table.sum(1), base.sum(1), rtol=1e-4)


# ── round 13: planner, inventory dispatcher ──────────────────────────────────
from rl.features import CHAMPION_FEATURES  # noqa: E402
def _tiny_dqn():
    from rl.train import build_model
    return build_model("dqn", 1, 0, 1, 1000, features=CHAMPION_FEATURES)


def test_planner_with_one_candidate_is_the_dqn():
    from rl.eval_round12 import run as dqn_run
    from rl.planner import ScenarioBank, run as plan_run
    from rl.wrapper import static_data
    m = _tiny_dqn()
    bank = ScenarioBank(static_data()["ids"])
    a = plan_run(m, WINDOW[0], k=1, s=1, bank=bank)
    b = dqn_run([m], None, "flow_w0.3_k20_need", WINDOW[0])
    assert a["total_reward"] == b["total_reward"] and a["changed"] == 0
    assert a["orders"] == b["orders"]


def test_planner_never_touches_the_real_run_or_its_future_trips():
    import copy
    from rl.planner import Planner, ScenarioBank
    from rl.wrapper import static_data
    env = RebalanceEnv(1, windows=WINDOW, features=CHAMPION_FEATURES)
    obs, _ = env.reset(seed=0)
    for _ in range(3):                        # a few orders in, mid-day
        obs, *_ = env.step(int(np.argmax(env.fb.action_mask())) or 1)
    bank = ScenarioBank(static_data()["ids"])
    assert bank.trips.start_min.max() < bank.n_days * MINUTES_PER_DAY   # bank ends Jul 17
    pl = Planner(_tiny_dqn(), env, bank, k=4, s=2, horizon=120)
    r = env.env
    before = (r.t, r.bikes.copy(), r.trip_ptr, copy.deepcopy(dict(r.returns)), dict(r.totals),
              [(tk.location, tk.load) for tk in r.trucks])
    a = pl.act(obs)
    after = (r.t, r.bikes, r.trip_ptr, dict(r.returns), dict(r.totals),
             [(tk.location, tk.load) for tk in r.trucks])
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert 0 <= a < env.action_space.n
    future, returns = bank.scenario(10, r.t % MINUTES_PER_DAY, r.t, 120)
    assert (future.start_min >= r.t).all() and (future.start_min < r.t + 120).all()
    assert min(returns) >= r.t
    assert pl.sim.trips is not r.trips and (pl.sim.trips.start_min >= r.t).all()


def test_expected_cost_matches_simple_cases():
    from rl.inventory import expected_cost
    cap = np.array([3, 5])
    zero = np.zeros((24, 2))
    assert np.allclose(expected_cost(zero, zero, cap, 60, 0, 4.0, 1.0), 0)
    rent = np.full((24, 2), 0.05)             # rentals only: more bikes, fewer lost
    F = expected_cost(rent, zero, cap, 120, 0, 4.0, 1.0)
    assert (np.diff(F[0, :4]) < 0).all() and np.isclose(F[0, 0], 4.0 * 0.05 * 120)
    F = expected_cost(zero, rent, cap, 120, 0, 4.0, 1.0)   # returns only: full blocks
    assert (np.diff(F[1, :6]) > 0).all() and np.isclose(F[1, 5], 1.0 * 0.05 * 120)


def test_inventory_dispatcher_orders_are_valid_and_a_high_theta_waits():
    from rl.inventory import InventoryDispatcher
    env = RebalanceEnv(1, windows=WINDOW, features=CHAMPION_FEATURES)
    env.reset(seed=0)
    while env.env.t < 600:                    # 10 am, inside the paid shift
        env.step(0)
    a = InventoryDispatcher(env, horizon=120, theta=0.0).act()
    assert 0 < a < env.action_space.n and env.fb.decode(a).any()
    assert InventoryDispatcher(env, horizon=120, theta=1e6).act() == 0



# ── round 14: aged-starts validation ─────────────────────────────────────────
def test_pick_checkpoint_prefers_a_plateau_over_a_lone_spike():
    from rl.validation import pick_checkpoint
    s = [5, 6, 20, 4, 5, 14, 15, 14, 6]
    assert pick_checkpoint(s, 1) == 2          # plain best: the spike
    assert pick_checkpoint(s, 3) == 6          # 3-checkpoint average: the plateau
    assert pick_checkpoint([3.0], 3) == 0


def test_default_pick_is_the_plain_best_s1():
    # round 14: averaging neighbouring checkpoints picked worse ones, so the default is width 1
    from rl.validation import PICK_WIDTH, pick_checkpoint
    assert PICK_WIDTH == 1
    assert pick_checkpoint([1.0, 2.0, 9.0, 2.0, 1.0, 5.0, 6.0, 5.0]) == 2


def test_aged_validation_scores_only_held_out_days_with_matching_no_truck_runs():
    from rl.validation import SCHEMES, VAL_DAYS
    assert all(scored == VAL_DAYS and w[1] == VAL_DAYS[1] for w, scored in SCHEMES["S1"])
    assert sorted(w[0] for w, _ in SCHEMES["S1"]) == ["2026-06-06", "2026-06-20",
                                                      "2026-07-04", "2026-07-18"]
    for scheme in SCHEMES.values():            # nothing on or after the test's first day
        assert all(w[1] < "2026-08-01" for w, _ in scheme)


def test_training_callback_keeps_checkpoints_and_the_picked_best(tmp_path):
    import json
    import pandas as pd
    from rl.train import build_model
    from rl.validation import make_callback
    m = build_model("dqn", 1, 0, 1, 300, features=CHAMPION_FEATURES)
    out = str(tmp_path)
    m.learn(250, callback=make_callback(out, 100, 1, None, CHAMPION_FEATURES))
    log = pd.read_csv(tmp_path / "validation.csv")
    assert list(log.steps) == [100, 200] and log.S1.notna().all()
    pick = json.load(open(tmp_path / "best_pick.json"))
    assert pick["steps"] in (100, 200) and (tmp_path / "best_model.zip").exists()
    assert (tmp_path / "checkpoints" / "model_100_steps.zip").exists()


# ── round 15: coordinated 2-truck DQN ────────────────────────────────────────
def test_daily_gain_fleet_cost_scales_with_number_trucks():
    from rl.validation import daily_gain
    wait = lambda o, e: 0                                  # noqa: E731
    day = ("2026-07-18", "2026-07-18")
    g1 = daily_gain(wait, day, None, number_trucks=1)
    g2 = daily_gain(wait, day, None, number_trucks=2)
    assert g1.shape == g2.shape == (1,)
    assert g1[0] < 0 and g2[0] == pytest.approx(2 * g1[0])  # idle fleet still costs its daily rate


def test_score_forwards_number_trucks_to_every_window(monkeypatch):
    import rl.validation as val
    seen = []

    def fake_daily_gain(choose, window, zone_scheme, features=None, stop_rule=None,
                        number_trucks=1):
        seen.append(number_trucks)
        return np.zeros(len(__import__("pandas").date_range(*window)))
    monkeypatch.setattr(val, "daily_gain", fake_daily_gain)
    val.score(lambda o, e: 0, "S0", None, number_trucks=2)
    assert seen == [2, 2]                                   # both S0 windows


def test_other_truck_block_shows_a_same_minute_pending_order_not_yet_applied():
    from rl import forecast as fc
    env = RebalanceEnv(number_trucks=2, windows=WINDOW)
    env.reset(seed=0)
    assert set(env.queue) == {0, 1}                         # both idle at the depot, minute 0
    k0, k1 = env.queue[0], env.queue[1]
    rng = np.random.default_rng(3)
    a0 = next(int(a) for a in rng.permutation(env.action_space.n) if env.fb.decode(a).any())
    row0 = env.fb.decode(a0)
    obs1, r, done, _, info = env.step(a0)                   # k0 decides; k1 is asked next
    assert not done and r == 0.0 and info["asking_truck"] == k1
    Z = env.zones.n_zones
    n_fc = len(fc.TARGETS) * Z
    base = 5 * Z + n_fc + 10                                # start of the (only) other-truck block
    zof = env.zones.zone_of_station
    assert np.array_equal(obs1[base: base + Z],
                          np.eye(Z)[zof[env.env.trucks[k0].location]])
    assert tuple(obs1[base + Z: base + Z + 2]) == (1.0, 1.0)   # shown busy, full minutes-left
    dest = zof[row0[1]]
    assert obs1[base + Z + 2: base + 2 * Z + 2].argmax() == dest
    assert obs1[base + 2 * Z + 2] == pytest.approx(row0[2] / env.env.truck_capacity)
    assert not env.env.trucks[k0].busy(env.env.t)           # not actually dispatched yet


def test_other_truck_block_reflects_a_real_mid_mission_destination_and_load():
    from rl import forecast as fc
    env = RebalanceEnv(number_trucks=2, windows=WINDOW)
    env.reset(seed=0)
    k0, k1 = env.queue[0], env.queue[1]
    rng = np.random.default_rng(4)
    a0 = next(int(a) for a in rng.permutation(env.action_space.n) if env.fb.decode(a).any())
    row0 = env.fb.decode(a0)
    env.step(a0)                        # k0 decides a real order (not yet applied)
    obs1, r, done, _, info = env.step(0)   # k1 waits; the minute is now simulated for real
    assert info["asking_truck"] == k1 and not done
    tk0 = env.env.trucks[k0]
    assert tk0.busy(env.env.t)           # k0 is now genuinely mid-mission
    Z = env.zones.n_zones
    n_fc = len(fc.TARGETS) * Z
    base = 5 * Z + n_fc + 10
    zof = env.zones.zone_of_station
    want_dest = zof[tk0.ops[-1][1]] if tk0.ops else zof[tk0.location]
    assert want_dest == zof[row0[1]]     # still heading to the ordered drop-off zone
    assert obs1[base + Z + 2: base + 2 * Z + 2].argmax() == want_dest
    assert obs1[base + 2 * Z + 2] == pytest.approx(tk0.load / env.env.truck_capacity)
    assert obs1[base + Z] == 1.0         # busy flag true (real state, not a pending override)


def test_two_truck_city_wide_training_run_uses_aged_validation(tmp_path, capsys):
    from rl.train import build_model
    from rl.validation import make_callback
    m = build_model("dqn", 2, 0, 1, 300, features=CHAMPION_FEATURES)
    out = str(tmp_path)
    m.learn(200, callback=make_callback(out, 100, 1, None, CHAMPION_FEATURES, number_trucks=2))
    import pandas as pd
    log = pd.read_csv(tmp_path / "validation.csv")
    assert list(log.steps) == [100, 200] and log.S1.notna().all()
    assert (tmp_path / "best_model.zip").exists()
