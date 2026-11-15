"""RL wrapper tests on real data (one short validation window)."""
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
    assert env.observation_space.shape == (obs_size(21, env.K),) == ((199,) if env.K == 1 else (222,))
    assert obs_size(20, env.K) == (190 if env.K == 1 else 212)                  # round 4
    assert obs_size(20, env.K, forecast=False) == (110 if env.K == 1 else 132)   # rounds 1-3


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


def test_duplicated_dqn_runs_both_trucks_on_every_order():
    import rl.policy as rp
    from bikeshare_env.rollout import simulate_day
    if "dqn_duplicated_2trucks" not in rp.POLICIES:
        pytest.skip("models/dqn_1trucks/best_model.zip not trained")
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
