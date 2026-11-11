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
    assert env.action_space.n == n_actions(20) == 1 + 20 * 19 * len(QTY_LEVELS) == 1141
    assert env.observation_space.shape == (obs_size(20, env.K),) == ((110,) if env.K == 1 else (132,))


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

    env = RebalanceEnv(number_trucks=1, windows=WINDOW)
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
    env = RebalanceEnv(number_trucks=1, windows=WINDOW)
    env.reset(seed=0)
    zof, a, done = env.zones.zone_of_station, FixShuttlePolicy.action(), False
    while not done:
        row = env.fb.decode(a)
        if row.any():                            # always downtown (12) -> zone 11
            assert (zof[row[0] - 1], zof[row[1]], row[2]) == (12, 11, 10)
        _, _, done, _, info = env.step(a)
    assert r.reward.sum() == pytest.approx(info["true_total"])
    assert env.env.totals["bikes_moved"] > 0
