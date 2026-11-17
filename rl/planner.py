"""Round 13, idea 1: lookahead planning on top of a trained DQN.

At each decision the planner takes the DQN's K best orders (by Q) and tries each in
a copy of the simulator for H minutes, against S demand scenarios. The DQN drives
the truck for the rest of each try. It then picks the order with the best mean
score: the try's dollars / reward_scale, plus (optionally) the DQN's own value of
the state where the try ends.

No future trips of the real run are used. A scenario is a past day of the same
type (weekday or weekend/holiday) from the training period: its trips from H
minutes ahead, and its bikes out with riders right now (where and when they come
back), stand in for the real ones. Bikes on docks, bikes held back after a full
dock, the truck and the feature history are copied from the real run. All orders
of one decision face the same scenarios (common random numbers).

K = 1 reproduces the DQN exactly.
"""
import copy
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from bikeshare_env.env import MINUTES_PER_DAY
from bikeshare_env.trips import TripEvents, load_period_trips
from rl import forecast as fc

SCENARIO_RANGE = ("2026-06-01", "2026-07-17")   # inside the training period
LOOKBACK_MIN = 240                               # rides started this long ago may still be out


def _off_days(start, n_days) -> np.ndarray:
    days = pd.date_range(start, periods=n_days)
    return np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))


class ScenarioBank:
    """Training-period days, used as demand scenarios for the planner."""

    def __init__(self, station_ids, date_range=SCENARIO_RANGE):
        self.trips = load_period_trips(*date_range, station_ids)
        self.n_days = (pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])).days + 1
        self.off = _off_days(date_range[0], self.n_days)

    def days_like(self, off_day: bool, minute_of_day: int, horizon: int) -> np.ndarray:
        """Bank days of the same type whose window fits inside the bank."""
        d = np.arange(self.n_days)
        lo = d * MINUTES_PER_DAY + minute_of_day - LOOKBACK_MIN >= 0
        hi = d * MINUTES_PER_DAY + minute_of_day + horizon <= self.n_days * MINUTES_PER_DAY
        ok = (self.off == off_day) & lo & hi
        return d[ok] if ok.any() else d[lo & hi]

    def scenario(self, day: int, minute_of_day: int, t: int, horizon: int):
        """(future trips, in-flight returns) of bank `day`, moved onto real minute t."""
        tr = self.trips
        anchor = day * MINUTES_PER_DAY + minute_of_day
        shift = t - anchor
        i0 = np.searchsorted(tr.start_min, anchor - LOOKBACK_MIN, side="left")
        i1 = np.searchsorted(tr.start_min, anchor, side="left")
        i2 = np.searchsorted(tr.start_min, anchor + horizon, side="left")
        out = tr.end_min[i0:i1] > anchor                         # still riding at `anchor`
        returns = defaultdict(list)
        for e, st, rent in zip(tr.end_min[i0:i1][out] + shift, tr.end_idx[i0:i1][out],
                               tr.rent_min[i0:i1][out]):
            returns[int(max(e, t))].append((int(st), int(rent)))
        sl = slice(i1, i2)
        future = TripEvents(tr.start_min[sl] + shift, tr.end_min[sl] + shift,
                            tr.start_idx[sl], tr.end_idx[sl], tr.rent_min[sl])
        return future, returns


def q_values(model, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.q_net(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()


class Planner:
    """Chooses truck 0's order in a 1-truck RebalanceEnv (see module docstring)."""

    def __init__(self, model, real, bank: ScenarioBank, k: int = 8, s: int = 6,
                 horizon: int = 360, value_weight: float = 0.0, seed: int = 0):
        from rl.features import FeatureBuilder
        from rl.wrapper import make_base_env
        self.model, self.real, self.bank = model, real, bank
        self.k, self.s, self.horizon, self.value_weight = int(k), int(s), int(horizon), value_weight
        self.rng = np.random.default_rng(seed)
        self.scale = real.reward_scale
        empty = TripEvents(*(np.zeros(0, np.int64) for _ in range(5)))
        self.sim = make_base_env(real.window, real.K, real.init_occupancy, trips=empty,
                                 zone_scheme=real.zone_scheme, stop_rule=real.stop_rule)
        self.sim.reset(seed=0)
        self.fb = FeatureBuilder(self.sim, real.zones, forecast=real.forecast,
                                 features=real.features)
        self.fb.reset()
        self.off = _off_days(real.env.start_date, real.env.n_days + 2)
        self.n_decisions = self.n_changed = 0

    # ── one decision ─────────────────────────────────────────────────────────
    def act(self, obs: np.ndarray) -> int:
        q = q_values(self.model, obs)
        top = [int(a) for a in np.argsort(-q, kind="stable")[: self.k]]
        self.n_decisions += 1
        if len(top) == 1:
            return top[0]
        r = self.real.env
        t = r.t
        mod = t % MINUTES_PER_DAY
        horizon = min(self.horizon, r.total_minutes - t)
        pool = self.bank.days_like(bool(self.off[t // MINUTES_PER_DAY]), mod, horizon)
        days = self.rng.choice(pool, size=self.s, replace=len(pool) < self.s)
        scen = [self.bank.scenario(int(d), mod, t, horizon) for d in days]
        score = np.zeros(len(top))
        for i, a in enumerate(top):
            for future, returns in scen:
                score[i] += self._rollout(a, future, returns, t + horizon)
        best = top[int(np.argmax(score))]
        self.n_changed += best != top[0]
        return best

    def _load(self, future, returns) -> None:
        """Copy the real run's state into the simulator copy, with scenario demand."""
        r, e, fb, rfb = self.real.env, self.sim, self.fb, self.real.fbs[0]
        e.t = r.t
        e.bikes = r.bikes.copy()
        e.trips = future
        e.trip_ptr = 0
        e.trip_started = np.zeros(len(future), dtype=bool)
        e.returns = defaultdict(list, {m: list(v) for m, v in returns.items()})
        e.overflow = defaultdict(list, {m: list(v) for m, v in r.overflow.items()})
        e.trucks = copy.deepcopy(r.trucks)
        e.totals = dict.fromkeys(r.totals, 0.0)
        fb.history = deque(rfb.history, maxlen=rfb.history.maxlen)
        fb.failures = deque(rfb.failures, maxlen=rfb.failures.maxlen)
        fb._trip_ptr = 0
        fb.last_pick, fb.last_drop = rfb.last_pick.copy(), rfb.last_drop.copy()

    def _rollout(self, first: int, future, returns, t_end: int) -> float:
        self._load(future, returns)
        e, fb = self.sim, self.fb
        noop = np.zeros((1, 3), dtype=np.int64)
        total, a = 0.0, first
        while e.t < t_end:
            if not e.trucks[0].busy(e.t):
                if a is None:
                    a = int(np.argmax(q_values(self.model, fb.obs(0))))
                row = fb.decode(a)
                fb.note(0, row)
                act, a = row[None], None
            else:
                act = noop
            _, rew, *_ = e.step(act)
            fb.record()
            total += rew
        out = total / self.scale
        if self.value_weight:
            out += self.value_weight * float(q_values(self.model, fb.obs(0)).max())
        return out


def run(model, window, k=8, s=6, horizon=360, value_weight=0.0, seed=0, bank=None,
        zone_scheme="flow_w0.3_k20_need", features="risk+travel+failures+memory",
        progress=None) -> dict:
    """One continuous planner run over `window` (eval_round12.run's numbers).
    `progress(daily_rewards)` is called after each finished day, if given."""
    from collections import Counter
    from rl.wrapper import RebalanceEnv, static_data
    env = RebalanceEnv(number_trucks=1, windows=[window], record_minutes=True,
                       zone_scheme=zone_scheme, features=features)
    obs, _ = env.reset(seed=0)
    bank = bank or ScenarioBank(static_data()["ids"])
    pl = Planner(model, env, bank, k=k, s=s, horizon=horizon, value_weight=value_weight,
                 seed=seed)
    zof = env.zones.zone_of_station
    orders, waits, done, day = Counter(), 0, False, 0
    while not done:
        if progress is not None and env.env.t // MINUTES_PER_DAY > day:
            day = env.env.t // MINUTES_PER_DAY
            progress(np.asarray(env.minute_rewards[: day * MINUTES_PER_DAY])
                     .reshape(-1, MINUTES_PER_DAY).sum(1))
        a = pl.act(obs)
        row = env.fb.decode(a)
        if row.any():
            orders[(int(zof[row[0] - 1]), int(zof[row[1]]), int(row[2]))] += 1
        else:
            waits += 1
        obs, _, done, _, _ = env.step(a)
    T = env.env.totals
    return {"total_reward": float(np.sum(env.minute_rewards)),
            "daily": np.asarray(env.minute_rewards).reshape(-1, MINUTES_PER_DAY).sum(1),
            "fulfilled_pct": 100 * T["trips_started"] / (T["trips_started"] + T["lost_trips"]),
            "lost_requests": int(T["lost_trips"]), "blocked_returns": int(T["overflow_returns"]),
            "truck_cost": T["labor_cost"] + T["fuel_cost"] + T["fixed_cost"],
            "bikes_moved": int(T["bikes_moved"]), "truck_km": float(T["truck_km"]),
            "waits": [waits], "orders": [[[*key, n] for key, n in orders.most_common()]],
            "decisions": pl.n_decisions, "changed": pl.n_changed}
