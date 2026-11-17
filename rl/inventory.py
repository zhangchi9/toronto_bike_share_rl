"""Round 13, idea 2: an inventory-model dispatcher (operations research, no learning).

Each station is modelled on its own as a queue: rentals and returns arrive at the
station's average rate for that hour and day type (weekday or weekend/holiday), from
the training period (May 1 - Jul 17). Backward dynamic programming gives, for every
station and every number of bikes x, the expected cost over the next H minutes:

    F(x) = $ per lost rental x E[rentals that find no bike]
         + $ per blocked return x E[returns that find no free dock]

(Raviv & Kolka style). A lost rental costs the unlock fee, the average ride's fee and
the lost-trip penalty; a blocked return costs the fee difference of an overflow return.

At each decision the dispatcher scores all 1,140 zone orders. It follows the same
station rule as the simulator (most bikes / most free docks first, at most 5 bikes a
stop) to see which stations an order would touch and how many bikes. Then:

    benefit = sum over drop-off stops of F(x) - F(x + n)
            - sum over pickup stops of F(x - n) - F(x)
    score   = benefit - fuel - overtime pay - theta x mission minutes

and takes the best order, or waits when no order scores above zero (and then looks
again RECHECK_MIN minutes later). theta is the
value of a truck-minute (the jobs the truck can't do meanwhile); H and theta are
tuned on the training period.
"""
import numpy as np
import pandas as pd

from bikeshare_env.env import MINUTES_PER_DAY
from rl import forecast as fc
from rl.features import MAX_BIKES_PER_STOP, QTY_LEVELS, encode_action

RATE_RANGE = ("2026-05-01", "2026-07-17")      # = wrapper.TRAIN_RANGE
RECHECK_MIN = 10        # after a wait, the next look is this many minutes later
_cache: dict = {}


def station_rates(station_ids) -> tuple[np.ndarray, np.ndarray, float]:
    """(rent, ret) per minute, shape (2 day types, 24 h, N), and the mean ride minutes."""
    key = ("rates", np.asarray(station_ids).tobytes())
    if key not in _cache:
        from bikeshare_env.trips import load_period_trips
        tr = load_period_trips(*RATE_RANGE, station_ids)
        n_days = (pd.Timestamp(RATE_RANGE[1]) - pd.Timestamp(RATE_RANGE[0])).days + 1
        days = pd.date_range(RATE_RANGE[0], periods=n_days + 1)
        off = np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
        n_type = np.array([(~off[:n_days]).sum(), off[:n_days].sum()], float)
        N = len(station_ids)

        def rate(minutes, idx):
            d = np.minimum(minutes // MINUTES_PER_DAY, n_days)
            h = (minutes % MINUTES_PER_DAY) // 60
            typ = off[d].astype(int)
            c = np.bincount((typ * 24 + h) * N + idx, minlength=2 * 24 * N).reshape(2, 24, N)
            return c / n_type[:, None, None] / 60.0

        keep = tr.end_min < (n_days * MINUTES_PER_DAY)
        _cache[key] = (rate(tr.start_min, tr.start_idx),
                       rate(tr.end_min[keep], tr.end_idx[keep]),
                       float(tr.rent_min.mean()))
    return _cache[key]


def expected_cost(rent, ret, capacity, horizon, start_hour, c_lost, c_block) -> np.ndarray:
    """(N, max capacity + 1): expected $ of lost rentals + blocked returns over `horizon`
    minutes from `start_hour`, starting with x bikes. rent/ret: (24, N) per minute."""
    N, C = len(capacity), int(capacity.max())
    x = np.arange(C + 1)[None, :]
    cap = capacity[:, None]
    down = np.maximum(x - 1, 0).repeat(N, 0)
    up = np.minimum(x + 1, cap)
    empty, full = (x == 0), (x >= cap)
    V = np.zeros((N, C + 1))
    for m in range(horizon - 1, -1, -1):          # backward over minutes
        h = (start_hour + m // 60) % 24
        p, q = rent[h][:, None], ret[h][:, None]
        sub = int(np.ceil(max(1.0, (p + q).max() * 1.25)))
        p, q = p / sub, q / sub
        for _ in range(sub):
            Vd = np.take_along_axis(V, down, 1) + c_lost * empty
            Vu = np.take_along_axis(V, up, 1) + c_block * full
            V = p * Vd + q * Vu + (1 - p - q) * V
    return V


class InventoryDispatcher:
    """Picks truck 0's order in a 1-truck RebalanceEnv (see module docstring)."""

    def __init__(self, env, horizon: int = 240, theta: float = 0.5):
        self.real = env
        e = env.env
        self.horizon, self.theta = int(horizon), float(theta)
        rent, ret, mean_ride = station_rates(e.station_ids)
        self.c_lost = e.unlock_fee + e.per_min_fee * mean_ride + e.lost_trip_penalty
        self.c_block = (e.per_min_fee - e.overflow_per_min_fee) * mean_ride
        key = ("F", e.station_ids.tobytes(), self.horizon)
        if key not in _cache:
            _cache[key] = np.stack([np.stack([
                expected_cost(rent[typ], ret[typ], e.capacity, self.horizon, h,
                              self.c_lost, self.c_block) for h in range(24)])
                for typ in range(2)])                  # (2, 24, N, C + 1)
        self.F = _cache[key]
        days = pd.date_range(e.start_date, periods=e.n_days + 2)
        self.off = np.asarray((days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
        self.hold_until = 0

    def _plan(self, stations, pickup: bool, want: int):
        """Follow the simulator's rule in one zone: stops, bikes per stop, cumulative."""
        e = self.real.env
        bikes = e.bikes[stations].astype(int)
        room = bikes if pickup else e.capacity[stations] - bikes
        order = np.argsort(-room, kind="stable")
        stops, amounts, left = [], [], want
        for i in order:
            if left <= 0 or room[i] <= 0:
                break
            n = int(min(MAX_BIKES_PER_STOP, room[i], left))
            stops.append(int(stations[i]))
            amounts.append(n)
            left -= n
        return stops, amounts

    def _value(self, F, stops, amounts, pickup: bool) -> float:
        e = self.real.env
        x = e.bikes[stops]
        n = np.asarray(amounts)
        if pickup:
            return float((F[stops, x - n] - F[stops, x]).sum())     # cost added
        return float((F[stops, x] - F[stops, x + n]).sum())         # cost removed

    def _route(self, frm, stops):
        e = self.real.env
        mins, km = 0.0, 0.0
        for st in stops:
            mins += np.ceil(e.time_min[frm, st]) if st != frm else 0
            km += e.dist_km[frm, st]
            frm = st
        return mins, km, frm

    def act(self) -> int:
        env, e = self.real, self.real.env
        fb = env.fb
        t = e.t
        if t < self.hold_until:           # nothing worth doing: look again in a while
            return 0
        a = self._best(env, e, fb, t)
        if a == 0:
            self.hold_until = t + RECHECK_MIN
        return a

    def _best(self, env, e, fb, t) -> int:
        F = self.F[int(self.off[t // MINUTES_PER_DAY]), (t % MINUTES_PER_DAY) // 60]
        tk = e.trucks[0]
        zones = fb.action_zones
        Z = len(zones)
        mod = t % MINUTES_PER_DAY
        overtime = 0.0 if e._in_shift(mod) else e._hourly_rate(mod // 60) / 60.0
        drop_plans = {}
        best, best_a = 0.0, 0
        for i, src in enumerate(zones):
            for q in QTY_LEVELS:
                q_eff = min(q, e.truck_capacity)
                stops, amounts = self._plan(fb.truck_stations[src], True, q_eff)
                load = int(sum(amounts))
                if not load:
                    continue
                cost_pick = self._value(F, stops, amounts, True)
                m1, k1, last = self._route(tk.location, stops)
                for j, dst in enumerate(zones):
                    if j == i:
                        continue
                    key = (j, load)
                    if key not in drop_plans:
                        drop_plans[key] = self._plan(fb.truck_stations[dst], False, load)
                    dstops, damounts = drop_plans[key]
                    if not dstops:
                        continue
                    gain = self._value(F, dstops, damounts, False)
                    m2, k2, _ = self._route(last, dstops)
                    minutes = m1 + m2 + load + sum(damounts)
                    score = (gain - cost_pick - e.fuel_cost_per_km * (k1 + k2)
                             - (overtime + self.theta) * minutes)
                    if score > best:
                        best, best_a = score, encode_action(i, j, q, Z)
        return best_a


def run(window, horizon=240, theta=0.5, zone_scheme="flow_w0.3_k20_need",
        features="risk+travel+failures+memory") -> dict:
    """One continuous dispatcher run over `window` (eval_round12.run's numbers)."""
    from collections import Counter
    from rl.wrapper import RebalanceEnv
    env = RebalanceEnv(number_trucks=1, windows=[window], record_minutes=True,
                       zone_scheme=zone_scheme, features=features)
    env.reset(seed=0)
    disp = InventoryDispatcher(env, horizon=horizon, theta=theta)
    zof = env.zones.zone_of_station
    orders, waits, done, n = Counter(), 0, False, 0
    while not done:
        a = disp.act()
        n += 1
        row = env.fb.decode(a)
        if row.any():
            orders[(int(zof[row[0] - 1]), int(zof[row[1]]), int(row[2]))] += 1
        else:
            waits += 1
        _, _, done, _, _ = env.step(a)
    T = env.env.totals
    return {"total_reward": float(np.sum(env.minute_rewards)),
            "daily": np.asarray(env.minute_rewards).reshape(-1, MINUTES_PER_DAY).sum(1),
            "fulfilled_pct": 100 * T["trips_started"] / (T["trips_started"] + T["lost_trips"]),
            "lost_requests": int(T["lost_trips"]), "blocked_returns": int(T["overflow_returns"]),
            "truck_cost": T["labor_cost"] + T["fuel_cost"] + T["fixed_cost"],
            "bikes_moved": int(T["bikes_moved"]), "truck_km": float(T["truck_km"]),
            "waits": [waits], "orders": [[[*key, c] for key, c in orders.most_common()]],
            "decisions": n}
