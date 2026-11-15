"""Round 10: station rules inside a zone for multi-stop missions.

The agents pick a pickup zone, a drop-off zone and an amount. A stop rule decides
which stations of those zones the truck visits, in what order, and how many bikes
it moves at each (bikeshare_env.env.BikeShareEnv(stop_rule=...)).

A rule is "<order>/<amount>":
  orders
    most       most bikes (pickup) / most free docks (drop-off); the round 3 rule
    fill       highest fill ratio (pickup) / lowest fill ratio (drop-off)
    nearest    first stop as `most`, then the nearest usable station (road minutes)
    need       projected state in 2 h: bikes now + the station's average hourly net
               flow (training period, weekday / off-day). Pickup: projected bikes
               above half the docks; drop-off: below half the docks
    need_near  `need` minus NEAR_BIKES_PER_MIN bikes per road minute from the truck
  amounts
    cap5       at most 5 bikes per station (the round 3 rule)
    cap10      at most 10
    target     move the station toward half full: pick up only the bikes above
               ceil(docks / 2), drop only up to floor(docks / 2); stations already on
               the wrong side of half are skipped
"most/cap5" through this module gives exactly the round 3 rule (stop_rule=None).
"""

import numpy as np
import pandas as pd

ORDERS = ("most", "fill", "nearest", "need", "need_near")
AMOUNTS = {"cap5": 5, "cap10": 10, "target": None}
RULES = ("most/cap5", "fill/cap5", "nearest/cap5", "need/cap5", "need_near/cap5",
         "most/cap10", "fill/target", "need/target", "need_near/target")
DEFAULT_RULE = "most/cap5"
NEED_HOURS = 2
NEAR_BIKES_PER_MIN = 0.5


class StopRule:
    def __init__(self, order: str, amount: str):
        if order not in ORDERS or amount not in AMOUNTS:
            raise ValueError(f"unknown stop rule {order}/{amount}; orders {ORDERS}, "
                             f"amounts {tuple(AMOUNTS)}")
        self.order, self.amount_rule = order, amount
        self.cap = AMOUNTS[amount]
        self._calendar = {}

    @property
    def name(self) -> str:
        return f"{self.order}/{self.amount_rule}"

    def __repr__(self):
        return f"StopRule({self.name!r})"

    # ── amounts ──────────────────────────────────────────────────────────────
    def _room(self, b, c, pickup: bool):
        """Bikes the rule lets the truck move at stations with b bikes, c docks."""
        if self.cap is None:                                   # target: half full
            return b - np.ceil(c / 2) if pickup else np.floor(c / 2) - b
        return np.minimum(b if pickup else c - b, self.cap)

    def amount(self, env, st: int, pickup: bool, want: int) -> int:
        room = self._room(float(env.bikes[st]), float(env.capacity[st]), pickup)
        return int(max(0, min(want, room)))

    # ── orders ───────────────────────────────────────────────────────────────
    def _off_days(self, env) -> np.ndarray:
        key = (env.start_date, env.n_days)
        if key not in self._calendar:
            from rl import forecast as fc
            days = pd.date_range(env.start_date, periods=env.n_days + 1)
            self._calendar[key] = np.asarray(
                (days.dayofweek >= 5) | days.strftime("%Y-%m-%d").isin(fc.HOLIDAYS))
        return self._calendar[key]

    def projected(self, env, cand: np.ndarray) -> np.ndarray:
        """Bikes at `cand` in NEED_HOURS hours: now + average hourly net flow
        (training-period profiles, as the round 9 risk features)."""
        from rl.features import _profiles_bikes
        P = _profiles_bikes(env.station_ids, env.capacity)
        off = self._off_days(env)
        t = env.t
        d, h = t // 1440, (t % 1440) // 60
        cols = [(h + i) % 24 + 24 * int(off[min(d + (h + i) // 24, len(off) - 1)])
                for i in range(NEED_HOURS)]
        return env.bikes[cand] + P[cand][:, cols].sum(1)

    def scores(self, env, cand: np.ndarray, pickup: bool, frm: int,
               first: bool = False) -> np.ndarray:
        b = env.bikes[cand].astype(float)
        c = env.capacity[cand].astype(float)
        o = self.order
        if o == "most" or (o == "nearest" and first):
            s = b if pickup else c - b
        elif o == "fill":
            s = b / c if pickup else -b / c
        elif o == "nearest":
            s = -env.time_min[frm, cand]
        else:                                                  # need, need_near
            p = self.projected(env, cand)
            s = p - c / 2 if pickup else c / 2 - p
            if o == "need_near":
                s = s - NEAR_BIKES_PER_MIN * env.time_min[frm, cand]
        s = np.asarray(s, float).copy()
        s[self._room(b, c, pickup) <= 0] = -np.inf
        return s


def parse_rule(spec) -> StopRule | None:
    """None / "" / "none" -> None (the env's built-in round 3 rule);
    "need_near/target" or "need_near-target" -> StopRule."""
    if spec is None or isinstance(spec, StopRule):
        return spec
    if spec in ("", "none"):
        return None
    order, sep, amount = spec.replace("-", "/").rpartition("/")
    if not sep:
        raise ValueError(f"stop rule {spec!r} should look like order/amount, e.g. need/cap5")
    return StopRule(order, amount)


def rule_name(spec) -> str:
    """Folder-safe name: "need_near/target" -> "need_near-target"; None -> "default"."""
    r = parse_rule(spec)
    return r.name.replace("/", "-") if r else "default"
