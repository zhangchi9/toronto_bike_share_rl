"""Expose trained RL models as dashboard / simulate_day policies.

Importing this module registers one policy per trained model found under
models/<algo>_<k>trucks/best_model.zip (e.g. "ppo_2trucks") into POLICIES.
Decisions follow the training rule exactly: every minute, each idle truck is
asked in order; busy trucks are skipped.

Since 2026-11-17 the dashboard shows 5 solutions only (DASHBOARD_POLICIES):
  no_rebalancing          no trucks
  fix_shuttle_policy      the fixed route the 1-truck PPO/A2C agents converged to,
                          written as a plain rule (no learning), 1 truck
  dqn_1truck              the round 9 champion DQN (seed 2: best S1 validation, round 14)
  dqn_planner_1truck      the round 13 lookahead planner on that same DQN (the champion)
  dqn_territory_2trucks   round 12's territory pair, seed 2 (best validation): one DQN
                          per half of the city
The older policies (PPO / A2C / DQN on the round 5 zones, the duplicated DQN, the
2-truck shuttle) stay defined here but are no longer registered.
"""
import os

import numpy as np

from bikeshare_env.policies import POLICIES, Policy
from rl import zones as zz
from rl.features import TRUCK_CAPACITY, FeatureBuilder, encode_action, obs_size, truck_zones
from rl.train import model_dir
from rl.wrapper import multi_stop_kwargs, static_data

ALGOS = {"ppo": "PPO", "a2c": "A2C", "dqn": "DQN"}


class ZoneActionPolicy(Policy):
    """Asks `choose(obs_vector)` for every idle truck each minute, like RebalanceEnv."""
    env_overrides = {"truck_capacity": TRUCK_CAPACITY}   # same trucks as in training
    multi_stop = True       # round-3 rule: <= 5 bikes per station, several stops per zone
    paired = False          # both trucks run every order together (env paired_trucks)
    forecast = True         # round-4 demand forecast features in the observation
    zone_scheme = None      # None = zones.ZONE_SCHEME (what the agents were trained on)
    features = ()           # round-9 extra feature groups (rl.features.FEATURE_GROUPS)
    stop_rule = None        # round-10 station rule (rl.stop_rules; None = round 3 rule)

    @classmethod
    def env_kwargs(cls) -> dict:
        kw = super().env_kwargs()
        if cls.multi_stop:
            kw.update(multi_stop_kwargs(cls.zone_scheme, cls.stop_rule))
        if cls.paired:
            kw["paired_trucks"] = True
        return kw

    def reset(self, env) -> None:
        super().reset(env)
        self.env = env
        self.fb = FeatureBuilder(env, zz.load_zones(self.zone_scheme), forecast=self.forecast,
                                 features=self.features)
        self.fb.reset()
        self.first = True

    def choose(self, obs_vector: np.ndarray) -> int:
        raise NotImplementedError

    def act(self, obs, info):
        if not self.first:
            self.fb.record()          # one entry per simulated minute, as in training
        self.first = False
        action = np.zeros((self.number_trucks, 3), dtype=np.int64)
        for k, tk in enumerate(self.env.trucks):
            if tk.busy(self.env.t):
                continue
            action[k] = self.fb.decode(self.choose(self.fb.obs(k)))
            self.fb.note(k, action[k])
        return action


class RLPolicy(ZoneActionPolicy):
    algo = ""
    path = ""
    zone_scheme = zz.ROUND5_SCHEME   # models/<algo>_<k>trucks were trained on these zones
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            from stable_baselines3 import A2C, DQN, PPO
            cls._model = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[cls.algo].load(cls.path, device="cpu")
        return cls._model

    def choose(self, obs_vector):
        return int(self.model().predict(obs_vector, deterministic=True)[0])


class FixShuttlePolicy(ZoneActionPolicy):
    """Always shuttle bikes from the downtown core (zone 12: Bay/Albert, Simcoe,
    York/Queens Quay) to the area just north (zone 11: Church-Wellesley, Jarvis,
    Sherbourne, Annex), with every idle truck, at any hour. No learning."""
    label = "Fixed shuttle (1 truck)"
    number_trucks = 1
    multi_stop = False      # keeps the single pickup / single drop-off station rule
    forecast = False        # it ignores its inputs
    zone_scheme = "kmeans"  # zones 12 and 11 are k-means zones: same rule as rounds 1-4
    source_zone, dest_zone, qty = 12, 11, 10

    @classmethod
    def action(cls) -> int:
        z = zz.load_zones(cls.zone_scheme)
        return encode_action(cls.source_zone, cls.dest_zone, cls.qty,
                             len(truck_zones(z, static_data()["ids"])))

    def choose(self, obs_vector):
        return self.action()


def register_trained_models() -> list[str]:
    added = []
    for algo, name in ALGOS.items():
        for k in (1, 2):
            path = os.path.join(model_dir(algo, k), "best_model.zip")
            key = f"{algo}_{k}trucks"
            if not os.path.exists(path) or key in POLICIES:
                continue
            POLICIES[key] = type(f"{name}{k}Policy", (RLPolicy,), {
                "label": f"{name} ({k} truck{'s' * (k > 1)})",
                "algo": algo, "path": path, "number_trucks": k,
            })
            added.append(key)
    return added


class DuplicatedTruckPolicy(RLPolicy):
    """Two trucks driven by the 1-truck DQN as if they were one. When both are
    idle, the model (shown truck 0 as the only truck) picks one order; both trucks
    run it, each collecting the ordered amount, truck 0 on the zone's ranks 1, 3,
    5, ... and truck 1 on ranks 2, 4, 6, ... (env paired_trucks)."""
    label = "DQN duplicated (2 trucks)"
    algo = "dqn"
    path = os.path.join(model_dir("dqn", 1), "best_model.zip")
    number_trucks = 2
    paired = True
    qty_divisor = 1         # 2 = each truck carries half the order (same bikes as 1 truck)

    def act(self, obs, info):
        if not self.first:
            self.fb.record()
        self.first = False
        action = np.zeros((self.number_trucks, 3), dtype=np.int64)
        if any(tk.busy(self.env.t) for tk in self.env.trucks):
            return action
        n1 = obs_size(self.fb.z.n_zones, 1, self.forecast)   # the 1-truck model's input
        row = self.fb.decode(self.choose(self.fb.obs(0)[:n1]))
        row[2] //= self.qty_divisor
        action[0] = row                           # paired env reads row 0 only
        return action


class FixShuttlePolicy2Trucks(FixShuttlePolicy):
    label = "Fixed shuttle (2 trucks)"
    number_trucks = 2


# ── the dashboard's current solutions (2026-11-17) ────────────────────────────
CHAMP_SEED = 2          # champion seed with the best S1 validation (round 14)
CHAMP_ZONES = "flow_w0.3_k20_need"
CHAMP_FEATURES = "risk+travel+failures+memory"
TERRITORY_ZONES = "flow_w0.3_k40_need"
TERRITORY_SEED = 2      # round 12 pair with the best validation


class ChampionDQNPolicy(RLPolicy):
    """The round 9 champion DQN on its own (1 truck, 20 zones, all four feature groups)."""
    label = "DQN (1 truck)"
    algo = "dqn"
    path = os.path.join("models", "round9", f"dqn_1trucks_{CHAMP_FEATURES}_s{CHAMP_SEED}",
                        "best_model.zip")
    number_trucks = 1
    zone_scheme = CHAMP_ZONES
    features = CHAMP_FEATURES
    _model = None


class PlannerPolicy(ChampionDQNPolicy):
    """Round 13's lookahead planner on the champion DQN, with its tuned settings
    (results/round13/planner_best.json). Slow: each order tries the DQN's 8 best
    orders against 6 past-day scenarios for 24 hours."""
    label = "DQN + lookahead planner (1 truck)"

    def reset(self, env) -> None:
        import json
        from types import SimpleNamespace

        import pandas as pd

        from rl.planner import Planner, ScenarioBank
        from rl.wrapper import static_data
        super().reset(env)
        best = json.load(open(os.path.join("results", "round13", "planner_best.json")))
        end = env.start_date + pd.Timedelta(days=env.n_days - 1)
        real = SimpleNamespace(            # what Planner reads from a 1-truck RebalanceEnv
            env=env, fbs=[self.fb], window=(str(env.start_date.date()), str(end.date())),
            K=1, init_occupancy=env.init_occupancy, zone_scheme=self.zone_scheme,
            stop_rule=self.stop_rule, zones=self.fb.z, forecast=self.forecast,
            features=self.fb.features, reward_scale=10.0)
        self.planner = Planner(self.model(), real, ScenarioBank(static_data()["ids"]),
                               k=best["k"], s=best["s"], horizon=best["horizon"],
                               value_weight=best["value_weight"],
                               seed=CHAMP_SEED)   # scenario draws as in round 13's test

    def choose(self, obs_vector):
        return self.planner.act(obs_vector)


class TerritoryPairPolicy(ZoneActionPolicy):
    """Round 12: truck 0 driven by territory A's DQN, truck 1 by territory B's. Each
    sees only its own zones plus one outside zone, and orders moves inside them."""
    label = "DQN (2 trucks, one territory each)"
    number_trucks = 2
    zone_scheme = TERRITORY_ZONES
    features = CHAMP_FEATURES
    paths = [os.path.join("models", "round12", f"dqn_1trucks_territory{t}_s{TERRITORY_SEED}",
                          "best_model.zip") for t in "AB"]
    path = paths[0]         # dashboard_data's fingerprint (rebuild when retrained)
    _models = None

    def reset(self, env) -> None:
        Policy.reset(self, env)
        from rl.territory import load_territories, territory_view
        self.env = env
        zones = zz.load_zones(self.zone_scheme)
        terr = load_territories(self.zone_scheme)
        self.fbs = [FeatureBuilder(env, territory_view(zones, t), forecast=self.forecast,
                                   features=self.features, action_zone_ids=np.arange(len(t)),
                                   others=False, forecast_zones=zones) for t in terr]
        for fb in self.fbs:
            fb.reset()
        self.first = True
        if TerritoryPairPolicy._models is None:
            from stable_baselines3 import DQN
            TerritoryPairPolicy._models = [DQN.load(p, device="cpu") for p in self.paths]

    def act(self, obs, info):
        if not self.first:
            for fb in self.fbs:
                fb.record()
        self.first = False
        action = np.zeros((self.number_trucks, 3), dtype=np.int64)
        for k, tk in enumerate(self.env.trucks):
            if tk.busy(self.env.t):
                continue
            fb = self.fbs[k]
            a = int(self._models[k].predict(fb.obs(k), deterministic=True)[0])
            action[k] = fb.decode(a)
            fb.note(k, action[k])
        return action


# Dropdown order (no_rebalancing is first, from bikeshare_env.policies).
DASHBOARD_POLICIES = {
    "fix_shuttle_policy": FixShuttlePolicy,
    "dqn_1truck": ChampionDQNPolicy,
    "dqn_planner_1truck": PlannerPolicy,
    "dqn_territory_2trucks": TerritoryPairPolicy,
}
for _key, _cls in DASHBOARD_POLICIES.items():
    if getattr(_cls, "path", "") and not os.path.exists(_cls.path):
        print(f"dashboard: {_key} skipped, {_cls.path} not found")
        continue
    POLICIES[_key] = _cls
