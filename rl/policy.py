"""Expose trained RL models as dashboard / simulate_day policies.

Importing this module registers one policy per trained model found under
models/<algo>_<k>trucks/best_model.zip (e.g. "ppo_2trucks") into POLICIES.
Decisions follow the training rule exactly: every minute, each idle truck is
asked in order; busy trucks are skipped.

Also registers "fix_shuttle_policy": the fixed route the 1-truck PPO/A2C agents
converged to, written as a plain rule (no learning), and "dqn_duplicated_2trucks":
two trucks driven by the 1-truck DQN as one (paired trucks).
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

    @classmethod
    def env_kwargs(cls) -> dict:
        kw = super().env_kwargs()
        if cls.multi_stop:
            kw.update(multi_stop_kwargs(cls.zone_scheme))
        if cls.paired:
            kw["paired_trucks"] = True
        return kw

    def reset(self, env) -> None:
        super().reset(env)
        self.env = env
        self.fb = FeatureBuilder(env, zz.load_zones(self.zone_scheme), forecast=self.forecast)
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
        return action


class RLPolicy(ZoneActionPolicy):
    algo = ""
    path = ""
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


# Dropdown order: No trucks, shuttle x2, then PPO / A2C / DQN x (1, 2) trucks.
POLICIES["fix_shuttle_policy"] = FixShuttlePolicy
POLICIES["fix_shuttle_policy_2trucks"] = FixShuttlePolicy2Trucks
register_trained_models()
if os.path.exists(DuplicatedTruckPolicy.path):
    POLICIES["dqn_duplicated_2trucks"] = DuplicatedTruckPolicy
