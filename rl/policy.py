"""Expose trained RL models as dashboard / simulate_day policies.

Importing this module registers one policy per trained model found under
models/<algo>_<k>trucks/best_model.zip (e.g. "ppo_2trucks") into POLICIES.
Decisions follow the training rule exactly: every minute, each idle truck is
asked in order; busy trucks are skipped.

Also registers "fix_shuttle_policy": the fixed route the 1-truck PPO/A2C agents
converged to, written as a plain rule (no learning).
"""
import os

import numpy as np

from bikeshare_env.policies import POLICIES, Policy
from rl.features import TRUCK_CAPACITY, FeatureBuilder, encode_action
from rl.train import model_dir
from rl.wrapper import static_data

ALGOS = {"ppo": "PPO", "a2c": "A2C", "dqn": "DQN"}


class ZoneActionPolicy(Policy):
    """Asks `choose(obs_vector)` for every idle truck each minute, like RebalanceEnv."""
    env_overrides = {"truck_capacity": TRUCK_CAPACITY}   # same trucks as in training

    def reset(self, env) -> None:
        super().reset(env)
        self.env = env
        self.fb = FeatureBuilder(env, static_data()["zones"])
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
    source_zone, dest_zone, qty = 12, 11, 10

    @classmethod
    def action(cls) -> int:
        return encode_action(cls.source_zone, cls.dest_zone, cls.qty,
                             static_data()["zones"].n_zones)

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


class FixShuttlePolicy2Trucks(FixShuttlePolicy):
    label = "Fixed shuttle (2 trucks)"
    number_trucks = 2


# Dropdown order: No trucks, shuttle x2, then PPO / A2C / DQN x (1, 2) trucks.
POLICIES["fix_shuttle_policy"] = FixShuttlePolicy
POLICIES["fix_shuttle_policy_2trucks"] = FixShuttlePolicy2Trucks
register_trained_models()
