"""Rebalancing policies. Each maps (obs, info) -> action for BikeShareEnv.

Add a policy by subclassing Policy and listing it in POLICIES; the dashboard's
policy dropdown is built from that dict.
"""
import numpy as np


class Policy:
    label = "Base policy"
    number_trucks = 2   # fleet the policy is run with (hired every day, used or not)
    env_overrides: dict = {}   # extra env settings the policy needs (e.g. truck_capacity)

    @classmethod
    def env_kwargs(cls) -> dict:
        """Env settings this policy runs with (override for computed settings)."""
        return dict(cls.env_overrides)

    def reset(self, env) -> None:
        """Called once after env.reset(); `env` is the unwrapped BikeShareEnv."""
        self.number_trucks = env.number_trucks

    def act(self, obs: dict, info: dict) -> np.ndarray:
        raise NotImplementedError


class NoRebalancingPolicy(Policy):
    """Dummy baseline: never hires a truck or dispatches a mission."""
    label = "No trucks (dummy)"
    number_trucks = 0   # hires no trucks, so pays no truck costs

    def act(self, obs, info):
        return np.zeros((self.number_trucks, 3), dtype=np.int64)


POLICIES: dict[str, type[Policy]] = {
    "no_rebalancing": NoRebalancingPolicy,
}
