"""Rebalancing policies. Each maps (obs, info) -> action for BikeShareEnv.

Add a policy by subclassing Policy and listing it in POLICIES; the dashboard's
policy dropdown is built from that dict.
"""
import numpy as np


class Policy:
    label = "Base policy"

    def reset(self, env) -> None:
        """Called once after env.reset(); `env` is the unwrapped BikeShareEnv."""
        self.n_trucks = env.n_trucks

    def act(self, obs: dict, info: dict) -> np.ndarray:
        raise NotImplementedError


class NoRebalancingPolicy(Policy):
    """Dummy baseline: never hires a truck or dispatches a mission."""
    label = "No rebalancing (dummy)"

    def act(self, obs, info):
        return np.zeros((self.n_trucks, 3), dtype=np.int64)


POLICIES: dict[str, type[Policy]] = {
    "no_rebalancing": NoRebalancingPolicy,
}
