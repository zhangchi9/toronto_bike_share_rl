"""Gymnasium environment for Bike Share Toronto rebalancing.

    import gymnasium as gym
    import bikeshare_env  # registers the env
    env = gym.make("BikeShareToronto-v0", init_occupancy=0.35,
                   simulation_period=("2026-06-01", "2026-06-07"))
"""
from gymnasium.envs.registration import register

register(
    id="BikeShareToronto-v0",
    entry_point="bikeshare_env.env:BikeShareEnv",
)
