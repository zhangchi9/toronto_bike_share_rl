"""Train a PPO / A2C / DQN rebalancing agent on May 1 - Jul 17, 2026.

    python -m rl.train --algo ppo --trucks 2 --timesteps 1000000

The best checkpoint on the validation windows (Jul 18-24, Jul 25-31) is saved
to models/<algo>_<k>trucks/best_model.zip; TensorBoard logs go to runs/.
"""
import argparse
import os

import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.wrapper import VAL_WINDOWS, RebalanceEnv

POLICY_KWARGS = {"net_arch": [256, 256]}
GAMMA = 0.999          # per-minute decisions: ~1,000-minute (most of a day) horizon


def model_dir(algo: str, trucks: int) -> str:
    return os.path.join("models", f"{algo}_{trucks}trucks")


def _make_env(trucks: int, seed: int):
    def thunk():
        env = Monitor(RebalanceEnv(number_trucks=trucks))
        env.reset(seed=seed)
        return env
    return thunk


EXPLORATION_STEPS = 200_000   # DQN: epsilon decays 1 -> 0.02 over this many decisions


def build_model(algo: str, trucks: int, seed: int, n_envs: int, timesteps: int = 1_000_000):
    common = dict(policy="MlpPolicy", gamma=GAMMA, policy_kwargs=POLICY_KWARGS,
                  seed=seed, device="cpu", verbose=0, tensorboard_log="runs")
    if algo == "dqn":
        venv = DummyVecEnv([_make_env(trucks, seed)])
        return DQN(env=venv, buffer_size=200_000, learning_starts=20_000,
                   batch_size=256, learning_rate=1e-4, train_freq=4,
                   target_update_interval=10_000,
                   exploration_fraction=min(1.0, EXPLORATION_STEPS / timesteps),
                   exploration_final_eps=0.02, **common)
    venv = SubprocVecEnv([_make_env(trucks, seed + i) for i in range(n_envs)])
    if algo == "ppo":
        return PPO(env=venv, n_steps=2048, batch_size=256, ent_coef=0.01, **common)
    if algo == "a2c":
        return A2C(env=venv, n_steps=64, ent_coef=0.01, **common)
    raise ValueError(f"unknown algo {algo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "a2c", "dqn"], required=True)
    p.add_argument("--trucks", type=int, choices=[1, 2], required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000, help="decisions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8, help="parallel envs (PPO/A2C)")
    p.add_argument("--eval-every", type=int, default=50_000, help="decisions")
    p.add_argument("--out", default=None, help="model folder (default models/<algo>_<k>trucks)")
    args = p.parse_args()

    torch.set_num_threads(2)
    out = args.out or model_dir(args.algo, args.trucks)
    os.makedirs(out, exist_ok=True)
    model = build_model(args.algo, args.trucks, args.seed, args.n_envs, args.timesteps)
    n_envs = model.get_env().num_envs

    eval_env = Monitor(RebalanceEnv(number_trucks=args.trucks, windows=VAL_WINDOWS))
    callback = EvalCallback(eval_env, n_eval_episodes=len(VAL_WINDOWS), deterministic=True,
                            eval_freq=max(1, args.eval_every // n_envs),
                            best_model_save_path=out, log_path=out, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback,
                tb_log_name=os.path.basename(os.path.normpath(out)), progress_bar=False)
    model.save(os.path.join(out, "final_model"))
    if not os.path.exists(os.path.join(out, "best_model.zip")):
        model.save(os.path.join(out, "best_model"))   # very short runs: no eval yet
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
