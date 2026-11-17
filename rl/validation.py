"""Round 14: validation schemes that score a policy without touching the test period.

Every scheme scores only the held-out validation days Jul 18-31 (S0, S1) or days
before Aug 1 (S2), as $ per week against no trucks started on the same day.

  S0 fresh weeks   two runs from a fresh network (35% full): Jul 18-24 and Jul 25-31.
                   This is the validation used in rounds 1-13.
  S1 aged starts   four runs from a fresh network starting Jul 18, Jul 4, Jun 20 and
                   Jun 6, each to Jul 31; only Jul 18-31 is scored. By then the network
                   has run 1-14, 15-28, 29-42 and 43-56 days, the same mix of ages as the
                   51-day test (1-51). The earlier days only set up the bikes' positions.
                   Named in advance as the main scheme of round 14.
  S2 long run      one run Jun 11 -> Jul 31 (51 days, the test's length), all scored.
                   Mostly training days, so it may favour overfit policies.

A policy is `choose(obs, env) -> action`, called once per truck asked (regardless of
how many trucks `env` has): `number_trucks` (round 15) picks the fleet size the scheme
runs with, default 1.

Since round 14, S1 is the default validation in training (rl/train.py): every checkpoint
is kept and scored with S1, and the model kept as best_model is the checkpoint with the
best S1 (pick_checkpoint, width 1). Round 14 first averaged each checkpoint with its two
neighbours (width 3), but the pick-then-check on the retrained champion seeds showed that
neighbouring checkpoints are nearly unrelated policies (lag-1 correlation of S1 0.05-0.15)
and the averaging picked worse ones (results/round14/schemes_checkpoints.md), so the
default became the plain best S1.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY, MINUTES_PER_DAY

VAL_DAYS = ("2026-07-18", "2026-07-31")
S1_STARTS = ("2026-07-18", "2026-07-04", "2026-06-20", "2026-06-06")
S2_WINDOW = ("2026-06-11", "2026-07-31")
SCHEMES = {
    "S0": [(("2026-07-18", "2026-07-24"), ("2026-07-18", "2026-07-24")),
           (("2026-07-25", "2026-07-31"), ("2026-07-25", "2026-07-31"))],
    "S1": [((s, VAL_DAYS[1]), VAL_DAYS) for s in S1_STARTS],
    "S2": [(S2_WINDOW, S2_WINDOW)],
}   # scheme -> [(run window, scored days)]


def daily_gain(choose, window, zone_scheme, features=None, stop_rule=None,
               number_trucks: int = 1) -> np.ndarray:
    """$ per day vs no trucks over `window`, both runs started fresh on its first day."""
    from rl.wrapper import RebalanceEnv, no_truck_rewards
    env = RebalanceEnv(number_trucks=number_trucks, windows=[window], record_minutes=True,
                       zone_scheme=zone_scheme, features=features, stop_rule=stop_rule)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(choose(obs, env))
    base = no_truck_rewards(window, DEFAULT_INIT_OCCUPANCY)
    return (np.asarray(env.minute_rewards) - base).reshape(-1, MINUTES_PER_DAY).sum(1)


def score(choose, scheme: str, zone_scheme: str, features=None, stop_rule=None,
          cache: dict | None = None, number_trucks: int = 1) -> dict:
    """{"score": $/week over the scheme's runs, "runs": [$/week per run],
        "halves": [$/week on Jul 18-24, on Jul 25-31] (S0, S1)}.
    `cache` (window -> daily gains) lets schemes share runs of the same policy."""
    cache = {} if cache is None else cache
    per_run, halves = [], []
    for window, scored in SCHEMES[scheme]:
        if window not in cache:
            cache[window] = daily_gain(choose, window, zone_scheme, features, stop_rule,
                                       number_trucks)
        days = pd.date_range(*window)
        g = pd.Series(cache[window], index=days)[scored[0]:scored[1]]
        per_run.append(float(g.sum()) / len(g) * 7)
        if scheme != "S2":
            halves.append([float(g[:"2026-07-24"].sum()), float(g["2026-07-25":].sum())])
    out = {"score": float(np.mean(per_run)), "runs": per_run}
    if halves:
        h = np.asarray(halves)
        # S0 scores each week in its own run; S1 scores both weeks in every run.
        out["halves"] = ([h[0, 0], h[1, 1]] if scheme == "S0" else h.mean(0).tolist())
    return out


# ── checkpoint picking (default since round 14) ─────────────────────────────
PICK_WIDTH = 1          # plain best S1 (round 14: averaging neighbours picked worse ones)


def pick_checkpoint(scores, width: int = PICK_WIDTH) -> int:
    """Index of the checkpoint with the best score averaged over `width` neighbours
    (fewer at the ends). width 1 = plain best."""
    s = np.asarray(scores, dtype=float)
    h = width // 2
    smooth = [s[max(0, i - h): i + h + 1].mean() for i in range(len(s))]
    return int(np.argmax(smooth))


def model_chooser(model):
    """`choose(obs, env)` for an SB3 model (Maskable PPO gets the env's action masks)."""
    if type(model).__name__ == "MaskablePPO":
        return lambda o, e: int(model.predict(o, action_masks=e.action_masks(),
                                              deterministic=True)[0])
    return lambda o, e: int(model.predict(o, deterministic=True)[0])


def make_callback(out: str, eval_every: int, n_envs: int, zone_scheme=None, features=None,
                  stop_rule=None, number_trucks: int = 1):
    """SB3 callback: every `eval_every` decisions, save a checkpoint to <out>/checkpoints,
    score it with S1 (and S0, for comparison with earlier rounds), log both to
    <out>/validation.csv, and copy the checkpoint picked by pick_checkpoint over the S1
    scores so far to <out>/best_model.zip (pick recorded in <out>/best_pick.json).
    `number_trucks` (round 15): the fleet size to validate with; the same one model is
    asked once per idle truck, as in training."""
    from stable_baselines3.common.callbacks import BaseCallback

    class S1Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.freq = max(1, eval_every // n_envs)
            self.rows = []

        def _on_step(self) -> bool:
            if self.n_calls % self.freq:
                return True
            steps = int(self.num_timesteps)
            path = os.path.join(out, "checkpoints", f"model_{steps}_steps")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            choose, cache = model_chooser(self.model), {}
            s1 = score(choose, "S1", zone_scheme, features, stop_rule, cache=cache,
                      number_trucks=number_trucks)
            s0 = score(choose, "S0", zone_scheme, features, stop_rule, cache=cache,
                      number_trucks=number_trucks)
            self.rows.append({"steps": steps, "S1": s1["score"], "S0": s0["score"],
                              "S1_runs": json.dumps([round(x, 1) for x in s1["runs"]])})
            pd.DataFrame(self.rows).to_csv(os.path.join(out, "validation.csv"), index=False)
            i = pick_checkpoint([r["S1"] for r in self.rows])
            best = self.rows[i]
            shutil.copyfile(os.path.join(out, "checkpoints", f"model_{best['steps']}_steps.zip"),
                            os.path.join(out, "best_model.zip"))
            pick = {"steps": best["steps"], "S1": best["S1"], "rule": f"S1, width {PICK_WIDTH}"}
            if PICK_WIDTH > 1:
                pick["S1_smoothed"] = float(np.mean([r["S1"] for r in self.rows][
                    max(0, i - PICK_WIDTH // 2): i + PICK_WIDTH // 2 + 1]))
            json.dump(pick, open(os.path.join(out, "best_pick.json"), "w"))
            print(f"S1 validation at {steps:,}: S1 {s1['score']:,.0f} $/week "
                  f"(S0 {s0['score']:,.0f}); kept {best['steps']:,}", flush=True)
            return True

    return S1Validation()


def picked(folder: str) -> tuple[int, float] | None:
    """(steps, S1 $/week) of the checkpoint kept by S1 picking, if the run used it."""
    p = os.path.join(folder, "best_pick.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return int(d["steps"]), float(d["S1"])
