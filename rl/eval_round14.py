"""Round 14: judge validation schemes (rl/validation.py). See results/round14/preregistration.md.

    python -m rl.eval_round14 models       [--workers 8]   # A: past models vs their test results
    python -m rl.eval_round14 checkpoints  [--workers 8]   # B: picking rules inside a run

No test run is made: A reads the test results already in results/round{7,8,9,11}.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

OUT = os.path.join("results", "round14")
SCHEME_NAMES = ("S0", "S1", "S2")
CHAMP_ZONES = "flow_w0.3_k20_need"
CHAMP_FEATURES = "risk+travel+failures+memory"


# ── A: past models ───────────────────────────────────────────────────────────
def catalogue() -> list[dict]:
    """Every 1-truck learned model with a test result and a saved best checkpoint."""
    rows = []

    def add(rnd, label, folder, algo, zones, features, seed, t):
        rows.append({"round": rnd, "label": label, "folder": folder, "algo": algo, "zones": zones,
                     "features": features, "seed": int(seed), "test": t.vs_no_trucks,
                     "test_first14": t.first_14_days, "test_rest": t.rest,
                     "val_logged": t.val_per_week})

    for t in pd.read_csv("results/round7/rl_test_results.csv").itertuples():
        z, s = t.zones, int(t.seed)
        f = (os.path.join("models", "round4", "dqn_1trucks") if z == "kmeans"
             else os.path.join("models", "dqn_1trucks")) if s == 0 else \
            os.path.join("models", "round7", f"dqn_1trucks_{z}_s{s}")
        add(7, f"DQN {z}", f, "dqn", z, None, s, t)
    for t in pd.read_csv("results/round8/rl_test_results.csv").itertuples():
        if t.policy != "DQN" or t.zones == "flow_w0.3":          # flow_w0.3 = round 7's
            continue
        s = int(t.seed)
        add(8, f"DQN {t.zones}", os.path.join("models", "round8", f"dqn_1trucks_{t.zones}_s{s}"),
            "dqn", t.zones, None, s, t)
    for t in pd.read_csv("results/round9/rl_test_results.csv").itertuples():
        if t.features == "base":                                 # = round 8 flow_w0.3_k20_need
            continue
        add(9, f"DQN {t.features}", os.path.join("models", "round9",
                                                 f"dqn_1trucks_{t.features}_s{t.seed}"),
            "dqn", CHAMP_ZONES, t.features, t.seed, t)
    for t in pd.read_csv("results/round11/rl_test_results.csv").itertuples():
        if t.algo == "dqn":                                      # = round 9 champion
            continue
        add(11, t.algo, os.path.join("models", "round11", f"{t.algo}_1trucks_s{t.seed}"),
            t.algo, CHAMP_ZONES, CHAMP_FEATURES, t.seed, t)
    return rows


def chooser(algo: str, path: str):
    if algo == "rainbow":
        from rl.rainbow import RainbowDQN
        m = RainbowDQN.load(path)
    else:
        import sb3_contrib
        import stable_baselines3 as sb3
        cls = {"dqn": sb3.DQN, "ppo": sb3.PPO, "a2c": sb3.A2C, "qrdqn": sb3_contrib.QRDQN,
               "maskppo": sb3_contrib.MaskablePPO}[algo]
        m = cls.load(path, device="cpu")
    if algo == "maskppo":
        return lambda o, e: int(m.predict(o, action_masks=e.action_masks(), deterministic=True)[0])
    return lambda o, e: int(m.predict(o, deterministic=True)[0])


def model_job(row: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from rl.validation import score
    path = os.path.join(row["folder"], "best_model.pt" if row["algo"] == "rainbow" else "best_model.zip")
    choose = chooser(row["algo"], path)
    cache, out = {}, dict(row)
    for s in SCHEME_NAMES:
        r = score(choose, s, row["zones"], row["features"], cache=cache)
        out[s] = r["score"]
        out[f"{s}_runs"] = r["runs"]
    return out


def judge_models(workers: int) -> None:
    from scipy import stats
    rows = catalogue()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(model_job, rows))
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "schemes_models.csv"), index=False)
    lines = ["# A: validation schemes vs the test, across past models\n",
             f"{len(df)} models (rounds 7, 8, 9, 11; 1 truck; best checkpoints). Spearman "
             "correlation (p) between each scheme's $/week and the test gain.\n",
             "| Scheme | vs test (51 days) | vs test days 1-14 | vs test days 15-51 |",
             "|---|---|---|---|"]
    for s in SCHEME_NAMES + ("val_logged",):
        cells = []
        for c in ("test", "test_first14", "test_rest"):
            rho, p = stats.spearmanr(df[s], df[c])
            cells.append(f"{rho:.2f} ({p:.3f})")
        name = {"val_logged": "logged validation (training eval)"}.get(s, s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "S0 recomputed here should equal the logged validation (same runs).",
              "", "## Per model", "",
              "| round | model | seed | S0 | S1 | S2 | test |", "|---|---|---|---|---|---|---|"]
    for r in df.sort_values("test", ascending=False).itertuples():
        lines.append(f"| {r.round} | {r.label} | {r.seed} | {r.S0:,.0f} | {r.S1:,.0f} | "
                     f"{r.S2:,.0f} | {r.test:,.0f} |")
    open(os.path.join(OUT, "schemes_models.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


# ── B: checkpoints inside a run ──────────────────────────────────────────────
def checkpoint_job(path: str) -> dict:
    import torch
    torch.set_num_threads(1)
    from rl.validation import score
    choose = chooser("dqn", path)
    cache, out = {}, {"path": path}
    for s in ("S0", "S1"):
        r = score(choose, s, CHAMP_ZONES, CHAMP_FEATURES, cache=cache)
        out[s], out[f"{s}_wk1"], out[f"{s}_wk2"] = r["score"], *r["halves"]
    return out


def judge_checkpoints(workers: int) -> None:
    from rl.validation import pick_checkpoint
    paths = sorted(glob.glob(os.path.join("models", "round14", "dqn_1trucks_s*", "checkpoints", "*.zip")))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(checkpoint_job, paths))
    df = pd.DataFrame(res)
    df["run"] = df.path.str.extract(r"(dqn_1trucks_s\d)")
    df["steps"] = df.path.str.extract(r"_(\d+)_steps").astype(int)
    df = df.sort_values(["run", "steps"])
    df.to_csv(os.path.join(OUT, "schemes_checkpoints.csv"), index=False)
    rules = {"R0 best S0": ("S0", 1), "R1 best S1": ("S1", 1),
             "R2 best S1, 3-checkpoint average": ("S1", 3),
             "R3 best S1, 5-checkpoint average": ("S1", 5)}
    lines = ["# B: checkpoint-picking rules, pick on one validation week, check on the other\n",
             "Each rule picks a checkpoint from one week's scores (Jul 18-24 or Jul 25-31) and is "
             "scored by S1 on the other week, in $ for that week; mean of both directions.\n",
             "| Rule | " + " | ".join(sorted(df.run.unique())) + " | mean | picks (steps) |",
             "|---|" + "---|" * (df.run.nunique() + 2)]
    for name, (s, w) in rules.items():
        vals, picks = [], []
        for run, g in df.groupby("run"):
            g = g.reset_index(drop=True)
            both = []
            for pick_wk, check_wk in (("wk1", "wk2"), ("wk2", "wk1")):
                i = pick_checkpoint(g[f"{s}_{pick_wk}"].to_numpy(), w)
                both.append(g.loc[i, f"S1_{check_wk}"])
                picks.append(f"{run[-2:]}:{g.loc[i, 'steps'] // 1000}k")
            vals.append(np.mean(both))
        lines.append(f"| {name} | " + " | ".join(f"{v:,.0f}" for v in vals) +
                     f" | {np.mean(vals):,.0f} | {', '.join(picks)} |")
    lines += ["", "## Every checkpoint ($/week)", "", "| run | steps | S0 | S1 |", "|---|---|---|---|"]
    lines += [f"| {r.run} | {r.steps:,} | {r.S0:,.0f} | {r.S1:,.0f} |" for r in df.itertuples()]
    open(os.path.join(OUT, "schemes_checkpoints.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["models", "checkpoints"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    (judge_models if a.part == "models" else judge_checkpoints)(a.workers)


if __name__ == "__main__":
    main()
