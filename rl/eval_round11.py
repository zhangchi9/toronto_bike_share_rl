"""Round 11: six RL algorithms, 1 truck, test Aug 1 -> Sep 20.

    python -m rl.eval_round11 [--out results/round11] [--workers 9]

All use zones flow_w0.3_k20_need, all four feature groups, today's station rule,
1M training decisions and the best checkpoint on the validation weeks:
  dqn      round 9 champion, unchanged (models/round9/dqn_1trucks_<features>_s<seed>)
  ppo, a2c                       stable-baselines3
  qrdqn, maskppo                 sb3-contrib (Maskable PPO tested with its masks)
  rainbow                        rl/rainbow.py
The five new ones are in models/round11/<algo>_1trucks_s<seed>/.
Writes rl_test_results.{md,csv}, round11.json and rl_test_daily.png to --out.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ZONES = "flow_w0.3_k20_need"
FEATURES = "risk+travel+failures+memory"
SEEDS = (0, 1, 2)
EARLY_DAYS = 14
ALGOS = ("dqn", "ppo", "a2c", "qrdqn", "maskppo", "rainbow")
LABEL = {"dqn": "DQN (champion)", "ppo": "PPO", "a2c": "A2C", "qrdqn": "QR-DQN",
         "maskppo": "Maskable PPO", "rainbow": "Rainbow-style DQN"}


def folder(algo: str, seed: int) -> str:
    if algo == "dqn":
        return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}")
    return os.path.join("models", "round11", f"{algo}_1trucks_s{seed}")


def load(algo: str, seed: int):
    f = folder(algo, seed)
    if algo == "rainbow":
        from rl.rainbow import RainbowDQN
        return RainbowDQN.load(os.path.join(f, "best_model.pt"))
    if algo in ("qrdqn", "maskppo"):
        import sb3_contrib
        cls = {"qrdqn": sb3_contrib.QRDQN, "maskppo": sb3_contrib.MaskablePPO}[algo]
    else:
        import stable_baselines3 as sb3
        cls = {"dqn": sb3.DQN, "ppo": sb3.PPO, "a2c": sb3.A2C}[algo]
    return cls.load(os.path.join(f, "best_model.zip"), device="cpu")


def test_one(args) -> dict:
    algo, seed = args
    import torch
    torch.set_num_threads(1)
    from rl.eval_round6 import run, validation
    m = load(algo, seed)
    if algo == "maskppo":
        choose = lambda o, e: int(m.predict(o, action_masks=e.action_masks(),  # noqa: E731
                                            deterministic=True)[0])
    else:
        choose = lambda o, e: int(m.predict(o, deterministic=True)[0])          # noqa: E731
    r = run(1, ZONES, choose, features=FEATURES)
    step, val = validation(folder(algo, seed), 1)
    return {"algo": algo, "seed": seed, "r": r, "val": val, "step": step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round11"))
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    from scipy import stats

    from rl.evaluate import run_no_trucks
    from rl.wrapper import TEST_PERIOD
    base = run_no_trucks()
    nt_total, nt_daily = base["total_reward"], np.asarray(base["daily_reward"])
    n_days = len(nt_daily)

    jobs = [(a, s) for a in ALGOS for s in SEEDS
            if os.path.exists(folder(a, s))]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(test_one, jobs):
            r = res["r"]
            diff = r["daily"] - nt_daily
            gain = float(diff.sum())
            rows.append({"algo": res["algo"], "seed": res["seed"],
                         "total_reward": r["total_reward"], "vs_no_trucks": gain,
                         "test_per_week": gain / n_days * 7,
                         "first_14_days": float(diff[:EARLY_DAYS].sum()),
                         "rest": float(diff[EARLY_DAYS:].sum()),
                         "days_ahead": int((diff > 0).sum()),
                         "val_per_week": res["val"], "val_best_step": res["step"],
                         **{c: r[c] for c in ("fulfilled_pct", "lost_requests",
                                              "blocked_returns", "truck_cost", "bikes_moved",
                                              "truck_km", "waits")},
                         "decisions": sum(o[3] for o in r["orders"]),
                         "n_routes": len(r["orders"]), "top_orders": r["orders"][:6],
                         "daily_vs_no_trucks": diff.tolist()})
            print(f"  {LABEL[res['algo']]:18s} seed {res['seed']}: {gain:+,.0f} "
                  f"(val {res['val']:,.0f}/week at {res['step']:,})", flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns=["top_orders", "daily_vs_no_trucks"]).to_csv(
        os.path.join(out, "rl_test_results.csv"), index=False)
    json.dump({"no_trucks": nt_total, "no_trucks_daily": nt_daily.tolist(), "zones": ZONES,
               "features": FEATURES, "rows": rows},
              open(os.path.join(out, "round11.json"), "w"), default=float)

    b = df[df.algo == "dqn"].vs_no_trucks
    rho = stats.spearmanr(df.val_per_week, df.vs_no_trucks)
    with open(os.path.join(out, "rl_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 11: six RL algorithms, 1 truck, zones {ZONES}, features {FEATURES}, "
                f"test {TEST_PERIOD[0]} to {TEST_PERIOD[1]}\n\nNo trucks: ${nt_total:,.0f}. "
                f"Spearman (validation vs test, {len(df)} runs): {rho.statistic:.2f} "
                f"(p {rho.pvalue:.2f}).\n\n")
        cols = ["algo", "seed", "vs_no_trucks", "test_per_week", "first_14_days", "rest",
                "days_ahead", "val_per_week", "val_best_step", "fulfilled_pct",
                "lost_requests", "blocked_returns", "truck_cost", "bikes_moved", "truck_km",
                "decisions", "n_routes", "waits"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.2f}" if c == "fulfilled_pct" else f"{v:,.0f}"
                elif isinstance(v, int) and c != "seed":
                    v = f"{v:,}"
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n## Per algorithm: mean of seeds (min to max); Welch t-test vs DQN\n\n"
                "| algorithm | vs no trucks | diff | p | validation $/week | first 14 days | "
                "rest | fulfilled % | truck km | bikes moved | routes |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for algo, g in df.groupby("algo", sort=False):
            def s(c):
                v = g[c]
                return f"{v.mean():,.0f} ({v.min():,.0f} to {v.max():,.0f})"
            x = g.vs_no_trucks
            p = "" if algo == "dqn" or len(x) < 2 else \
                f"{stats.ttest_ind(x, b, equal_var=False).pvalue:.2f}"
            f.write(f"| {LABEL[algo]} | {s('vs_no_trucks')} | {x.mean() - b.mean():+,.0f} | "
                    f"{p} | {s('val_per_week')} | {s('first_14_days')} | {s('rest')} | "
                    f"{g.fulfilled_pct.mean():.2f} | {g.truck_km.mean():,.0f} | "
                    f"{g.bikes_moved.mean():,.0f} | {g.n_routes.mean():.1f} |\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1])
    colors = dict(zip(ALGOS, plt.get_cmap("tab10").colors))
    for r in rows:
        ax.plot(days, np.cumsum(r["daily_vs_no_trucks"]), color=colors[r["algo"]],
                ls=("-", "--", ":")[r["seed"]], lw=1.2,
                label=f"{LABEL[r['algo']]}, seed {r['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(days[EARLY_DAYS], color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative $ vs no trucks")
    ax.set_title("Round 11 test: six RL algorithms, 1 truck, 3 seeds each")
    ax.legend(fontsize=6.5, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rl_test_daily.png"), dpi=120)
    print(f"wrote {out}/rl_test_results.md, .csv, round11.json, rl_test_daily.png")


if __name__ == "__main__":
    main()
