"""Round 10: screen station rules inside a zone (rl.stop_rules) without retraining.

    python -m rl.stop_search [--out results/round10] [--workers 14]

Zones flow_w0.3_k20_need. Every rule in stop_rules.RULES is scored three ways; the
test period (Aug 1 - Sep 20) is never touched:
  - champion zero-shot: the round 9 DQN models (all four feature groups, 3 seeds),
    unchanged, on the validation weeks (Jul 18-24, Jul 25-31), in $/week vs no
    trucks, the measure training used to pick checkpoints;
  - greedy dispatcher (rl.zone_search.run_greedy) over the training period;
  - fixed routes: the 10 best routes of the round 8 screen for these zones, each
    repeated over the training period; best and mean gain.
Finalists: the 3 non-default rules with the best average rank over the champion's
mean, greedy and best route (ties: champion's mean). The champion was trained on
the default rule, so its scores favour rules close to it; the two no-learning
scores do not. Writes screen.md / screen.csv / finalists.json.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from rl.stop_rules import DEFAULT_RULE, RULES

ZONES = "flow_w0.3_k20_need"
FEATURES = "risk+travel+failures+memory"
SEEDS = (0, 1, 2)
VAL_WEEKS = (("2026-07-18", "2026-07-24"), ("2026-07-25", "2026-07-31"))
N_ROUTES = 10
N_FINALISTS = 3


def champion_folder(seed: int) -> str:
    return os.path.join("models", "round9", f"dqn_1trucks_{FEATURES}_s{seed}")


def run_champion(args) -> dict:
    """One champion seed, one validation week, one rule: gain over no trucks."""
    rule, seed, week = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN
    from rl.eval_round6 import run
    from rl.wrapper import DEFAULT_INIT_OCCUPANCY, no_truck_rewards
    m = DQN.load(os.path.join(champion_folder(seed), "best_model.zip"), device="cpu")
    r = run(1, ZONES, lambda o, e: int(m.predict(o, deterministic=True)[0]),
            features=FEATURES, stop_rule=rule, window=week)
    base = float(no_truck_rewards(week, DEFAULT_INIT_OCCUPANCY).sum())
    return {"rule": rule, "seed": seed, "week": week[0],
            "gain": r["total_reward"] - base, "bikes_moved": r["bikes_moved"],
            "truck_km": r["truck_km"], "lost": r["lost_requests"],
            "blocked": r["blocked_returns"], "decisions": sum(o[3] for o in r["orders"])}


def run_greedy(rule):
    from rl.zone_search import run_greedy as greedy
    return {"rule": rule, **greedy(ZONES, rule)}


def run_route(args):
    from rl.zone_search import run_route as route
    return {"rule": args[-1], **route(args)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "round10"))
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from rl.wrapper import static_data
    static_data()
    routes = pd.read_csv(os.path.join("results", "round8", "routes.csv"))
    top = routes[routes.scheme == ZONES].nlargest(N_ROUTES, "gain")
    champ_jobs = [(r, s, w) for r in RULES for s in SEEDS for w in VAL_WEEKS]
    route_jobs = [(ZONES, int(t.src), int(t.dst), int(t.qty), r)
                  for r in RULES for t in top.itertuples()]
    print(f"{len(RULES)} rules: {len(champ_jobs)} champion weeks, {len(RULES)} greedy runs, "
          f"{len(route_jobs)} route runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        g_fut = pool.map(run_greedy, RULES)
        r_fut = pool.map(run_route, route_jobs)
        champ = []
        for i, c in enumerate(pool.map(run_champion, champ_jobs)):
            champ.append(c)
            print(f"  champion {c['rule']:18s} seed {c['seed']} {c['week']}: "
                  f"{c['gain']:+,.0f}", flush=True)
        greedy = list(g_fut)
        rts = list(r_fut)
    ch, gr, rt = pd.DataFrame(champ), pd.DataFrame(greedy), pd.DataFrame(rts)
    ch.to_csv(os.path.join(args.out, "champion_weeks.csv"), index=False)
    rt.to_csv(os.path.join(args.out, "routes.csv"), index=False)

    per_seed = ch.groupby(["rule", "seed"])["gain"].mean().unstack()     # $/week
    rows = []
    for rule in RULES:
        c = ch[ch.rule == rule]
        r = rt[rt.rule == rule]
        g = gr[gr.rule == rule].iloc[0]
        rows.append({"rule": rule,
                     **{f"champion_s{s}": per_seed.loc[rule, s] for s in SEEDS},
                     "champion_mean": per_seed.loc[rule].mean(),
                     "champion_bikes_moved": c.bikes_moved.mean(),
                     "champion_km": c.truck_km.mean(),
                     "greedy_gain": g.greedy_gain, "greedy_bikes_moved": g.greedy_bikes_moved,
                     "best_route_gain": r.gain.max(), "mean_route_gain": r.gain.mean()})
    df = pd.DataFrame(rows)
    cand = df[df.rule != DEFAULT_RULE].copy()
    score_cols = ["champion_mean", "greedy_gain", "best_route_gain"]
    cand["avg_rank"] = cand[score_cols].rank(ascending=False).mean(1)
    cand = cand.sort_values(["avg_rank", "champion_mean"], ascending=[True, False])
    finalists = cand.rule.head(N_FINALISTS).tolist()
    df = df.merge(cand[["rule", "avg_rank"]], on="rule", how="left")
    df["finalist"] = df.rule.isin(finalists)
    df.to_csv(os.path.join(args.out, "screen.csv"), index=False)
    json.dump({"finalists": finalists, "default": DEFAULT_RULE},
              open(os.path.join(args.out, "finalists.json"), "w"), indent=1)

    with open(os.path.join(args.out, "screen.md"), "w", encoding="utf-8") as f:
        f.write(f"# Round 10 station-rule screen, zones {ZONES}\n\n"
                "champion = round 9 DQN (all four feature groups), unchanged, on the "
                "validation weeks Jul 18-31, $/week vs no trucks. greedy / routes = gain "
                "over no trucks on the training period (May 1 - Jul 17, one run). "
                f"Finalists: best average rank over {', '.join(score_cols)}.\n\n")
        cols = ["rule"] + [f"champion_s{s}" for s in SEEDS] + [
            "champion_mean", "champion_bikes_moved", "champion_km", "greedy_gain",
            "best_route_gain", "mean_route_gain", "avg_rank", "finalist"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, (bool, np.bool_)):
                    v = "yes" if v else ""
                elif isinstance(v, float):
                    v = "" if np.isnan(v) else (f"{v:.2f}" if c == "avg_rank" else f"{v:,.0f}")
                cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
    print(df[["rule", "champion_mean", "greedy_gain", "best_route_gain", "avg_rank",
              "finalist"]].to_string(index=False))
    print("finalists:", finalists)


if __name__ == "__main__":
    main()
