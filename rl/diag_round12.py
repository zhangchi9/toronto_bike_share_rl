"""Round 12 additivity diagnostics (seed from argv), current checkpoints in CK."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CK = sys.argv[1]
SEED = int(sys.argv[2])
SCHEME = "flow_w0.3_k40_need"
F = "risk+travel+failures+memory"


class Wait:
    def predict(self, o, deterministic=True):
        return np.int64(0), None


def model(t):
    from stable_baselines3 import DQN
    return DQN.load(os.path.join(CK, f"dqn_1trucks_territory{t}_s{SEED}", "best_model.zip"), device="cpu")


def stats(env, terr_of):
    tr = env.trips
    started = env.trip_started
    T = env.totals
    out = {k: float(T[k]) for k in ("unlock_revenue", "return_revenue", "overflow_revenue",
                                    "lost_penalty", "labor_cost", "fuel_cost", "fixed_cost",
                                    "trips_started", "lost_trips", "overflow_returns", "bikes_moved",
                                    "truck_km")}
    rev = T["unlock_revenue"] + T["return_revenue"] + T["overflow_revenue"]
    out["reward"] = rev - T["lost_penalty"] - T["labor_cost"] - T["fuel_cost"] - T["fixed_cost"]
    for name, g in (("A", 0), ("B", 1), ("other", 2)):
        m = terr_of[tr.start_idx] == g
        out[f"started_{name}"] = int((started & m).sum())
        out[f"lost_{name}"] = int((~started & m).sum())
    return out


def closed(kind):
    """Closed loop through RebalanceEnv; record every order (minute, truck, row)."""
    from rl.territory import load_territories
    from rl.wrapper import RebalanceEnv, TEST_PERIOD
    A, B = load_territories(SCHEME)
    if kind == "A":
        ms, terr = [model("A")], [A]
    elif kind == "B":
        ms, terr = [model("B")], [B]
    elif kind == "pair":
        ms, terr = [model("A"), model("B")], [A, B]
    elif kind == "pair_Awaits":
        ms, terr = [Wait(), model("B")], [A, B]
    env = RebalanceEnv(len(ms), windows=[TEST_PERIOD], record_minutes=True, zone_scheme=SCHEME,
                       features=F, territories=terr)
    obs, _ = env.reset(seed=0)
    log, done = [], False
    while not done:
        k = env.queue[0]
        a = int(ms[k].predict(obs, deterministic=True)[0])
        row = env.fb.decode(a)
        if row.any():
            log.append((int(env.env.t), k, row.tolist()))
        obs, _, done, _, _ = env.step(a)
    return env.env, log


def replay(logs_by_truck):
    """Open loop: each truck issues its recorded orders at the recorded minute (or as
    soon as it is idle after that minute)."""
    from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
    from rl.wrapper import TEST_PERIOD, make_base_env
    K = len(logs_by_truck)
    env = make_base_env(TEST_PERIOD, K, DEFAULT_INIT_OCCUPANCY, zone_scheme=SCHEME)
    env.reset(seed=0)
    ptr = [0] * K
    late = 0
    for m in range(env.total_minutes):
        act = np.zeros((K, 3), np.int64)
        for k, log in enumerate(logs_by_truck):
            if ptr[k] < len(log) and log[ptr[k]][0] <= m and not env.trucks[k].busy(env.t):
                if log[ptr[k]][0] < m:
                    late += 1
                act[k] = log[ptr[k]][2]
                ptr[k] += 1
        env.step(act)
    return env, late


def work(kind):
    from rl.territory import load_territories
    from rl.wrapper import static_data
    from rl import zones as zz
    A, B = load_territories(SCHEME)
    zof = zz.load_zones(SCHEME).zone_of_station
    terr_of = np.where(np.isin(zof, A), 0, np.where(np.isin(zof, B), 1, 2))
    if kind == "none":
        from rl.wrapper import make_base_env, TEST_PERIOD
        from bikeshare_env.env import DEFAULT_INIT_OCCUPANCY
        env = make_base_env(TEST_PERIOD, 0, DEFAULT_INIT_OCCUPANCY)
        env.reset(seed=0)
        noop = np.zeros((0, 3), np.int64)
        for _ in range(env.total_minutes):
            env.step(noop)
        return kind, stats(env, terr_of), None
    env, log = closed(kind)
    return kind, stats(env, terr_of), log


def main():
    from rl.territory import load_territories
    from rl import zones as zz
    A, B = load_territories(SCHEME)
    zof = zz.load_zones(SCHEME).zone_of_station
    terr_of = np.where(np.isin(zof, A), 0, np.where(np.isin(zof, B), 1, 2))
    res, logs = {}, {}
    with ProcessPoolExecutor(5) as pool:
        for kind, st, log in pool.map(work, ["none", "A", "B", "pair", "pair_Awaits"]):
            res[kind], logs[kind] = st, log
            print(kind, "reward", round(st["reward"]), flush=True)
    eA, lateA = replay([logs["A"]])
    res["replay_A"] = stats(eA, terr_of)
    eAB, lateAB = replay([logs["A"], logs["B"]])
    res["replay_A+B"] = stats(eAB, terr_of)
    print("replay late orders A", lateA, "A+B", lateAB, flush=True)
    keys = list(res["none"].keys())
    n = res["none"]
    print(f"{'':18s}" + "".join(f"{k[:14]:>15s}" for k in ("none", "A", "B", "pair", "pair_Awaits",
                                                          "replay_A", "replay_A+B")))
    for key in keys:
        vals = [res[k][key] for k in ("none", "A", "B", "pair", "pair_Awaits", "replay_A", "replay_A+B")]
        print(f"{key:18s}" + "".join(f"{v:>15,.0f}" for v in vals))
    print("\nInteraction = X(A and B together) - X(A) - X(B) + X(none):")
    for key in keys:
        cl = res["pair"][key] - res["A"][key] - res["B"][key] + n[key]
        op = res["replay_A+B"][key] - res["A"][key] - res["B"][key] + n[key]
        print(f"  {key:18s} closed loop {cl:>12,.0f}   open-loop replay {op:>12,.0f}")
    json.dump({"res": res, "logs": {k: v for k, v in logs.items() if v is not None}},
              open(os.path.join(os.path.dirname(CK), f"diag12_s{SEED}.json"), "w"))


if __name__ == "__main__":
    main()
