# BikeShareToronto-v0: a rebalancing environment

A [Gymnasium](https://gymnasium.farama.org/) environment for Bike Share Toronto. Customer demand is
replayed **minute by minute from the real 2026 trip log**, and your policy decides where to send
trucks to move bikes between the 1,072 stations. The reward is rental revenue minus the cost of
rebalancing.

```python
import gymnasium as gym
import bikeshare_env  # registers BikeShareToronto-v0

env = gym.make("BikeShareToronto-v0",
               init_occupancy=0.35,
               simulation_period=("2026-06-15", "2026-06-21"),
               number_trucks=2)
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()          # replace with your policy
    obs, reward, done, truncated, info = env.step(action)
print(info["totals"])
```

---

## 1. Setup

Run everything from the repository root (the package imports `data_loader.py` from there).

```bash
pip install -r requirements.txt
```

The environment needs four data files under `data/`. They're gitignored, so build them once:

| File | How to build it | What it is |
|---|---|---|
| `data/bikeshare-ridership-2026.parquet` | Download the 2026 ridership CSV from the City of Toronto Open Data portal ("Bike Share Toronto Ridership Data"), unzip it to `data/bikeshare-ridership-2026/`, then run `python preprocess.py` | 6.83M trips, Jan 1 – Sep 30 2026 |
| `data/stations.json` | `python fetch_stations.py` (live GBFS feed) | 1,072 stations: id, name, lat/lon, dock `capacity` |
| `data/road_dist_km.npy`, `data/road_time_min.npy`, `data/road_station_ids.npy` | `python build_road_distances.py` (downloads the OSM drive network on first run, then caches it to `data/toronto_drive.graphml`) | Station-to-station **driving** distance (km) and free-flow time (min) |

If you re-run `fetch_stations.py` and the station list changes, re-run `build_road_distances.py` too.
If the road matrices are missing, `gym.make` raises an error telling you so.

Check the install:

```bash
python -m pytest tests -q
```

---

## 2. How the simulation works

### Time
- A `step()` simulates `step_minutes` minutes (default **1**). Then, if **every truck is busy**,
  it keeps simulating until at least one truck is idle (`skip_when_all_busy=True`, the default).
  The agent is only asked for an action when it can actually dispatch a truck, and the reward
  returned covers all the minutes that passed. `info["minutes_elapsed"]` says how many.
- An episode covers the **whole `simulation_period`**, starting at 00:00 of the first date and
  ending after 23:59 of the last date. That's 1,440 minutes per day; the number of steps is fewer
  whenever trucks are busy. With `skip_when_all_busy=False`, every step is exactly
  `step_minutes` minutes.
- Bike positions carry over between days. `terminated` becomes `True` at the end of the period;
  `truncated` is always `False`.

### Initial state
Each station starts with `ceil(capacity × init_occupancy)` bikes, capped at capacity. All trucks
start idle at the depot station.

The default `init_occupancy=0.35` gives 7,824 bikes on 20,780 docks. That's close to what the live
GBFS feed shows at midnight (~7.9k rentable of ~8.8k docked bikes) and to the 6–10k distinct bikes
per day in the summer 2026 trip data. Use a lower value (≈0.25) for winter months, when far fewer
bikes are in service.

### Customers (replayed from the trip log)
Every trip in the log is a **request** at its start station and start minute:

| Situation | What happens | Reward |
|---|---|---|
| Start station has a bike | Rental starts; the bike leaves the station | **+$1.00** unlock fee |
| Start station is empty | Request is **lost**; no return happens later | **−$0.50** penalty |
| Return: end station has a free dock | Bike docks | **+$0.15 × rented minutes** |
| Return: end station is full | Bike is **blocked**: unavailable for **15 min** | $0 at that moment |
| 15 min after a blocked return | Bike is placed at the **nearest station by road distance** that has a free dock (possibly the original one) | **+$0.10 × rented minutes** (the 15 min aren't billed) |

- Rented minutes = `ceil(Trip_Duration / 60)`, minimum 1, taken from the log.
- User type (member/casual) and bike model don't affect pricing.
- Riders don't walk to another station when theirs is empty, and they always return to the end
  station recorded in the log.

### Order of events within one minute
1. Your action is applied: idle trucks start their new missions.
2. Truck missions advance (arrive, load, unload).
3. Returns due this minute dock, or get blocked.
4. Blocked bikes whose 15 minutes are up get placed.
5. Rentals starting this minute are served or lost.
6. Overtime labor is charged for busy trucks outside the shift.

### Trucks (your policy's lever)
A **mission** is *pick up up to `qty` bikes at station A and drop them at station B*. It runs as:

1. Drive from the truck's current station to A.
2. Load `min(qty, bikes at A on arrival, truck space)`: 1 min per bike.
3. Drive A → B.
4. Unload as many as B has free docks for: 1 min per bike.
5. If bikes are left over, drive to the next-nearest station with free docks and unload there
   (repeating as needed). If no station has room, the leftovers stay on the truck and are
   unloaded along with the next mission's bikes.

The truck then waits, idle, wherever it finished.

**Missions can't be cancelled.** When all trucks are busy, the env doesn't ask for a decision at
all (see *Time*). When only some are busy, the agent is asked, and the busy trucks' rows are
ignored. Use `info["idle_trucks"]` or `obs["truck_busy"]` to see which trucks will accept an
order.

Missions that pick up from or drop off at the 7 Toronto Island stations are ignored (ferry-only, no
road access), and leftovers are never dropped there. Change this with `truck_blocked_stations`.

### Costs
The fleet size `number_trucks` is a **hyperparameter you choose, not something the policy
optimizes**. Every truck is hired for every simulated day, whether it moves or not.

| Cost | Rule (defaults) |
|---|---|
| Truck fixed cost | **$100** per truck per day, charged at 00:00 of each day for the whole fleet |
| Driver shift | Charged at the same moment: the full shift **09:00–17:00** at the hourly rate = 8 h × $20 = **$160** per truck |
| Overtime | Each minute a truck is busy **outside** the shift: **$20/h** between 09:00–17:00, **$40/h** otherwise |
| Fuel | **$0.30 per km** of road distance, for every leg (including empty drives to the pickup), charged when the leg starts |

Driving distances and times come from the real road network: OSMnx with Dijkstra, one-way streets
respected. Times are free-flow, with no traffic. Median road distance is 1.21× the straight-line
distance.

### Reward
Per step:

```
reward = unlock_revenue + return_revenue + overflow_revenue
         - lost_penalty - labor_cost - fuel_cost - fixed_cost
```

A day with no rebalancing earns about **$103k** (see §6), so the reward scale is large. Consider
normalizing it for learning.

---

## 3. Spaces

`N` = number of stations (1,072), `K` = `number_trucks` (default 2). Station **index** `i` = row `i` of
`data/stations.json`. Map indices to GBFS station ids with `env.unwrapped.station_ids[i]`.

### Action: `MultiDiscrete`, shape `(K, 3)`
One row per truck: `[pickup, dropoff, qty]`

| Field | Range | Meaning |
|---|---|---|
| `pickup` | `0 … N` | `0` = **do nothing**; otherwise pick up at station index `pickup − 1` |
| `dropoff` | `0 … N−1` | Drop-off station index |
| `qty` | `0 … truck_capacity` | Bikes requested (`0` = do nothing) |

A row is also ignored when the truck is busy, when pickup and dropoff are the same station, or when
either station is truck-blocked. A no-op action is `np.zeros((K, 3), dtype=np.int64)`.

> The random policy (`env.action_space.sample()`) dispatches a mission on almost every step, which
> is expensive. Sample sparsely for exploration.

### Observation: `Dict` of `float32` arrays
| Key | Shape | Meaning |
|---|---|---|
| `bikes` | `(N,)` | Bikes docked at each station |
| `free_docks` | `(N,)` | `capacity − bikes` |
| `minute_of_day` | `(1,)` | 0–1439 |
| `day_of_week` | `(1,)` | 0 = Monday … 6 = Sunday |
| `truck_location` | `(K,)` | Station index the truck is at, or is currently driving to |
| `truck_busy` | `(K,)` | 1 if on a mission |
| `truck_minutes_left` | `(K,)` | Minutes until the truck's current leg/load/unload finishes |
| `truck_load` | `(K,)` | Bikes on the truck |

The observation does **not** reveal future demand or when riding bikes will come back; the policy
has to learn those patterns. For algorithms that need a flat vector, wrap with
`gymnasium.wrappers.FlattenObservation` (size `2N + 2 + 4K` = 2,154 with defaults).

### `info`
| Key | Meaning |
|---|---|
| `unlock_revenue`, `return_revenue`, `overflow_revenue`, `lost_penalty`, `labor_cost`, `fuel_cost`, `fixed_cost` | This step's reward components ($) |
| `trips_started`, `lost_trips`, `returns_docked`, `overflow_returns`, `overflow_placed`, `bikes_moved`, `truck_km` | This step's counts |
| `minutes_elapsed` | Simulated minutes this step covered (more than `step_minutes` when the env skipped ahead while all trucks were busy) |
| `totals` | The same keys, summed since `reset()` |
| `idle_trucks` | `(K,)` bool: trucks that will accept a new mission |
| `minute` | Minutes elapsed since the start of the period |

Bikes blocked right now = `totals["overflow_returns"] − totals["overflow_placed"]`.

---

## 4. Constructor parameters

Every parameter can be passed to `gym.make("BikeShareToronto-v0", ...)`.

| Parameter | Default | Description |
|---|---|---|
| `init_occupancy` | `0.35` | Initial bikes per station = `ceil(capacity × init_occupancy)` |
| `simulation_period` | `("2026-06-15", "2026-06-15")` | Inclusive `(start, end)` dates within 2026-01-01 … 2026-09-30 |
| `step_minutes` | `1` | Minimum minutes simulated per `step()`; the action is applied at the first one |
| `skip_when_all_busy` | `True` | While every truck is busy, keep simulating inside `step()` until one is idle, so no decision is requested when none can be made |
| `unlock_fee` | `1.0` | $ per started rental |
| `per_min_fee` | `0.15` | $ per rented minute on a normal return |
| `overflow_per_min_fee` | `0.10` | $ per rented minute when the return was blocked |
| `overflow_delay_min` | `15` | Minutes a blocked bike is unavailable |
| `lost_trip_penalty` | `0.5` | $ subtracted per lost request |
| `number_trucks` | `2` | Fleet size (hyperparameter). Each truck costs $100 + its shift every day, used or not. `0` = no trucks: the action space is empty, shape `(0, 3)` |
| `truck_capacity` | `20` | Bikes per truck |
| `load_min_per_bike` | `1.0` | Minutes to load or unload one bike |
| `depot_station` | `None` | Station **id** where trucks start; default = station nearest downtown (Eaton Centre) |
| `truck_blocked_stations` | Toronto Island ids | Station ids trucks can't serve |
| `shift` | `(9, 17)` | Paid driver shift, hours |
| `day_hours` | `(9, 17)` | Hours billed at `day_rate`; others at `night_rate` |
| `day_rate`, `night_rate` | `20.0`, `40.0` | Labor $ per hour per truck |
| `truck_fixed_cost` | `100.0` | $ per truck per day |
| `fuel_cost_per_km` | `0.3` | $ per road km |
| `station_zone`, `max_bikes_per_stop` | `None`, `None` | **Multi-stop missions**, on only when both are set. `station_zone` = one zone id per station index. The truck picks up at most `max_bikes_per_stop` bikes at `pickup`, then moves on to the unvisited station in the same zone with the most bikes, until it has `qty` bikes or the zone has none left (if it collected none, the mission ends there). It then drops at most `max_bikes_per_stop` at `dropoff`, then at the unvisited station in that zone with the most free docks, until empty. If the drop-off zone fills up, the rest go to the nearest free docks with no per-stop limit. Used by the RL agents from round 3 (5 bikes per stop) |
| `paired_trucks` | `False` | **Two trucks run every order together** (needs multi-stop and `number_trucks=2`). Only row 0 of the action is read, and only when both trucks are idle. Both trucks collect `qty` bikes each: the pickup zone's stations are ranked by bikes (`pickup` first), truck 0 visits ranks 1, 3, 5, ... and truck 1 ranks 2, 4, 6, ..., at most `max_bikes_per_stop` per station. When the first truck has collected, the drop-off zone is ranked by free docks (`dropoff` first) and split the same way. Stations emptied or filled meanwhile are skipped; if a truck runs out of drop-off ranks, the rest go to the nearest free docks |
| `stations`, `trips`, `road_dist_km`, `road_time_min` | `None` | Inject your own data instead of loading from `data/` (used by the unit tests) |

---

## 5. Writing a policy

Policies live in `bikeshare_env/policies.py`. Subclass `Policy` and register it in `POLICIES`, and
it appears automatically in the dashboard's **Policy** dropdown.

```python
import numpy as np
from bikeshare_env.policies import Policy, POLICIES

class MoveFromFullestToEmptiest(Policy):
    label = "Fullest -> emptiest (example)"
    number_trucks = 2                               # fleet this policy runs with

    def act(self, obs, info):
        action = np.zeros((self.number_trucks, 3), dtype=np.int64)
        if obs["minute_of_day"][0] % 60:            # decide once an hour
            return action
        fill = obs["bikes"] / np.maximum(obs["bikes"] + obs["free_docks"], 1)
        for k in np.flatnonzero(info["idle_trucks"]):
            a, b = int(np.argmax(fill)), int(np.argmin(fill))
            action[k] = [a + 1, b, 10]               # pickup is index + 1
            fill[a], fill[b] = 0.5, 0.5             # don't send two trucks to the same pair
        return action

POLICIES["fullest_to_emptiest"] = MoveFromFullestToEmptiest
```

Each policy declares the fleet it runs with in `number_trucks` (default 2). The dashboard and
`simulate_day` build the env with that fleet. The dummy `no_rebalancing` policy declares **0**, so
it pays no truck costs.

`Policy.reset(env)` is called once with the unwrapped env, so you can read static data such as
`env.capacity`, `env.dist_km` or `env.station_ids`. It also sets `self.number_trucks` to the env's
actual fleet size.

Run one day under a policy and get per-minute series (what the dashboard plots):

```python
from bikeshare_env.rollout import simulate_day

r = simulate_day("2026-06-15", "no_rebalancing")   # key from POLICIES
r.reward.sum(), r.fulfilled.sum(), r.lost.sum(), r.blocked.sum(), r.pending.max()
```

`simulate_day` runs with `skip_when_all_busy=False`, so it records every minute. `DayRollout` holds `(1440,)` arrays `reward`, `fulfilled`, `lost`, `active`, `blocked` and
`pending`, plus `bikes` `(1440, N)`, `trips`, `trip_started` and `totals`.

### Dashboard

```bash
python app.py        # http://localhost:8050
```

Pick a date and a policy. The server simulates that day (about 1 s, cached), and the page animates
it: station fill level, rides in progress, and live metrics under the map.

---

## 6. Baseline numbers

No rebalancing (`no_rebalancing`, 0 trucks), `init_occupancy=0.35`, 2026-06-15 (a Monday):

| Metric | Value |
|---|---|
| Requests | 34,429 |
| Fulfilled / lost | 31,187 (90.6%) / 3,242 |
| Blocked returns | 1,983 (peak 188 bikes held at once, ≈09:00) |
| Total reward | $103,429 |

A useful policy should beat this after paying for its trucks. That's harder than it looks: the
naive *fullest → emptiest* example in §5 serves 135 more requests on the same day but scores
**$102,818**, below the baseline, because its 2 trucks cost $1,125 in labor and fixed fees and
gain less than that in revenue.

---

## 7. Assumptions and limitations

- **Demand is the observed trips.** The log only contains trips that actually happened under the
  real operator's rebalancing. People who found an empty station in reality aren't in the data,
  so true demand is somewhat higher than what's replayed.
- About 1.5% of trips are dropped because their station isn't in the current GBFS station list.
- Rides still in progress when the episode ends never return and earn no return fee.
- Truck travel times are free-flow (no traffic, no parking time).
- A blocked bike is placed at the nearest free dock however far away that is. There's no explicit
  cost for moving it.
- No bike breakdowns, e-bike charging, or operator rebalancing beyond what your trucks do.
- Labor is billed per truck. There's no separate crew model, and no requirement to return to the
  depot.

## Files

| File | Contents |
|---|---|
| `env.py` | `BikeShareEnv`, constants (`DEFAULT_INIT_OCCUPANCY`, `ISLAND_STATIONS`) |
| `trips.py` | `load_period_trips()` → `TripEvents` (minute-indexed numpy arrays) |
| `road_network.py` | `load_road_matrices()`, `haversine_km()` |
| `policies.py` | `Policy`, `NoRebalancingPolicy`, the `POLICIES` registry |
| `rollout.py` | `simulate_day()` → `DayRollout` |
| `../build_road_distances.py` | Builds the road matrices |
| `../tests/test_env.py` | Unit tests on small synthetic station networks |
