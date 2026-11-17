# Round 15: a coordinated 2-truck DQN, written down before any test run

Written 2026-11-16, before training finished or any score was compared with the test.

## Why 1 truck is short, and why round 12's 2 trucks lost anyway (non-test evidence)

1. The champion's truck is busy 99.2% of the time (checked on a 2-week slice, seed 0's
   plain DQN) and runs about 11 missions/day: it is saturated, not under-used.
2. Even running continuously, 39% of minutes have >=1 zone in acute crisis (>=60% of its
   stations empty or full) somewhere the truck isn't; 17% have 2+ such zones at once, up to 7.
3. The no-truck demand pattern has a recurring far-apart (15.5 min road time) empty/full
   mismatch (zone 7 needs bikes, zone 0 needs pickup) in about 20% of all test-period minutes.
4. Evening rush hour (4-7pm) produces about 2,500 lost/blocked events in that 4-hour window;
   one truck can realistically move only about 55-120 bikes in the same window.
5. Round 12 already tried 2 trucks and lost ($236,685 vs the champion's $257,204 at the time,
   now $333,282 after round 13's lookahead planner). Its own diagnostics put the cause on
   coordination, not capacity: the 2 trucks had fully independent policies on a hard territory
   split, while 31% of real trips cross between the two territories
   (results/round12/cross_trips.json); the pair earned $133k-145k less than the sum of its
   own two solo halves (results/round12/additivity_check_s1_300k.json).
6. That's explained by two gaps in the code (rl/wrapper.py, rl/features.py), confirmed by
   reading it, not by running anything: when two trucks go idle in the same minute, the second
   one asked can't see the first one's just-chosen order (the env only applies decided orders
   once every idle truck that minute has answered); and the "other truck" feature block only
   ever showed a truck's location, busy flag and minutes left, never its current mission's
   destination or load.

## The idea (rl/features.py, rl/wrapper.py, rl/validation.py)

One coordinated, whole-city DQN, 2 trucks, no territory split: both trucks share one action
space over all 20 mainland zones + island, like the champion, each asked the same way a
1-truck DQN is asked (once per idle truck per minute).

- A truck's observation of the other truck now also carries the other truck's current
  mission's destination zone (one-hot) and its load, not just its location, busy flag and
  minutes left.
- When two trucks are idle in the same minute, the second one asked sees the first one's
  just-decided order in that same block (destination = its drop-off zone, load = its
  quantity), even though the order isn't applied to the env until both have answered.
- Same DQN hyperparameters, feature groups (risk+travel+failures+memory), zones
  (flow_w0.3_k20_need) and station rule as the champion; 3 seeds, 1M decisions each.
- Validated with round 14's S1 aged-starts scheme, generalized to 2 trucks (same 4 runs,
  same scoring window, just 2 trucks sharing the one model being evaluated); checkpoint
  picking is the same rule as the champion's own training (best S1 averaged with its two
  neighbouring checkpoints).

K=1 is unaffected: a lone truck never reaches the "other truck" block at all, so the 1-truck
champion and the round-13 planner built on it are unchanged (checked: the champion's seed 0
still gives exactly +219,556 on the test after these edits).

## How it is judged

No test data in any choice. Training and checkpoint picking use only S1 (non-test days).
Once training finishes, one full test run (Aug 1 - Sep 20), compared against:
- the current champion: $333,282 (the 1-truck lookahead planner, 3-seed mean)
- the no-truck baseline: $6,057,953
- round 12's bar to clear decisively, not just tie: $236,685 (the independent territory pair)

The "vs no trucks" number already nets out each run's own truck cost(s) (fixed + shift +
fuel + labor), so this 2-truck run's number compares directly with the 1-truck numbers above,
with no separate cost adjustment.
