# Duplicated DQN truck: test results, 2026-08-01 to 2026-09-20

One continuous 51-day run. Rewards are the env's true $ reward; every truck costs $260/day (fixed + shift) plus night labor and fuel.

| policy | trucks | total_reward | vs_no_trucks | days_ahead | fulfilled_pct | lost_requests | blocked_returns | truck_cost | bikes_moved | truck_km | orders | minutes_one_truck_waits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No trucks (benchmark) | 0 | $6,057,953 | $0 | – | 85.15 | 302,201 | 105,163 | $0 | 0 | 0.00 | – | – |
| DQN (1 truck) | 1 | $6,335,107 | $277,154 | 49.00 | 90.15 | 200,523 | 153,739 | $51,435 | 20,720 | 18,451 | 519 | 0.00 |
| DQN duplicated, full order each | 2 | $6,282,635 | $224,682 | 49.00 | 90.17 | 200,127 | 131,772 | $100,622 | 38,621 | 37,332 | 508 | 4,636 |
| DQN duplicated, half order each | 2 | $6,311,496 | $253,543 | 49.00 | 90.56 | 192,201 | 135,704 | $100,955 | 36,569 | 40,205 | 934 | 5,419 |
| DQN (2 trucks, trained) | 2 | $5,964,500 | $-93,453 | 17.00 | 85.14 | 302,507 | 91,114 | $83,395 | 21,424 | 46,669 | – | – |
