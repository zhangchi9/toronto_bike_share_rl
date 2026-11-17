# Round 15: a coordinated, whole-city 2-truck DQN, test 2026-08-01 to 2026-09-20

No trucks: $6,057,953. Training and checkpoint picking used only round 14's S1 aged-starts validation; this test run was made once.

| seed | vs_no_trucks | test_per_week | first_14_days | rest | days_ahead | val_per_week | val_from | fulfilled_pct | lost_requests | blocked_returns | truck_cost | bikes_moved | truck_km | waits | n_routes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 107,503 | 14,755 | 21,757 | 85,746 | 44 | 22,087 | best checkpoint 50,000 (S1 aged starts) | 88.34 | 237,407 | 131,715 | 101,859 | 28,892 | 50,505 | 3,481 / 4,231 | 24 / 22 |
| 1 | 101,717 | 13,961 | 8,899 | 92,819 | 44 | 20,497 | best checkpoint 650,000 (S1 aged starts) | 88.65 | 231,082 | 144,998 | 114,426 | 27,322 | 75,418 | 0 / 0 | 22 / 23 |
| 2 | 6,898 | 947 | -11,034 | 17,933 | 26 | 4,726 | best checkpoint 950,000 (S1 aged starts) | 87.07 | 263,226 | 131,204 | 106,748 | 34,422 | 49,939 | 51 / 37 | 25 / 25 |

## Mean vs no trucks (min to max); Welch t-test vs the 1-truck planner champion

| run | n | vs no trucks | diff vs planner | p |
|---|---|---|---|---|
| Coordinated 2-truck DQN (round 15) | 3 | 72,040 (6,898 to 107,503) | -261,243 | 0.01 |
| 1-truck lookahead planner (champion) | 3 | 333,282 (320,379 to 342,964) | +0 |  |
| 1-truck DQN alone (round 9) | 3 | 257,204 (219,556 to 279,788) | -76,079 | 0.04 |
| Independent territory pair (round 12) | 3 | 236,685 (229,076 to 243,118) | -96,597 | 0.00 |

Coordinated pair vs round 12's independent pair: -164,646 (Welch p 0.04)
