# Round 13: two new ways to beat the 1-truck champion, test 2026-08-01 to 2026-09-20

No trucks: $6,057,953. All settings were picked on the validation weeks (Jul 18-31); each test run was made once.

| kind | seed | vs_no_trucks | test_per_week | first_14_days | rest | days_ahead | val_per_week | val_from | fulfilled_pct | lost_requests | blocked_returns | truck_cost | bikes_moved | truck_km | decisions | waits | changed_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | 0 | 219,556 | 30,135 | 38,906 | 180,649 | 49 | 17,233 | best checkpoint 650,000 | 89.48 | 214,207 | 158,607 | 52,917 | 17,866 | 23,484 | 779 | 42 |  |
| champion | 1 | 279,788 | 38,402 | 44,246 | 235,542 | 49 | 18,268 | best checkpoint 600,000 | 90.40 | 195,485 | 202,598 | 53,304 | 17,855 | 24,681 | 449 | 0 |  |
| champion | 2 | 272,267 | 37,370 | 43,051 | 229,216 | 49 | 17,929 | best checkpoint 500,000 | 90.70 | 189,258 | 232,922 | 53,634 | 16,725 | 25,780 | 432 | 0 |  |
| planner | 0 | 336,504 | 46,187 | 56,343 | 280,161 | 49 | 20,309 | H 1440 min, value weight 0 | 90.72 | 188,833 | 161,402 | 52,649 | 19,010 | 22,496 | 536 | 0 | 88.99 |
| planner | 1 | 342,964 | 47,073 | 58,361 | 284,603 | 51 | 22,618 | H 1440 min, value weight 0 | 90.70 | 189,283 | 156,356 | 52,358 | 19,395 | 21,527 | 486 | 0 | 91.56 |
| planner | 2 | 320,379 | 43,974 | 52,303 | 268,076 | 50 | 20,844 | H 1440 min, value weight 0 | 90.24 | 198,699 | 139,139 | 52,733 | 18,625 | 22,777 | 476 | 0 | 87.82 |
| inventory | 0 | 283,597 | 38,925 | 48,517 | 235,079 | 51 | 19,808 | H 1440 min, theta 0 | 89.38 | 216,208 | 100,507 | 51,192 | 21,405 | 17,640 | 551 | 0 |  |

## Mean of runs (min to max); Welch t-test vs the champion

| run | n | vs no trucks | diff | p | validation $/week | first 14 days | rest | fulfilled % | blocked returns | truck cost | truck km | bikes moved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DQN champion (round 9) | 3 | 257,204 (219,556 to 279,788) | +0 |  | 17,810 (17,233 to 18,268) | 42,068 (38,906 to 44,246) | 215,136 (180,649 to 235,542) | 90.19 | 198,042 | 53,285 | 24,648 | 17,482 |
| Idea 1: lookahead planner on the champion | 3 | 333,282 (320,379 to 342,964) | +76,079 | 0.04 | 21,257 (20,309 to 22,618) | 55,669 (52,303 to 58,361) | 277,614 (268,076 to 284,603) | 90.55 | 152,299 | 52,580 | 22,267 | 19,010 |
| Idea 2: inventory-model dispatcher | 1 | 283,597 | +26,393 | 0.30 (1 run) | 19,808 | 48,517 | 235,079 | 89.38 | 100,507 | 51,192 | 17,640 | 21,405 |

## Planner vs the same champion seed (paired)

| seed | champion | planner | diff | days the planner beat the champion |
|---|---|---|---|---|
| 0 | 219,556 | 336,504 | +116,948 | 51 of 51 |
| 1 | 279,788 | 342,964 | +63,176 | 49 of 51 |
| 2 | 272,267 | 320,379 | +48,112 | 45 of 51 |

Mean paired diff +76,079; paired t-test p 0.07

Planner vs the inventory dispatcher: +49,686 (one-sample t-test p 0.02)
