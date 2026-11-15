# Round 10 station-rule screen, zones flow_w0.3_k20_need

champion = round 9 DQN (all four feature groups), unchanged, on the validation weeks Jul 18-31, $/week vs no trucks. greedy / routes = gain over no trucks on the training period (May 1 - Jul 17, one run). Finalists: best average rank over champion_mean, greedy_gain, best_route_gain.

| rule | champion_s0 | champion_s1 | champion_s2 | champion_mean | champion_bikes_moved | champion_km | greedy_gain | best_route_gain | mean_route_gain | avg_rank | finalist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| most/cap5 | 17,233 | 18,268 | 17,929 | 17,810 | 2,241 | 3,933 | 58,183 | 235,446 | 161,920 |  |  |
| fill/cap5 | 18,703 | 18,533 | 17,861 | 18,366 | 2,233 | 3,882 | 67,950 | 231,266 | 163,127 | 3.00 | yes |
| nearest/cap5 | 15,639 | 17,266 | 16,636 | 16,513 | 2,953 | 2,418 | -18,468 | 209,429 | 140,087 | 6.00 |  |
| need/cap5 | 18,516 | 17,161 | 17,299 | 17,659 | 2,241 | 3,885 | 78,608 | 230,366 | 162,708 | 3.00 | yes |
| need_near/cap5 | 20,326 | 20,192 | 20,060 | 20,193 | 2,619 | 3,072 | 85,617 | 234,443 | 156,349 | 1.00 | yes |
| most/cap10 | 17,188 | 17,553 | 17,281 | 17,340 | 2,875 | 3,036 | 55,394 | 233,354 | 155,767 | 4.33 |  |
| fill/target | 4,368 | 6,498 | 6,617 | 5,828 | 1,086 | 3,051 | 58,915 | 178,843 | 165,067 | 7.33 |  |
| need/target | 5,013 | 6,722 | 6,882 | 6,206 | 1,113 | 3,080 | 68,606 | 186,256 | 164,555 | 5.67 |  |
| need_near/target | 5,586 | 7,673 | 7,320 | 6,860 | 1,150 | 2,866 | 66,657 | 188,292 | 166,194 | 5.67 |  |
