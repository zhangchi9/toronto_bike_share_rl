# B: checkpoint-picking rules, pick on one validation week, check on the other

Each rule picks a checkpoint from one week's scores (Jul 18-24 or Jul 25-31) and is scored by S1 on the other week, in $ for that week; mean of both directions.

| Rule | dqn_1trucks_s0 | dqn_1trucks_s1 | dqn_1trucks_s2 | mean | picks (steps) |
|---|---|---|---|---|---|
| R0 best S0 | 38,794 | 41,654 | 45,841 | 42,096 | s0:650k, s0:650k, s1:600k, s1:600k, s2:500k, s2:500k |
| R1 best S1 | 39,899 | 40,088 | 45,841 | 41,943 | s0:450k, s0:450k, s1:800k, s1:600k, s2:500k, s2:500k |
| R2 best S1, 3-checkpoint average | 37,959 | 40,156 | 33,084 | 37,066 | s0:850k, s0:850k, s1:750k, s1:800k, s2:450k, s2:450k |
| R3 best S1, 5-checkpoint average | 30,593 | 26,690 | 33,084 | 30,123 | s0:900k, s0:900k, s1:700k, s1:700k, s2:450k, s2:450k |

## Every checkpoint ($/week)

| run | steps | S0 | S1 |
|---|---|---|---|
| dqn_1trucks_s0 | 50,000 | 10,299 | 9,000 |
| dqn_1trucks_s0 | 100,000 | 13,829 | -5,397 |
| dqn_1trucks_s0 | 150,000 | 10,809 | 23,382 |
| dqn_1trucks_s0 | 200,000 | 1,437 | -30,586 |
| dqn_1trucks_s0 | 250,000 | 12,566 | 23,448 |
| dqn_1trucks_s0 | 300,000 | 9,145 | 18,771 |
| dqn_1trucks_s0 | 350,000 | 10,915 | -7,399 |
| dqn_1trucks_s0 | 400,000 | 14,257 | 30,514 |
| dqn_1trucks_s0 | 450,000 | 14,853 | 39,899 |
| dqn_1trucks_s0 | 500,000 | 14,428 | 17,947 |
| dqn_1trucks_s0 | 550,000 | 14,712 | 30,843 |
| dqn_1trucks_s0 | 600,000 | 8,917 | 27,550 |
| dqn_1trucks_s0 | 650,000 | 17,233 | 38,794 |
| dqn_1trucks_s0 | 700,000 | 16,230 | 24,799 |
| dqn_1trucks_s0 | 750,000 | 15,891 | 9,933 |
| dqn_1trucks_s0 | 800,000 | 11,248 | 35,342 |
| dqn_1trucks_s0 | 850,000 | 9,467 | 37,959 |
| dqn_1trucks_s0 | 900,000 | 10,675 | 30,593 |
| dqn_1trucks_s0 | 950,000 | 4,911 | 33,300 |
| dqn_1trucks_s0 | 1,000,000 | 11,286 | 31,380 |
| dqn_1trucks_s1 | 50,000 | 12,495 | 20,979 |
| dqn_1trucks_s1 | 100,000 | 9,531 | 14,120 |
| dqn_1trucks_s1 | 150,000 | 6,037 | 13,300 |
| dqn_1trucks_s1 | 200,000 | 2,621 | -214 |
| dqn_1trucks_s1 | 250,000 | 6,684 | -1,040 |
| dqn_1trucks_s1 | 300,000 | 9,483 | 16,466 |
| dqn_1trucks_s1 | 350,000 | 13,814 | 32,451 |
| dqn_1trucks_s1 | 400,000 | 17,187 | 33,442 |
| dqn_1trucks_s1 | 450,000 | 9,538 | 30,053 |
| dqn_1trucks_s1 | 500,000 | 12,675 | 35,812 |
| dqn_1trucks_s1 | 550,000 | -791 | -8,940 |
| dqn_1trucks_s1 | 600,000 | 18,268 | 41,654 |
| dqn_1trucks_s1 | 650,000 | 12,925 | 35,724 |
| dqn_1trucks_s1 | 700,000 | 14,697 | 26,690 |
| dqn_1trucks_s1 | 750,000 | 15,065 | 38,268 |
| dqn_1trucks_s1 | 800,000 | 17,173 | 40,170 |
| dqn_1trucks_s1 | 850,000 | 13,249 | 28,318 |
| dqn_1trucks_s1 | 900,000 | 13,857 | 19,180 |
| dqn_1trucks_s1 | 950,000 | 14,684 | 31,018 |
| dqn_1trucks_s1 | 1,000,000 | 2,372 | -5,817 |
| dqn_1trucks_s2 | 50,000 | 909 | 11,939 |
| dqn_1trucks_s2 | 100,000 | 6,658 | -1,019 |
| dqn_1trucks_s2 | 150,000 | 14,500 | 18,730 |
| dqn_1trucks_s2 | 200,000 | 5,931 | 20,134 |
| dqn_1trucks_s2 | 250,000 | 520 | 17,181 |
| dqn_1trucks_s2 | 300,000 | 3,929 | -25,198 |
| dqn_1trucks_s2 | 350,000 | 12,431 | 38,347 |
| dqn_1trucks_s2 | 400,000 | 16,712 | 39,925 |
| dqn_1trucks_s2 | 450,000 | 15,185 | 33,084 |
| dqn_1trucks_s2 | 500,000 | 17,929 | 45,841 |
| dqn_1trucks_s2 | 550,000 | 14,405 | 19,237 |
| dqn_1trucks_s2 | 600,000 | 10,187 | 19,846 |
| dqn_1trucks_s2 | 650,000 | 2,279 | 7,241 |
| dqn_1trucks_s2 | 700,000 | 9,044 | 13,734 |
| dqn_1trucks_s2 | 750,000 | 6,003 | 30,082 |
| dqn_1trucks_s2 | 800,000 | 3,820 | 7,929 |
| dqn_1trucks_s2 | 850,000 | 14,069 | 33,082 |
| dqn_1trucks_s2 | 900,000 | 7,274 | 21,411 |
| dqn_1trucks_s2 | 950,000 | 7,352 | 35,611 |
| dqn_1trucks_s2 | 1,000,000 | 9,842 | 22,114 |
