# Round 14: validation schemes, written down before any comparison with the test

Written 2026-11-16, before any scheme score was compared with a test result.

## Why today's validation misleads (non-test evidence)

1. Validation (two fresh 1-week episodes, Jul 18-24 and Jul 25-31) mostly measures the first
   days after the network starts 35% full. In one continuous run from May 1, the same
   champion models earn $34.8k-58.1k/week on Jul 18-31, against $17.2k-18.3k/week in fresh
   1-week episodes; the seeds rank differently.
2. Checkpoint picking takes the best of 20 evaluations on the same 2 weeks. Across 27 past
   runs, the picked checkpoint averaged $16.2k/week and its two neighbours $9.7k/week.

## Schemes (rl/validation.py); all score only days before Aug 1

- S0 fresh weeks: today's validation.
- S1 aged starts (MAIN): 4 runs from a fresh network starting Jul 18, Jul 4, Jun 20 and Jun 6,
  each to Jul 31; only Jul 18-31 is scored, against no trucks started the same day. Ages
  1-56 days, like the test's 1-51.
- S2 long run: one run Jun 11 -> Jul 31 (51 days), all scored.

## How they are judged

A. Across past models (one check, no new test runs): Spearman correlation of each scheme's
   score with the test gain (Aug 1 - Sep 20) already in results/round{7,8,9,11}, over every
   1-truck learned model with a saved best checkpoint (duplicates removed). Also against test
   days 1-14 and 15-51. The check measures; it picks no model and does not change the main
   scheme.

B. Within a training run (round 14 retrains of the champion's settings, seeds 0-2, every
   checkpoint kept; no test runs): picking rules are judged by pick-then-check. A rule picks
   a checkpoint using Jul 18-24 only and is scored by S1 on Jul 25-31, and the other way round
   (mean of both). Rules:
   - R0: best S0 checkpoint (today's rule)
   - R1: best S1 checkpoint
   - R2 (MAIN): best S1 averaged with its two neighbouring checkpoints
   - R3: best S1 averaged over five neighbouring checkpoints
