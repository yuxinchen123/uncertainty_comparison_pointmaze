# Map screening for the reward-decline study

## 1. Purpose

Choose the smallest PointMaze map on which the reward-decline problem appears, so that fixes can
be tested quickly. The rule was fixed before the runs: the smallest map where

1. no bonus reaches the goal in at most 25% of copies (exploration is needed there);
2. the best RND setting's final-10% mean episode return is at most 60% of its peak, with the peak
   in the first half of the run;
3. the oracle 1/sqrt(n) bonus shows the same fall.

If no small map passes, the small setting is the large maze at learning rate 1e-3 for 50M steps.

## 2. Key hyperparameters

| map | open cells | start | goal | cap (steps) | steps per copy |
|---|---|---|---|---|---|
| large | 46 | (7,1) | (1,10) | 400 | 50M |
| medium | 26 | (6,1) | (1,6) | 400 | 30M |
| medium | 26 | (6,1) | (1,6) | 250 | 30M |
| U-maze | 7 | (3,1) | (1,1) | 150 | 30M |

- Continuing task (+1 per step inside the 0.45 m goal radius, episodes end only at the cap), no
  reset noise, as in the document's earlier runs.
- Bonuses: RND with intrinsic weights 1, 10, 100, 1000 packed in one 512-copy program; oracle
  1/sqrt(n) with weight 1; no bonus. Learning rate 1e-3 annealed linearly over the run; full-batch
  updates (one gradient step per 128-step x 4-environment rollout); 128 paired seeds (0-127) per
  setting; every other knob at the trainer default.
- Records every 250 iterations with `--diagnostics` (per-episode tracker and policy/critic numbers)
  and coverage.

## 3. Code and config changes

The reward-decline study's environment flags (`--map`, `--start-cell`, `--goal-cell`,
`--episode-steps` now setting the cap) and `--diagnostics`; every other switch off. Summaries by
`scripts/summarize_episode_records.py`. Queue: `queue/`, built by `code/build_queue.py`; workers
`code/worker.sh` through the shared `worker_manager.py`.

## 4. Git state

Commit 1ac4d07 (pushed) plus the run folder's own scaffolding and `scripts/summarize_episode_records.py`,
committed before the full submission (hash recorded below when submitted).
