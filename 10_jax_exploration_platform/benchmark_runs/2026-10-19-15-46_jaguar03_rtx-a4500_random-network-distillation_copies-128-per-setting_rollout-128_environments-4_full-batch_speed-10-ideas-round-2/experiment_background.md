# Ten speed ideas on jaguar03 (RTX A4500), round 2

## 1. Purpose

Find execution choices that make the reward-decline study's training units finish sooner on
jaguar03's RTX A4500 cards, before the study's batches are submitted. The serval05 study of
2026-10-13 already covered flat parameters, scan unroll, grouped dispatch and copy count alone;
these ten ideas cover how work is laid out on a card, XLA's execution flags, setup time, the
processor count, and the cost of the study's own measurements.

| idea | what changes against the baseline |
|---|---|
| 01 | 4 settings x 128 copies in one program (512 copies) |
| 02 | 8 settings x 128 copies in one program (1,024 copies) |
| 03 | two baseline programs sharing one card |
| 04 | four baseline programs sharing one card |
| 05 | XLA command buffers off (`--xla_gpu_enable_command_buffer=`) |
| 06 | XLA command buffers for every supported kind, including while loops |
| 07 | XLA latency-hiding scheduler on |
| 08 | persistent compilation cache: a cold run, then a warm run reading the cache |
| 09 | 2 processors for the program instead of 8 |
| 10 | the study's diagnostics on (episode tracker, policy and critic numbers) |

Each round puts the baseline on one card and ideas on the other three at the same time; the
baseline card turns over every round and each idea runs twice on two different cards.

## 2. Key hyperparameters

- Program: PPO with random network distillation, `full_batch` (one gradient step per rollout),
  rollout 128 steps, 4 environments per copy, 128 copies per setting, learning rate 1e-3,
  intrinsic weight 100 for setting 0 (packed settings add weights 1, 10, 1000, 3, 30, 300, 3000),
  coverage tracking on, large PointMaze, seed 0.
- Timing: 10 checked iterations (the first includes compiling), then three 8-second blocks of
  32-iteration groups; the middle block is the result.
- Agreement: parameters of setting 0's 128 copies after the first 10 iterations are saved (`.npz`)
  and compared with the baseline's.
- Job: jaguar03, 4 RTX A4500 cards, 32 processors (8 per card), 64 GB, reservation `sl5nw_165`.

## 3. Code and config changes

- New: `benchmarks/harness/bench_jaguar03_ideas.py` (one measurement), `code/driver.py` (rounds),
  `code/job.sh` (the Slurm job).
- The training code includes the reward-decline study's switches, all off except in idea 10.

## 4. Git state

Filled in at submission (see the line below).
