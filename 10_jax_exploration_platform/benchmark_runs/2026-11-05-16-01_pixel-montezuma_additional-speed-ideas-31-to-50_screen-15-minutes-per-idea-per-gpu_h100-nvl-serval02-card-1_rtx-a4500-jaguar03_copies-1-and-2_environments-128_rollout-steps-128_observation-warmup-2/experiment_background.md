# Additional speed ideas 31 to 50: fifteen-minute screens

## Purpose

Phase 2 of the pixel Montezuma speed study (plan `~/.claude/plans/2026-10-31-00-02_rnd-merge-atari-infrastructure-optimization.md`,
sections 5 and 190). The first thirty ideas are finished (120 comparisons, campaign folder
`benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_.../`). This folder holds the twenty
additional ideas, chosen in four batches of five from measurements, each screened on an H100 NVL and an RTX A4500 at
one and two independent copies.

User rule 2026-11-05: each idea runs at most 15 minutes of wall time on each GPU, compilation included. Each screen
measures steady-state seconds per complete update, paired against the reference in the same process.

Measured starting point (2026-10-31 trace, 2026-11-05 common timing): with deterministic XLA operations, convolution
weight-gradient kernels take 86% of H100 GPU time; one H100 update takes 13.1 s with one copy and about 45 s with two
(the RTX A4500: 4.6 s and 11 s before the rendering changes). With determinism off the H100 update took 1.81 s.
Batch 1 therefore targets the convolution gradients and the two-copy slowdown while keeping deterministic operations.

## Key hyperparameters

| setting | value |
|---|---|
| learning configuration | original CleanRL PPO+RND (128 environments, 128 rollout steps, 4 epochs, 4 minibatches, learning rate 1e-4) |
| precision | float32 networks at highest matrix precision (no TF32), float64 statistics |
| determinism | `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true` (as in all earlier formal measurements) |
| reference | original port + cached room backgrounds + render only the last two game states |
| observation-normalization warmup | 2 random-action rollouts (CleanRL: 50); this changes no update's work, only the screen's start-up time |
| copies | 1 and 2; seeds start at 1 |
| time per screen | 840 s budget for timing decisions, hard limit 900 s (`timeout` / Slurm `--time=00:15:00`) |
| timed pairs | one complete update per variant per pair, alternating order, up to 7 pairs while time remains |
| decision | accepted: at least 5 pairs, every pair faster, median speed ratio at least 1.01; rejected: correctness failure or median at most 1.0; otherwise inconclusive |
| correctness | one complete update on identical supplied inputs plus one complete real-game iteration; float32 absolute 1e-5 / relative 1e-4, float64 1e-10 (`paired_benchmark.correctness_checks`) |

Hardware: H100 NVL on serval02 card 1 (UUID GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3, study exception of plan section 6)
for one-copy units, H100 NVL on serval05 through its shared queue for two-copy units, RTX A4500 on jaguar03 through
Slurm (reservation `sl5nw_168`, one card per job).

## Code and config changes

- `src/exploration_platform/atari_pixels/networks.py`: `convolution(..., algorithm=...)` with `compiler` (unchanged
  default), `patch_weight_gradient`, `patch_matrix`, `space_to_depth`; `convolution_batching="sequential"`.
- `src/exploration_platform/atari_pixels/config.py`: new field `convolution_algorithm` (default `compiler`); new
  choice `sequential` for `convolution_batching`. Defaults reproduce the earlier code exactly.
- `scripts/atari_pixel_campaign/screen_idea.py`: the fifteen-minute screen and the reference profile.
- `tests/test_pixel_convolution_algorithms.py`: 15 checks (values, input and weight gradients against the compiler's
  convolution for all three CleanRL layers, one and two copies; bit-exact copy independence). Passed on an RTX A4500,
  job 6830306, 2026-11-05.

Batch 1 (ideas 31 to 34; idea 35 is chosen from the reference profile in this folder):

| idea | mechanism | switch |
|---|---|---|
| 31 | convolution weight gradient as one matrix product of copied image windows | `convolution_algorithm=patch_weight_gradient` |
| 32 | whole convolution as matrix products of copied image windows | `convolution_algorithm=patch_matrix` |
| 33 | strided convolutions as stride-1 convolutions on rearranged pixels | `convolution_algorithm=space_to_depth` |
| 34 | copies one after another in the convolutions | `convolution_batching=sequential` |

## Git state

Branch `merge-atari-cleanrl`, base commit `67d7e6c07cc5472ac36dc401b789089a6047b7f3`; the code above is committed and pushed before the first screen (commit recorded in
`manifest.yaml` and in the plan).
