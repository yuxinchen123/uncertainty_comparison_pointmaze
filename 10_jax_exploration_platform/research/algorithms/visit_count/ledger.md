# visit_count — round ledger

The oracle visit-count family: `gt_position_velocity_sqrt` (bonus `1/sqrt(n)`) and
`gt_position_velocity_linear` (`1/n`), counted over discretised position and velocity per copy.
Specification: `src/exploration_platform/bonuses/visit_count/spec.md` (`gt_position_velocity@1`
with update schedule `post_rollout_update@1`).

Times are Pacific with a `PT` marker; the machines run on Eastern and the times were converted for
display. Result files are not committed (the repository ignores `*.json`), so each round keeps its
own `result.json` and the table below carries what it said.

## The gate this family had to pass

Total environment steps per second at least PPO with random network distillation's, at the same
copy count, on the same card, with the bar re-measured in the same session rather than quoted from
a file. The copy count is the science run's: 3 learning rates x 11 intrinsic weights x 256 copies
= **8,448**, one update per batch, 128 rollout steps of 4 environments per copy.

## Rounds

| round | when | what was tried | outcome |
|---|---|---|---|
| 1 — [the fused form against the distillation bar](rounds/2026-08-16-r01-fused-form-against-the-distillation-bar/) | 2026-08-16 19:05–20:21 PT | nothing changed: measure the shipped form at 8,448 copies against the bar, then profile the iteration to find where a change would have to go | **Both presets pass, at 2.44 times the bar, 22 of 22 paired rounds.** The profile then closed the round: the bonus's whole share of an iteration is **about 1.2 percent**, so that was the ceiling on every change that had been planned. Five directions discarded on that ceiling, one recorded unmeasured for a future round that needs it |

## Where this family stands

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory | against the bar |
|---|---|---|---|---|---|---|---|
| `rnd_next_state` (the bar) | 8,448 | 0.08402 | 51.48 M | 6.09 k | 0.0456 | 28.16 GiB | 1.000 |
| `gt_position_velocity_sqrt` | 8,448 | 0.03450 | 125.38 M | 14.84 k | 0.0187 | 7.78 GiB | **2.436** |
| `gt_position_velocity_linear` | 8,448 | 0.03427 | 126.21 M | 14.94 k | 0.0186 | 6.89 GiB | **2.449** |

Measured 2026-08-16 on the H100 NVL of serval05 under the lock, jax 0.11.0, platform commit
`f3fd968`, sync timing, medians over 11 paired rounds of 10 iterations. The two presets differ only
in the decay exponent, so the 0.7 percent between them is noise, not a difference between them.

At 8,448 copies each visit-count arm covers 4.33 M environment steps per iteration, so the first
batch's 10 M steps per copy is 19,532 iterations — about **11 minutes** per arm on this card, and
about 27 minutes for the distillation arm beside it.

## Correctness, after the round

No source file changed in this round, so every gate stands exactly where the stage that built the
family left it, and all of them were re-run at the end of the round to say so: the golden-parity
comparison against the frozen `09_parallelization` trainer, the fused-against-reference
equivalence on supplied trajectories, the count table against `07_reconstruction`'s own wrapper
(0 of 10,800 entries differ), copy isolation, compiled against uncompiled, and the learning-sanity
check. The record of those runs is `tests/golden_09/parity_results.md`.
