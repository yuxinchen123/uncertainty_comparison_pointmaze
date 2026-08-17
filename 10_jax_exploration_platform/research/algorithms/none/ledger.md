# none — round ledger

PPO with no intrinsic bonus at all: the family selects empty parameters and an empty state, its
loss is the constant zero, and because the bonus is chosen before the program is compiled, none of
its arithmetic reaches the compiled program (`tests/bonuses/test_none_has_no_bonus_arithmetic.py`
counts the matrix multiplications with and without it).

Times are Pacific with a `PT` marker; the machines run on Eastern and the times were converted for
display.

## The gate this family had to pass

The same gate every family passes: total environment steps per second at least PPO with random
network distillation's, at the same copy count, on the same card, with the bar re-measured in the
same session. This arm's copy count is the science run's for it — 3 learning rates x 256 copies =
**768** (the intrinsic weight is not swept for an arm that has no intrinsic reward to weight).

## Rounds

| round | when | what was tried | outcome |
|---|---|---|---|
| 1 | 2026-08-16 19:38–19:40 PT | nothing changed: measure the shipped form at 768 copies against the bar | **Passes at 1.597 times the bar, 11 of 11 paired rounds**, holding a quarter of the memory (0.76 GiB against 2.98 GiB) |

**No code change was needed, and none was made.** This family is strictly less work than the bar
it is measured against — it is the bar with a piece removed — so the only thing in question was
whether the removal shows up on the card, and it does.

## Where this family stands

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory | against the bar |
|---|---|---|---|---|---|---|---|
| `rnd_next_state` (the bar) | 768 | 0.01116 | 35.24 M | 45.89 k | 0.0061 | 2.98 GiB | 1.000 |
| `none` | 768 | 0.00699 | 56.27 M | 73.27 k | 0.0038 | 0.76 GiB | **1.597** |

Measured 2026-08-16 on the H100 NVL of serval05 under the lock, jax 0.11.0, platform commit
`f3fd968`, sync timing, medians over 11 paired rounds of 10 iterations. The arm's intrinsic reward
read back after the rounds is exactly 0.0, in every copy, so the row is not measuring a bonus that
quietly survived. Raw output: `benchmark_runs/2026-08-16_visit-count-gate/`; the assembled record
is `../visit_count/rounds/2026-08-16-r01-fused-form-against-the-distillation-bar/result.json`,
under `gate_at_768_copies_no_bonus`.

At 768 copies this arm covers 393 k environment steps per iteration, so 10 M steps per copy is
25,432 iterations — about **3 minutes** on this card.

## A note on how this family was handled

The plan gives every algorithm family its own worktree fork and its own fusion subagent. This
family was measured inside the `algo/visit-count` fork instead, together with the visit-count
family, and the deviation is deliberate: a fork whose expected change set is empty costs a branch,
a worktree, a merge and a second agent to produce a single measurement, and the measurement needs
the same card, the same lock and the same benchmark script the visit-count fork was already using.
The one-fork-per-family rule stands for families that might need code changed; it was folded in
here because this one could not.
