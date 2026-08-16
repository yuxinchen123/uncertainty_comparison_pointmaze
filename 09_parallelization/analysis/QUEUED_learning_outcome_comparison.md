# Queued: do the two implementations LEARN the same, and does reduced precision change behaviour?

Not yet launched. Waiting on two pieces of work: the round-six PyTorch optimisation at 1024–4096
copies, and the jaguar03 copies-per-worker plateau sweep. When both have reported, are merged, and
`report.md` has been regenerated, launch one agent with the task below.

Everything measured in this project so far is **speed**. This is the first experiment about
**behaviour**: whether the two implementations, given the same algorithm and the same environment,
actually learn the same thing — and whether the card's reduced-precision matrix mode changes that.

## The two questions

1. **PyTorch against JAX.** Same algorithm, same environment, so with enough seeds they should
   converge to the same place. If they do not, the first hypothesis is a defect in one of them, not
   a genuine difference.
2. **Reduced precision against exact single precision**, within each framework, everything else held
   identical. Both trainers ship with the card's reduced-precision matrix mode — PyTorch asks for it
   with `tf32=True`, JAX takes it by default — and the two frameworks' forward passes differ by
   2.2e-03 in that mode against 1.2e-06 in exact single precision (measured 2026-08-15). That is far
   below anything that should move where training converges, which makes it a real question rather
   than a rhetorical one.

## The runs

Four configurations: {PyTorch, JAX} x {reduced precision, exact single precision}.

Each configuration: **8 learning rates, 1024 copies per rate, 10 million environment steps per
copy.** Both trainers already carry the learning-rate sweep across copy groups, so 8 x 1024 = 8,192
copies is one run rather than eight. Splitting into 256- or 512-copy chunks run sequentially is
allowed if memory or scheduling makes it easier; 8,192 copies measured 30 GB of the card's 94, so it
should fit whole.

Every configuration uses the same learning rates, the same seeds, the same step budget, and the same
environment settings. The only thing that varies between two runs being compared is the one thing
under test.

## Parity comes first, before any training

Confirm before spending the card, and write down what was checked:

- **Initialisation follows the same algorithm** in both — same distributions, same gains per layer,
  same fan-in convention, same layer ordering, same treatment of biases and of the log-standard
  deviation. Identical weights are NOT required and should not be forced; the point is that the two
  draw from the same distribution so that, over enough seeds, they explore the same space.
- Same optimiser and its defaults, same advantage normalisation, same observation and reward
  statistics and when they update, same episode boundary and reset handling, same intrinsic-reward
  normalisation, same clipping and entropy terms, same evaluation cadence and definition.
- Anything that differs and cannot be made to match is recorded explicitly as a known difference,
  with its expected effect.

## If the results do not match

Treat a mismatch as a suspected defect. Find it, fix it, re-run, and **document what was fixed** in
`ppo/torch_ppo/progress_and_changes.md` or `ppo/jax_ppo/progress_and_changes.md` (whichever side the
defect was on), with the evidence that identified it and the measurement showing the fix worked. If
the mismatch turns out to be genuine and not a defect, document that too, with the reason.

The bar is not identical curves. With 1024 seeds per rate the bar is that the seed distributions
overlap: compare distributions and their uncertainty, not single curves by eye.

## Plots

- **Three of the eight learning rates**, chosen to span the useful range, on a learning-curve plot:
  **PyTorch solid, JAX dashed**, one colour per learning rate so the pairs read together.
- **Two plots for the precision question**, one per framework: reduced precision against exact single
  precision, same three learning rates, everything else held identical.
- Learning curves carry an uncertainty band across seeds, not just the mean, because the whole
  question is whether two distributions agree.

## Afterwards

Update `report.md` through its generator (never by hand), as a new section at the end, with key
numbers in tables rather than in prose. The reading-state machinery will mark it unread; do not mark
anything read.

## Practical notes

- All GPU work through `locks/gpu_run.sh`. This is hours of card time — write per-item progress to a
  file as it goes, and make the runs resumable, so a kill costs one chunk rather than the campaign.
- The precision knobs: PyTorch `PPOConfig(tf32=...)`; JAX
  `jax.config.update("jax_default_matmul_precision", "highest")` for exact single precision, and its
  default for reduced. Verify each run actually used the precision it claims — a configuration that
  silently ignored the knob would produce a null result that looks like a finding.

---

## Executed 2026-08-16

Run folder: `train_runs/2026-08-16-00-50_learning_outcome_torch-vs-jax_precision-reduced-vs-exact_pointmaze-large-topright_rates-3e-6-to-1e-2-x8_copies-1024-per-rate_T-128_N-4_style-B-epoch-minibatch_10M-step-per-copy_paired-seeds_seed-0`
(parity audit in its `parity_check.md`, written analysis in `analysis/analysis.md`, driver and
analysis code in `code/`). Report section: "Do the two implementations learn the same thing?",
last section of `report/2026-08-15-pointmaze-gpu-parallelization/report.md`.

Answer to both questions: **no difference the run can resolve.** PyTorch against JAX, blocked over
the eight rates, +0.376 reward per copy per iteration with interval [-0.874, +1.626] against an
interquartile spread across copies of 91.6. Reduced against exact single precision: PyTorch +0.841
[-0.395, +2.077], JAX +0.348 [-0.878, +1.575]. Reduced precision's only measured consequence is
speed — 43% in PyTorch and 49% in JAX at 8,192 copies.
