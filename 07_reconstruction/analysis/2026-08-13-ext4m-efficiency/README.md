# ext4m efficiency auto-research (2026-08-13)

Make the run-6 4M-step training (`ext4m`) run faster on CPU **without invalidating a single result
already produced**, then hand the surviving changes to the 96-hour training.

Ordered by the user on 2026-08-13 (03:30), modelled on the autoresearch loop
(`reference_autoresearch/`, cloned from github.com/karpathy/autoresearch): establish a baseline,
change one thing, measure, keep or discard, log every attempt, repeat. Two differences from that
loop, both from the user's instruction:

1. **The metric is speed under a hard constraint.** A candidate is only "applied" if the run it
   produces is *the same run*: the record JSON — every eval row, every training-episode row, every
   visit count — identical to the baseline's, modulo `runtime_seconds`. Speed without that is not
   an improvement, it is a different experiment.
2. **Validity-breaking ideas are not thrown away.** The meaningful ones live, unapplied, in
   `deferred_experiments/` with the measurement or reasoning behind them, for a future run that is
   allowed to start a fresh comparison.

Scope, per the user: **CPU only, PyTorch only** (no JAX port, no CUDA kernels, no C++ env — those
belong to `future_changes.md`).

## Result (campaign closed 2026-08-13 10:45, six experiments, ~5.5 h of exclusive-node measurement)

**+9% throughput, with every recorded number unchanged** — 223 of 223 record comparisons identical.
Per worker under production packing: 15.366 → 16.821 steps/s, i.e. a 4M run drops from **72.3 h to
66.1 h**, about **6.2 h per run and ~5,600 worker-hours across the sweep's 900 runs**.

Roughly 7 of those 9 points need **no code change at all**: `LD_PRELOAD=libtcmalloc_minimal.so.4`
plus two switches that already ship in `train.py` and had never been turned on
(`opt_polyak_foreach`, `opt_torch_reward`). The other ~2 points are three small code changes, held
for the next sweep boundary rather than applied under 450 running runs. Details, the identity
argument for each, and the two commands that apply them: `analysis/applied_changes.md`.

## Where things are

| path | what |
|---|---|
| `analysis/progress_and_changes.md` | the experiment log — every attempt, its measured effect, kept or discarded |
| `analysis/applied_changes.md` | the changes that survived: what, why, measured gain, identity proof, how they reach the 96-h run |
| `analysis/future_changes.md` | meaningful changes that would break result validity — described, costed, NOT applied |
| `analysis/code/run_conditions.py` | the measurement driver (one subprocess per condition, speed + record identity) |
| `analysis/code/patched_entry.py` | candidate optimizations, monkey-patched into the trainer at run time |
| `deferred_experiments/` | code for the validity-breaking ideas, deliberately not wired into `train.py` |
| `slurm/bench.slurm` | one measurement job = every condition, sequentially, on an exclusive node |
| `slurm/submitted_jobids_effres.txt` | my own job ids — the only list anything here may cancel from |
| `data/markers/` | the three production configs, copied read-only out of the live ext4m queue |
| `data/results_<jobid>.json` | raw measurements |

## The measurement, and why it is trustworthy

- **The workload is the production workload.** Each condition runs the real `train4m.py` driven by a
  real ext4m queue marker (`data/markers/`: `run5origrnd`, `alg23`, `gtposvel` — the sweep's three
  configurations), with production env knobs, `OMP_NUM_THREADS=1`, the shared env's python, and only
  the horizon shortened (10,000 steps, `eval_freq` 2,500) so a condition costs minutes.
- **One job at a time, on an idle node it owns.** `sbatch --exclusive -w <node>`; the conditions run
  sequentially inside that one job, so no other process — mine or anyone's — shares the machine
  while a measurement is taken (the user's clean-result rule).
  **Node: `cortado01`** (48 cores, idle, same class as the production node `cortado09`). The user
  asked for `puma01`; `puma01` is inside reservation `sl5nw_156` whose `Users=` list is `sl5nw`
  alone, so this account cannot place a job there — an idle whole node of a production class is the
  faithful substitute, and every comparison is baseline-vs-candidate *within the same job on that
  same node*, so the node choice cancels out of every ratio.
- **The live sweep is never touched.** 450 ext4m runs are training out of this repo right now, so no
  candidate edits `train.py`, `rnd.py` or anything else under `07_reconstruction/`: candidates are
  applied by `patched_entry.py` inside the measurement process only. Nothing here cancels, requeues
  or writes to the sweep's queue, and the markers were copied, not moved.

## Conditions

| condition | what it changes | expected to be record-identical? |
|---|---|---|
| `baseline` | nothing — what the sweep runs today | (reference) |
| `optflags` | `--opt_polyak_foreach=True --opt_torch_reward=True`, two switches that already ship in `train.py` (built and measured 2026-06-25 at +6.1%) but have never been turned on in a production sweep | yes, by construction (proved bit-exact by `analysis/2026-06-25-run-profiling/analysis/code/test_optimizations.py`) |
| `adamforeach` | `torch.optim.Adam/SGD(foreach=True)`: on CPU torch defaults to the single-tensor Python loop, so SAC's three optimizers + the RND predictor walk their parameter lists every gradient step | expected — same element-wise math, verified by the record check |
| `normcache` | cache the observation-normalization mean/var tensors between `RunningMeanStd` updates instead of rebuilding them on every call | expected — same values |
| `gcfreeze` | `gc.freeze()` after construction so the long-lived env/model objects stop being rescanned | yes — bookkeeping only |
| `tcmalloc` | `LD_PRELOAD=libtcmalloc_minimal.so.4` | yes — allocator only |
| `combined` | everything above together | the one that matters for the 96-h run |

## The loop

1. Submit one `bench.slurm` job with the condition list; **never** submit a second while it runs.
2. When it finishes, read `data/results_<jobid>.json`: `steps_per_s`, `ratio_vs_baseline`, and
   `validity` (`IDENTICAL` or the exact field where it diverged).
3. Append one row per attempt to `analysis/progress_and_changes.md` — including the failures.
4. Keep what is faster *and* identical; move what is faster *and* different into
   `analysis/future_changes.md` + `deferred_experiments/`; discard what is neither.
5. Re-measure the survivors together (`combined`) with repeats, on all three configs, before
   applying anything to a 96-hour run.
