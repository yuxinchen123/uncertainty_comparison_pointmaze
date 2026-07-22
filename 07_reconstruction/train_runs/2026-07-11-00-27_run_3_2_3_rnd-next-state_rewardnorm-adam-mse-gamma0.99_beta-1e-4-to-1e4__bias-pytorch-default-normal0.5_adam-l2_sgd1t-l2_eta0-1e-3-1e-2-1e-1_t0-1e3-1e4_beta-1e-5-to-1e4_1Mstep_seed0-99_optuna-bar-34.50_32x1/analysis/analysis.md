# Run 3.2.3 + 3.2.4 analysis (snapshot 2026-07-17; both sweeps still finishing)

The authoritative write-up lives in `development_document/main.tex` (subsubsections
"Train run 3.2.3" and "Train run 3.2.4", Tables 49–50). This folder holds the generator and the
generated table bodies that `main.tex` inputs:

- `code/make_results.py` — reads both runs' per-run JSONs (completed records only; score = the
  last `train_history` row's `train/mean_extrinsic_reward`) and writes the two tables. Rerun it
  with `/p/rlprojects/RND/.venvs/exploration/bin/python` when the sweeps complete to refresh
  every number in place.
- `plots/reward_table.tex` — run 3.2.3: best configuration (>= 10 seeds) per arm x bias scheme,
  plus the run-3.2.2 C2 RND-benchmark reference (recomputed from its records; bar = its mean).
- `plots/validation_table.tex` — run 3.2.4: the top-5 (V1–V5, ranked by in-sample mean at the
  2026-07-14 selection) on 100 fresh seeds 500–599, no pruning, with the gap to the benchmark
  and the one-sided confidence.

Status at this snapshot: run 3.2.3 has 132/149 configurations decided (all by the bar rule;
10,240 of 14,900 queued runs pruned before running) with a refill wave finishing the 138-run
tail; run 3.2.4 has ~75/100 seeds per configuration done. V1's n excludes two NaN-diverged
seeds (575, 598) — a model failure (SAC actor NaN under the largest initial bonus), reported,
not requeued.
