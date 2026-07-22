# Convergence run 1 — analysis (2026-07-19)

Fits of the convergence-rate sweep `2026-07-19-03-32_convergence-run1` (4320 runs: 3 point sets
x 4 initializations x 12 optimizer configurations x 30 seeds, 4096 full-batch distillation
steps each). The full writeup, both design tables, and the results live in
`development_document/main.tex`, Section "Convergence rate runs" (`sec:convergence-rate-runs`);
this file records what the analysis computed and where its artifacts are.

## Data and exclusions

- Input: `../data/2026-07-19-03-32_convergence-run1/local/*.json` — 4320 records, all with
  `completed: true`; per record, per-point l2 readouts at 80 checkpoint steps ({0} plus
  round of $2^{j/8}$ for $j = 0..96$, deduplicated).
- Excluded from every fit: 180 records with `diverged: true` — exactly SGD-1/t at
  learning rate 0.1 on env 2 (cell midpoints) for initializations I3 (full PyTorch default)
  and I4 (normal-0.5 bias), all 30 seeds, at every t0 value (6 cells x 30). Cause: env 2's
  raw inputs reach an x magnitude of 5.5 with no observation normalization, and the 0.1-rate
  full-batch step is unstable under the two largest-scale initializations. These cells show
  "div." in the tables; their aggregate column drops out too (the aggregate needs all three
  point sets of a seed). Post-exclusion cell sizes are n = 30 everywhere else.

## Method (formulas at the point of use, per the analysis convention)

- Per-point normalization: $y_i(n) = b_i(n) / b_i(0)$ where $b_i(n)$ is the exact l2 readout
  at point $x_i$ after $n$ updates (the code's 1e-4 readout floor is dropped in logging).
- Fitted model, window $n \ge 2$: $y(n) = c + a\,(n + n_0)^{-\alpha}$ — nonlinear least
  squares with residuals on $\log y$ (a floored fit, not a straight-line log regression),
  bounded parameters, 6 starts. The reported slope is $-\hat\alpha$.
- Two granularities, reported separately: per-seed fits (30 slopes -> mean and standard error
  $s/\sqrt{30}$ over seeds) and one fit of the seed-averaged curve.
- Ranking: closeness of the per-seed aggregate slope (all 308 points pooled) to $-1/2$.

## Headline results

1. Best configuration: I1 (zero-bias) + SGD-1/t, learning rate 1e-2, t0 1e2 — aggregate slope
   $-0.503 \pm 0.005$ (seed-average fit $-0.502$): almost exactly the count-based reference.
2. Adam never comes close to $-1/2$: its best (I1, learning rate 1e-3) fits $-0.83$; all Adam
   cells decay faster than the reference.
3. The two granularities agree on the aggregate column (median gap 0.002, max 0.04) but split
   by up to 0.53 on env 1.2 — real seed heterogeneity in the top-right cell, exactly what
   reporting both is for.
4. The curves are not single power laws: the local-exponent panels show an early geometric
   ramp, a mid plateau, and a late steepening; the very steep small-learning-rate slopes
   (about $-1.7$) come from that late phase.

## Artifacts

- `code/fit_convergence.py` -> `data/fits.json` (all per-cell fits, curves, local exponents).
- `code/make_results.py` -> `plots/slope_table_perseed.tex`, `plots/slope_table_seedavg.tex`,
  `plots/opt_effect_table.tex` (the tables the tex `\input`s).
- `code/make_plots.py` -> `plots/curves_env11.pdf`, `plots/curves_env12.pdf`,
  `plots/curves_env2.pdf`, `plots/opt_effect.pdf`.
- Regenerate with `/p/rlprojects/RND/.venvs/exploration/bin/python` in `code/`, in that order.
