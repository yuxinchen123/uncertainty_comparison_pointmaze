# Train run 1 (before reorganization) — aggregate of W&B sweep `y24vyh06`

Aggregate of the SAC + intrinsic-bonus sweep `sweep_my_rnd` (id **`y24vyh06`**,
project `catresearch/rnd_07_reconstruction`) on `PointMaze_Large-v3` with the fixed
**`bottom_right`** goal — the configuration used before the later reorganization to
the `top_right` goal documented in `main.tex`. This folder produces the §5.1
"Train run 1" subsection of `development_document/main.tex`: one swept-hyperparameter
table, one best-beta results table, one bar plot, and one learning-curve line plot.

## 1. Data source

- Pulled live from W&B on 2026-06-24 (see `data/y24vyh06/manifest.json`).
- **8000** launched runs. State counts: **3292 finished**, **4708 crashed**.
- Only **finished** runs enter every aggregate below (crashed / running excluded).
- Slim per-run summary table: `data/y24vyh06/runs_slim.csv` (one row per run; config
  + summary scalars only, no heatmap-media columns).
- Per-seed learning curves for the best-beta groups: `data/y24vyh06/history_best_beta.csv`
  (16,840 rows = 421 runs × 40 eval points, eval logged every 50,000 steps).

## 2. Filter and grouping

- **Filter:** `state == "finished"`.
- **Group:** by `(algorithm, beta)`. The grid is 10 algorithms × 8 `beta` values
  × up to 100 seeds; each finished `(algorithm, beta)` group has ≈ 40 seeds.
- **Best beta per algorithm:** for each algorithm keep the single `beta` whose
  finished runs have the largest mean of the primary metric
  `eval/mean_extrinsic_reward` (the 10,000-episode evaluation at 2,000,000 steps).

Primary metric per group of $n$ seeds with per-seed final rewards $R_i$:

$$\bar R = \dfrac{1}{n}\sum_{i=1}^{n} R_i, \qquad \mathrm{SE} = \dfrac{s}{\sqrt{n}}, \qquad s = \sqrt{\dfrac{1}{n-1}\sum_{i=1}^{n}\left(R_i-\bar R\right)^2}.$$

## 3. Verification against the W&B dashboard

The dashboard view (grouped by `algorithm` then `beta`, collapsed to algorithm level)
shows the **`train/mean_extrinsic_reward`** column averaged over all 8 betas and all
seeds of each algorithm. Reproducing that exact quantity from the downloaded data
matches the dashboard to every shown digit — confirming the download is complete and
correct (`data/y24vyh06/verify_train_level.csv`). Best value **bold**, second
<u>underlined</u>; higher is better.

| Algorithm | runs | `train/mean_extrinsic_reward` (mine) | dashboard | match |
|---|---:|---:|---:|:--:|
| `rnd_next_state` | 318 | **33.7973** | 33.79733 | ✓ |
| `rnd_state_action` | 325 | <u>32.0206</u> | 32.02058 | ✓ |
| `gt_position_velocity` | 328 | 31.2493 | 31.24933 | ✓ |
| `rnd_state` | 325 | 31.0023 | 31.00231 | ✓ |
| `rnd_state_action_next_state` | 325 | 28.4505 | 28.45046 | ✓ |
| `gt_position` | 337 | 27.7469 | 27.74694 | ✓ |
| `rnd_elliptical` | 338 | 25.2448 | 25.24476 | ✓ |
| `rnd_next_state_position_only` | 321 | 17.8957 | 17.89573 | ✓ |
| `rnd_linear_next_state` | 334 | 17.6592 | 17.65919 | ✓ |
| `no_exploration` | 341 | 16.8512 | 16.85120 | ✓ |

Note this dashboard quantity (algorithm-level mean **train** reward, all betas) is a
sanity check on the data, not the deliverable. The deliverable below uses the project
primary metric **`eval/mean_extrinsic_reward`** and the **best** beta per algorithm.

## 4. Best beta per algorithm (deliverable)

Final eval extrinsic reward $\bar R$ at 2,000,000 steps for each algorithm at its best
coefficient (`data/y24vyh06/best_beta_per_algorithm.csv`). Rows sorted by $\bar R$
descending; best **bold**, second <u>underlined</u>; higher is better.

| Algorithm | best beta | $\bar R$ (↓) | SE | n |
|---|---:|---:|---:|---:|
| `gt_position_velocity` | 1 | **74.86** | 4.63 | 43 |
| `rnd_elliptical` | 0.01 | <u>71.12</u> | 6.09 | 44 |
| `rnd_state` | 100 | 69.90 | 4.80 | 41 |
| `rnd_next_state` | 100 | 62.03 | 5.34 | 40 |
| `gt_position` | 10 | 59.37 | 5.32 | 45 |
| `rnd_state_action_next_state` | 100 | 57.04 | 4.97 | 40 |
| `rnd_state_action` | 10 | 52.53 | 4.87 | 43 |
| `rnd_linear_next_state` | 10 | 32.18 | 5.60 | 42 |
| `rnd_next_state_position_only` | 1000 | 29.99 | 5.97 | 38 |
| `no_exploration` | 0.0001 | 25.60 | 6.09 | 45 |

Takeaways: the `gt_position_velocity` count oracle is strongest (74.9); the learned
`rnd_elliptical` bonus is a close second (71.1) and `rnd_state` third (69.9); all three
are well above the `no_exploration` baseline (25.6). Position-only RND and the linear
RND head trail.

## 5. Figures

- `plots/bar_best_beta_eval_reward.{pdf,png}` — bar chart of $\bar R \pm \mathrm{SE}$,
  best beta per algorithm, sorted best-first (Figure 2 in `main.tex`).
- `plots/line_best_beta_eval_curve.{pdf,png}` — eval reward vs training step, one line
  per algorithm's best-beta group, shaded band $= \pm\,\mathrm{SE}$ over seeds, legend
  ordered by final reward (Figure 3 in `main.tex`). Colors match the bar chart.

## 6. Reproduce

Run from this folder with `conda run -n exploration python …`:

1. `code/01_download_slim.py --sweep-id y24vyh06` — slim summary for all 8000 runs →
   `data/y24vyh06/runs_slim.csv` (+ `manifest.json`). Fast: a server-side `sweep`
   filter returns each run's summary in the page, so 8000 runs pull in under a minute.
2. `code/02_download_history.py` — best-beta per algorithm by eval reward, then the
   eval-reward time series for those ~421 runs → `data/y24vyh06/history_best_beta.csv`.
   Run on Slurm via `code/02_run_history_download.slurm` (pinned to jaguar03, which has
   outbound internet; uses my active reservation `--reservation=sl5nw_151`).
3. `code/03_table_and_bar.py` — grouped tables (`all_groups.csv`,
   `best_beta_per_algorithm.csv`, `verify_train_level.csv`), the bar plot, and it
   splices the results table directly into `../../main.tex` between the
   `% >>> AUTO-GENERATED TABLE START: trainrun1-best` markers (no separate `.tex`
   fragment; the generate-latex-table convention).
4. `code/04_line_plot.py` — the learning-curve line plot (+ `curve_stats.csv`).

`data/` is gitignored (regenerable); `code/`, `plots/` (PNGs), and this `analysis.md`
are committed. The §5.1 tables and figures live inline in `main.tex` — the data table
is spliced by script 3, so no separate `.tex` files are kept here.
