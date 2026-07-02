# Train run 3.1.1 — analysis

Status: **aggregated 2026-07-02** (`SWEEP_ID=2026-06-26-05-45_4algo-betaridge-input`); the sweep continues
to fill the 100-seed grid. Aggregate pooled finished runs only (a run writes its JSON once at completion,
so cancelled/incomplete runs contribute no file): 4054 of 9100 finished, 0 failed, every one of the 91
configurations at >= 30 finished seeds (min 37, median 45).

## Results (at aggregation)

Best combination of parameters per method, scored on the final training-episode extrinsic reward
$\bar{R} = \frac{1}{n}\sum_i R_i$ (mean over the past 100 completed training episodes at $10^6$ steps;
$\mathrm{SE} = s/\sqrt{n}$), configurations eligible at $n \ge 30$:

| Method | input* | lambda* | beta* | Rbar | SE | n |
|---|---|---|---|---|---|---|
| rnd_elliptical_global (sample) | s' | 1 | 100 | **37.41** | 5.69 | 47 |
| rnd_next_state | s' (fixed) | — | 100 | <u>36.95</u> | 3.92 | 48 |
| rnd_elliptical_global (add) | s' | 1 | 10 | 28.57 | 4.69 | 53 |
| rnd_elliptical (batch) | s' | 1e-2 | 1e-2 | 15.65 | 3.75 | 45 |

Reading (all at 1M steps, training-episode reward):
- The **global elliptical (sample-time)** at its best config edges out RND-next-state (37.41 vs 36.95,
  overlapping SEs — statistically a tie), and both track run-2's `rnd_state` eval curve (~35.8).
- **Next-state input wins for every elliptical method** — none of the three best elliptical configs uses
  the (s,a) input. The large ridge $\lambda=1$ wins for both global variants; the batch variant prefers
  the small ridge and a small beta and clearly trails (15.65).
- Add-time covariance (the true visitation count) is WORSE than sample-time here (28.57 vs 37.41).
- The run-2 `gt_position_velocity` oracle (~67.8 eval reward) remains far above every learned bonus.
- Caveat: the run-2 reference lines are **eval** reward while the run-3.1.1 lines are **training-episode**
  reward (this run logs no standalone eval); related but not identical metrics — stated in the figure
  caption.

Spliced into `development_document/main.tex` §5.3.2 as Table `tab:trainrun3-1-1-best` and Figure
`fig:trainrun3-1-1-curve` (seven lines: 4 methods solid + 3 run-2 references dashed
[`gt_position_velocity` β=1, `rnd_state` β=100, `rnd_elliptical` β=0.01]; a run-4 JAX overlay was tried
and removed at the user's request — run 4 under-trains and only reaches 500K steps).

## Why the run-3.1.1 batch elliptical (15.65) is far below run-2's `rnd_elliptical` (44.49)

Not a sweep-selection artifact: at the configuration closest to run 2 — (s,a) input, λ=1e-2, β=0.01 —
the run-3.1.1 batch elliptical scores only **9.03 ± 3.22 (n=39)** (full 28-config grid inspected). The
two runs use DIFFERENT algorithms. Verified against the run-2-era code (`git show
293ec3e:.../elliptical_bonus.py` and `.../methods/__init__.py`); three implementation deltas:

1. **Feature normalization**: run 2 = raw frozen-encoder features (unbounded, state-dependent norm);
   run 3.1.1 = unit-norm everywhere ($\|\varphi\|_2 \le 1$).
2. **Ridge**: run 2 hard-coded $\lambda = 10^{-6}$; the run-3.1.1 grid only has $\{1, 10^{-2}\}$
   ($10^{-6}$ was never swept).
3. **Bonus clip**: run 2 had none; run 3.1.1 clips at 5.

Mechanism: with unit-norm features the bonus is bounded by $\sqrt{1/\lambda}$ — at most 10 for
λ=1e-2 (then clipped to 5) and at most 1 for λ=1 — while run-2's raw features with λ=1e-6 and no clip
gave novelty bonuses up to ~$10^3\,\|z_\varphi\|$ with much stronger novel-vs-visited contrast. β cannot
restore the contrast (it rescales the level, and the clip flattens exactly the most-novel states). The
eval-vs-train metric difference contributes little: run-2 `rnd_state` eval (35.79) ≈ run-3.1.1
`rnd_next_state` train (36.95). Isolating which delta dominates (λ=1e-6 / no clip / raw features under
the current code) is a candidate follow-up run. This explanation is also in §5.3.2 ("Why the batch
elliptical scores far below run 2's rnd_elliptical").

## Independent review (2026-07-02)

An independent review agent re-parsed all 4054 raw JSONs (without reusing the analysis code) and
recomputed every published number: **all 8 checks passed** — config grid exactly 91 (28+28+28+7), every
run's final step = 1e6 with 20 train-history points, all four best-config rows reproduced to 4 decimals
(each a true argmax among n>=30 configs), bold/underline/sort correct, figure legend/config match, run-2
reference finals reproduced (gt 67.76 n=97; rnd_state 35.79 n=99), and the 4054/0-failed/min-37/median-45
coverage claims exact. Two notes, no corrections: (1) the top two rows overlap within one SE — a sentence
stating this was added to §5.3.2; (2) `make_reward_table.py` counts rows rather than distinct seeds for n
(currently identical — verified no duplicate seeds; would only matter if a seed were ever re-run within
one sweep).

## What this analysis produces
- `plots/reward_table.tex` — the best-tuned coefficient `\beta^\star` table: one row per method
  (A1 batch elliptical, A2 global-sample, A3 global-add, A4 RND next-state), each at its single best
  configuration (the `(\beta, \lambda, input)` with the highest mean **training-episode** extrinsic reward
  over seeds), with `\bar R`, SE, n. Best bold, second underlined; sorted by `\bar R` descending.
- `plots/line_reward_curve.{pdf,png}` — training-episode extrinsic reward over training for each method at
  its best config (mean ± SE over seeds, each line truncated to steps reached by `>=30` seeds), with the
  run-2 `gt_position_velocity` **eval**-reward curve overlaid as the oracle reference (run 2 logged eval
  reward only; the run-3.1.1 lines are training reward — stated on the plot/caption).

Both are spliced into `development_document/main.tex` under
`\subsubsection{Train run 3.1.1 ...}` (`\label{sec:train-run-3-1-1}`).

## How to (re)build
```
conda run -n exploration python code/make_all.py ../data/2026-06-26-05-45_4algo-betaridge-input plots
```
`code/` holds `common.py` (loader + config-id grouping), `make_reward_table.py`, `make_reward_curve.py`,
`make_all.py`, and `test_analysis.py` (`pytest code/test_analysis.py`). Scoring uses the training-episode
reward `train/mean_extrinsic_reward` (standalone eval is OFF in this run).

## Convention notes
- A "config" key is `method|beta|ridge|input` (RND has `ridge=none|input=none`), kept as a string so RND's
  absent ridge/input round-trips cleanly (a float `None` would become NaN in a pandas groupby and silently
  fail to match).
- `min seeds = 30` gates both the table (a config must have `>=30` seeds to be eligible) and each plotted
  curve segment.
