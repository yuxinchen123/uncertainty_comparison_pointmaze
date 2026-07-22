# Run 3.2.1 (optimizer sweep) and run 3.2.2 (fresh-seed check) — analysis

Two runs, read together. Run 3.2.1 swept optimizer and bonus-readout variants of RND next-state and
picked the best configuration in each family. Run 3.2.2 then re-ran the two best configurations plus the
standard RND baseline on a fresh, disjoint set of seeds to check whether the advantage was real.

All numbers are computed from the per-run JSON files by `code/make_results.py`, which asserts them against
the values below before writing the tables and figure. Regenerate with
`conda run -n exploration python code/make_results.py`.

## What is measured

Each run's score is $R_i$, its final training-episode extrinsic reward (the mean extrinsic reward over the
past 100 completed training episodes at $10^6$ steps; standalone evaluation is off, as in run 3.1.1). A
configuration pools its finished seeds:

- mean reward $\bar{R} = \dfrac{1}{n}\sum_{i=1}^{n} R_i$,
- standard error $\mathrm{SE} = \dfrac{s}{\sqrt{n}}$ with $s$ the sample standard deviation,
- success rate $= \dfrac{1}{n}\,\#\{\, i : R_i > 5 \,\}$ (the per-seed rewards split into a group near
  zero and a group well above 20; the value 5 sits in the gap).

Only completed runs are counted. A point on a curve is drawn only where at least 10 seeds reached that
step.

## Run 3.2.1 — best configuration per optimizer and readout (`plots/reward_table.tex`)

Six rows, sorted by $\bar{R}$: the five families swept in run 3.2.1 (SGD-$1/t$ and AdaGrad, each with the
mse and the l2 bonus readout, plus Adam with the l2 readout), each at its own best configuration, together
with the canonical Adam / mse RND baseline recomputed from run 3.1.1 (seed range 0-99).

- The two SGD-$1/t$ configurations top the table: $\bar{R} = 47.64$ (mse, $\beta=10$) and $45.33$
  (l2, $\beta=1$), both well above the Adam / mse baseline at $35.15$.
- AdaGrad / mse ($38.35$) also edges past the baseline; AdaGrad / l2 ($34.11$) matches it; Adam / l2
  ($26.15$) is last.

Read on its own, this table says the SGD-$1/t$ optimizer beats the standard RND baseline by about 10-12
reward. That reading is misleading, for the reason below.

Each SGD-$1/t$ value is the largest mean over a grid of 5 step sizes $\eta_0$ $\times$ 2 offsets $t_0$
$\times$ 8 coefficients $\beta$ $=$ 80 configurations per readout (160 SGD-$1/t$ configurations in total),
each with roughly 45-50 seeds. Taking the maximum over many noisy configurations makes the selected mean
biased upward: the configuration that happens to land highest is partly high because of favorable seed
noise, not only because of its settings (the standard selection-bias effect, also called the winner's
curse). The Adam / mse baseline row is not subject to this — it is a single fixed configuration — so the
table compares an optimistically selected number against an unselected one.

## Run 3.2.2 — the same configurations on fresh seeds (`plots/validation_table.tex`)

Run 3.2.2 re-ran three configurations on 100 fresh seeds (100-199), disjoint from the seeds used in run
3.2.1 (0-49) and run 3.1.1 (0-99), so the selection bias is removed:

- C1 — SGD-$1/t$ / mse / $\beta=10$ (the best 3.2.1 configuration),
- C0 — SGD-$1/t$ / l2 / $\beta=1$ (the specific l2 configuration re-run: $\eta_0=0.1$, $t_0=10^3$),
- C2 — Adam / mse / $\beta=10^2$ (the RND baseline).

The table reports, for each candidate, the gap to the baseline $\Delta = \bar{R}_{\text{cand}} -
\bar{R}_{\text{bench}}$ and a one-sided confidence that the candidate beats the baseline,
$\Phi\!\left(\dfrac{\Delta}{\sqrt{\mathrm{SE}_{\text{cand}}^2 + \mathrm{SE}_{\text{bench}}^2}}\right)$,
where $\Phi$ is the standard-normal cumulative distribution function.

On fresh seeds the SGD-$1/t$ advantage collapses:

1. **C1 barely leads the baseline.** $\bar{R} = 36.92$ against the baseline's $34.19$, a gap of only
   $+2.74$, which is $73\%$ one-sided confidence — not a significant difference.
2. **C0 is below the baseline.** $\bar{R} = 33.50$, a gap of $-0.69$ ($44\%$ confidence), i.e. slightly
   worse than the baseline.
3. **The two best SGD-$1/t$ configurations are, on fresh seeds, statistically on par with the standard
   Adam / mse RND baseline — not better.**

## The selection-bias finding

Comparing each configuration's 3.2.1 value with its fresh-seed 3.2.2 value shows the pattern directly:

1. **The baseline reproduced.** Adam / mse held at $34.19$ on fresh seeds, against $35.15$ in run 3.1.1 —
   within one standard error. The baseline was not selected from a grid, so it does not move.
2. **The selected SGD-$1/t$ configurations dropped sharply.** C1 fell from $47.64$ to $36.92$
   ($47.6 \to 36.9$). C0 fell from $42.35$ to $33.50$ ($42.4 \to 33.5$); the best SGD-$1/t$ / l2
   configuration in 3.2.1 was higher still at $45.33$, so the l2 family drop reads $45.3/42.4 \to 33.5$.
3. **A large drop only for the configurations that were selected, with the unselected baseline holding
   steady, is the signature of selection bias.** The 3.2.1 SGD-$1/t$ lead was mostly the upward bias of
   picking the best of many noisy cells, not a real optimizer advantage.

## Figure (`plots/line_reward_curve.pdf`, `.png`)

Training-episode extrinsic reward against environment step, pooled over seeds (mean with a $\pm 1$
standard-error band), each line drawn up to the largest step reached by at least 10 seeds:

- five solid lines — the five swept best configurations from run 3.2.1 (the two SGD-$1/t$, the two
  AdaGrad, and Adam / l2);
- one black dashed line — the Adam / mse RND baseline from run 3.1.1 (training reward, same metric as the
  solid lines);
- one gray dashed line — the run-2 `gt_position_velocity` visit-count oracle, included for reference.

Caveat: the oracle line is **evaluation** reward (run 2 logged evaluation only), while every other line is
**training-episode** reward. These are related but not identical metrics — the same mismatch noted in the
run-3.1.1 figure. The oracle (about 68 at $10^6$ steps) stays far above every learned bonus.

## Artifacts

- `plots/reward_table.tex` — run-3.2.1 six-row table (bare `tabular`; goes into `main.tex`).
- `plots/validation_table.tex` — run-3.2.2 three-row table (bare `tabular`).
- `plots/line_reward_curve.pdf`, `plots/line_reward_curve.png` — the reward-curve figure.
- `code/common.py`, `code/make_results.py` — the loader and the generator (compute + assert, no
  hardcoded numbers).
