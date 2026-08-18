# Queued question, to be answered when this sweep drains and the document is updated

**Aggregated over all of a configuration's copies, does every configuration in this sweep have a
curve that first goes up and then comes down?**

The answer decides what else gets written:

- **If yes** — give the step-by-step algorithm, as a numbered list, by which that was determined.
- **If no** — give the best configuration that does NOT have that shape, one line per algorithm
  arm (`rnd_next_state` and `gt_position_velocity_sqrt`).
- Put those lines in a **separate table and plot only if they are not already the configurations
  the results table and the curve figure carry.** A configuration that is both the arm's best and
  not rise-then-fall is already shown; repeating it would add nothing.

## How it is answered

`analysis/code/shape_report.py` runs the whole thing and writes `analysis/shape_classification.md`
plus the per-configuration JSON. It reads every configuration through `code/aggregate.curve_of`,
the same pooling the results table and the figure use, so shapes and scores cannot disagree.
`analysis/code/curve_shape.py` holds the classifier and its thresholds.

The classifier was written and checked against the **completed 10M-step batch** before this run's
data existed (`--run <the 2026-08-16-20-39 folder>`), which is where its two defects were found and
fixed: a rank-correlation gate that mislabelled real rise-then-fall curves as noise, and a missing
signal floor that let floating-point dust in an all-zero curve read as a decline.

## What the 10M-step batch answered

Not every configuration rises then falls: of its 102 configurations, 17% rise then fall, 22% are
still rising at the end of the budget, 5% rise then flatten, and 57% never reach the goal at all.
The eight highest-scoring configurations are all **still rising**, with the same label under all
nine threshold variants — which is why this 1000M-step sweep exists, and it means the same
question at a hundred times the budget is genuinely open.
