# 2026-08-17-20-22_pm-jax-run11-extension-2-tables

The table generator of train run 1.1 **after its second extension**. It replaces
`2026-08-16-22-27_pm-jax-run11-extension-tables/`, which replaced
`2026-08-16-21-00_pm-jax-run11-tables/`; both stay in place unchanged as the record of what the
document's table said before each extension ran, and neither is called any more.

`make_all.py` writes one table into `platform_development_document.tex`:

| marker name | what it is |
|---|---|
| `pm-jax-run11-results` | three blocks separated by two double rules: the four arms at their best configuration in the ORIGINAL 10M-step sweep; the same four configurations as the FIRST extension re-ran them at 1000M steps with 1,024 copies; and the two arms whose WHOLE grid the SECOND extension re-ran at 1000M steps, each at the configuration that grid chooses, over 128 copies |

Everything between `% >>> AUTO-GENERATED TABLE START: pm-jax-run11-results` and
`% <<< AUTO-GENERATED TABLE END: pm-jax-run11-results` is replaced wholesale. The caption, the
label and the `\resizebox` stay outside the markers and are hand-edited; the caption is where both
double rules are documented, per the house rule that every deliberate double rule says in its
caption what it separates. A missing or duplicated marker is a hard failure, not a silent no-op.

## Where the three blocks' numbers come from

Each block's numbers come from the run that produced them, through that run's OWN
`code/aggregate.py`, imported:

| block | run folder |
|---|---|
| above the first double rule | `runs/2026-08-16-20-39_jax-ppo_..._copies-per-cell-256_first-batch` |
| between the two double rules | `runs/2026-08-16-22-47_jax-ppo_..._copies-1024_extension-of-first-batch` |
| below the second double rule | `runs/2026-08-17-10-45_jax-ppo_..._copies-per-cell-128_extension-2-of-first-batch` |

All three modules are files named `aggregate.py`, so a plain `import aggregate` would return
whichever was imported first and silently score one run with another's module. `load_run_module`
loads each by path under its own module name instead. The generator itself loads no shard, scores
no copy and ranks nothing, so the writeup and any run's 20-minute status table cannot disagree.

Bold and underline are computed **within each block**. The blocks are different step budgets over
different copy counts, so marking one against another would compare measurements that are not
comparable.

## What the third block is, and why it has two rows and not four

The first extension took each arm's best configuration **as chosen at 10M steps** and re-ran it at
1000M. The second extension re-ran the whole learning-rate by intrinsic-weight grid of the two arms
that were still improving, so its rows are those two arms at the configuration a 1000M-step sweep
chooses for them. The other two arms were not re-run, so they have no row in the third block.

## The figure

The training-curve figure is not this folder's: it is drawn by
`runs/<the second extension>/analysis/code/make_reward_curves.py`, which imports the same
`best_per_arm` this generator uses so that the curves shown are the configurations the table names.
The figure keeps the file name and the document label it has had since the original run. Its
dashed curves are the configurations chosen at 10M steps, as the first extension ran them; its
solid curves are the configurations the second extension's own 1000M-step sweep chooses. Both sets
run to the same 1000M steps per copy, so line style is the only thing separating "chosen by the
short budget" from "chosen by the long one".

## Running it

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python make_all.py
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -cd -pdf -interaction=nonstopmode \
  /p/rlprojects/RND/10_jax_exploration_platform/development_document/platform_development_document.tex
grep -nE 'Overfull \\hbox' ../../platform_development_document.log
```

Then rasterize the table's page and look at it, per the development document's README.
