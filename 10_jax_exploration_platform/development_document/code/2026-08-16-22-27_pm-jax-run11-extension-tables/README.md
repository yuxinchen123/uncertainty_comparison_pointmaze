# 2026-08-16-22-27_pm-jax-run11-extension-tables

The table generator of train run 1.1 **after its extension**. It replaces
`2026-08-16-21-00_pm-jax-run11-tables/`, which stays in place unchanged as the record of what the
document's table said before the extension ran, and is no longer called.

`make_all.py` writes one table into `platform_development_document.tex`:

| marker name | what it is |
|---|---|
| `pm-jax-run11-results` | the four arms at their best configuration in the ORIGINAL sweep, a double rule, then the same four configurations as the EXTENSION re-ran them at 1,024 copies for 1,000,038,400 environment steps per copy |

Everything between `% >>> AUTO-GENERATED TABLE START: pm-jax-run11-results` and
`% <<< AUTO-GENERATED TABLE END: pm-jax-run11-results` is replaced wholesale. The caption, the
label and the `\resizebox` stay outside the markers and are hand-edited; the caption is where the
double rule is documented, per the house rule that every deliberate double rule says in its caption
what it separates. A missing or duplicated marker is a hard failure, not a silent no-op.

## Where the two blocks' numbers come from

Each block's numbers come from the run that produced them, through that run's OWN
`code/aggregate.py`, imported:

| block | run folder |
|---|---|
| above the double rule | `runs/2026-08-16-20-39_jax-ppo_..._copies-per-cell-256_first-batch` |
| below the double rule | `runs/2026-08-16-22-47_jax-ppo_..._copies-1024_extension-of-first-batch` |

Both modules are files named `aggregate.py`, so a plain `import aggregate` would return whichever
was imported first and silently score one run with the other's module. `load_run_module` loads each
by path under its own module name instead. The generator itself loads no shard, scores no copy and
ranks nothing, so the writeup and either run's 20-minute status table cannot disagree.

Bold and underline are computed **within each block**. The two blocks are different step budgets
over different copy counts, so marking one against the other would compare measurements that are
not comparable.

The training-curve figure is not this folder's: it is drawn by
`runs/<the extension run>/analysis/code/make_reward_curves.py`, which imports the same two
functions so that the curves shown are the configurations the table names. The figure keeps the
file name and the document label it had before the extension, and its curves are the extension's —
the shorter budget's curves are removed rather than overplotted, because two budgets of one arm in
one panel read as two algorithms. The parent run's own plot file is untouched in the parent run
folder.

## Running it

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python make_all.py
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -cd -pdf -interaction=nonstopmode \
  /p/rlprojects/RND/10_jax_exploration_platform/development_document/platform_development_document.tex
grep -nE 'Overfull \\hbox' ../../platform_development_document.log
```

Then rasterize the table's page and look at it, per the development document's README.
