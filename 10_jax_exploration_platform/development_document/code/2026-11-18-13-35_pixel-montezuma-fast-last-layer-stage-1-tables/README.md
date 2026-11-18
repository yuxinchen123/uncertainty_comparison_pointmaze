# Pixel Montezuma, fast last layer against RND: stage-1 tables

`make_all.py` rebuilds the generated blocks of
`/p/rlprojects/RND/10_jax_exploration_platform/development_document/pixel_montezuma_fast_last_layer.tex` (section
"Pixel Montezuma: the fast last layer against RND" of the JAX platform development document) from the records of the
stage-1 run folder
`/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000/`.

## Generated blocks

| marker name | what it holds |
|---|---|
| `pixel-montezuma-fll-numbers` | the `\PixelFll...` macros the hand-written prose uses (reading time, run counts, exits, checks) |
| `tab:pixel-montezuma-fll-status` | runs planned, started, finished and out of the start room, per arm |
| `tab:pixel-montezuma-fll-early-rnd` | first key, first door, first exit of RND per intrinsic weight, and the archived baseline |
| `tab:pixel-montezuma-fll-early-group-128`, `-16`, `-4` | the same for the fast last layer, one table per sharing group |
| `tab:pixel-montezuma-fll-checks` | the live report's standing checks, as counts |

Prose, captions, labels and the two settings tables (the game, what is swept) are hand-written and stay outside the
markers.

## Refreshing

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-11-18-13-35_pixel-montezuma-fast-last-layer-stage-1-tables
python3 make_all.py
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -cd -pdf -interaction=nonstopmode ../../RND_JAX_exploration_development_document.tex
```

- Standard library only, and it reads the run folder without writing to it, so it may run on a login node while the
  runs train. Its one side effect there is a moment under the queue's lock while it reads the queue records, as the
  live report does at every monitoring pass.
- It reads the records through the live report's own functions (`scripts/pixel_sweep_campaign/report.py`: `collect`,
  `configurations`, `first_step`, `non_finite`), so the document and `early_signals.md` read the same numbers. If
  `report.py` renames one of them, this script stops with an error rather than writing different numbers.
- A run counts as started here once it has trained at least one update; the live report also counts a unit that a
  worker has claimed and is still compiling.
- After a refresh, reread the prose: the counts follow the macros, but a sentence about what the tables show is
  hand-written. The rules of `/p/rlprojects/.claude/skills/rnd-experiment-tex-track/SKILL.md` ("After writing or
  regenerating") apply: rebuild the PDF, grep the log for `Overfull \hbox`, look at the rendered pages.

## Tests

```bash
python3 -m unittest test_make_all.py
```
