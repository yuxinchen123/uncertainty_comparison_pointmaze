# code

One folder per train run's table generator, named `<YYYY-MM-DD-HH-MM>_<run-slug>-tables/` with the
timestamp in Pacific time. It is empty until a run has written completed records.

Each folder holds `make_all.py` and its own `README.md`. `make_all.py` imports the run's own status
and metrics modules — it never re-implements loading, scoring, ranking or best/second marking — and
splices only the block between the `% >>> AUTO-GENERATED TABLE START: <name>` and
`% <<< AUTO-GENERATED TABLE END: <name>` markers in `RND_JAX_exploration_development_document.tex`, failing
hard when a marker is missing or appears twice. The caption, the label and the `\resizebox` stay
outside the markers.

The exemplar to copy rather than reinvent:
`/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-31-16-22_pm-am-run1-interim-tables/`.

Run a generator with the project environment by full path:

```bash
/p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
```

## Single-file generators

- `beat_rnd_tables.py` writes every table body, both figures (`beat_rnd_plots/`) and the `\BeatRnd...` numbers of
  `beat_rnd_pointmaze_antmaze.tex` (the section "PointMaze and AntMaze: beat CleanRL PPO-RND"). It reads the run
  records of the auto research folder, the PointMaze RND baseline folder and the AntMaze baseline folder through
  `scripts/maze_campaign/analysis/`, so its numbers equal those of the generated standing page and batch
  summaries. It needs numpy, so it runs on a compute node, together with its tests:

  ```bash
  cd /p/rlprojects/RND/10_jax_exploration_platform
  bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" \
      -s development_document/code/beat_rnd_tables_job.py development_document/code/test_beat_rnd_tables.py
  ```

  A unit of an arm that is not complete stops it with an error naming the unit.
