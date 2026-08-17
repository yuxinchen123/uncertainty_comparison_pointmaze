# code

One folder per train run's table generator, named `<YYYY-MM-DD-HH-MM>_<run-slug>-tables/` with the
timestamp in Pacific time. It is empty until a run has written completed records.

Each folder holds `make_all.py` and its own `README.md`. `make_all.py` imports the run's own status
and metrics modules — it never re-implements loading, scoring, ranking or best/second marking — and
splices only the block between the `% >>> AUTO-GENERATED TABLE START: <name>` and
`% <<< AUTO-GENERATED TABLE END: <name>` markers in `platform_development_document.tex`, failing
hard when a marker is missing or appears twice. The caption, the label and the `\resizebox` stay
outside the markers.

The exemplar to copy rather than reinvent:
`/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-31-16-22_pm-am-run1-interim-tables/`.

Run a generator with the project environment by full path:

```bash
/p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
```
