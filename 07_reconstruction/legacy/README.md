# legacy/ — frozen pre-reorganization reference (do not run as-is)

These files are kept only as a record of how `07_reconstruction` evolved before the reorganization into
the `rnd_exploration` package. **Their imports were intentionally not rewritten** — they reference the
pre-restructure top-level layout (`env_wrapper/`, `intrinsic/`, `utilities/`, `distance_to_GT/`) and, for
`02`, modules that were deleted in the reorganization (`intrinsic/intrinsic_replay_buffer.py`,
`intrinsic/RLeXplore_utilities/`) plus the external `rllte` package. They will not import against the
current tree.

To run any of them, check out the last commit before the reorganization:

```
git checkout 3b212d7  # pre-reorganization HEAD (branch Use-RLexplore-RND)
```

Contents:
- `01_gt.py` + `01_wandb_sweep.yaml` — precursor: visit-count (oracle) bonus only.
- `02_rnd_rlexplore.py` + `02_wandb_sweep.yaml` — precursor: RND bonus via the external RLeXplore (`rllte`).
- `03_rnd_my_implementation.py` + `03_wandb_sweep.yaml` — precursor: self-contained RND bonus.
- `debug/` — diagnostic scripts + `report.md` write-ups (env wrapper-stack diff; RND-semantics change;
  visit-count flat-bonus bug). Useful institutional notes; the scripts target the old layout.

The live driver is now `../train.py` (replaces `04_many_exploration_method.py`); the algorithm dispatch lives
in `../src/rnd_exploration/methods/__init__.py` (the registry).
