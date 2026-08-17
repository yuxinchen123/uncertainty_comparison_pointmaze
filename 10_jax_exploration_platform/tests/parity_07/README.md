# parity_07

Where a family of this platform is a re-implementation of one in `07_reconstruction/`, it is
checked against that code — the code, running in its own interpreter, not a transcription of it.

| file | what it checks |
|---|---|
| `test_visit_count_against_07.py` | the visit-count table and the bonus of every step, against `PositionVelocityVisitCountWrapper` and `VisitCount`, on one fixed trajectory |
| `dump_07_visit_counts.py` | the other half of that check: it runs under `07_reconstruction`'s interpreter, drives its wrapper over the supplied trajectory with a stub environment, and writes what it counted |

The two projects use different interpreters — `07_reconstruction` runs on
`/p/rlprojects/RND/.venvs/exploration` (torch, gymnasium) and the platform on
`/p/rlprojects/RND/.venvs/platform_jax` (jax, no gymnasium) — so the check runs the other side as
a subprocess and compares what it wrote. Nothing under `07_reconstruction/` is modified; it is
imported and called.

Run it with the platform's interpreter; it invokes the other one itself:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/parity_07/test_visit_count_against_07.py
```

## Result, 2026-08-16

265-step trajectory sweeping the whole world and past its edges, resting 40 steps on the start
cell, and driving both velocity components to and beyond the +-5 m/s the environment clips to.

| preset | table entries differing | visits counted | largest count | worst bonus difference over 265 steps |
|---|---|---|---|---|
| `gt_position_velocity_sqrt` | 0 of 10,800 | 178 of 265 steps (the rest landed on wall cells) | 42 | 1.49e-08 |
| `gt_position_velocity_linear` | 0 of 10,800 | 178 of 265 steps | 42 | 9.93e-09 |

The count tables are identical entry for entry. The bonus differs in the eighth decimal because
`n**decay` is evaluated in float32 here and in float64 there.
