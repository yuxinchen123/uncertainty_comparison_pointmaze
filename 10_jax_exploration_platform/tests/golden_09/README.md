# golden_09

The gate that keeps the platform's code equal to the frozen `09_parallelization` baseline.

`test_golden_parity.py` runs two trainers in one process — the split package at
`src/exploration_platform/` and the frozen single module at
`09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py` — builds the same small configuration in both,
starts them from identical state with identical random keys, runs three iterations in both update
styles, and requires every reported number and every array of the state to be **bit-identical**,
not merely close.

The two sides keep their state in different shapes, so the comparison is by NAME rather than by
position: each side is turned into a dictionary of arrays under one agreed set of names
(`params/actor/W0`, `obs_rms/mean`, ...), the two key sets must match exactly, and every array must
match byte for byte. Nothing is skipped. The platform's two arrays with no counterpart in the
baseline — the run key and the iteration counter, which the baseline kept on the host — are checked
separately for being exactly what the baseline's driver would have passed in.

Run it on the processor:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/golden_09/test_golden_parity.py
```

and on the graphics card, through the serval05 lock (one graphics-card command at a time):

```
bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
  "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python \
   /p/rlprojects/RND/10_jax_exploration_platform/tests/golden_09/test_golden_parity.py"
```

This test has to stay green through the whole refactor of the trainer. Only the platform side of
this file is ever changed — the `09_parallelization` side is the reference and is never edited, and
the requirement is never relaxed from byte equality to a tolerance.
