# golden_09

The gate that keeps the platform's code equal to the frozen `09_parallelization` baseline.

`test_golden_parity.py` loads two modules into one process — the copy at
`src/exploration_platform/agents/ppo/jax_ppo_rnd.py` and the original at
`09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py` — builds the same small configuration in both, starts
them from identical state with identical random keys, runs three iterations in both update styles,
and requires every reported number and every final parameter to be **bit-identical**, not merely
close.

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

This test has to stay green through the whole refactor of the trainer. When the split into agent,
bonus and composition changes how the platform's side is built, change only the platform side of this
file — the `09_parallelization` side is the reference and is never edited.
