# pointmaze

Two files, copied from `09_parallelization` (see `../../../../ORIGIN.md`):

- `pm_common.py` — the physics constants, the maze maps, the environment configuration and the
  precomputed wall geometry. Numpy only. The constants were probe-verified against
  Gymnasium-Robotics 1.3.1.
- `jax_pointmaze.py` — the batched JAX stepper: state is arrays of shape `[copies, envs, ...]`,
  the step is a pure function, and the reset draw is frozen so it is bit-identical to the PyTorch
  and CUDA steppers 09 compared it against.

The configuration these files default to is the specification the platform trains on:
`pointmaze_large_cont400_nonoise@1`, written out in `../../../../ORIGIN.md`.
