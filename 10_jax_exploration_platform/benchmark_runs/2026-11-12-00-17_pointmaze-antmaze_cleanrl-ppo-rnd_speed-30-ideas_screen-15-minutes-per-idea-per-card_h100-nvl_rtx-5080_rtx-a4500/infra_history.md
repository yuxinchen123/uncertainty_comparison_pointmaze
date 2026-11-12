# Infrastructure history

Times are Pacific.

- 2026-11-12 00:17 PT: folder created. serval05's card is in use by another account (pyt9dp), so profiles go to serval02 card 1 until it is free.
- 2026-11-12 00:22 PT: test jobs 6873325, 6873326 and 6873328 failed within a second: `platform_jax` has no pytest.
  The test job now uses `atari_pixels_jax` (a copy of `platform_jax` with the same JAX, NumPy and MuJoCo versions,
  plus pytest). The job script also names the `cpu` partition now.
- 2026-11-12 00:23 PT: environment build job 6873341 failed within a second: the conda path of the older build
  script does not exist on puma01. The script now uses
  `/sw/ubuntu2204/current/software/common/core/miniforge/26.7.2/bin/conda`.
