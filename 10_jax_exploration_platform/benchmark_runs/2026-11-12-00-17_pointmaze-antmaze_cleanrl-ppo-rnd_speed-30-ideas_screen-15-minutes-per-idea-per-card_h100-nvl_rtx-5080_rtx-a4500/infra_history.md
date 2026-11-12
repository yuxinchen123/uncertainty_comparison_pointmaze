# Infrastructure history

Times are Pacific.

- 2026-11-12 00:17 PT: folder created. serval05's card is in use by another account (pyt9dp), so profiles go to serval02 card 1 until it is free.
- 2026-11-12 00:22 PT: test jobs 6873325, 6873326 and 6873328 failed within a second: `platform_jax` has no pytest.
  The test job keeps the `platform_jax` interpreter and adds a folder of links to pytest's own pure-Python packages
  from `atari_pixels_jax` to its path, so JAX, NumPy and MuJoCo still come from `platform_jax`. The job script also
  names the `cpu` partition now.
- 2026-11-12 00:23 PT: environment build job 6873341 failed within a second: the conda path of the older build
  script does not exist on puma01. The script now uses
  `/sw/ubuntu2204/current/software/common/core/miniforge/26.7.2/bin/conda`.
- 2026-11-12 00:24 PT: environment build job 6873344 failed within a second: that conda starts with
  `#!/usr/bin/env python` and puma01 has no `python` on its path. The script now puts miniforge's own `bin` first
  on the path.
