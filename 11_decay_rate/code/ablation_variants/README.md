# Ablation variant snapshots

Each file is the exact `method.py` snapshot of one ablation experiment (abl_2xx), committed so
the one-line parameter changes are reproducible from git. Base methods and the changed line:

- `coinflip_d128_uniform.py` (abl_201) — exp-019 method with `D_COINS = 128` (was 512).
- `coinflip_tau02_uniform.py` (abl_202) — exp-019 method with `TAU_ADD = 0.2` (was 0.05); the
  nonuniform run abl_203 used the same snapshot.
- `elliptical_sigma02_uniform.py` (abl_204) — exp-032 method with `SIGMA_MAX = 0.2` (was 0.35).
- Horizon runs hor_301/302/303 reuse the unmodified committed methods of exps 010, 019, 009
  at `--n_steps 32768`.
- `deepshrink_wide_atari.py` / `deepcfn_wide_atari.py` (exps 059/060, val_204/205) — the
  exp-048/049 methods with image-trunk feature width raised 256 -> 1024, after the Atari runs
  showed a representational floor at ~0.25-0.29 with 512 distinct frames > 256 features.
