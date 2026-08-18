# Ablation variant snapshots

Each file is the exact `method.py` snapshot of one ablation experiment (abl_2xx), committed so
the one-line parameter changes are reproducible from git. Base methods and the changed line:

- `abl_201_coinflip_d128_uniform.py` — exp-019 method with `D_COINS = 128` (was 512).
- `abl_202_coinflip_tau02_uniform.py` — exp-019 method with `TAU_ADD = 0.2` (was 0.05); the
  nonuniform run abl_203 used the same snapshot.
- `abl_204_elliptical_sigma02_uniform.py` — exp-032 method with `SIGMA_MAX = 0.2` (was 0.35).
- Horizon runs hor_301/302/303 reuse the unmodified committed methods of exps 010, 019, 009
  at `--n_steps 32768`.
