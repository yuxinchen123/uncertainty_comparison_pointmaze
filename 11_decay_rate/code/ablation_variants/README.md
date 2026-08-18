# Ablation variant snapshots

Each file is the exact `method.py` snapshot of one ablation experiment (abl_2xx), committed so
the one-line parameter changes are reproducible from git. Base methods and the changed line:

- `coinflip_d128_uniform.py` (abl_201) — exp-019 method with `D_COINS = 128` (was 512).
- `coinflip_tau02_uniform.py` (abl_202) — exp-019 method with `TAU_ADD = 0.2` (was 0.05); the
  nonuniform run abl_203 used the same snapshot.
- `elliptical_sigma02_uniform.py` (abl_204) — exp-032 method with `SIGMA_MAX = 0.2` (was 0.35).
- Horizon runs hor_301/302/303 reuse the unmodified committed methods of exps 010, 019, 009
  at `--n_steps 32768`.
