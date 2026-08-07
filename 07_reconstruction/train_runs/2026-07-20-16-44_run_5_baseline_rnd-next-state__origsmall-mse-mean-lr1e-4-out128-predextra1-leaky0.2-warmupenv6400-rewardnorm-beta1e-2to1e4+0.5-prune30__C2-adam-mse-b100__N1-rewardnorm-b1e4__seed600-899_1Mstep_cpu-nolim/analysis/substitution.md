# Train run 5 — per-run substitution test

Pre-registered rule: an old run is *substantially different* from the run-5 fresh line (and so is NOT pooled) iff **p < 0.01 AND |d| > 0.3** (Welch t on the final training-episode reward). A run that passes BOTH gates (not substantially different) may be pooled after an executed-path code-diff audit.

- **benchmark (run-3.2.2 data)** vs run-5 benchmark: mean_old=34.50 (n=100), mean_new=35.16 (n=300); Welch p=0.856, Cohen d=0.02, KS=0.05 -> **poolable (pending code-diff)**.
- **benchmark (run-3.1.1 data)** vs run-5 benchmark: mean_old=6.66 (n=33), mean_new=35.16 (n=300); Welch p=1.12e-09, Cohen d=0.93, KS=0.57 -> **differs (keep separate)**.
- **reward-norm (run-3.2.3 data)** vs run-5 reward-norm: mean_old=45.62 (n=100), mean_new=46.84 (n=300); Welch p=0.731, Cohen d=0.04, KS=0.06 -> **poolable (pending code-diff)**.
- **reward-norm (run-3.2.4 data)** vs run-5 reward-norm: mean_old=43.71 (n=100), mean_new=46.84 (n=300); Welch p=0.399, Cohen d=0.10, KS=0.10 -> **poolable (pending code-diff)**.
