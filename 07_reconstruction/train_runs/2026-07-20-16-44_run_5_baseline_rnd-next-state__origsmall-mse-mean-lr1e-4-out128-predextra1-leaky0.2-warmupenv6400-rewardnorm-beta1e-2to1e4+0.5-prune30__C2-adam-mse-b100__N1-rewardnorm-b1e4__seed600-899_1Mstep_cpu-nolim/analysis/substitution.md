# Train run 5 — per-run substitution test

Pre-registered rule: an old run is *substantially different* from the run-5 fresh line (and so is NOT pooled) iff **p < 0.01 AND |d| > 0.3** (Welch t on the final training-episode reward). A run that passes BOTH gates (not substantially different) may be pooled after an executed-path code-diff audit.

- **C2 (run 3.2.2)** vs run-5 C2: mean_old=34.50 (n=100), mean_new=36.19 (n=234); Welch p=0.65, Cohen d=0.05, KS=0.07 -> **poolable (pending code-diff)**.
- **C2 (run 3.1.1)** vs run-5 C2: mean_old=6.66 (n=33), mean_new=36.19 (n=234); Welch p=4.97e-10, Cohen d=0.99, KS=0.59 -> **differs (keep separate)**.
- **N1 (run 3.2.3)** vs run-5 N1: mean_old=45.62 (n=100), mean_new=47.56 (n=233); Welch p=0.595, Cohen d=0.06, KS=0.09 -> **poolable (pending code-diff)**.
- **N1 (run 3.2.4)** vs run-5 N1: mean_old=43.71 (n=100), mean_new=47.56 (n=233); Welch p=0.313, Cohen d=0.12, KS=0.11 -> **poolable (pending code-diff)**.
