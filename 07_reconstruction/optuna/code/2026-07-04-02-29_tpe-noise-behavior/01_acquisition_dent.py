"""Point 1: quantify how many bad draws at a good location it takes to push TPE's proposal off the peak.

Build a study whose good group (the top-decile "below" trials) is 20 observations clustered near x=8
with values 45-55, and add a growing number of bad observations (value ~0) at exactly x=8. For each
count, drive the installed sampler internals to get the acquisition log l(x) - log g(x) on a fine grid,
and report where the argmax sits and how far the acquisition at x=8 is below its maximum. This isolates
the smoothing (bandwidths + magic-clip floor) and the argmax-over-candidates return: one bad point dents
the ratio at 8 only mildly; several bad points move the proposal to the shoulder of the cluster.

To make the 20 good points BE the good group, TPE's split size gamma = ceil(0.1 n) must be 20, so the
study also holds 180 mediocre background observations (values ~10, spread across 0..10). These are the
many ordinary trials a real search accumulates; they form the "rest" group. The 20 good (values 45-55)
are the top 20 by value, so the split puts exactly them in the good group and everything else, including
the bad points at x=8, in the rest group.
"""

import numpy as np

import optuna
from optuna.samplers import TPESampler

from common import SPACE, add_observation, acquisition_on_grid, substream


def build_study(n_bad, seed):
    """Study whose good group is 20 observations near x=8 (values 45-55), plus 180 background trials,
    plus n_bad bad observations at exactly x=8 (value ~0)."""
    # fresh maximize study; the sampler seed only affects candidate draws, not the grid acquisition we compute
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    # 180 background observations first: x spread across the whole range, values ~10 (well below the good 45-55)
    bx = substream(seed, "bg_x")
    bv = substream(seed, "bg_val")
    for _ in range(180):
        add_observation(study, float(bx.uniform(0.0, 10.0)), float(bv.normal(10.0, 4.0)))
    # 20 good observations: x clustered around 8 (SD 0.6, clipped to [0,10]), values uniform in [45,55]
    xr = substream(seed, "good_x")
    vr = substream(seed, "good_val")
    for _ in range(20):
        x = float(np.clip(xr.normal(8.0, 0.6), 0.0, 10.0))
        add_observation(study, x, float(vr.uniform(45.0, 55.0)))
    # n_bad bad observations at exactly x=8.0 with value ~0 (tiny positive noise so they are distinct but near 0)
    br = substream(seed, "bad_val")
    for _ in range(n_bad):
        add_observation(study, 8.0, float(abs(br.normal(0.0, 0.3))))
    return study


def main():
    """Sweep n_bad in {0,1,3,6} and report the argmax of the acquisition and the notch at x=8."""
    print("optuna version:", optuna.__version__)
    seed = 0
    grid = np.linspace(0.0, 10.0, 1001)   # step 0.01, so x=8.0 is exactly on the grid
    i8 = int(np.argmin(np.abs(grid - 8.0)))
    sampler = TPESampler(seed=seed)

    print("\nGood group = 20 observations near x=8 (values 45-55); 180 background trials (values ~10);")
    print("bad observations added at exactly x=8 (value ~0). gamma=ceil(0.1*n)=20 so the 20 good ARE the good group.")
    print("acquisition = log l(x) - log g(x) on a 0..10 grid (step 0.01), via the installed sampler internals.\n")
    header = f"{'n_bad':>5} | {'below/good':>10} | {'above/rest':>10} | {'argmax x':>9} | {'acq@peak(argmax)':>16} | {'acq@x=8':>9} | {'acq@8 - max':>12}"
    print(header)
    print("-" * len(header))
    for n_bad in (0, 1, 3, 6):
        study = build_study(n_bad, seed)
        acq, n_below, n_above = acquisition_on_grid(study, sampler, grid)
        j = int(np.argmax(acq))
        argmax_x = float(grid[j])
        acq_max = float(acq[j])
        acq_at8 = float(acq[i8])
        print(f"{n_bad:>5} | {n_below:>10} | {n_above:>10} | {argmax_x:>9.3f} | "
              f"{acq_max:>16.4f} | {acq_at8:>9.4f} | {acq_at8 - acq_max:>12.4f}")

    # Show the full acquisition shape near the peak for the 0-bad and 6-bad cases so the notch is visible.
    print("\nAcquisition near the peak (x from 6.5 to 9.5, step 0.25): value relative to that case's own max.")
    near = np.arange(6.5, 9.51, 0.25)
    for n_bad in (0, 1, 3, 6):
        study = build_study(n_bad, seed)
        acq, _, _ = acquisition_on_grid(study, sampler, near)
        rel = acq - acq.max()
        cells = "  ".join(f"{v:6.2f}" for v in rel)
        print(f"  n_bad={n_bad}: " + cells)
    print("  x grid   : " + "  ".join(f"{x:6.2f}" for x in near))


if __name__ == "__main__":
    main()
