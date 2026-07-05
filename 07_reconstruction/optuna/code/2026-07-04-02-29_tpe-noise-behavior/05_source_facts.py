"""Point 5: confirm from the installed source and empirically how TPE handles repeated / same-x trials.

(a) Nothing in TPESampler._sample or the split path averages duplicate or nearby parameter values or
    models noise: each trial enters the good/rest split individually, by its own value. We print the
    source of the split functions and show that two trials at the SAME x but different values land in
    different groups.
(b) With several trials at the SAME x, both the good density l and the rest density g get kernels at
    that x, and the acquisition there reflects how many of that x's draws reached the top quantile.
    We build 12 draws at x=8 of which 4 are in the good group, and 12 draws at x=2 all in the good
    group, then compare the acquisition at x=8 (4/12 good) with x=2 (12/12 good).
"""

import inspect

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import (
    _split_trials,
    _split_complete_trials,
    _split_complete_trials_single_objective,
)
from optuna.trial import TrialState

from common import SPACE, add_observation, substream

optuna.logging.set_verbosity(optuna.logging.WARNING)


def print_source():
    """Print the installed source of the split functions that back fact (a)."""
    # the three functions on the split path; note there is no averaging or param-deduplication anywhere
    print("optuna version:", optuna.__version__)
    print("\n--- TPESampler._sample (fetch finished trials, split, fit, sample, argmax) ---")
    print(inspect.getsource(TPESampler._sample))
    print("--- _split_complete_trials (dispatch to single/multi objective) ---")
    print(inspect.getsource(_split_complete_trials))
    print("--- _split_complete_trials_single_objective (SORT BY VALUE, take top n_below) ---")
    print(inspect.getsource(_split_complete_trials_single_objective))
    print("Note: the good/rest split sorts trials by trial.value and slices sorted[:n_below]; no step")
    print("averages duplicate or nearby params or models per-x noise. Each trial is placed by its own value.")


def same_x_split_check():
    """Show two trials at the SAME x but different values fall in different split groups (fact a)."""
    # study with 4 trials, two of them at exactly x=5 (one high value, one low value)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
    add_observation(study, 5.0, 90.0)   # x=5, high value -> should be in the good group
    add_observation(study, 5.0, 10.0)   # x=5, low value  -> should be in the rest group
    add_observation(study, 1.0, 80.0)
    add_observation(study, 9.0, 20.0)
    # split with n_below=2 so exactly the two highest values are "good"
    trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
    below, above = _split_trials(study, trials, 2, False)
    below_vals = {t.params["x"]: t.value for t in below}
    above_vals = {t.params["x"]: t.value for t in above}
    print("\n--- Same-x trials split individually by value (n_below=2) ---")
    print("two trials at x=5.0: value 90 and value 10.")
    print("good group (below) trials (x, value):", [(t.params["x"], t.value) for t in below])
    print("rest group (above) trials (x, value):", [(t.params["x"], t.value) for t in above])
    x5_below = sum(1 for t in below if t.params["x"] == 5.0)
    x5_above = sum(1 for t in above if t.params["x"] == 5.0)
    print(f"x=5.0 draws in good group: {x5_below}, in rest group: {x5_above} "
          f"-> the same x is split across both groups by each trial's value, not merged.")


def repeated_x_acquisition():
    """Compare the acquisition at x=8 (4/12 draws good) with x=2 (12/12 draws good) -- fact (b)."""
    # build 12 draws at x=2 (all high) and 12 draws at x=8 (4 high, 8 low)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
    v2 = substream(0, "x2_val")
    for _ in range(12):
        add_observation(study, 2.0, float(v2.uniform(90.0, 100.0)))   # x=2: all 12 high -> all in good group
    vh = substream(0, "x8_high")
    for _ in range(4):
        add_observation(study, 8.0, float(vh.uniform(70.0, 75.0)))    # x=8: 4 high -> in good group
    vl = substream(0, "x8_low")
    for _ in range(8):
        add_observation(study, 8.0, float(vl.uniform(10.0, 20.0)))    # x=8: 8 low -> in rest group

    # set the good-group size to 16 so the good group is exactly {12 at x=2, 4 at x=8}; the rest is {8 at x=8}
    sampler = TPESampler(seed=0)
    trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
    n_below = 16
    below, above = _split_trials(study, trials, n_below, False)
    # confirm the constructed split: count good/rest draws at each x
    below_at = {2.0: 0, 8.0: 0}
    above_at = {2.0: 0, 8.0: 0}
    for t in below:
        below_at[t.params["x"]] += 1
    for t in above:
        above_at[t.params["x"]] += 1
    print("\n--- Repeated-x acquisition (good-group size set to 16) ---")
    print(f"x=2.0 -> good group: {below_at[2.0]:>2}, rest group: {above_at[2.0]:>2}  (12/12 draws in the top quantile)")
    print(f"x=8.0 -> good group: {below_at[8.0]:>2}, rest group: {above_at[8.0]:>2}  ( 4/12 draws in the top quantile)")

    # fit l from the good group and g from the rest, then evaluate the acquisition at x=2 and x=8
    mpe_below = sampler._build_parzen_estimator(study, SPACE, below, handle_below=True)
    mpe_above = sampler._build_parzen_estimator(study, SPACE, above, handle_below=False)
    pts = {"x": np.array([2.0, 8.0])}
    log_l = mpe_below.log_pdf(pts)
    log_g = mpe_above.log_pdf(pts)
    acq = sampler._compute_acquisition_func(pts, mpe_below, mpe_above)
    print(f"\n{'x':>4} | {'log l':>9} | {'log g':>9} | {'acq = log l - log g':>20}")
    print("-" * 50)
    for i, x in enumerate([2.0, 8.0]):
        print(f"{x:>4.1f} | {log_l[i]:>9.4f} | {log_g[i]:>9.4f} | {acq[i]:>20.4f}")
    print(f"\nacq(x=2) - acq(x=8) = {acq[0] - acq[1]:.4f}: the acquisition at a repeated x reflects how many")
    print("of that x's draws reached the top quantile. x=2 (12/12 good, 0 in rest) has both a higher l and a")
    print("near-prior g, so its acquisition is far above x=8 (4 good, 8 in rest -> a large g at x=8).")


def main():
    """Run the source print and both empirical checks for point 5."""
    print_source()
    same_x_split_check()
    repeated_x_acquisition()


if __name__ == "__main__":
    main()
