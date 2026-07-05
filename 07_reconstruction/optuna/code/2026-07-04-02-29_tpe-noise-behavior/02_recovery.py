"""Point 2: does TPE return to the true-best region after unlucky early failures there, and how fast?

Landscape: true mean m(x) = 50*exp(-(x-8)^2/2) on [0,10], peak at x=8. Per-seed reward is the bimodal
model (success near 80 w.p. m/80, else near 0). Seed a study with 20 informative random observations and
3 forced failure observations (value ~0) placed at exactly x=8 (the true best), then run TPE for 100 more
honest single-seed trials. Track the fraction of proposals landing within |x-8|<1 across four 25-trial
windows, at 3 base seeds, with and without the 3 forced failures. The comparison shows how quickly TPE
re-concentrates on the peak despite the early failures there.
"""

import numpy as np

import optuna

from common import SPACE, bimodal_reward, peak_mean, run_tpe, substream


def seeded_observations(seed, with_failures):
    """Build the pre-loaded observations: 20 informative random draws, optionally plus 3 forced failures at x=8."""
    # 20 informative observations: random x in [0,10] with one honest bimodal reward each at m(x)
    obs = []
    xr = substream(seed, "init_x")
    for i in range(20):
        x = float(xr.uniform(0.0, 10.0))
        r = bimodal_reward(substream(seed, "init_reward", i), peak_mean(x))
        obs.append((x, r))
    # optionally add 3 forced failures at exactly x=8 (value ~0): unlucky early draws at the true best
    if with_failures:
        fr = substream(seed, "forced_fail")
        for _ in range(3):
            obs.append((8.0, float(abs(fr.normal(0.0, 0.5)))))
    return obs


def window_fractions(xs):
    """Fraction of proposed x within |x-8|<1, computed over each successive 25-trial window."""
    # split the 100 proposals into four windows and report the near-peak hit fraction of each
    arr = np.asarray(xs)
    fracs = []
    for lo in (0, 25, 50, 75):
        w = arr[lo:lo + 25]
        fracs.append(float(np.mean(np.abs(w - 8.0) < 1.0)))
    return fracs


def value_fn_for(seed):
    """Return an objective that reports one honest bimodal reward at m(x), keyed by trial index."""
    # each trial index gets its own keyed reward generator so the run is reproducible regardless of order
    def value_fn(i, x):
        return bimodal_reward(substream(seed, "reward", i), peak_mean(x))
    return value_fn


def main():
    """Run TPE for 100 trials from both seedings at 3 base seeds and print the per-window near-peak fractions."""
    print("optuna version:", optuna.__version__)
    print("\nLandscape m(x) = 50*exp(-(x-8)^2/2), peak at x=8; single-seed bimodal rewards.")
    print("Seed = 20 informative random observations, then TPE for 100 trials.")
    print("Fraction of proposals with |x-8|<1 per 25-trial window.\n")

    seeds = (0, 1, 2)
    header = f"{'setup':>16} | {'base seed':>9} | {'trials 1-25':>11} | {'26-50':>7} | {'51-75':>7} | {'76-100':>7}"
    print(header)
    print("-" * len(header))
    # accumulate per-window fractions across seeds so we can also print the average row per setup
    summary = {True: [], False: []}
    for with_failures in (True, False):
        label = "3 forced fails" if with_failures else "no forced fails"
        for seed in seeds:
            obs = seeded_observations(seed, with_failures)
            _, _, xs = run_tpe(seed, 100, value_fn_for(seed), seeded_obs=obs)
            fr = window_fractions(xs)
            summary[with_failures].append(fr)
            print(f"{label:>16} | {seed:>9} | {fr[0]:>11.2f} | {fr[1]:>7.2f} | {fr[2]:>7.2f} | {fr[3]:>7.2f}")
        avg = np.mean(summary[with_failures], axis=0)
        print(f"{label:>16} | {'mean':>9} | {avg[0]:>11.2f} | {avg[1]:>7.2f} | {avg[2]:>7.2f} | {avg[3]:>7.2f}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
