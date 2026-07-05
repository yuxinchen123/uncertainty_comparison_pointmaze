"""Point 3 (central): under single-seed feedback TPE targets the top-quantile draw, not the mean.

Two regions on [0,10]:
  region A (|x-3|<=0.5): reward ~ Normal(60, 5)                      -> true mean 60, consistent good learner
  region B (|x-7|<=0.5): success Normal(80,10) w.p. 0.5 else Normal(0,3) -> true mean 40, half succeed
                                                                           near 80 and half collapse near 0
  elsewhere:             reward ~ Normal(5, 3)
A has the higher true mean (60 vs 40). But TPE's good group is the top gamma(n) of single noisy values,
and B's ~80 successes populate that top group, so single-seed TPE concentrates on B. Averaging 10 seeds
per trial turns each value into an estimate of the mean, and the concentration flips to A. Same 150
seed-run budget both ways: 150 single-seed trials vs 15 trials of 10-seed means.

Region B uses the project's own bimodal per-seed regime (success Normal(80,10), collapse Normal(0,3)).
The task's original numbers (A Normal(30,2); B 80 w.p. 0.15 else Normal(15,3)) do NOT produce this flip,
because A's consistent 30 fills the top decile far more often than B's rare 80s -- the contrast run at
the bottom shows single-seed TPE stays on A there. B's success probability has to be high enough (0.5
here) for its high draws to actually dominate the top decile.
"""

import numpy as np

import optuna

from common import SPACE, region_reward, region_reward_task_original, run_tpe, substream

optuna.logging.set_verbosity(optuna.logging.WARNING)


def region_of(x):
    """Label a proposed x as 'A' (near 3), 'B' (near 7), or 'elsewhere'."""
    # region membership uses the same half-width-0.5 windows as the reward model
    if abs(x - 3.0) <= 0.5:
        return "A"
    if abs(x - 7.0) <= 0.5:
        return "B"
    return "elsewhere"


def single_seed_value(seed, reward_fn):
    """Objective returning one region reward per trial, keyed by trial index."""
    # one keyed generator per trial so the single-seed run is reproducible regardless of order
    def value_fn(i, x):
        return reward_fn(substream(seed, "single", i), x)
    return value_fn


def mean_of_10_value(seed, reward_fn):
    """Objective returning the mean of 10 region rewards per trial, keyed by (trial index, seed index)."""
    # 10 independent keyed generators per trial; the reported value estimates the true mean at x
    def value_fn(i, x):
        draws = [reward_fn(substream(seed, "mean10", i, s), x) for s in range(10)]
        return float(np.mean(draws))
    return value_fn


def counts_in_window(xs, window):
    """Count how many of the last `window` proposals fall in region A, region B, and elsewhere."""
    # tally the region label of each proposal in the counting window
    tail = xs[-window:]
    c = {"A": 0, "B": 0, "elsewhere": 0}
    for x in tail:
        c[region_of(x)] += 1
    return c


def concentration_region(counts):
    """Return the region ('A'/'B'/'tie') that a proposal-count window concentrated on."""
    # a run concentrated on whichever of A / B got more proposals in the counting window
    if counts["A"] > counts["B"]:
        return "A"
    if counts["B"] > counts["A"]:
        return "B"
    return "tie"


def run_block(reward_fn):
    """Run both feedback modes at 3 seeds for one reward landscape and print a labelled block."""
    header = f"{'feedback':>18} | {'seed':>4} | {'window':>15} | {'in A':>5} | {'in B':>5} | {'else':>5} | {'best x':>7} | {'best value':>10}"
    print(header)
    print("-" * len(header))
    # single-seed: 150 trials, count region membership over the last 50 proposals
    for seed in (0, 1, 2):
        study, _, xs = run_tpe(seed, 150, single_seed_value(seed, reward_fn))
        c = counts_in_window(xs, 50)
        print(f"{'single seed':>18} | {seed:>4} | {'last 50 of 150':>15} | {c['A']:>5} | {c['B']:>5} | "
              f"{c['elsewhere']:>5} | {study.best_trial.params['x']:>7.3f} | {study.best_trial.value:>10.3f}")
    print("-" * len(header))
    # mean of 10 seeds: 15 trials (same 150 seed-runs); startup lowered to 3 so 12 trials are TPE-guided
    for seed in (0, 1, 2):
        study, _, xs = run_tpe(seed, 15, mean_of_10_value(seed, reward_fn), n_startup_trials=3)
        c = counts_in_window(xs, 12)
        print(f"{'mean of 10 seeds':>18} | {seed:>4} | {'last 12 of 15':>15} | {c['A']:>5} | {c['B']:>5} | "
              f"{c['elsewhere']:>5} | {study.best_trial.params['x']:>7.3f} | {study.best_trial.value:>10.3f}")
    print("-" * len(header))


def aggregate_over_seeds(reward_fn, n_seeds):
    """Tally, over n_seeds, which of A/B each feedback mode concentrated its late proposals on.

    Both modes scored the same way: the region (A vs B) that got more proposals in the mode's late
    window (single-seed: last 50 of 150; mean-of-10: last 12 of 15). Single-seed is bistable, so the
    tally quantifies the majority behaviour rather than resting on any one seed.
    """
    # classify each run so the majority behaviour is quantified without cherry-picking a single seed
    single = {"A": 0, "B": 0, "tie": 0}
    mean10 = {"A": 0, "B": 0, "tie": 0}
    for seed in range(n_seeds):
        _, _, xs = run_tpe(seed, 150, single_seed_value(seed, reward_fn))
        single[concentration_region(counts_in_window(xs, 50))] += 1
        _, _, xs = run_tpe(seed, 15, mean_of_10_value(seed, reward_fn), n_startup_trials=3)
        mean10[concentration_region(counts_in_window(xs, 12))] += 1
    return single, mean10


def main():
    """Print the region concentration and best-trial location for the main landscape and the contrast."""
    print("optuna version:", optuna.__version__)

    print("\n=== Main landscape (project bimodal regime for B) ===")
    print("region A near x=3: Normal(60,5), true mean 60 (higher, consistent -- no draws above ~75).")
    print("region B near x=7: Normal(80,10) w.p. 0.5 else Normal(0,3), true mean 40 (lower, but its ~80")
    print("successes are the highest single draws around, so they own the top decile).")
    print("elsewhere: Normal(5,3). Budget = 150 seed-runs each way. Region A near x=3, region B near x=7.\n")
    run_block(region_reward)
    print("\nSingle-seed TPE is pulled to B (true mean 40); its best value is a lone ~80-100 success, far above")
    print("any region's true mean. The 10-seed-mean objective concentrates on A (true mean 60) with a best")
    print("value ~60, an honest estimate of A's mean. Single-seed is bistable (a seed whose early draws lock")
    print("onto A's steady 60 stays there), so the tally over 12 seeds below quantifies the majority behaviour.")

    # 12-seed tally so the headline does not depend on any one seed
    n_seeds = 12
    single, mean10 = aggregate_over_seeds(region_reward, n_seeds)
    print(f"\nOver {n_seeds} base seeds, region of late-window concentration (A vs B):")
    print(f"  single seed     : B (lower mean 40): {single['B']:>2}   A (higher mean 60): {single['A']:>2}   tie: {single['tie']:>2}")
    print(f"  mean of 10 seeds: B (lower mean 40): {mean10['B']:>2}   A (higher mean 60): {mean10['A']:>2}   tie: {mean10['tie']:>2}")
    print("Single-seed favours the lower-mean jackpot region B in the majority of seeds; averaging 10 seeds")
    print("shifts the majority to the true-best region A. Ten seeds do not fully resolve B's large variance,")
    print("so the mean-of-10 tally is not unanimous -- point 4 shows 30-seed means resolve it cleanly.")

    print("\n=== Contrast: the task's original numbers (do NOT flip) ===")
    print("region A near x=3: Normal(30,2), true mean 30; region B near x=7: 80 w.p. 0.15 else Normal(15,3),")
    print("true mean 24.75; elsewhere Normal(5,2). Here A's steady 30s fill the top decile, so single-seed")
    print("TPE stays on A -- B's 15%-rare 80s are too few to take over the good group.\n")
    run_block(region_reward_task_original)
    single_o, mean10_o = aggregate_over_seeds(region_reward_task_original, n_seeds)
    print(f"\nOver {n_seeds} base seeds (task-original numbers), region of late-window concentration (A vs B):")
    print(f"  single seed     : B (lower mean 24.75): {single_o['B']:>2}   A (higher mean 30): {single_o['A']:>2}   tie: {single_o['tie']:>2}")
    print(f"  mean of 10 seeds: B (lower mean 24.75): {mean10_o['B']:>2}   A (higher mean 30): {mean10_o['A']:>2}   tie: {mean10_o['tie']:>2}")


if __name__ == "__main__":
    main()
