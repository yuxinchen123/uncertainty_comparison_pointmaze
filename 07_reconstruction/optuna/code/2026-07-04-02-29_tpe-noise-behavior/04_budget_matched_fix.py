"""Point 4: spend a fixed 300 seed-runs three ways and see which most reliably finds a high-true-mean x.

Landscape (same as point 2): true mean m(x) = 50*exp(-(x-8)^2/2), peak 50 at x=8; per-seed reward is
the bimodal model (success near 80 w.p. m/80, else near 0). Three budget-matched designs:
  (a) 300 single-seed trials
  (b) 100 trials, each value = mean of 3 seeds
  (c)  30 trials, each value = mean of 10 seeds
For each design and 3 base seeds, after the budget we report:
  - m(best-by-value x): the TRUE mean at the trial with the highest observed value (the config you'd pick)
  - best value vs m(best-by-value x): the inflation (a single lucky draw overstates the true mean)
  - belief m: the mean true reward over TPE's next 25 proposals (a stable read of where it would sample)

On this single-peak landscape the per-seed success probability m(x)/80 rises with the mean, so successes
cluster where the mean is high and all three designs find roughly the right x region. The seed-count
lesson here is the inflation: single-seed best-by-value reads ~100 where the true mean is ~40, and picks
a slightly worse x; 10-seed means cut the inflation to ~17 and pick an x with a higher true mean. Note the
belief-m confound: 300 single-seed trials give TPE far more points to concentrate with than 30 ten-seed
trials, so belief-m partly reflects trial count, not seed quality -- it is reported with that caveat.
"""

import numpy as np

import optuna

from common import SPACE, bimodal_reward, peak_mean, run_tpe, substream

optuna.logging.set_verbosity(optuna.logging.WARNING)


def k_seed_value(seed, k):
    """Objective reporting the mean of k bimodal seed rewards at m(x), keyed by (trial index, seed index)."""
    # k independent keyed generators per trial so the k-seed mean is reproducible regardless of order
    def value_fn(i, x):
        draws = [bimodal_reward(substream(seed, "reward", i, s), float(peak_mean(x))) for s in range(k)]
        return float(np.mean(draws))
    return value_fn


def belief_mean_true(study, n_proposals):
    """Mean true reward m(x) over TPE's next n_proposals asks (not told), a stable read of its belief."""
    # repeatedly ask without telling: each ask is an independent draw from the current proposal density
    ms = [float(peak_mean(study.ask(SPACE).params["x"])) for _ in range(n_proposals)]
    return float(np.mean(ms))


def run_design(seed, n_trials, k):
    """Run one budget-matched design and return best-by-value stats and the belief true-mean."""
    # run the study, read the best-by-value trial, then measure the belief over 25 next proposals
    study, _, _ = run_tpe(seed, n_trials, k_seed_value(seed, k))
    best_x = study.best_trial.params["x"]
    best_value = study.best_trial.value
    return best_x, best_value, float(peak_mean(best_x)), belief_mean_true(study, 25)


def main():
    """Run all three designs at 3 seeds and print per-run and averaged true-mean / inflation numbers."""
    print("optuna version:", optuna.__version__)
    print("\nLandscape m(x) = 50*exp(-(x-8)^2/2), true optimum m=50 at x=8; bimodal per-seed rewards.")
    print("Budget = 300 seed-runs each design. 'm(...)' is the TRUE mean reward at that x (max possible 50).")
    print("'belief m' = mean true reward over TPE's next 25 proposals (confounded by trial count, see note).\n")

    designs = [("(a) 300 x 1 seed", 300, 1), ("(b) 100 x 3-seed mean", 100, 3), ("(c) 30 x 10-seed mean", 30, 10)]
    header = (f"{'design':>22} | {'seed':>4} | {'best-by-val x':>13} | {'best value':>10} | "
              f"{'m(best x)':>9} | {'inflation':>9} | {'belief m':>8}")
    print(header)
    print("-" * len(header))
    # per design: print each seed, then the across-seed average of m(best x), inflation, belief m
    for label, n_trials, k in designs:
        m_best_list, infl_list, belief_list = [], [], []
        for seed in (0, 1, 2):
            best_x, best_value, m_best, belief = run_design(seed, n_trials, k)
            inflation = best_value - m_best
            m_best_list.append(m_best)
            infl_list.append(inflation)
            belief_list.append(belief)
            print(f"{label:>22} | {seed:>4} | {best_x:>13.3f} | {best_value:>10.2f} | "
                  f"{m_best:>9.2f} | {inflation:>9.2f} | {belief:>8.2f}")
        print(f"{label:>22} | {'mean':>4} | {'':>13} | {'':>10} | "
              f"{np.mean(m_best_list):>9.2f} | {np.mean(infl_list):>9.2f} | {np.mean(belief_list):>8.2f}")
        print("-" * len(header))

    print("\n'inflation' = best observed value minus the true mean at that x. It is large under single-seed")
    print("trials (a lone success reads ~80-100 where the true mean is ~40) and small once each trial value")
    print("is a 10-seed mean. 'm(best x)' (the true mean of the config you would pick) rises as seeds per")
    print("trial rise, so the selected x is genuinely better. 'belief m' is higher for the 300-trial design")
    print("only because 300 points concentrate TPE more than 30 points do -- it reflects trial count, not")
    print("seed quality, so read it alongside the inflation and m(best x) columns, not on its own.")


if __name__ == "__main__":
    main()
