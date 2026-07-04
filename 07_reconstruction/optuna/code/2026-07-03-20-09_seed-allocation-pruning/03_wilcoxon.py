"""Point 3: Optuna's built-in WilcoxonPruner (added v3.6.0, present in 4.9.0).

Reporting convention from the docstring: the objective evaluates ONE instance (here one seed) at a
time and calls trial.report(value_of_that_seed, step=instance_id). The pruner runs a Wilcoxon
signed-rank test between the CURRENT trial's per-step values and the CURRENT BEST completed trial's
values at the SAME steps, and prunes when it is confident (up to p_threshold) the current cell is
worse than the best cell. When should_prune() is True the docstring says to RETURN the running mean
(so Optuna records a predicted value) rather than raise TrialPruned.

This is a relative test (worse-than-the-best-so-far), NOT the user's absolute bar at 35. We enqueue
the strong cell 0 first so it completes and becomes the reference every later cell is compared to.
"""

import numpy as np
import optuna

import sim

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    # One cell per trial; evaluate seeds one at a time and report each seed's own reward at step=seed.
    cell = trial.suggest_int("cell", 0, sim.N_CELLS - 1)
    rewards = []
    for seed_index in range(sim.MAX_SEEDS):
        # Draw this single seed's reward and report it as this step's value (per-instance value).
        r = sim.draw_reward(0, cell, seed_index)
        rewards.append(r)
        trial.report(r, step=seed_index)
        # Follow the docstring: when the pruner is confident this cell is worse, return the estimate.
        if trial.should_prune():
            return float(np.mean(rewards))
    return float(np.mean(rewards))


def main():
    # Print the exact reporting convention lines from the docstring, then run the demo.
    print("WilcoxonPruner reporting convention (from the docstring):")
    print("  * call Trial.report(value, step) once per instance (here: per seed)")
    print("  * the reported value is that instance's score, NOT a running mean")
    print("  * it tests the current trial vs the current BEST trial at matching step ids")
    print("  * on should_prune()==True, RETURN the running mean (do not raise TrialPruned)\n")

    # p_threshold controls aggressiveness; n_startup_steps delays pruning until enough paired steps.
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1, n_startup_steps=sim.MIN_SEEDS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=pruner,
    )
    # Enqueue cell 0 (the strongest) first so it completes and becomes the comparison reference.
    for cell in range(sim.N_CELLS):
        study.enqueue_trial({"cell": cell})
    study.optimize(objective, n_trials=sim.N_CELLS)

    # Per-cell table: seeds used, final mean estimate, whether the pruner stopped it early.
    print(f"{'cell':>4} {'true_mean':>9} {'seeds':>5} {'mean_est':>8} {'stopped_early':>13} {'state':>18}")
    total = 0
    for t in sorted(study.trials, key=lambda t: t.params["cell"]):
        cell = t.params["cell"]
        seeds = len(t.intermediate_values)
        mean_est = float(np.mean(list(t.intermediate_values.values())))
        stopped_early = seeds < sim.MAX_SEEDS
        total += seeds
        print(f"{cell:>4} {sim.TRUE_MEANS[cell]:>9.1f} {seeds:>5} {mean_est:>8.2f} "
              f"{str(stopped_early):>13} {str(t.state):>18}")

    print(f"\ntotal seed-runs used: {total}  (full grid would use {sim.N_CELLS * sim.MAX_SEEDS})")
    print("decision rule: WilcoxonPruner prunes a cell when a signed-rank test says it is")
    print("statistically worse than the BEST cell so far (relative). The user's rule prunes when a")
    print(f"cell's own 95% upper bound falls below the fixed bar {sim.THRESHOLD} (absolute). A cell above")
    print("35 but below the best cell (e.g. cell 1 vs cell 0) can be pruned by Wilcoxon yet kept by the")
    print("user's absolute rule; that is the key difference.")


if __name__ == "__main__":
    main()
