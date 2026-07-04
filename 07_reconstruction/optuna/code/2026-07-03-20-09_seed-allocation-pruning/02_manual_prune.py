"""Point 2: the SAME rule with NO custom pruner object at all — the check lives inside the objective.

This is the simplest expression and the one the tutorial leads with: when n>=10 and the 95% upper
bound has fallen below 35, the objective itself raises optuna.TrialPruned(). No pruner subclass, no
should_prune(); we still call trial.report(...) purely so the partial seed curve is saved in the study.
We confirm the trial ends in state TrialState.PRUNED.
"""

import optuna

import sim

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    # One cell per trial; loop seeds, and self-prune when the upper 95% bound drops below the bar.
    cell = trial.suggest_int("cell", 0, sim.N_CELLS - 1)
    rewards = []
    for seed_index in range(sim.MAX_SEEDS):
        # Draw this seed's reward and recompute the running mean / sd / upper bound.
        rewards.append(sim.draw_reward(0, cell, seed_index))
        mean, sd, upper = sim.upper_ci(rewards)
        n = len(rewards)
        # Report the running mean only so the partial curve is stored (no pruner reads it here).
        trial.report(mean, step=n)
        # The rule, inline: after the warmup count, raise TrialPruned when the bound is below 35.
        if n >= sim.MIN_SEEDS and upper < sim.THRESHOLD:
            raise optuna.TrialPruned()
    return mean


def main():
    # Run all 12 cells; no pruner passed to create_study for the rule (the objective does it).
    print(f"Manual rule inside objective: raise TrialPruned() when n>=10 and "
          f"mean + 1.96*sd/sqrt(n) < {sim.THRESHOLD}")
    print("No custom pruner object is used for the rule.\n")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"cell": list(range(sim.N_CELLS))}, seed=0),
    )
    study.optimize(objective, n_trials=sim.N_CELLS)

    # Report each trial's final state and the seed count it stopped at.
    print(f"{'cell':>4} {'true_mean':>9} {'seeds':>5} {'last_mean':>9} {'state':>18}")
    total = 0
    for t in sorted(study.trials, key=lambda t: t.params["cell"]):
        # A trial's stored seed count is the number of intermediate values it reported.
        cell = t.params["cell"]
        seeds = len(t.intermediate_values)
        last_mean = t.intermediate_values[max(t.intermediate_values)]
        total += seeds
        print(f"{cell:>4} {sim.TRUE_MEANS[cell]:>9.1f} {seeds:>5} {last_mean:>9.2f} {str(t.state):>18}")

    # Confirm the two possible end states and the total cost.
    states = {str(t.state) for t in study.trials}
    print(f"\ndistinct end states observed: {sorted(states)}")
    print(f"total seed-runs used: {total}  (full grid would use {sim.N_CELLS * sim.MAX_SEEDS})")
    print(f"pruner passed to create_study for the rule: none "
          f"(default pruner is {type(study.pruner).__name__}, unused here)")


if __name__ == "__main__":
    main()
