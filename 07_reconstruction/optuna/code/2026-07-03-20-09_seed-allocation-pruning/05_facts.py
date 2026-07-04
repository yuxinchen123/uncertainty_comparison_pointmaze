"""Point 5: pin down the exact pruning interface facts on Optuna 4.9.0.

(a) trial.report(value, step) + trial.should_prune() is the pruning interface, and pruning is
    COOPERATIVE: should_prune() returning True does not stop anything by itself; the objective must
    raise optuna.TrialPruned(). We show a trial that ignores a should_prune()==True and runs to the end.
(b) A pruned trial's state is TrialState.PRUNED, and its intermediate_values survive in study.trials
    so a later analysis can still read the partial seed curve.
(c) should_prune() can be called even when no pruner was passed to create_study(); the DEFAULT pruner
    is MedianPruner (not a no-op), so we print type(study.pruner) to show it.
"""

import optuna

import sim

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AlwaysTruePruner(optuna.pruners.BasePruner):
    """A pruner whose prune() always returns True, to prove should_prune() alone stops nothing."""

    def prune(self, study, trial):
        # Always claim the trial should be pruned; the objective decides whether to act on it.
        return True


def fact_a_cooperative():
    # (a) Show should_prune()==True does NOT stop a trial; only raising TrialPruned() does.
    print("=== (a) pruning is cooperative: should_prune() True does nothing until you raise ===")

    def objective_ignores(trial):
        # Report seeds and observe should_prune()==True at every step, but never raise: it completes.
        trial.suggest_int("cell", 0, 0)
        saw_true = 0
        for s in range(sim.MAX_SEEDS):
            trial.report(sim.draw_reward(0, 0, s), step=s + 1)
            if trial.should_prune():
                saw_true += 1
        trial.set_user_attr("times_should_prune_true", saw_true)
        return 1.0

    study = optuna.create_study(direction="maximize", pruner=AlwaysTruePruner(),
                                sampler=optuna.samplers.GridSampler({"cell": [0]}, seed=0))
    study.optimize(objective_ignores, n_trials=1)
    t = study.trials[0]
    print(f"  should_prune() returned True {t.user_attrs['times_should_prune_true']} times, "
          f"objective never raised -> final state {t.state}, seeds run {len(t.intermediate_values)}")

    def objective_raises(trial):
        # Same pruner, but this objective cooperates: raise TrialPruned at the first True.
        trial.suggest_int("cell", 0, 0)
        for s in range(sim.MAX_SEEDS):
            trial.report(sim.draw_reward(0, 0, s), step=s + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return 1.0

    study2 = optuna.create_study(direction="maximize", pruner=AlwaysTruePruner(),
                                 sampler=optuna.samplers.GridSampler({"cell": [0]}, seed=0))
    study2.optimize(objective_raises, n_trials=1)
    t2 = study2.trials[0]
    print(f"  same pruner, objective raises TrialPruned -> final state {t2.state}, "
          f"seeds run {len(t2.intermediate_values)}\n")


def fact_b_state_and_curve_survive():
    # (b) A pruned trial ends PRUNED and keeps its intermediate_values (the partial seed curve).
    print("=== (b) a pruned trial's state is PRUNED and its intermediate_values survive ===")

    def objective(trial):
        # Prune this cell after 12 seeds by raising, to leave a partial 12-point curve behind.
        trial.suggest_int("cell", 0, 0)
        rewards = []
        for s in range(sim.MAX_SEEDS):
            rewards.append(sim.draw_reward(0, 11, s))  # cell 11: true mean 0.1, clearly below 35
            trial.report(sum(rewards) / len(rewards), step=len(rewards))
            if len(rewards) >= 12:
                raise optuna.TrialPruned()
        return sum(rewards) / len(rewards)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler({"cell": [0]}, seed=0))
    study.optimize(objective, n_trials=1)
    t = study.trials[0]
    # Read the surviving partial curve straight off the stored trial.
    curve = t.intermediate_values
    print(f"  stored trial state: {t.state}")
    print(f"  intermediate_values kept: {len(curve)} points, steps {sorted(curve)[:3]}...{sorted(curve)[-1]}")
    print(f"  last stored running mean: {curve[max(curve)]:.3f}")
    print(f"  the partial seed curve is fully readable after pruning\n")


def fact_c_default_pruner():
    # (c) should_prune() works with NO pruner passed; the default pruner is MedianPruner.
    print("=== (c) should_prune() with no pruner passed: default pruner is MedianPruner ===")
    observed = {}

    def objective(trial):
        # No pruner was passed to create_study; should_prune() is still callable and returns a bool.
        trial.suggest_int("cell", 0, 0)
        for s in range(sim.MAX_SEEDS):
            trial.report(sim.draw_reward(0, 0, s), step=s + 1)
            observed[s + 1] = trial.should_prune()
        return 1.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler({"cell": [0]}, seed=0))
    print(f"  type(study.pruner) when none was passed: {type(study.pruner)}")
    study.optimize(objective, n_trials=1)
    # should_prune returns a bool at every step; with only one trial MedianPruner has no baseline.
    print(f"  should_prune() is callable with the default pruner; returns bool. "
          f"Sample: step1={observed[1]}, step25={observed[25]}, step50={observed[50]}")
    print(f"  (with a single trial MedianPruner has no other trials to compare against, so False)\n")


def main():
    fact_a_cooperative()
    fact_b_state_and_curve_survive()
    fact_c_default_pruner()


if __name__ == "__main__":
    main()
