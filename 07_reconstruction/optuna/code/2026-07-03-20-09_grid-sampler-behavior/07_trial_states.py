"""Verify trial states: produce COMPLETE, PRUNED, and FAIL trials and count them from trial.state."""
from collections import Counter
import optuna
from optuna.trial import TrialState


def objective(trial):
    """Objective that completes, prunes, or fails depending on the suggested integer bucket."""
    # one integer axis chooses the fate of the trial
    bucket = trial.suggest_int("bucket", 0, 2)
    # bucket 0 -> raise TrialPruned so the trial is recorded PRUNED
    if bucket == 0:
        raise optuna.TrialPruned()
    # bucket 1 -> raise a plain error so the trial is recorded FAIL (caught by optimize's catch=)
    if bucket == 1:
        raise ValueError("deliberate failure to produce a FAIL state")
    # bucket 2 -> return normally so the trial is recorded COMPLETE
    return float(bucket)


def main():
    """Run trials that hit all three states and print the counts read from trial.state."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__)

    # grid over the three buckets, repeated via a second axis so we get several of each state
    search_space = {"bucket": [0, 1, 2], "rep": list(range(4))}
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # catch=(ValueError,) so a raised ValueError is recorded as FAIL instead of stopping optimize
    study.optimize(objective, n_trials=None, catch=(ValueError,))

    # count states by reading trial.state on every trial
    counts = Counter(t.state for t in study.trials)
    print("total trials:", len(study.trials))
    print("COMPLETE:", counts[TrialState.COMPLETE])
    print("PRUNED:", counts[TrialState.PRUNED])
    print("FAIL:", counts[TrialState.FAIL])

    # show that trial.state is where each is read from, with one example per state
    for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
        example = next(t for t in study.trials if t.state == state)
        print(f"example {state.name}: trial #{example.number}, params={example.params}, value={example.value}")

    # convenience helpers that filter by state
    print("len(study.get_trials(states=(TrialState.COMPLETE,))):",
          len(study.get_trials(states=(TrialState.COMPLETE,))))


if __name__ == "__main__":
    main()
