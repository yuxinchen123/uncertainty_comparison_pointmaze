import optuna, numpy as np
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ask n trials from a study but never tell a result, and collect the suggested points.
def ask_without_telling(sampler, n=40):
    study = optuna.create_study(sampler=sampler, direction="minimize")
    pts = [(t.suggest_float("x", -5, 5), t.suggest_float("y", -5, 5))
           for t in (study.ask() for _ in range(n))]
    return np.array(pts), study

# TPESampler with a startup threshold of 10, but with zero completed trials.
tpe_pts, tpe_study = ask_without_telling(TPESampler(seed=0, n_startup_trials=10))
# A plain RandomSampler with the same seed.
rnd_pts, _ = ask_without_telling(RandomSampler(seed=0))

# n_startup_trials counts COMPLETED trials (0 here), so all 40 TPE asks are random.
n_match = int(np.all(np.isclose(tpe_pts, rnd_pts), axis=1).sum())
print(f"TPE asks identical to RandomSampler(seed=0): {n_match} / 40")
states = {}
for t in tpe_study.get_trials(deepcopy=False):
    states[t.state.name] = states.get(t.state.name, 0) + 1
print("TPE-study trial states after 40 asks, 0 tells:", states)
