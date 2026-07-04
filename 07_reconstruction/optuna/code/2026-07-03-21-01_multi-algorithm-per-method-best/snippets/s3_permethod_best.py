import hashlib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHODS = ["adam", "adagrad", "sgd1t"]


def reward(method, readout, log10_beta, eta0=None, t0=None):
    # noise-free bump per method + reproducible additive noise SD 6 keyed by the configuration
    center, amp = {"adam": (2.0, 20.0), "adagrad": (1.0, 27.0), "sgd1t": (2.0, 45.0)}[method]
    mean = amp * np.exp(-0.5 * ((log10_beta - center) / 0.65) ** 2)
    if method == "adam" and readout == "l2":
        mean += 3.0
    if method == "sgd1t":
        mean += -4.0 * (np.log10(eta0) + 2.0) ** 2 - 2.0 * abs(np.log10(t0) - 4.0)
    key = "::".join(str(p) for p in (method, readout, round(log10_beta, 6), eta0, t0))
    gen = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return mean + gen.normal(0.0, 6.0)


def objective(trial):
    # mixed study: method is suggested; eta0/t0 only in the sgd1t branch
    method = trial.suggest_categorical("method", METHODS)
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    if method == "sgd1t":
        return reward(method, readout, log10_beta,
                      trial.suggest_float("eta0", 1e-3, 1e-1, log=True),
                      trial.suggest_categorical("t0", [1e3, 1e4]))
    return reward(method, readout, log10_beta)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=90)

# recover the best trial PER method: filter COMPLETE trials on params["method"], take argmax
winners = {}
for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
    m = t.params.get("method")
    if m is not None and (m not in winners or t.value > winners[m].value):
        winners[m] = t

for m in METHODS:
    t = winners[m]
    n_m = sum(x.params.get("method") == m for x in study.trials)
    print(f"{m:>8}: best value={t.value:6.2f}  from {n_m:2d} trials  params={t.params}")
