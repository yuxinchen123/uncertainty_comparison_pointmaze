import hashlib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHODS = ["adam", "adagrad", "sgd1t"]
TRUE_PEAK = {"adam": 23.0, "adagrad": 27.0, "sgd1t": 45.0}


def reward_mean(method, readout, log10_beta, eta0=None, t0=None):
    # noise-free mean: Gaussian bump per method, +3 for adam l2, mild eta0/t0 dependence for sgd1t
    peak = {"adam": (2.0, 20.0), "adagrad": (1.0, 27.0), "sgd1t": (2.0, 45.0)}[method]
    mean = peak[1] * np.exp(-0.5 * ((log10_beta - peak[0]) / 0.65) ** 2)
    if method == "adam" and readout == "l2":
        mean += 3.0
    if method == "sgd1t":
        mean += -4.0 * (np.log10(eta0) + 2.0) ** 2 - 2.0 * abs(np.log10(t0) - 4.0)
    return mean


def reward(method, readout, log10_beta, eta0=None, t0=None):
    # reproducible additive noise SD 6, keyed by the configuration
    key = "::".join(str(p) for p in (method, readout, round(log10_beta, 6), eta0, t0))
    gen = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return reward_mean(method, readout, log10_beta, eta0, t0) + gen.normal(0.0, 6.0)


def suggest(trial, method):
    # suggest one method's axes; sgd1t alone carries eta0/t0
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    if method == "sgd1t":
        return reward(method, readout, log10_beta,
                      trial.suggest_float("eta0", 1e-3, 1e-1, log=True),
                      trial.suggest_categorical("t0", [1e3, 1e4]))
    return reward(method, readout, log10_beta)


def true_mean_of(method, p):
    # noise-free mean at a trial's chosen params (search quality, no max-of-noise bias)
    if method == "sgd1t":
        return reward_mean(method, p["readout"], p["log10_beta"], p["eta0"], p["t0"])
    return reward_mean(method, p["readout"], p["log10_beta"])


seed = 0
# shared study: 90 trials, method is a suggested categorical
shared = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
shared.optimize(lambda t: suggest(t, t.suggest_categorical("method", METHODS)), n_trials=90)

# split: three studies, 30 trials each (same 90-trial total), method fixed per study
split = {}
for m in METHODS:
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(lambda t, m=m: suggest(t, m), n_trials=30)
    split[m] = st

print(f"{'method':>8} {'shared_n':>9} {'shared_true':>12} {'split_n':>8} {'split_true':>11} {'true_peak':>10}")
for m in METHODS:
    sh = [t for t in shared.trials if t.params.get("method") == m]
    sh_true = max(true_mean_of(m, t.params) for t in sh)
    sp_true = max(true_mean_of(m, t.params) for t in split[m].trials)
    print(f"{m:>8} {len(sh):>9} {sh_true:>12.2f} {30:>8} {sp_true:>11.2f} {TRUE_PEAK[m]:>10.1f}")
