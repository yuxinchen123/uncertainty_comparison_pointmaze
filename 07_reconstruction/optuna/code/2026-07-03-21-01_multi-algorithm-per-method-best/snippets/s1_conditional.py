import hashlib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def reward(method, readout, log10_beta, eta0=None, t0=None):
    # noise-free bump per method (adam peak 20 @ beta=1e2, adagrad 27 @ 1e1, sgd1t 45 @ 1e2)
    center, amp = {"adam": (2.0, 20.0), "adagrad": (1.0, 27.0), "sgd1t": (2.0, 45.0)}[method]
    mean = amp * np.exp(-0.5 * ((log10_beta - center) / 0.65) ** 2)
    if method == "adam" and readout == "l2":
        mean += 3.0
    if method == "sgd1t":
        mean += -4.0 * (np.log10(eta0) + 2.0) ** 2 - 2.0 * abs(np.log10(t0) - 4.0)
    # reproducible additive noise SD 6: one generator per configuration, keyed by a stable string
    key = "::".join(str(p) for p in (method, readout, round(log10_beta, 6), eta0, t0))
    gen = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return mean + gen.normal(0.0, 6.0)


def objective(trial):
    # method is a top-level categorical; eta0/t0 are suggested ONLY in the sgd1t branch
    method = trial.suggest_categorical("method", ["adam", "adagrad", "sgd1t"])
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    if method == "sgd1t":
        eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
        t0 = trial.suggest_categorical("t0", [1e3, 1e4])
        return reward(method, readout, log10_beta, eta0, t0)
    return reward(method, readout, log10_beta)


# default TPESampler is multivariate=False -> independent per-parameter estimators
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60)

df = study.trials_dataframe()
cols = ["number", "params_method", "params_log10_beta", "params_eta0", "params_t0"]
print(df[cols].head(6).to_string(index=False))
n_sgd = sum(t.params.get("method") == "sgd1t" for t in study.trials)
n_eta0 = sum("eta0" in t.params for t in study.trials)
print(f"\ntrials total={len(study.trials)}  sgd1t={n_sgd}  with eta0={n_eta0}  "
      f"(eta0 present only in sgd1t trials: {n_eta0 == n_sgd})")
non_sgd = df[df["params_method"] != "sgd1t"]
print(f"non-sgd1t rows all have eta0/t0 = NaN: "
      f"{non_sgd['params_eta0'].isna().all() and non_sgd['params_t0'].isna().all()}")
