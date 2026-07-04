import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# the canonical reference cell for adam: mse readout at beta=1e2 (log10_beta=2.0)
ADAM_REFERENCE = {"readout": "mse", "log10_beta": 2.0}


def adam_objective(trial):
    # adam has no eta0/t0; only readout and beta are searched
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    return -(log10_beta - 2.0) ** 2 + (1.0 if readout == "mse" else 0.0)


study = optuna.create_study(direction="maximize", study_name="3_2_1_adam",
                            sampler=optuna.samplers.TPESampler(seed=0))

# enqueue the reference cell into THIS study; it runs before any sampler suggestion
study.enqueue_trial(ADAM_REFERENCE)
study.optimize(adam_objective, n_trials=8)

t0 = study.trials[0]
print(f"trial 0 params : {t0.params}")
print(f"matches ref    : {t0.params == ADAM_REFERENCE}")
print(f"trial 0 state  : {t0.state.name}")
print(f"best value     : {study.best_value:.3f}")
