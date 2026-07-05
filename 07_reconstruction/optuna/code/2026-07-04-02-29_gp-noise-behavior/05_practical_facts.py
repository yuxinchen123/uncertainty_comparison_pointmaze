"""Point 5: practical facts for the tutorial.

(a) GPSampler per-ask wall-clock cost at 150 / 200 / 300 completed trials (single float parameter).
(b) Re-check that RUNNING trials get a best-value constant-liar imputation automatically.
(c) Failure modes with the project's bimodal values (standardization + extreme bimodality): what
    actually happens when the completed values are all near-identical, or split hard between ~0 and
    ~80, or exactly constant.
"""

import hashlib
import time
import warnings

import numpy as np
import torch

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space
from optuna._gp.acqf import LogEI

optuna.logging.set_verbosity(optuna.logging.WARNING)
DIST = optuna.distributions.FloatDistribution(0.0, 10.0)


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal(rng, mean):
    # project noise model: success -> Normal(80,10), else -> Normal(0,3)
    if rng.random() < mean / 80.0:
        return float(rng.normal(80.0, 10.0))
    return float(rng.normal(0.0, 3.0))


def preloaded_study(n, seed):
    # a study with n completed trials at random x, valued by one bimodal draw at true mean 20+2x
    sampler = optuna.samplers.GPSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    xr = substream(seed, "cost", "x", n).uniform(0.0, 10.0, n)
    dr = substream(seed, "cost", "draw", n)
    for xv in xr:
        study.add_trial(optuna.trial.create_trial(
            params={"x": float(xv)}, distributions={"x": DIST}, value=bimodal(dr, 20.0 + 2.0 * xv)))
    return study


# ---- (a) per-ask cost ----
print("(a) GPSampler per-ask wall-clock cost (one float parameter, one full GP fit + acqf optimize).")
# one warmup ask so the reported times exclude the one-time torch/scipy initialization cost
_w = preloaded_study(150, seed=99)
_t = _w.ask(); _t.suggest_float("x", 0.0, 10.0); _w.tell(_t, 0.0)
print(f"    {'n_completed':>11} {'ask#1':>10} {'ask#2':>10} {'ask#3':>10}")
for n in (150, 200, 300):
    study = preloaded_study(n, seed=0)
    times = []
    for _ in range(3):
        # time one full suggestion: study.ask + first suggest triggers the GP fit and acqf optimize
        t0 = time.perf_counter()
        trial = study.ask()
        x = trial.suggest_float("x", 0.0, 10.0)
        times.append(time.perf_counter() - t0)
        study.tell(trial, 0.0)  # tell a value so the next ask sees n+1 completed trials
    print(f"    {n:>11} {times[0]*1000:>8.1f}ms {times[1]*1000:>8.1f}ms {times[2]*1000:>8.1f}ms")
print()

# ---- (b) running-trial constant-liar re-check ----
print("(b) Running-trial handling: LogEI imputes the best (max) completed value at running points.")
study = preloaded_study(30, seed=1)
ts = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
X = np.array([[t.params["x"] / 10.0] for t in ts])
y_raw = np.array([t.value for t in ts])
y = (y_raw - y_raw.mean()) / max(1e-10, np.std(y_raw))
gpr = gp.fit_kernel_params(
    X=X, Y=y, is_categorical=np.zeros(1, dtype=bool), log_prior=prior.default_log_prior,
    minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR, deterministic_objective=False, gpr_cache=None)
search_space = gp_search_space.SearchSpace({"x": DIST})
running = np.array([[0.42], [0.77]])  # two pretend running trials
acqf = LogEI(gpr, search_space, threshold=float(y.max()), normalized_params_of_running_trials=running)
imputed = acqf._gpr._y_all[-running.shape[0]:].detach().numpy()  # values LogEI assigned to running pts
print(f"    max completed (standardized) y      = {float(y.max()):.4f}")
print(f"    value LogEI assigned to the 2 running points = {np.array2string(imputed, precision=4)}")
print(f"    match best-value constant liar?     = {np.allclose(imputed, y.max())}")
print("    (source: optuna/_gp/acqf.py LogEI -> constant_liar_value = self._gpr._y_train.max())")
print()

# ---- (c) failure modes with bimodal values ----
print("(c) Failure modes with bimodal / degenerate completed values (captured warnings + does it run).")


def probe(label, values):
    # fit optuna's GP on a hand-built value list after GPSampler-style standardization; report issues
    X = np.linspace(0.05, 0.95, len(values)).reshape(-1, 1)
    v = np.array(values, dtype=float)
    means, stds = v.mean(), np.std(v)
    y = (v - means) / np.maximum(1e-10, stds)  # exactly optuna's _standardize_values denominator
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gpr = gp.fit_kernel_params(
            X=X, Y=y, is_categorical=np.zeros(1, dtype=bool), log_prior=prior.default_log_prior,
            minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR, deterministic_objective=False,
            gpr_cache=None)
        pm, pv = gpr.posterior(torch.from_numpy(np.array([0.5])))
    msgs = [str(w.message) for w in caught]
    print(f"    {label}")
    print(f"      raw std={stds:9.4f}  standardized y range=[{y.min():.2f},{y.max():.2f}]  "
          f"fitted noise_var={gpr.noise_var.item():.4f}")
    print(f"      posterior at x=0.5: mean={float(pm.item()):.3f} std={float(np.sqrt(pv.item())):.3f}  "
          f"warnings={msgs if msgs else 'none'}")


# extreme bimodality: values split hard between ~0 (failure) and ~80 (success)
probe("extreme bimodal (0/80 mix)", [0.5, 79.0, 1.2, 81.0, 0.3, 78.0, 2.1, 82.0, 0.9, 80.5])
# all-failures start: every completed trial is a near-0 failure (tiny but nonzero spread)
probe("all-failures (all ~0)", [0.4, 1.1, 0.2, 2.3, 0.7, 1.9, 0.1, 0.8, 1.5, 0.3])
# exactly constant: every value identical (std is exactly 0 -> denominator falls back to 1e-10)
probe("exactly constant (all 40.0)", [40.0] * 10)
print()

# full-sampler probe: run a real GPSampler where every completed value is exactly constant, to
# confirm the whole ask path (fit + acqf optimize) does not crash on a degenerate flat objective
print("    full-sampler probe: 25-trial GPSampler run, objective returns a constant 40.0 every time")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s = optuna.samplers.GPSampler(seed=7)
    st = optuna.create_study(direction="maximize", sampler=s)
    st.optimize(lambda t: 40.0 + 0.0 * t.suggest_float("x", 0.0, 10.0), n_trials=25)
    gp_msgs = [str(w.message) for w in caught if "kernel" in str(w.message).lower()]
print(f"      completed {len(st.trials)} trials, best value {st.best_value:.1f}, "
      f"kernel-fit warnings={gp_msgs if gp_msgs else 'none'}")
print()
print("Reading: standardization divides by np.maximum(1e-10, std). Extreme bimodality is handled")
print("(the spread just becomes fitted noise). The exactly-constant case is the edge: std=0 makes")
print("the standardized values 0/1e-10 = 0, so the GP sees a flat objective; the full sampler path")
print("completes all 25 trials with no crash and no kernel-fit warning.")
