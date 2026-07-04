# constant_liar spreads a parallel batch: 8 asks with no results reported, with vs without it
import hashlib, itertools
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.distributions import FloatDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)

PEAK = np.array([-6., -2.])
DIST = {"r": FloatDistribution(-6, -2), "b": FloatDistribution(-3, -1)}

def substream(base, *parts):                       # one RNG per named quantity, hashed key
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)

def bump(r, b):                                    # noiseless signal peaked at PEAK, width 1 decade
    return 50.0 * np.exp(-((r + 6) ** 2 + (b + 2) ** 2) / 2.0)

def batch(constant_liar):                          # seed 30 completed trials, then ask 8 (RUNNING)
    sampler = TPESampler(seed=0, constant_liar=constant_liar)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for i in range(30):
        r = substream(0, "r", i).uniform(-6, -2)   # per-quantity seeding, not one shared stream
        b = substream(0, "b", i).uniform(-3, -1)
        noise = substream(0, "noise", i).normal(0, 2)
        study.add_trial(optuna.trial.create_trial(
            params={"r": r, "b": b}, distributions=DIST, value=float(bump(r, b) + noise)))
    pts = []
    for _ in range(8):                             # ask() leaves each trial RUNNING (no result told)
        t = study.ask(DIST)
        pts.append([t.params["r"], t.params["b"]])
    return np.array(pts)

def spread(pts):                                   # mean pairwise distance across the 8 points
    return float(np.mean([np.hypot(*(a - b)) for a, b in itertools.combinations(pts, 2)]))

off, on = batch(False), batch(True)
print(f"constant_liar=False  mean pairwise distance: {spread(off):.3f}  (clustered)")
print(f"constant_liar=True   mean pairwise distance: {spread(on):.3f}  (spread out)")
