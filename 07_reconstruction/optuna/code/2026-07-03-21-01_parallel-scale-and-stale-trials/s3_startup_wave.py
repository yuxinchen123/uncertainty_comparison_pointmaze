"""Verify that a wave of asks with zero tells is pure random search under TPESampler.

TPESampler counts COMPLETED trials against n_startup_trials. If nothing has been
told yet, every ask falls back to the sampler's internal RandomSampler. We ask 40
trials with no tells in between and compare the points to a standalone
RandomSampler(seed=0). This mirrors a sweep launch where hundreds of workers all
ask before any result comes back.
"""

import numpy as np

import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


# Suggest two floats on a trial and return them as a pair.
def suggest_point(trial):
    # a small 2-D search space, same for both samplers
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x, y)


# Ask n trials from a study WITHOUT telling any result; collect the suggested points.
def ask_without_telling(sampler, n):
    # in-memory study; ask n trials back-to-back, never call tell
    study = optuna.create_study(sampler=sampler, direction="minimize")
    points = []
    for _ in range(n):
        trial = study.ask()
        points.append(suggest_point(trial))
    return np.array(points), study


def main():
    n = 40
    # TPESampler with a startup threshold of 10, but we never complete a trial
    tpe_points, tpe_study = ask_without_telling(
        optuna.samplers.TPESampler(seed=0, n_startup_trials=10, constant_liar=False), n
    )
    # a standalone RandomSampler with the same seed
    rnd_points, _ = ask_without_telling(optuna.samplers.RandomSampler(seed=0), n)

    # how many of the 40 TPE asks exactly match the RandomSampler asks?
    identical = np.all(np.isclose(tpe_points, rnd_points, rtol=0, atol=0), axis=1)
    n_identical = int(identical.sum())
    print(f"TPE asks that are bit-identical to RandomSampler(seed=0): {n_identical} / {n}")
    print(f"first mismatch index (or -1 if none): {int(np.argmin(identical)) if n_identical < n else -1}")

    # states of all 40 TPE-study trials: they never completed, so all RUNNING
    states = {}
    for t in tpe_study.get_trials(deepcopy=False):
        states[t.state.name] = states.get(t.state.name, 0) + 1
    print(f"TPE-study trial states after 40 asks, 0 tells: {states}")

    # show the first 3 and last 3 asked points from each, to eyeball that 11..40 are still random
    print("\nidx |    TPE (x, y)         |   Random (x, y)")
    for i in list(range(3)) + list(range(n - 3, n)):
        tx, ty = tpe_points[i]
        rx, ry = rnd_points[i]
        print(f"{i:3d} | ({tx:7.4f}, {ty:7.4f}) | ({rx:7.4f}, {ry:7.4f})")

    # a crude spread check: std of asks 11..40 (0-indexed 10..39) is not collapsed
    tail = tpe_points[10:]
    print(f"\nstd of TPE asks 11..40 on x = {tail[:,0].std():.3f}, on y = {tail[:,1].std():.3f} "
          f"(full range is 10 wide; a fitted model would concentrate these)")


if __name__ == "__main__":
    main()
