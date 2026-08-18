"""Classify a configuration's learning curve as rise-then-fall, or say what else it is.

The question this answers: aggregated over all of a configuration's copies, does its extrinsic
reward curve go UP and then come back DOWN? A curve that rises and then merely flattens is not a
rise-then-fall, and neither is one still climbing at the end of the budget; the fall has to be
larger than the noise on the curve and a real fraction of what the rise gained.

Every threshold below is stated as a fraction of the run, not as a number of windows, so the same
classifier applies to a 98-window run and a 9,766-window run without retuning.
"""
import numpy as np

# every threshold in one place, so the sensitivity check can vary them
DEFAULTS = dict(
    smooth_fraction=0.10,      # centred moving average width, as a fraction of the windows
    edge_fraction=0.05,        # a peak this close to either end is not an interior peak
    noise_multiple=3.0,        # a rise or fall must exceed this many standard errors
    fall_fraction=0.05,        # ... and the fall must be at least this fraction of the rise
    trend_correlation=0.5,     # reported on each side of the peak; diagnostic, never a gate
    min_peak=1e-3,             # mean episode return below which the goal was never really reached
)


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, shrinking the window at the two ends rather than padding."""
    if width <= 1:
        return y.astype(float)
    out = np.empty(len(y), dtype=float)
    half = width // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def _spearman(y: np.ndarray) -> float:
    """Rank correlation of y against its own index: +1 rising throughout, -1 falling throughout."""
    n = len(y)
    if n < 3:
        return 0.0
    rank_y = np.argsort(np.argsort(y)).astype(float)
    rank_x = np.arange(n, dtype=float)
    ry, rx = rank_y - rank_y.mean(), rank_x - rank_x.mean()
    denominator = np.sqrt((ry * ry).sum() * (rx * rx).sum())
    return float((ry * rx).sum() / denominator) if denominator > 0 else 0.0


def classify(means, errors, **overrides) -> dict:
    """Label one aggregated curve. `means`/`errors` are per window, over the configuration's copies.

    before: 9,766 window means with their standard errors over 128 copies;
    after:  one label of six, with the peak's position and the rise and fall that produced it
    """
    settings = {**DEFAULTS, **overrides}
    y, se = np.asarray(means, dtype=float), np.asarray(errors, dtype=float)
    windows = len(y)
    if windows < 10:
        return {"shape": "too_short", "windows": windows}

    width = max(3, int(round(settings["smooth_fraction"] * windows)))
    edge = max(1, int(round(settings["edge_fraction"] * windows)))
    smooth = _moving_average(y, width)

    # the noise on a smoothed point: the typical standard error over copies, reduced by the
    # averaging. Using the median keeps one loud window from setting the scale for the whole curve.
    noise = float(np.median(se)) / np.sqrt(width) if len(se) else 0.0
    threshold = settings["noise_multiple"] * noise

    # A configuration whose best window still averages under `min_peak` reward per episode never
    # learned to reach the goal: the goal is worth about one per episode, so this is under one
    # episode in a thousand. There is no shape in that curve, and the standard error over copies
    # collapses to zero when every copy is identically zero, which leaves the noise threshold at
    # zero and lets floating-point dust be read as a trend. Answer before that can happen.
    if float(np.max(_moving_average(y, width))) < settings["min_peak"]:
        return {"shape": "no_learning", "windows": windows, "smooth_width": width,
                "peak_value": float(np.max(_moving_average(y, width))), "rise": 0.0, "fall": 0.0,
                "interior_peak": False, "rise_significant": False, "fall_significant": False,
                "separate_peaks": 0, "multi_peaked": False}

    peak = int(np.argmax(smooth))
    rise = float(smooth[peak] - smooth[0])
    fall = float(smooth[peak] - smooth[-1])
    interior = edge <= peak <= windows - 1 - edge
    rise_real = rise > threshold
    fall_real = fall > threshold and fall >= settings["fall_fraction"] * max(rise, 1e-12)

    # the two trend correlations are REPORTED, never gated on: on a noisy curve a real rise-then-
    # fall scores well below any fixed threshold, so gating on them mislabels shape as noise
    before = _spearman(smooth[: peak + 1]) if peak >= 2 else 0.0
    after = _spearman(smooth[peak:]) if windows - peak >= 3 else 0.0

    # multi-modality, as an annotation: interior maxima that come back within one threshold of the
    # global peak after dipping more than a threshold below it. Two or more means the curve does
    # not simply go up and come down, whatever its endpoints do.
    peaks, below = 0, True
    for value in smooth:
        if value >= smooth[peak] - threshold and below:
            peaks, below = peaks + 1, False
        elif value < smooth[peak] - 2 * threshold:
            below = True

    if not rise_real and not fall_real:
        shape = "flat"
    elif interior and rise_real and fall_real:
        shape = "rise_then_fall"
    elif rise_real and not fall_real:
        shape = "still_rising" if peak > windows - 1 - edge else "rise_then_plateau"
    else:
        shape = "declining"

    return {"shape": shape, "windows": windows, "smooth_width": width, "edge_windows": edge,
            "peak_window": peak, "peak_fraction": peak / (windows - 1),
            "peak_value": float(smooth[peak]), "first_value": float(smooth[0]),
            "last_value": float(smooth[-1]), "rise": rise, "fall": fall,
            "noise_threshold": threshold, "interior_peak": interior,
            "rise_significant": rise_real, "fall_significant": fall_real,
            "trend_before_peak": before, "trend_after_peak": after,
            "separate_peaks": peaks, "multi_peaked": peaks >= 2}


def sensitivity(means, errors) -> dict:
    """Re-label the same curve under looser and tighter thresholds; a label that survives is stable."""
    variants = {
        "baseline": {},
        "smoother": {"smooth_fraction": 0.10},
        "rougher": {"smooth_fraction": 0.05},
        "strict_noise": {"noise_multiple": 5.0},
        "loose_noise": {"noise_multiple": 2.0},
        "strict_fall": {"fall_fraction": 0.10},
        "very_smooth": {"smooth_fraction": 0.20},
        "wide_edge": {"edge_fraction": 0.10},
        "strict_signal": {"min_peak": 1e-2},
    }
    labels = {name: classify(means, errors, **kw)["shape"] for name, kw in variants.items()}
    return {"labels": labels, "stable": len(set(labels.values())) == 1}


def copy_level_agreement(per_copy_curves, **overrides) -> dict:
    """How many individual copies share the aggregate's label — the averaging caveat, measured."""
    labels = [classify(curve, np.zeros(len(curve)) + 1e-12, **overrides)["shape"]
              for curve in per_copy_curves]
    counts = {label: labels.count(label) for label in set(labels)}
    return {"copies": len(labels), "counts": counts}
