"""The fixed metric battery for the decay-rate experiments (program.md, "Metrics").

Everything here is computed from per-cell records of the shape written by run_experiment.py:

    record = {
      "point_set": str, "a_seed": int, "diverged": bool,
      "checkpoint_steps": [0, 1, 2, 3, ...],          # stored optimizer-step indices
      "bonus": [[b_i at step 0], [b_i at step 1], ...],  # one row per stored step, P columns
      "visit_counts": [[m_i at step 0], ...],         # same shape; m_i(n) = times point i trained
    }

The target curve at position i and checkpoint n is T_i(n) = min(1, m_i(n)^(-1/2)) (T = 1 where
m = 0). The per-position deviation is the RMS of log b - log T over in-window checkpoints.
FIXED harness module: the experiment loop never edits it.
"""
import numpy as np

from .fitting import BURN_IN, fit_power_floor

BONUS_FLOOR = 1e-12  # guards log(0) on exactly-converged curves; hits are counted and reported


def target_curve(visit_counts: np.ndarray) -> np.ndarray:
    """T = min(1, m^(-1/2)) elementwise, with T = 1 where m = 0 (the count oracle's bonus)."""
    # before: m = [[0, 0], [1, 1], [4, 2]]; after: T = [[1, 1], [1, 1], [0.5, 0.7071]]
    m = np.asarray(visit_counts, dtype=float)
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, np.where(m > 0, m, 1.0) ** -0.5)


def deviation_per_position(record: dict) -> np.ndarray:
    """Per-position RMS log-deviation from the target curve over in-window checkpoints.

    Returns a (P,) array: dev_i = sqrt(mean_n (log b_i(n) - log T_i(n))^2) for checkpoint steps
    n >= BURN_IN. Non-finite or non-positive bonuses make the affected positions inf."""
    steps = np.asarray(record["checkpoint_steps"])
    b = np.asarray(record["bonus"], dtype=float)          # (T, P)
    t = target_curve(record["visit_counts"])              # (T, P)
    win = steps >= BURN_IN
    # log-space residual per (checkpoint, position); bad values (<= 0, nan, inf) poison their
    # position with inf rather than being silently clipped into a good-looking score
    bw, tw = b[win], t[win]
    bad = ~np.isfinite(bw) | (bw <= 0)
    r = np.log(np.maximum(bw, BONUS_FLOOR)) - np.log(tw)
    dev = np.sqrt((r ** 2).mean(axis=0))
    dev[bad.any(axis=0)] = np.inf
    return dev


def start_deviation(record: dict) -> np.ndarray:
    """Per-position |log b_i(0)| — the distance of the initial bonus from 1, per position."""
    b0 = np.asarray(record["bonus"], dtype=float)[0]
    out = np.abs(np.log(np.maximum(b0, BONUS_FLOOR)))
    out[~np.isfinite(b0) | (b0 <= 0)] = np.inf
    return out


def compute_metrics(records: list, fit_slopes: bool = True) -> dict:
    """The full metric battery over a list of per-cell records (all point sets together).

    Ranking metrics (program.md): dev_worst (primary), dev_mean, start_dev; plus the slope
    battery from per-position fits on seed-mean curves, per point set and pooled."""
    live = [r for r in records if not r.get("diverged")]
    n_diverged = len(records) - len(live)
    if not live:
        return {"dev_worst": float("inf"), "dev_mean": float("inf"),
                "start_dev": float("inf"), "n_diverged": n_diverged, "n_records": len(records)}

    # per-record (seed x point-set cell) deviation vectors; the worst position is taken within
    # each record, then averaged over records — "mean over seeds of the max over positions"
    devs = [deviation_per_position(r) for r in live]
    starts = [start_deviation(r) for r in live]
    dev_worst = float(np.mean([d.max() for d in devs]))
    dev_mean = float(np.mean(np.concatenate(devs)))
    dev_p95 = float(np.mean([np.quantile(d, 0.95) for d in devs]))
    start_all = np.concatenate(starts)
    start_dev = float(np.sqrt((start_all ** 2).mean()))
    start_max = float(start_all.max())
    floor_hits = int(sum((np.asarray(r["bonus"], dtype=float) <= BONUS_FLOOR).sum() for r in live))

    out = {"dev_worst": dev_worst, "dev_mean": dev_mean, "dev_p95": dev_p95,
           "start_dev": start_dev, "start_max": start_max,
           "n_diverged": n_diverged, "n_records": len(records), "floor_hits": floor_hits,
           "per_env": {}}

    # slope battery per point set: average the normalized-by-nothing bonus curves over seeds
    # (curves are the method's own readout; no extra normalization here), fit each position's
    # seed-mean curve, and fit the grand-mean curve for the aggregate slope
    for env in sorted({r["point_set"] for r in live}):
        env_recs = [r for r in live if r["point_set"] == env]
        steps = np.asarray(env_recs[0]["checkpoint_steps"])
        # before: bonus arrays of shape (T, P) per seed; after: seed-mean array (T, P)
        b_mean = np.mean([np.asarray(r["bonus"], dtype=float) for r in env_recs], axis=0)
        env_out = {"n_seeds": len(env_recs)}
        if fit_slopes and np.isfinite(b_mean).all() and (b_mean > 0).all():
            slopes = np.array([fit_power_floor(steps, b_mean[:, i])["slope"]
                               for i in range(b_mean.shape[1])])
            agg = fit_power_floor(steps, b_mean.mean(axis=1))
            env_out.update({
                "slope_mean": float(slopes.mean()), "slope_std": float(slopes.std()),
                "slope_min": float(slopes.min()), "slope_max": float(slopes.max()),
                "slope_absdev_mean": float(np.abs(slopes + 0.5).mean()),
                "agg_slope": agg["slope"], "agg_floor": agg["c"],
                "per_position_slopes": slopes.tolist(),
            })
        out["per_env"][env] = env_out

    # pooled slope summary over every fitted point set (what results.tsv logs)
    pooled = [s for e in out["per_env"].values() for s in e.get("per_position_slopes", [])]
    if pooled:
        pooled = np.array(pooled)
        out["slope_mean"] = float(pooled.mean())
        out["slope_std"] = float(pooled.std())
        out["slope_min"] = float(pooled.min())
        out["slope_max"] = float(pooled.max())
    return out


def summary_block(metrics: dict, runtime_seconds: float) -> str:
    """The printed end-of-experiment summary (the autoresearch-style '---' block)."""
    lines = ["---"]
    for k in ("dev_worst", "dev_mean", "dev_p95", "start_dev", "start_max",
              "slope_mean", "slope_std", "slope_min", "slope_max"):
        if k in metrics:
            lines.append(f"{k}: {metrics[k]:.6f}")
    for env, e in metrics.get("per_env", {}).items():
        if "agg_slope" in e:
            lines.append(f"agg_slope[{env}]: {e['agg_slope']:.4f}")
    lines.append(f"n_diverged: {metrics.get('n_diverged', 0)}")
    lines.append(f"floor_hits: {metrics.get('floor_hits', 0)}")
    lines.append(f"runtime_seconds: {runtime_seconds:.1f}")
    return "\n".join(lines)
