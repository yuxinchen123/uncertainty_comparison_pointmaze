#!/usr/bin/env python3
"""
Download a slim, analysis-ready table for one W&B sweep (default y24vyh06,
"run 1 before reorganization": PointMaze_Large-v3, goal=bottom_right).

Only the columns the train-run-1 aggregate needs are pulled, so the result is a
small CSV (no per-run config/summary files, no heatmap-media columns). Uses the
server-side `sweep` filter via `api.runs(...)`, which returns each run's summary
in the page response, so there is no per-run round trip.

Output (under --out, default: this folder's ../data/):
  <sweep_id>/runs_slim.csv   one row per run: ids, state, config, summary scalars
  <sweep_id>/manifest.json   sweep path, download time, run count, state counts

Requires valid W&B credentials (wandb login / WANDB_API_KEY).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# This script's own folder, so --out defaults next to code/ -> ../data/.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data"

# Config keys copied verbatim from each run's config dict.
CONFIG_KEYS = [
    "algorithm",
    "beta",
    "a_seed",
    "goal_position",
    "total_timesteps",
    "env_max_episode",
    "discount_factor",
    "env_name",
    "apply_termination_wrapper",
    "rnd_distance",
    "rnd_obs_norm",
    "rnd_output_dim",
    "n_predictors",
]
# Summary scalars kept for the aggregate (train + eval reward, step, distances).
SUMMARY_KEYS = [
    "step",
    "train/mean_extrinsic_reward",
    "train/mean_intrinsic_reward",
    "eval/mean_extrinsic_reward",
    "eval/mean_intrinsic_reward",
    "eval/mean_ep_length",
    "eval/n_eval_episodes",
    "distance_to_gt/min_c_l1_diff",
    "distance_to_gt/min_c_l2_diff",
    "distance_to_gt/min_c_l1_inv",
    "distance_to_gt/min_c_l2_inv",
    "distance_to_gt/normalized_l2",
    "distance_to_gt/normalized_angle_rad",
]


def scalar_or_blank(value: Any) -> Any:
    """Return W&B config/summary values as CSV-safe scalars (blank for non-scalars)."""
    # before: value may be a dict (e.g. {"desc":..., "value":...}) or media ref
    # after: a plain scalar, or "" for anything non-scalar
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return ""


def download_slim(entity: str, project: str, sweep_id: str, out_root: Path) -> None:
    """Pull the sweep's runs into a slim CSV plus a manifest with state counts."""
    import wandb

    api = wandb.Api(timeout=180)

    # Server-side filter to this sweep; summary metrics come back in the page, so
    # iterating does not trigger a per-run fetch. per_page caps the page size.
    runs = api.runs(
        f"{entity}/{project}",
        filters={"sweep": sweep_id},
        per_page=500,
    )

    out_dir = out_root / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "runs_slim.csv"

    fieldnames = (
        ["run_id", "run_name", "state", "sweep_id", "url"]
        + [f"config.{k}" for k in CONFIG_KEYS]
        + [f"summary.{k}" for k in SUMMARY_KEYS]
    )

    state_counts: Counter[str] = Counter()
    n = 0
    # Stream rows straight to disk so an 8k-run sweep never sits fully in memory.
    with open(csv_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            cfg = dict(run.config or {})
            summ = dict(run.summary or {})
            row: Dict[str, Any] = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "sweep_id": sweep_id,
                "url": run.url,
            }
            row.update({f"config.{k}": scalar_or_blank(cfg.get(k)) for k in CONFIG_KEYS})
            row.update({f"summary.{k}": scalar_or_blank(summ.get(k)) for k in SUMMARY_KEYS})
            writer.writerow(row)
            state_counts[run.state] += 1
            n += 1
            if n % 250 == 0:
                print(f"  [{sweep_id}] {n} runs ...", file=sys.stderr, flush=True)

    manifest = {
        "entity": entity,
        "project": project,
        "sweep_id": sweep_id,
        "sweep_path": f"{entity}/{project}/{sweep_id}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": n,
        "state_counts": dict(state_counts),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Done {sweep_id}: {n} runs -> {csv_path}", file=sys.stderr)
    print(f"  state counts: {dict(state_counts)}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Slim per-run download for one W&B sweep.")
    p.add_argument("--entity", default="catresearch")
    p.add_argument("--project", default="rnd_07_reconstruction")
    p.add_argument("--sweep-id", default="y24vyh06")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_ROOT)
    args = p.parse_args()

    download_slim(args.entity, args.project, args.sweep_id, args.out)


if __name__ == "__main__":
    main()
