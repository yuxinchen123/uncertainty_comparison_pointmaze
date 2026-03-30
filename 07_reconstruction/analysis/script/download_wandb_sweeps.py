#!/usr/bin/env python3
"""
Download Weights & Biases sweep runs for offline analysis.

Requires valid W&B credentials (e.g. `wandb login` or WANDB_API_KEY).

Default targets are the two sweeps documented in ../note.md:
  entity: catresearch (override with --entity or env WANDB_ENTITY)
  entity/project: catresearch/rnd_07_reconstruction
  sweep IDs: y24vyh06, j40bkl6y

Output layout under --out (default: analysis/data/ next to this script):

  data/
    <sweep_id>/
      manifest.json          # sweep path, download time, run ids
      runs_index.csv         # one row per run: ids, state, flat config & summary scalars
      runs/
        <run_id>/
          config.json
          summary.json
          history.jsonl      # only if --include-history (one JSON object per logged step)

Exports include every W&B run state. All analysis in ../analysis.md that aggregates metrics
assumes finished runs only (`state == finished`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

# -----------------------------------------------------------------------------
# Path defaults
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = ANALYSIS_ROOT / "data"


def _scalar(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def json_sanitize(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    return str(x)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten_dict(v, key))
        elif _scalar(v):
            out[key] = v
        else:
            out[key] = str(v)
    return out


def iter_history_rows(run: Any) -> Iterator[Dict[str, Any]]:
    """Yield metric rows using scan_history (memory-friendly for long runs)."""
    for row in run.scan_history():
        yield row


def download_sweep(
    api: Any,
    entity: str,
    project: str,
    sweep_id: str,
    out_root: Path,
    *,
    include_history: bool,
) -> None:
    sweep_path = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    sweep_dir = out_root / sweep_id
    runs_dir = sweep_dir / "runs"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "entity": entity,
        "project": project,
        "sweep_id": sweep_id,
        "sweep_path": sweep_path,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_history": include_history,
        "run_ids": [],
    }

    runs = list(sweep.runs)
    index_rows: List[Dict[str, Any]] = []
    all_keys: set[str] = set()

    for i, run in enumerate(runs):
        rid = run.id
        manifest["run_ids"].append(rid)
        run_sub = runs_dir / rid
        run_sub.mkdir(parents=True, exist_ok=True)

        cfg = dict(run.config) if run.config else {}
        summ = dict(run.summary or {})

        with open(run_sub / "config.json", "w", encoding="utf-8") as f:
            json.dump(json_sanitize(cfg), f, indent=2, sort_keys=True)
        with open(run_sub / "summary.json", "w", encoding="utf-8") as f:
            json.dump(json_sanitize(summ), f, indent=2, sort_keys=True)

        if include_history:
            hist_path = run_sub / "history.jsonl"
            with open(hist_path, "w", encoding="utf-8") as hf:
                for row in iter_history_rows(run):
                    if not row:
                        continue
                    safe = {str(k): json_sanitize(v) for k, v in row.items()}
                    hf.write(json.dumps(safe, sort_keys=True) + "\n")
        row_index: Dict[str, Any] = {
            "run_id": rid,
            "run_name": run.name,
            "state": getattr(run, "state", ""),
            "sweep_id": sweep_id,
            "url": run.url,
        }
        row_index.update({f"config.{k}": v for k, v in flatten_dict(cfg).items()})
        row_index.update({f"summary.{k}": v for k, v in flatten_dict(summ).items()})
        all_keys.update(row_index.keys())
        index_rows.append(row_index)

        if (i + 1) % 50 == 0:
            print(f"  [{sweep_id}] processed {i + 1}/{len(runs)} runs", file=sys.stderr)

    # Unify columns for runs_index.csv
    ordered_cols = ["run_id", "run_name", "state", "sweep_id", "url"]
    extra = sorted(k for k in all_keys if k not in ordered_cols)
    fieldnames_final = ordered_cols + extra

    index_path = sweep_dir / "runs_index.csv"
    with open(index_path, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames_final, extrasaction="ignore")
        w.writeheader()
        for r in index_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_final})

    with open(sweep_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(
        f"Done {sweep_path}: {len(runs)} runs -> {sweep_dir} "
        f"({'with' if include_history else 'no'} per-run history)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download W&B sweep runs to analysis/data/")
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "catresearch"),
        help="W&B entity/team (default: env WANDB_ENTITY or catresearch)",
    )
    parser.add_argument(
        "--project",
        default="rnd_07_reconstruction",
        help="W&B project name",
    )
    parser.add_argument(
        "--sweep-ids",
        nargs="+",
        default=["y24vyh06", "j40bkl6y"],
        help="Sweep IDs (short id from W&B sweep URL)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Output directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also download each run's time-series history.jsonl (much larger/slower)",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("Install wandb: pip install wandb", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)

    for sid in args.sweep_ids:
        download_sweep(
            api,
            args.entity,
            args.project,
            sid,
            args.out,
            include_history=args.include_history,
        )


if __name__ == "__main__":
    main()
