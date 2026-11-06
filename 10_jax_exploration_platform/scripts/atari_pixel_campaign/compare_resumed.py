"""Compare an uninterrupted run folder with a stopped-and-resumed one: state and metrics, bit for bit."""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np


def leaves(value):
    """Every array of the saved state with its tree path (JAX structures included)."""
    import jax
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]}


uninterrupted, resumed, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
states = [pickle.load(open(folder / "checkpoint.pkl", "rb"))["state"] for folder in (uninterrupted, resumed)]
first, second = leaves(states[0]), leaves(states[1])
differing = [name for name in first if first[name].shape != second[name].shape
             or not np.array_equal(first[name], second[name], equal_nan=True)]
metrics = [[json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()] for folder in (uninterrupted, resumed)]
strip = lambda rows: [{key: value for key, value in row.items() if key != "unix"} for row in rows]
record = {"state_leaves": len(first), "differing_state_leaves": differing,
          "metric_updates": [len(rows) for rows in metrics],
          "metrics_identical": strip(metrics[0]) == strip(metrics[1]),
          "passed": not differing and strip(metrics[0]) == strip(metrics[1])}
output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
