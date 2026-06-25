#!/usr/bin/env python3
"""Shared constants for the train-run-1 aggregate: paths, metric, algorithm colors."""
from __future__ import annotations

from pathlib import Path

import matplotlib

CODE_DIR = Path(__file__).resolve().parent
FOLDER = CODE_DIR.parent
SWEEP_DIR = FOLDER / "data" / "y24vyh06"
PLOTS_DIR = FOLDER / "plots"
# Paper root: <development_document>/main.tex, two levels above this analysis folder
# (FOLDER -> train_run_aggregate -> development_document).
MAIN_TEX = FOLDER.parent.parent / "main.tex"

# Primary metric: final eval extrinsic reward (10000-episode eval at 2M steps).
METRIC = "eval/mean_extrinsic_reward"

# Fixed left-to-right / legend order: 10 sweep algorithms grouped by family
# (baseline, count oracles, RND variants, elliptical). Plots and tables follow it.
ALGORITHMS = [
    "no_exploration",
    "gt_position",
    "gt_position_velocity",
    "rnd_next_state",
    "rnd_next_state_position_only",
    "rnd_state",
    "rnd_state_action",
    "rnd_state_action_next_state",
    "rnd_linear_next_state",
    "rnd_elliptical",
]

# One stable color per algorithm (tab10), so the bar plot and the line plot agree.
_TAB10 = matplotlib.colormaps["tab10"]
ALGO_COLOR = {a: _TAB10(i) for i, a in enumerate(ALGORITHMS)}
