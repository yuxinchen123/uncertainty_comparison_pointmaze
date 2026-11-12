"""Where the time of one AntMaze environment step goes on a graphics card, piece by piece.

One environment step is frame_skip (5) MJX physics steps plus the reward, the reset and the
observation. This script times cumulative prefixes of the physics step. Each prefix is its own
compiled function that returns every array it computes, so the difference between two
neighbouring prefixes is the cost of the stages that were added:

  1  kinematics, centre of mass, composite inertia, mass-matrix factor
  2  plus collision detection
  3  plus constraint build
  4  plus velocity, actuation and acceleration
  5  plus the constraint solve            (the stages of mjx.forward)
  6  plus the integration                 (the stages of mjx.step)
  7  the environment's own step           (adds reward, reset, observation)

How one timed call is built:

- The start state is reached by 100 environment steps of uniform random torques from the spawn,
  so the ants touch the floor and the walls.
- One call runs 10 environment steps: a scan over environment steps around a scan over the 5
  frames, and inside each frame the prefix is batched over the environments with jax.vmap, the
  same way the environment batches mjx.step.
- Prefixes 1 to 6 are given, at every frame, the positions, velocities and solver start values
  recorded from the real physics step at that frame (same start state, same torques). The work is
  then done on the states the real step sees, and no result repeats from frame to frame, so the
  compiler cannot move the work out of the loop.
- Feeding the recorded states costs a little. It is measured by also timing mjx.step advancing
  its own state: prefix 6 minus that time is the feeding cost, which is taken off piece 1, and
  piece 7 is the environment step minus that time. The seven pieces then sum to the environment
  step exactly.
- Each function is called once (tracing, compilation, one execution), then twice more to warm
  up, then timed in rounds that go through all functions of the cell in turn. The reported time
  is the median over the rounds.
- The functions of a split cell are then timed a second time back to back (two warm-up calls,
  then all rounds of one function in a row), the order in which a training run calls the step.
  The split table uses the back-to-back times when a cell has them.

Other cells time the whole environment step only: the Large map, solver and contact settings
(at 4,096 environments, or at the sizes of --variation-batches), wider batches each in its own
cell (part "wide") or all sizes timed in turn in one cell (part "batches"), the frame loop
unrolled, and one compiler option.

Usage (on a machine with a graphics card, never on a login node):
  PYTHONNOUSERSITE=1 PYTHONPATH=<platform>/src python profile_antmaze_physics_split.py \
      --out <folder>/antmaze_physics_split.json --machine "serval02 card 1" --unit <unit name>
Rebuild the Markdown tables from the record (standard library only, any machine):
  python profile_antmaze_physics_split.py --out <folder>/antmaze_physics_split.json --render
"""
import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

BASE_SEED = 0
SETTLE_ENV_STEPS = 100      # random-torque environment steps before the start state is taken
ENV_STEPS_PER_CALL = 10     # environment steps one timed call runs
MIN_ROUNDS = 11             # a cell with fewer timed rounds is reported as not measured
NO_GPU_EXIT_CODE = 76       # JAX started without the graphics card

PIECES = (
    "kinematics, centre of mass, composite inertia, mass-matrix factor",
    "collision detection",
    "constraint build",
    "velocity, actuation and acceleration",
    "constraint solve",
    "integration",
    "reward, reset and observation",
)
PHYSICS_OWN_STATE = "physics step advancing its own state"
ENVIRONMENT_STEP = "environment step"
DEFAULT_SETTINGS = "default: 4 solver iterations, 8 line-search iterations, 32 contact points"
VARIATIONS = (
    (DEFAULT_SETTINGS, {}),
    ("solver iterations 1", {"solver_iterations": 1}),
    ("solver iterations 2", {"solver_iterations": 2}),
    ("line-search iterations 4", {"ls_iterations": 4}),
    ("contact points 16", {"max_contact_points": 16}),
)
FRAME_LOOP_UNROLLED = "frame loop unrolled (unroll 5)"
COMPILER_OPTION = "compiler option xla_gpu_graph_min_graph_size=1"
# seconds a cell is expected to take before any cell of the unit has been timed
FIRST_GUESS_SECONDS = {"split": 240.0, "large map": 90.0, "variations": 200.0, "wide batch": 100.0,
                       "batch sizes": 200.0, "unroll and compiler option": 240.0}


# ----------------------------------------------------------------------------------------------
# arithmetic of the tables (standard library only)
# ----------------------------------------------------------------------------------------------

def split_rows(prefix_ms, physics_ms, env_ms):
    """Per piece of one environment step: the prefix time, the time the piece adds, its share."""
    # GOAL: turn the six prefix times and the two whole-step times into seven pieces that sum to
    # the environment step.
    # before: prefix_ms = [2.0, 5.0, 6.0, 8.0, 15.0, 16.0]   (prefixes 1-6 on recorded states)
    #         physics_ms = 15.5   (mjx.step advancing its own state)
    #         env_ms = 16.0       (the environment's own step)
    # after:  feeding recorded states costs 16.0 - 15.5 = 0.5 ms, so the pieces add
    #         [1.5, 3.0, 1.0, 2.0, 7.0, 1.0, 0.5] ms, which sum to env_ms = 16.0
    feed_ms = prefix_ms[5] - physics_ms
    added = ([prefix_ms[0] - feed_ms] + [b - a for a, b in zip(prefix_ms, prefix_ms[1:])]
             + [env_ms - physics_ms])
    rows = [{"piece": piece, "prefix_ms": prefix, "added_ms": add, "share": add / env_ms}
            for piece, prefix, add in zip(PIECES, list(prefix_ms) + [env_ms], added)]
    return rows, feed_ms


def two_point_line(value_small, value_large, n_small, n_large):
    """The line through two (batch size, time) points: (time at batch 0, time per 1,000 environments)."""
    # before: 8.0 ms at 1,024 environments and 17.6 ms at 4,096
    # after:  slope 9.6 / 3,072 = 0.003125 ms per environment -> (4.8 ms fixed, 3.125 ms per 1,000)
    slope = (value_large - value_small) / (n_large - n_small)
    return value_small - slope * n_small, 1000.0 * slope


def throughput_row(n_envs, seconds_per_env_step):
    """The five throughput columns for one batch size."""
    per_env = 1.0 / seconds_per_env_step
    return {"environments": n_envs,
            "seconds_per_environment_step": seconds_per_env_step,
            "total_steps_per_second_millions": n_envs * per_env / 1e6,
            "steps_per_second_per_environment": per_env,
            "hours_per_million_steps_per_environment": 1e6 / (3600.0 * per_env)}


def solver_arithmetic(default_ms, one_iteration_ms, two_iterations_ms, four_line_search_ms):
    """What the settings rows say about the solver, in milliseconds per environment step."""
    # before: default (4 solver iterations, 8 line-search iterations) 17.8 ms, 1 solver iteration 8.0,
    #         2 solver iterations 11.8, 4 line-search iterations 14.1
    # after:  one iteration inside the loop (17.8 - 11.8) / 2 = 3.0; outside the iterations
    #         11.8 - 2 x 3.0 = 5.8; one pass without the loop 8.0 - 5.8 = 2.2; the loop form adds
    #         3.0 - 2.2 = 0.8 per iteration; the 8 line-search iterations of the 4 solver
    #         iterations 2 x (17.8 - 14.1) = 7.4
    in_loop = (default_ms - two_iterations_ms) / 2.0
    outside = two_iterations_ms - 2.0 * in_loop
    without_loop = one_iteration_ms - outside
    return {"one solver iteration inside the loop": in_loop,
            "one solver pass without the loop": without_loop,
            "added per iteration by the loop form": in_loop - without_loop,
            "the 8 line-search iterations of all 4 solver iterations": 2.0 * (default_ms - four_line_search_ms),
            "everything outside the solver iterations": outside}


TIMING_KEYS = {"in turn": "call_seconds", "back to back": "call_seconds_back_to_back"}


def median_ms(cell, function_name, order="in turn"):
    """Median milliseconds per environment step of one timed function of a cell, in one timing order."""
    seconds = cell["functions"][function_name][TIMING_KEYS[order]]
    return 1000.0 * statistics.median(seconds) / cell["env_steps_per_call"]


def split_order(cell):
    """The timing order a split cell is read in: back to back when every function has it, else in turn."""
    key = TIMING_KEYS["back to back"]
    complete = all(len(function.get(key, [])) >= MIN_ROUNDS for function in cell["functions"].values())
    return "back to back" if complete else "in turn"


def usable_cells(record):
    """The latest usable cell per name: finished, enough rounds, the card not shared, run kept."""
    chosen = {}
    for run in record["runs"]:
        if run.get("discarded"):
            continue
        for cell in run["cells"]:
            if (cell["status"] == "measured" and cell["rounds_done"] >= MIN_ROUNDS
                    and not cell["other_processes_on_card"]):
                chosen[cell["name"]] = dict(cell, machine=run["machine"], unit=run["unit"])
    return chosen


def unusable_cells(record):
    """Every cell that is not used in the tables, with the reason, in the order it was recorded."""
    rows, chosen = [], usable_cells(record)
    for run in record["runs"]:
        for cell in run["cells"]:
            if run.get("discarded"):
                reason = f"run discarded: {run['discarded']}"
            elif cell["status"] == "not measured":
                reason = cell["reason"]
            elif cell["status"] != "measured":
                reason = f"the unit ended during this cell after {cell.get('rounds_done', 0)} rounds"
            elif cell["rounds_done"] < MIN_ROUNDS:
                reason = f"only {cell['rounds_done']} rounds"
            elif cell["other_processes_on_card"]:
                reason = f"another process was on the card: {cell['other_processes_on_card']}"
            elif chosen[cell["name"]]["unit"] != run["unit"]:
                reason = f"replaced by the later measurement of this cell in unit `{chosen[cell['name']]['unit']}`"
            else:
                continue
            rows.append((cell["name"], run["unit"], reason))
    return rows


# ----------------------------------------------------------------------------------------------
# the Markdown tables (standard library only)
# ----------------------------------------------------------------------------------------------

def split_name(map_name, n_envs):
    """Name of the cell that splits the physics step at one batch size."""
    return f"split of the physics step, {map_name}, {n_envs:,} environments"


def wide_name(n_envs):
    """Name of the cell that times the environment step at one wide batch."""
    return f"wide batch, umaze, {n_envs:,} environments"


def variations_name(n_envs):
    """Name of the cell that times the solver and contact settings at one batch size."""
    return f"variations of the environment step, umaze, {n_envs:,} environments"


LARGE_MAP_NAME = "large map, environment step"
BATCH_SIZES_NAME = "batch sizes, umaze, whole environment step"
UNROLL_NAME = "frame loop unrolled and compiler option, umaze, 4,096 environments"


def where_and_when(cell):
    """One phrase saying on which machine and at what Pacific time a cell was measured."""
    return f"{cell['machine']}, {cell['measured_pacific']}"


def render_split(cells, sizes):
    """The split table at the two batch sizes, the growth table, and their notes."""
    lines = ["## The physics step, piece by piece (U-maze)", ""]
    have = [n for n in sizes if split_name("umaze", n) in cells]
    if not have:
        return lines + ["Not measured.", ""]
    # GOAL: per batch size, the seven pieces from the eight timed functions of the cell.
    tables, feeds, firsts, orders = {}, {}, {}, {}
    for n in have:
        cell = cells[split_name("umaze", n)]
        orders[n] = split_order(cell)
        prefix_ms = [median_ms(cell, f"stages through: {piece}", orders[n]) for piece in PIECES[:6]]
        tables[n], feeds[n] = split_rows(prefix_ms, median_ms(cell, PHYSICS_OWN_STATE, orders[n]),
                                         median_ms(cell, ENVIRONMENT_STEP, orders[n]))
        names = [f"stages through: {piece}" for piece in PIECES[:6]] + [ENVIRONMENT_STEP]
        firsts[n] = [cell["functions"][name]["first_call_seconds"] for name in names]
    lines += [
        "Define $t_k(n)$ as the median milliseconds per environment step of the function that runs"
        " pieces 1 to $k$ at $n$ environments (the column \"through this piece\"), $c(n)$ as the"
        " cost of feeding recorded states, and $p(n)$ as the time of the physics step advancing its"
        " own state.",
        "",
        "- Added by piece $k$ for $k$ from 2 to 6: $a_k(n) = t_k(n) - t_{k-1}(n)$.",
        "- Piece 1: $a_1(n) = t_1(n) - c(n)$, with $c(n) = t_6(n) - p(n)$.",
        "- Piece 7: $a_7(n) = t_7(n) - p(n)$, where $t_7(n)$ is the environment's own step.",
        "- Share of piece $k$: $\\dfrac{a_k(n)}{t_7(n)}$. The seven shares sum to 100%.",
        "- First call: seconds of the first call of the function (tracing, compilation and one"
        " execution of 10 environment steps).",
        "",
    ]
    head = "| piece |" + "".join(f" {n:,} env.:<br>through this<br>piece (ms) | added<br>(ms) | share |"
                                for n in have) + " first call (s):<br>" + "; ".join(f"{n:,}" for n in have) + " |"
    lines += [head, "|---|" + "---:|---:|---:|" * len(have) + "---:|"]
    for k, piece in enumerate(PIECES):
        row = f"| {k + 1}. {piece} |"
        for n in have:
            r = tables[n][k]
            row += f" {r['prefix_ms']:.3f} | {r['added_ms']:.3f} | {100 * r['share']:.1f}% |"
        lines.append(row + " " + "; ".join(f"{firsts[n][k]:.1f}" for n in have) + " |")
    lines.append("")
    # GOAL: the facts behind the table, one bullet per batch size.
    for n in have:
        cell = cells[split_name("umaze", n)]
        facts, check = cell["start_state"], cell.get("composition_check")
        lines.append(
            f"- {n:,} environments ({where_and_when(cell)}, {cell['rounds_done']} rounds of"
            f" {cell['env_steps_per_call']} environment steps, each function timed {orders[n]}): physics"
            f" step advancing its own state {median_ms(cell, PHYSICS_OWN_STATE, orders[n]):.3f} ms;"
            f" feeding recorded states {feeds[n]:.3f} ms;"
            f" {facts['contact_slots_per_environment']} contact slots and"
            f" {facts['constraint_rows_per_environment']} constraint rows per environment;"
            f" {facts['mean_contacts_within_margin_per_environment']:.2f} contacts within their margin"
            f" per environment at the start state, {100 * facts['share_of_environments_with_a_contact']:.1f}%"
            f" of environments with at least one."
            + (f" Stage list against `mjx.step`, one physics step: largest relative difference"
               f" {check['largest_relative_difference']:.2e}"
               f" ({'passed' if check['passed'] else 'FAILED'})." if check else ""))
    lines.append("")
    # GOAL: the same functions in the two timing orders, where both were recorded.
    both = [n for n in have if orders[n] == "back to back"]
    if both:
        lines += ["### The two timing orders", "",
                  "Each function was timed twice: in rounds that go through the eight functions in turn, and"
                  " back to back (two warm-up calls, then all rounds of one function in a row). The table"
                  " above uses the back-to-back times, the order in which a training run calls the step.", "",
                  "| function |" + "".join(f" {n:,} env.:<br>in turn (ms) | back to<br>back (ms) |"
                                          " in turn<br>against<br>back to back |" for n in both),
                  "|---|" + "---:|---:|---:|" * len(both)]
        names = [f"stages through: {piece}" for piece in PIECES[:6]] + [PHYSICS_OWN_STATE, ENVIRONMENT_STEP]
        for name in names:
            row = f"| {name} |"
            for n in both:
                cell = cells[split_name("umaze", n)]
                turn, back = median_ms(cell, name), median_ms(cell, name, "back to back")
                row += f" {turn:.3f} | {back:.3f} | {100 * (turn / back - 1):+.1f}% |"
            lines.append(row)
        lines.append("")
    # GOAL: the part of each piece that does not grow with the batch, from the two sizes.
    if len(have) == 2:
        small, large = have
        lines += [
            "### Fixed part and per-environment part of each piece", "",
            f"A line through the two batch sizes: with $a_k(n)$ as above, the slope is"
            f" $b_k = \\dfrac{{a_k({large}) - a_k({small})}}{{{large} - {small}}}$ and the fixed part is"
            f" $f_k = a_k({small}) - {small} \\times b_k$, the time the piece would take at a batch of"
            " zero. Two points give no check of the line, so read the fixed part as approximate; a piece"
            " whose two times differ by less than the spread of the timings is marked not resolved.", "",
            "| piece | fixed part<br>(ms) | per 1,000<br>environments (ms) |"
            f" fixed part as a share<br>of the piece at {small:,} | at {large:,} |",
            "|---|---:|---:|---:|---:|"]
        totals = [0.0, 0.0]
        for k, piece in enumerate(PIECES):
            a_small, a_large = tables[small][k]["added_ms"], tables[large][k]["added_ms"]
            fixed, per_thousand = two_point_line(a_small, a_large, small, large)
            totals = [totals[0] + fixed, totals[1] + per_thousand]
            resolved = 0.0 <= fixed <= min(a_small, a_large)
            lines.append(f"| {k + 1}. {piece} | {fixed:.3f} | {per_thousand:.3f} | " + (
                f"{100 * fixed / a_small:.0f}% | {100 * fixed / a_large:.0f}% |" if resolved
                else "not resolved | not resolved |"))
        env_small, env_large = tables[small][6]["prefix_ms"], tables[large][6]["prefix_ms"]
        lines += [f"| whole environment step | {totals[0]:.3f} | {totals[1]:.3f} |"
                  f" {100 * totals[0] / env_small:.0f}% | {100 * totals[0] / env_large:.0f}% |", ""]
    return lines


def render_full_step_table(title, cell, baseline_name, intro):
    """A table of whole environment steps of one cell, each row compared with the cell's baseline row."""
    lines = [f"## {title}", ""]
    if cell is None:
        return lines + ["Not measured.", ""]
    lines += [intro + f" Measured on {where_and_when(cell)}, {cell['rounds_done']} rounds of"
              f" {cell['env_steps_per_call']} environment steps, timed in turn.", "",
              "| setting | environments | environment<br>step (ms) | time against<br>the first row |"
              " total steps per<br>second (millions) | first<br>call (s) | environments reset for<br>a"
              " non-finite state<br>in the 100 start steps |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    base_ms = median_ms(cell, baseline_name)
    for name, function in cell["functions"].items():
        ms = median_ms(cell, name)
        n_envs = function["environments"]
        lines.append(
            f"| {name} | {n_envs:,} | {ms:.3f} | {100 * ms / base_ms:.1f}% |"
            f" {throughput_row(n_envs, ms / 1000.0)['total_steps_per_second_millions']:.3f} |"
            f" {function['first_call_seconds']:.1f} |"
            f" {function['start_state']['environments_reset_for_a_non_finite_state']:,} |")
    return lines + [""]


def render_solver_arithmetic(cells, sizes):
    """Estimates about the solver computed from the settings rows, per batch size that was measured."""
    measured = [n for n in sorted(sizes) if variations_name(n) in cells]
    if not measured:
        return []
    names = ("solver iterations 1", "solver iterations 2", "line-search iterations 4")
    columns = {n: solver_arithmetic(median_ms(cells[variations_name(n)], DEFAULT_SETTINGS),
                                    *(median_ms(cells[variations_name(n)], name) for name in names))
               for n in measured}
    steps = {n: median_ms(cells[variations_name(n)], DEFAULT_SETTINGS) for n in measured}
    lines = ["### What the settings rows say about the solver (estimates)", "",
             "Define $T(i)$ as the milliseconds per environment step with $i$ solver iterations and"
             " $T_4$ as the time with 4 line-search iterations instead of 8, other settings at their"
             " defaults. All five quantities assume the time is a straight line in the number of"
             " iterations, and the pass without the loop comes from the 1-iteration model, whose physics"
             " is unstable (see the resets above), so read them as estimates.", "",
             "- One solver iteration inside the loop: $u = \\dfrac{T(4) - T(2)}{2}$.",
             "- Everything outside the solver iterations: $o = T(2) - 2 \\times u$.",
             "- One solver pass without the loop: $w = T(1) - o$.",
             "- Added per iteration by the loop form: $u - w$.",
             "- The 8 line-search iterations of all 4 solver iterations: $2 \\times (T(4) - T_4)$.", "",
             "| quantity |" + "".join(f" {n:,} env.<br>(ms) | share of the<br>environment step |" for n in measured),
             "|---|" + "---:|---:|" * len(measured)]
    for quantity in next(iter(columns.values())):
        lines.append(f"| {quantity} |" + "".join(
            f" {columns[n][quantity]:.3f} | {100 * columns[n][quantity] / steps[n]:.1f}% |" for n in measured))
    return lines + [""]


def default_readings(cells, sizes):
    """Every usable reading of the default U-maze environment step: (environments, ms, cell and order, time, kind)."""
    rows = []
    for n in sizes:
        # GOAL: the split cell first, one row per timing order it holds.
        cell = cells.get(split_name("umaze", n))
        if cell is not None:
            for order in ("in turn", "back to back"):
                if TIMING_KEYS[order] in cell["functions"][ENVIRONMENT_STEP]:
                    rows.append((n, median_ms(cell, ENVIRONMENT_STEP, order), f"{cell['name']}, timed {order}",
                                 cell["measured_pacific"], f"split {order}"))
        # GOAL: then the cells that time only whole environment steps.
        sources = [(variations_name(n), DEFAULT_SETTINGS), (wide_name(n), ENVIRONMENT_STEP),
                   (UNROLL_NAME, DEFAULT_SETTINGS if n == 4096 else None),
                   (BATCH_SIZES_NAME, f"{n:,} environments")]
        for cell_name, function_name in sources:
            cell = cells.get(cell_name)
            if cell is not None and function_name in cell["functions"]:
                rows.append((n, median_ms(cell, function_name), cell_name, cell["measured_pacific"], "whole"))
    return rows


def render_wide(cells, sizes):
    """The throughput table over batch sizes with the five columns of the throughput rule, and all readings."""
    lines = ["## Wider batches (U-maze, whole environment step)", "",
             "Define $n$ as the number of environments and $s$ as the median seconds per environment"
             " step (one step of all $n$ environments, 5 physics steps each).", "",
             "- Total steps per second: $\\dfrac{n}{s}$, shown in millions.",
             "- Steps per second per environment: $r = \\dfrac{1}{s}$.",
             "- Hours per million steps per environment: $\\dfrac{10^{6}}{3600 \\times r}$.", ""]
    # GOAL: one row per batch size, from the cell that timed all sizes in turn when it exists.
    together = cells.get(BATCH_SIZES_NAME)
    if together is not None:
        lines += [f"All batch sizes were timed in turn in one cell ({where_and_when(together)},"
                  f" {together['rounds_done']} rounds of {together['env_steps_per_call']} environment steps).", ""]
    else:
        lines += ["Each batch size was timed in its own cell; the same function read in two cells differs by"
                  " up to about 5% (table of all readings below).", ""]
    lines += ["| copies<br>(environments) | seconds per iteration<br>(per environment step) |"
              " total steps per<br>second (millions) | steps per second<br>per environment |"
              " hours per million steps<br>per environment | measured |",
              "|---:|---:|---:|---:|---:|---|"]
    previous = None
    for n in sizes:
        if together is not None:
            cell, name = together, f"{n:,} environments"
        else:
            cell, name = cells.get(split_name("umaze", n)) or cells.get(wide_name(n)), ENVIRONMENT_STEP
        if cell is None or name not in cell["functions"]:
            lines.append(f"| {n:,} | not measured | | | | |")
            continue
        row = throughput_row(n, median_ms(cell, name) / 1000.0)
        gain = "" if previous is None else (
            f" ({100 * (row['total_steps_per_second_millions'] / previous - 1):+.0f}% total)")
        previous = row["total_steps_per_second_millions"]
        lines.append(f"| {n:,} | {row['seconds_per_environment_step']:.5f} |"
                     f" {row['total_steps_per_second_millions']:.3f}{gain} |"
                     f" {row['steps_per_second_per_environment']:.1f} |"
                     f" {row['hours_per_million_steps_per_environment']:.2f} | {where_and_when(cell)} |")
    lines += ["", "The percentage beside a total is its change from the row above.", ""]
    # GOAL: every reading of the same default function, so the spread between cells can be seen.
    readings = default_readings(cells, sizes)
    if readings:
        lines += ["### Every reading of the default environment step (U-maze)", "",
                  "The same compiled function was timed in several cells. Two settings are compared only"
                  " inside one cell, where the functions are timed in turn.", "",
                  "| environments | environment<br>step (ms) | cell | measured |", "|---:|---:|---|---|"]
        lines += [f"| {n:,} | {ms:.3f} | {label} | {when} |" for n, ms, label, when, _ in readings]
        lines.append("")
        # GOAL: say how far the readings are apart, split cells against the cells of whole steps only.
        for n in sizes:
            whole = [ms for m, ms, _, _, kind in readings if m == n and kind == "whole"]
            if len(whole) < 2:
                continue
            text = (f"- {n:,} environments: the {len(whole)} cells that time only whole environment steps"
                    f" read {min(whole):.3f} to {max(whole):.3f} ms"
                    f" ({100 * (max(whole) / min(whole) - 1):.1f}% apart)")
            for order in ("in turn", "back to back"):
                for ms in [ms for m, ms, _, _, kind in readings if m == n and kind == f"split {order}"]:
                    text += (f"; the split cell timed {order} reads {ms:.3f} ms,"
                             f" {100 * (ms / statistics.median(whole) - 1):+.1f}% from their median")
            lines.append(text + ".")
        split_ms = [(n, ms) for n, ms, _, _, kind in readings if kind.startswith("split")]
        whole_median = {n: statistics.median([ms for m, ms, _, _, kind in readings if m == n and kind == "whole"])
                        for n, _ in split_ms if any(m == n and kind == "whole" for m, _, _, _, kind in readings)}
        if any(ms > 1.03 * whole_median[n] for n, ms in split_ms if n in whole_median):
            lines += ["", "The split cells read the whole environment step higher than the cells that time only"
                      " whole steps. The reason was not found. The shares in the split table are ratios of times"
                      " taken inside one cell; the milliseconds of the split table carry this difference."]
        lines.append("")
    return lines


def render_large(cells, split_sizes):
    """The Large map's environment step beside the U-maze's at the same batch sizes."""
    lines = ["## Large map (whole environment step)", ""]
    cell = cells.get(LARGE_MAP_NAME)
    if cell is None:
        return lines + ["Not measured.", ""]
    lines += [f"The Large map has 31 wall boxes, the U-maze 8. Measured on {where_and_when(cell)},"
              f" {cell['rounds_done']} rounds of {cell['env_steps_per_call']} environment steps. The U-maze"
              " column is the median of the U-maze readings from the cells that time only whole environment"
              " steps (listed under the wider batches); the two maps were not timed in turn in one cell.", "",
              "| environments | Large map:<br>environment step (ms) | U-maze:<br>environment step (ms) |"
              " Large against<br>U-maze | Large: first<br>call (s) |",
              "|---:|---:|---:|---:|---:|"]
    readings = default_readings(cells, split_sizes)
    for n in sorted(split_sizes):
        name = f"large, {n:,} environments"
        if name not in cell["functions"]:
            continue
        large_ms = median_ms(cell, name)
        whole = [ms for m, ms, _, _, kind in readings if m == n and kind == "whole"]
        umaze_text, ratio_text = "not measured", ""
        if whole:
            umaze_ms = statistics.median(whole)
            umaze_text, ratio_text = f"{umaze_ms:.3f}", f"{100 * (large_ms / umaze_ms - 1):+.1f}%"
        lines.append(f"| {n:,} | {large_ms:.3f} | {umaze_text} | {ratio_text} |"
                     f" {cell['functions'][name]['first_call_seconds']:.1f} |")
    return lines + [""]


def render(record, split_sizes=(1024, 4096), sizes=(1024, 2048, 4096, 8192, 16384)):
    """The whole Markdown report from the record."""
    cells = usable_cells(record)
    kept = [run for run in record["runs"] if not run.get("discarded")]
    lines = ["# AntMaze physics step: where the time goes", "",
             "Generated by `benchmarks/env/profile_antmaze_physics_split.py --render` from"
             " `antmaze_physics_split.json`; change the script, not this file. Times are Pacific,"
             " converted from machines running on Eastern.", ""]
    # GOAL: one bullet per unit with the machine, the card, the versions and the wall time.
    for run in kept:
        v = run["versions"]
        lines.append(
            f"- Unit `{run['unit']}`: {run['machine']} ({run['card']['name']}, host {run['host']}),"
            f" started {run['started_pacific']}, "
            + (f"{run['wall_seconds'] / 60:.1f} minutes of wall time" if run.get("finished")
               else "ended before its last cell finished")
            + f"; JAX {v['jax']}, jaxlib {v['jaxlib']}, MuJoCo {v['mujoco']}, MJX {v['mujoco-mjx']},"
              f" Python {run['python']}, driver {run['card']['driver']}.")
    lines.append("")
    lines += render_split(cells, split_sizes)
    lines += render_large(cells, split_sizes)
    # GOAL: one settings table per batch size that was measured, the larger batch first.
    measured = [n for n in sorted(sizes, reverse=True) if variations_name(n) in cells] or [4096]
    for n in measured:
        lines += render_full_step_table(
            f"Solver and contact settings (U-maze, {n:,} environments)", cells.get(variations_name(n)),
            DEFAULT_SETTINGS,
            "Each row changes one setting of the default model. With 1 solver iteration MJX runs the"
            " solver pass once without its loop; with 2 or more it runs a loop that stops when every"
            " environment has converged or reached the limit.")
    lines += render_solver_arithmetic(cells, sizes)
    lines += render_wide(cells, sizes)
    lines += render_full_step_table(
        "Frame loop unrolled, and one compiler option (U-maze, 4,096 environments)",
        cells.get(UNROLL_NAME), DEFAULT_SETTINGS,
        "The same model and start state compiled three ways: as the environment is written (the 5"
        " frames in a loop), with the 5 frames unrolled into one program, and with the compiler"
        " option `xla_gpu_graph_min_graph_size=1` given to `jax.jit`.")
    # GOAL: everything that was planned or started and is not in the tables.
    lines += ["## Not measured or not used", ""]
    unused = unusable_cells(record)
    lines += [f"- {name} (unit `{unit}`): {reason}."
              + (" A usable measurement of this cell from another unit is in the tables."
                 if name in cells and not reason.startswith("replaced") else "")
              for name, unit, reason in unused] or ["Nothing."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# the record file
# ----------------------------------------------------------------------------------------------

def pacific(epoch):
    """An epoch time shown in Pacific time with the zone named."""
    return datetime.fromtimestamp(epoch, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")


def flush(record, out):
    """Write the whole record to disk through a temporary file, so a reader never sees half a file."""
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out)


def sha256_of(path):
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def card_record():
    """UUID, name, memory and driver of the card this process can see, from nvidia-smi."""
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    for row in rows:
        index, uuid, name, memory, driver = [field.strip() for field in row.split(",")]
        if visible in (index, uuid):
            return {"uuid": uuid, "index": index, "name": name, "memory": memory, "driver": driver}
    raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} names no card in nvidia-smi: {rows}")


def other_processes_on_card(card_uuid):
    """Compute processes on our card that are not this process, as 'pid user memory' strings."""
    rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    others = []
    for row in rows:
        uuid, pid, memory = [field.strip() for field in row.split(",")]
        if uuid == card_uuid and int(pid) != os.getpid():
            user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
            others.append(f"pid {pid} user {user or 'unknown'} {memory}")
    return others


# ----------------------------------------------------------------------------------------------
# random torques
# ----------------------------------------------------------------------------------------------

def substream_key(name, *coords):
    """A JAX random key for one named random quantity, independent of every other quantity."""
    import jax
    # before: name = "settle torques", coords = (3,), BASE_SEED = 0
    # after:  PRNGKey(low 32 bits of sha256("0::settle torques::3"))
    text = "::".join(str(part) for part in (BASE_SEED, name, *coords))
    return jax.random.PRNGKey(int(hashlib.sha256(text.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def uniform_torques(name, chunk, n_envs, act_dim):
    """Uniform torques in [-1, 1], shape [ENV_STEPS_PER_CALL, n_envs, act_dim], one stream per environment."""
    import jax
    import jax.numpy as jnp
    key = substream_key(name, chunk)
    # GOAL: environment i draws from its own stream, so a larger batch only appends environments.
    # before: per environment i, a [ENV_STEPS_PER_CALL, act_dim] block from fold_in(key, i),
    #         stacked to [n_envs, ENV_STEPS_PER_CALL, act_dim]
    # after:  axes swapped to [ENV_STEPS_PER_CALL, n_envs, act_dim], the order the scans consume
    per_env = jax.vmap(lambda i: jax.random.uniform(
        jax.random.fold_in(key, i), (ENV_STEPS_PER_CALL, act_dim), jnp.float32, -1.0, 1.0))(jnp.arange(n_envs))
    return jnp.swapaxes(per_env, 0, 1)


# ----------------------------------------------------------------------------------------------
# the compiled functions
# ----------------------------------------------------------------------------------------------

def stage_groups():
    """The stages of one MJX physics step in the order mjx.step runs them, grouped into pieces 1 to 6."""
    from mujoco import mjx
    return (
        (mjx.kinematics, mjx.com_pos, mjx.camlight, mjx.tendon, mjx.crb, mjx.tendon_armature, mjx.factor_m),
        (mjx.collision,),
        (mjx.make_constraint, mjx.transmission),
        (mjx.sensor_pos, mjx.fwd_velocity, mjx.sensor_vel, mjx.fwd_actuation, mjx.fwd_acceleration),
        (mjx.solve, mjx.sensor_acc),
        (mjx.implicit,),
    )


def run_stages(stages, model, data):
    """Apply the stage functions in order to one environment's MJX data."""
    for stage in stages:
        data = stage(model, data)
    return data


def build_prefix_call(env, stages):
    """Compiled function: the stage list at every frame of every environment step, on recorded states."""
    import jax

    def call(data, torques, recorded):
        """data [B] MJX state; torques [steps, B, 8]; recorded: arrays [steps, frame_skip, B, .]."""
        with jax.enable_x64(False):
            def env_step(d, step_inputs):
                """One environment step: hold the torque for frame_skip frames."""
                torque, frames = step_inputs
                d = d.replace(ctrl=torque)

                def frame(d, rec):
                    """One frame: set the recorded state, then run the stages over the batch."""
                    d = d.replace(qpos=rec["qpos"], qvel=rec["qvel"], qacc_warmstart=rec["qacc_warmstart"])
                    return jax.vmap(functools.partial(run_stages, stages), in_axes=(None, 0))(env.model, d), None
                d, _ = jax.lax.scan(frame, d, frames)
                return d, None
            data, _ = jax.lax.scan(env_step, data, (torques, recorded))
            return data
    return jax.jit(call)


def build_physics_call(env, record):
    """Compiled function: mjx.step advancing its own state, as in the environment's frame loop.

    With record=True it also returns the positions, velocities and solver start values each
    frame started from, shape [steps, frame_skip, B, .].
    """
    import jax
    from mujoco import mjx

    def call(data, torques):
        """data [B] MJX state; torques [steps, B, 8]."""
        with jax.enable_x64(False):
            def env_step(d, torque):
                """One environment step: hold the torque for frame_skip physics steps."""
                d = d.replace(ctrl=torque)

                def frame(d, _):
                    """One physics step; the state it started from is the frame's output when recording."""
                    rec = {"qpos": d.qpos, "qvel": d.qvel, "qacc_warmstart": d.qacc_warmstart} if record else None
                    return jax.vmap(mjx.step, in_axes=(None, 0))(env.model, d), rec
                return jax.lax.scan(frame, d, None, length=env.cfg.frame_skip)
            data, recorded = jax.lax.scan(env_step, data, torques)
            return (data, recorded) if record else data
    return jax.jit(call)


@contextlib.contextmanager
def frame_loop_unroll(frame_skip, unroll):
    """While active, the environment's frame loop is traced with this unroll; yields the patch count."""
    import jax
    original, seen = jax.lax.scan, {"inside": False, "patched": 0}

    def scan(f, init, xs=None, length=None, **kwargs):
        """jax.lax.scan, with unroll added to the outermost scan that has no inputs and length frame_skip."""
        if seen["inside"] or xs is not None or length != frame_skip:
            return original(f, init, xs, length=length, **kwargs)
        seen["inside"], seen["patched"] = True, seen["patched"] + 1
        try:
            return original(f, init, xs, length=length, unroll=unroll, **kwargs)
        finally:
            seen["inside"] = False
    jax.lax.scan = scan
    try:
        yield seen
    finally:
        jax.lax.scan = original


def build_env_call(env, frame_unroll=None, compiler_options=None):
    """Compiled function: the environment's own step for every torque, returning the state and all outputs."""
    import jax

    def call(state, torques):
        """state: the environment state; torques [steps, B, 8]."""
        def body(st, torque):
            """One environment step; its five outputs are stacked over the steps."""
            st, obs, reward, terminated, truncated, final_obs = env.step(
                st, torque.reshape(env.C, env.N, env.act_dim))
            return st, (obs, reward, terminated, truncated, final_obs)
        if frame_unroll is None:
            return jax.lax.scan(body, state, torques)
        # GOAL: trace the environment's unchanged code with its frame loop unrolled.
        with frame_loop_unroll(env.cfg.frame_skip, frame_unroll) as seen:
            out = jax.lax.scan(body, state, torques)
        if not seen["patched"]:
            raise RuntimeError("the environment's frame loop was not found, so nothing was unrolled")
        return out
    return jax.jit(call, compiler_options=compiler_options)


def composition_difference(env, stages, data):
    """Largest relative difference, over every array, between one physics step from the stage list and mjx.step."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    def worst_difference(d):
        """One physics step computed both ways from the same state, compared array by array."""
        with jax.enable_x64(False):
            composed = jax.vmap(functools.partial(run_stages, stages), in_axes=(None, 0))(env.model, d)
            reference = jax.vmap(mjx.step, in_axes=(None, 0))(env.model, d)
            # GOAL: |x - y| / (1 + |y|) per element; a value that is non-finite on both sides counts as equal.
            worst = jnp.zeros((), jnp.float32)
            for x, y in zip(jax.tree.leaves(composed), jax.tree.leaves(reference)):
                if x.size == 0:
                    continue
                x, y = x.astype(jnp.float32), y.astype(jnp.float32)
                diff = jnp.where(~jnp.isfinite(x) & ~jnp.isfinite(y), 0.0, jnp.abs(x - y) / (1.0 + jnp.abs(y)))
                worst = jnp.maximum(worst, jnp.nan_to_num(diff, nan=jnp.inf, posinf=jnp.inf).max())
            return worst
    return float(jax.jit(worst_difference)(data))


# ----------------------------------------------------------------------------------------------
# measuring
# ----------------------------------------------------------------------------------------------

def settle(env, env_call):
    """The state after SETTLE_ENV_STEPS random-torque environment steps, and the seconds of the first call."""
    import jax
    state, first_call_seconds = env.reset(), None
    for chunk in range(SETTLE_ENV_STEPS // ENV_STEPS_PER_CALL):
        torques = uniform_torques("settle torques", chunk, env.B, env.act_dim)
        jax.block_until_ready(torques)
        t0 = time.perf_counter()
        state, _ = env_call(state, torques)
        jax.block_until_ready(state)
        if first_call_seconds is None:
            first_call_seconds = time.perf_counter() - t0
    return state, first_call_seconds


def state_facts(state):
    """What the start state looks like: contact slots, constraint rows, contacts, non-finite resets."""
    import jax.numpy as jnp
    impl = state.data._impl
    near = impl.contact.dist < impl.contact.includemargin
    return {
        "contact_slots_per_environment": int(impl.contact.dist.shape[-1]),
        "constraint_rows_per_environment": int(impl.efc_J.shape[-2]),
        "mean_contacts_within_margin_per_environment": float(near.sum(-1).mean()),
        "share_of_environments_with_a_contact": float(near.any(-1).mean()),
        "environments_reset_for_a_non_finite_state": int(state.nan_count.sum()),
        "state_is_finite": bool(jnp.isfinite(state.data.qpos).all() & jnp.isfinite(state.data.qvel).all()),
    }


def time_functions(cell, functions, rounds, save, back_to_back=False):
    """First call, two warm-up calls, then timed rounds that go through the functions in turn.

    functions: (name, call with no arguments, seconds of an earlier first call or None). The
    timings go into cell["functions"][name] and are saved after the first calls and every round.
    With back_to_back=True every function is then timed a second time, all its rounds in a row.
    """
    import jax
    # GOAL: the first call of each function traces, compiles and runs it; two more calls warm it up.
    for name, call, first_call_seconds in functions:
        if first_call_seconds is None:
            t0 = time.perf_counter()
            jax.block_until_ready(call())
            first_call_seconds = time.perf_counter() - t0
        for _ in range(2):
            jax.block_until_ready(call())
        cell["functions"].setdefault(name, {}).update(first_call_seconds=first_call_seconds, call_seconds=[])
        print(f"  compiled in {first_call_seconds:6.1f} s: {name}", flush=True)
    save()
    # GOAL: every round times each function once, so a slow minute of the machine reaches all of them.
    for r in range(rounds):
        for name, call, _ in functions:
            t0 = time.perf_counter()
            jax.block_until_ready(call())
            cell["functions"][name]["call_seconds"].append(time.perf_counter() - t0)
        cell["rounds_done"] = r + 1
        save()
    for name, _, _ in functions:
        print(f"  {median_ms(cell, name):9.3f} ms per environment step: {name}", flush=True)
    if not back_to_back:
        return
    # GOAL: the order a training run uses: one function called again and again, after two warm-up calls.
    for name, call, _ in functions:
        for _ in range(2):
            jax.block_until_ready(call())
        times = cell["functions"][name][TIMING_KEYS["back to back"]] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            jax.block_until_ready(call())
            times.append(time.perf_counter() - t0)
        save()
        print(f"  {median_ms(cell, name, 'back to back'):9.3f} ms per environment step, back to back: {name}",
              flush=True)


def measure_split(cell, map_name, n_envs, check_stage_list, rounds, save):
    """Time the six prefixes, the physics step on its own state and the environment step at one batch size."""
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    # GOAL: the start state, the torques of the timed steps, and the states the real step visits.
    env = JaxAntMaze(preset(map_name), n_envs, 1)
    env_call = build_env_call(env)
    start, env_first_call = settle(env, env_call)
    cell["start_state"] = state_facts(start)
    torques = uniform_torques("timed torques", 0, n_envs, env.act_dim)
    _, recorded = build_physics_call(env, record=True)(start.data, torques)
    # GOAL: the stage list must reproduce mjx.step, or the pieces would not be the pieces of the real step.
    groups = stage_groups()
    if check_stage_list:
        worst = composition_difference(env, sum(groups, ()), start.data)
        cell["composition_check"] = {"largest_relative_difference": worst, "passed": worst <= 1e-2}
        print(f"  stage list against mjx.step, largest relative difference: {worst:.2e}", flush=True)
    # GOAL: the eight timed functions of the cell, all starting from the same state.
    functions = [(f"stages through: {PIECES[k]}",
                  functools.partial(build_prefix_call(env, sum(groups[:k + 1], ())), start.data, torques, recorded),
                  None) for k in range(6)]
    functions.append((PHYSICS_OWN_STATE,
                      functools.partial(build_physics_call(env, record=False), start.data, torques), None))
    functions.append((ENVIRONMENT_STEP, functools.partial(env_call, start, torques), env_first_call))
    time_functions(cell, functions, rounds, save, back_to_back=True)


def measure_environment_steps(cell, entries, rounds, save):
    """Time the whole environment step for each entry: its own model, start state and compiled function.

    entries: dicts with name, map, environments, and optionally changes (configuration fields),
    frame_unroll and compiler_options.
    """
    import dataclasses
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    functions = []
    for entry in entries:
        # GOAL: build and compile this entry, and bring it to its own start state.
        cfg = dataclasses.replace(preset(entry["map"]), **entry.get("changes", {}))
        env = JaxAntMaze(cfg, entry["environments"], 1)
        env_call = build_env_call(env, entry.get("frame_unroll"), entry.get("compiler_options"))
        start, first_call = settle(env, env_call)
        torques = uniform_torques("timed torques", 0, env.B, env.act_dim)
        cell["functions"][entry["name"]] = {"map": entry["map"], "environments": entry["environments"],
                                            "changes": entry.get("changes", {}),
                                            "start_state": state_facts(start)}
        functions.append((entry["name"], functools.partial(env_call, start, torques), first_call))
    time_functions(cell, functions, rounds, save)


def planned_cells(args):
    """The cells of this unit in the order they are measured: (kind, name, function that fills the cell)."""
    full = lambda entries: functools.partial(measure_environment_steps, entries=entries)
    plan = []
    if "split" in args.parts:
        for i, n in enumerate(args.split_batches):
            plan.append(("split", split_name("umaze", n), functools.partial(
                measure_split, map_name="umaze", n_envs=n, check_stage_list=(i == 0))))
    if "batches" in args.parts:
        plan.append(("batch sizes", BATCH_SIZES_NAME, full(
            [{"name": f"{n:,} environments", "map": "umaze", "environments": n} for n in args.batch_sizes])))
    if "variations" in args.parts:
        for n in args.variation_batches:
            plan.append(("variations", variations_name(n), full(
                [{"name": name, "map": "umaze", "environments": n, "changes": changes}
                 for name, changes in VARIATIONS])))
    if "wide" in args.parts:
        for n in args.wide_batches:
            plan.append(("wide batch", wide_name(n), full(
                [{"name": ENVIRONMENT_STEP, "map": "umaze", "environments": n}])))
    if "large" in args.parts:
        plan.append(("large map", LARGE_MAP_NAME, full(
            [{"name": f"large, {n:,} environments", "map": "large", "environments": n}
             for n in sorted(args.split_batches)])))
    if "unroll" in args.parts:
        plan.append(("unroll and compiler option", UNROLL_NAME, full([
            {"name": DEFAULT_SETTINGS, "map": "umaze", "environments": 4096},
            {"name": FRAME_LOOP_UNROLLED, "map": "umaze", "environments": 4096, "frame_unroll": 5},
            {"name": COMPILER_OPTION, "map": "umaze", "environments": 4096,
             "compiler_options": {"xla_gpu_graph_min_graph_size": 1}}])))
    return plan


def main():
    """Measure the planned cells inside the unit's time limit, saving after every round; or render."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the JSON record file; runs are appended to it")
    ap.add_argument("--render", action="store_true", help="only rebuild the Markdown from the record")
    ap.add_argument("--machine", default=None, help='for example "serval02 card 1"')
    ap.add_argument("--unit", default=None, help="name of the unit this run belongs to")
    ap.add_argument("--parts", nargs="+", default=["split", "large", "variations", "wide"],
                    choices=["split", "large", "variations", "wide", "unroll", "batches"])
    ap.add_argument("--split-batches", nargs="+", type=int, default=[4096, 1024])
    ap.add_argument("--variation-batches", nargs="+", type=int, default=[4096])
    ap.add_argument("--wide-batches", nargs="+", type=int, default=[8192, 16384])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384],
                    help="batch sizes of the cell that times all of them in turn (part: batches)")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--unit-start-epoch", type=float, default=None, help="when the unit started (date +%%s)")
    ap.add_argument("--unit-minutes", type=float, default=15.0, help="wall-time limit of the whole unit")
    args = ap.parse_args()
    out = Path(args.out)
    markdown = out.with_suffix(".md")
    if args.render:
        markdown.write_text(render(json.loads(out.read_text())))
        print(f"wrote {markdown}")
        return
    if args.machine is None or args.unit is None:
        ap.error("--machine and --unit are required for a measurement")
    if args.rounds < MIN_ROUNDS:
        ap.error(f"--rounds must be at least {MIN_ROUNDS}")
    unit_start = args.unit_start_epoch or time.time()
    # no new cell starts unless it is expected to end 45 seconds before the unit's limit
    deadline = unit_start + 60.0 * args.unit_minutes - 45.0

    # GOAL: refuse to measure on the processor, with an exit code the unit script recognises.
    import jax
    import exploration_platform  # noqa: F401  (switches on 64-bit statistics before any array exists, as in training)
    if jax.devices()[0].platform != "gpu":
        print(f"JAX did not load the graphics card (device {jax.devices()[0]}); nothing measured")
        sys.exit(NO_GPU_EXIT_CODE)

    # GOAL: the run's identity: machine, card, versions, sources, time.
    card = card_record()
    here = Path(__file__).resolve()
    env_dir = BASE / "src" / "exploration_platform" / "envs" / "antmaze"
    run = {
        "unit": args.unit, "machine": args.machine, "host": socket.gethostname(), "card": card,
        "started_pacific": pacific(unit_start), "started_iso": datetime.now().astimezone().isoformat(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "numpy")},
        "python": sys.version.split()[0],
        "source_sha256": {path.name: sha256_of(path)
                          for path in (here, env_dir / "mjx_antmaze.py", env_dir / "am_common.py")},
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "settings": {"rounds": args.rounds, "env_steps_per_call": ENV_STEPS_PER_CALL,
                     "settle_env_steps": SETTLE_ENV_STEPS, "base_seed": BASE_SEED},
        "cells": [], "finished": False,
    }
    record = json.loads(out.read_text()) if out.exists() else {"runs": []}
    record["runs"].append(run)
    save = functools.partial(flush, record, out)
    save()
    print(f"{run['started_pacific']} unit {args.unit} on {args.machine}: {card['name']} {card['uuid']},"
          f" JAX {run['versions']['jax']}, MuJoCo {run['versions']['mujoco']}", flush=True)

    # GOAL: measure cell after cell; a cell that is not expected to fit is recorded as not measured.
    pace = 1.0   # wall time of the last finished cell divided by its first guess
    for kind, name, measure in planned_cells(args):
        needed, left = FIRST_GUESS_SECONDS[kind] * pace, deadline - time.time()
        if needed > left:
            run["cells"].append({"kind": kind, "name": name, "status": "not measured",
                                 "reason": f"{left:.0f} s were left of the {args.unit_minutes:.0f}-minute"
                                           f" unit and the cell was expected to take {needed:.0f} s"})
            print(f"not measured: {name} ({left:.0f} s left, {needed:.0f} s expected)", flush=True)
            save()
            continue
        t0 = time.time()
        cell = {"kind": kind, "name": name, "status": "running", "measured_pacific": pacific(t0),
                "env_steps_per_call": ENV_STEPS_PER_CALL, "rounds_done": 0, "functions": {},
                "other_processes_on_card": other_processes_on_card(card["uuid"])}
        run["cells"].append(cell)
        print(f"{pacific(t0)} cell: {name}", flush=True)
        measure(cell, rounds=args.rounds, save=save)
        cell["other_processes_on_card"] += other_processes_on_card(card["uuid"])
        cell["wall_seconds"] = time.time() - t0
        cell["peak_device_bytes_of_the_process_so_far"] = jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]
        cell["status"] = "measured"
        pace = min(3.0, max(0.5, cell["wall_seconds"] / FIRST_GUESS_SECONDS[kind]))
        save()
        print(f"  cell took {cell['wall_seconds']:.0f} s; {deadline - time.time():.0f} s left", flush=True)

    # GOAL: close the run and write the tables.
    run["finished"], run["wall_seconds"] = True, time.time() - unit_start
    save()
    markdown.write_text(render(record))
    print(f"{pacific(time.time())} unit finished in {run['wall_seconds'] / 60:.1f} minutes; wrote {out} and"
          f" {markdown}", flush=True)


if __name__ == "__main__":
    main()
