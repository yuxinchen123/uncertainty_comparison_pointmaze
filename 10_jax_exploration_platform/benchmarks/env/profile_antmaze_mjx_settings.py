"""Speed and physics checks of the settings of the MJX AntMaze stepper, on a graphics card.

A model is the default U-maze AntMaze with some configuration fields changed. It is written on the
command line as `field=value,field=value` (`default` for no change) and shown in the tables in
words. Three parts, each appended to one record file:

  speed     the whole environment step of several models at one batch size, timed in turn in one
            cell with the timing functions of profile_antmaze_physics_split.py (start state after
            100 random-torque steps, calls of 10 environment steps, median over the rounds).
  equality  the two speed options against the stepper as shipped: one physics step computed both
            ways from the same states, 50 environment steps run freely, and 100 environment steps
            from the spawn with only the next step's inputs carried. Every comparison is also
            made between two calls of the same compiled program, which shows how far the card
            itself differs from run to run.
  checks    per model, three rollouts of one compiled function that steps the environment and,
            after every step, looks at the new state:
              random torques                 700 steps from the spawn (non-finite respawns, contact
                                             limits, solver passes, distance travelled);
              torques that drive into walls  700 steps from starts spread over the start cell; even
                                             environments hold one torque vector, odd ones switch
                                             the sign of each torque every 10 steps;
              no torques                     1,200 steps from the spawn (resting height).
            The contact counts compare the model's own contacts with those of the same model
            without contact limits (every geometry pair tested, every contact kept).

A fourth part, `gates`, runs on processors (never on a login node): the three gates of
tests/envs/test_antmaze_parity_cpu_mujoco.py with the model in place of the default.

Usage (graphics card):
  PYTHONNOUSERSITE=1 PYTHONPATH=<platform>/src python profile_antmaze_mjx_settings.py \
      --out <folder>/antmaze_mjx_settings.json --machine "serval02 card 1" --unit <unit name> \
      --part speed --group "the two speed options" --batch 4096 --models default solver_loop=fixed_count
Rebuild the Markdown tables from the record (standard library only, any machine):
  python profile_antmaze_mjx_settings.py --out <folder>/antmaze_mjx_settings.json --render
"""
import argparse
import contextlib
import dataclasses
import functools
import importlib.metadata
import io
import json
import re
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_antmaze_physics_split as split  # noqa: E402  (timing, record file, random torques)

BASE = split.BASE
DEFAULT_MODEL = "default model"
RESTING_HEIGHT = 0.38248          # metres, the reference's resting height (spec.md, gate 2)
WALL_HEIGHT = 2.0                 # metres: a torso above this is over the walls
DEEP_PENETRATION = 0.05           # metres: a wall contact deeper than this counts as passing into the wall
CHECK_CHUNK = 50                  # environment steps per call of the check function
TRAVEL_STEPS = 700                # steps of the two torque rollouts (one U-maze episode)
REST_STEPS = 1200                 # steps of the rollout without torques
START_SPREAD = 1.1                # metres: wall-rollout starts are the cell centre +- this in x and y
SWITCH_STEPS = 10                 # the odd environments of the wall rollout switch torque signs this often
TRAVEL_SEQUENCES = 16             # the first environments of the random rollout, reported on their own
RANDOM_ROLLOUT = "random torques"
WALL_ROLLOUT = "torques that drive into walls"
REST_ROLLOUT = "no torques"
EQUALITY_NAME = "the two speed options against the stepper as shipped"
NOTABLE_DIFFERENCE = 1e-4         # a relative difference above this is counted in the equality table
SOLVER_SAMPLE = 512               # environments whose solver is compared with a converged solve after every step
REFERENCE_PASSES = 8              # solver passes of the converged solve
REFERENCE_LINE_SEARCH = 16        # line-search iterations of the converged solve
FIRST_GUESS_SECONDS = {"speed": 30.0, "equality": 240.0, "checks": 170.0}   # speed: per model
FIELD_WORDS = {
    "max_contact_points": "contact points {}",
    "max_geom_pairs": "geometry pairs {}",
    "ls_iterations": "line-search iterations {}",
    "solver_iterations": "solver iterations {}",
    "solver_loop": {"fixed_count": "fixed-count solver loop",
                    "fixed_count_scan": "fixed-count solver loop as one scan"},
    "carried_state": {"next_step_inputs": "only the next step's inputs carried"},
}
GATE_BOUNDS = {
    "one step, no contact (m)": 1e-4, "one step, in contact (m)": 5e-2,
    "before the first contact (m)": 1e-4, "20 steps (m)": 4.0,
    "handover drift (m)": 1e-6, "resting heights apart (m)": 1e-4,
}


# ----------------------------------------------------------------------------------------------
# models and table arithmetic (standard library only)
# ----------------------------------------------------------------------------------------------

def parse_changes(text):
    """The configuration fields a model changes, from its command-line form."""
    # before: "max_contact_points=16,solver_loop=fixed_count"     before: "default"
    # after:  {"max_contact_points": 16, "solver_loop": "fixed_count"}   after: {}
    if text == "default":
        return {}
    changes = {}
    for part in text.split(","):
        field, value = part.split("=")
        if field not in FIELD_WORDS:
            raise ValueError(f"unknown field {field!r}; choose from {sorted(FIELD_WORDS)}")
        changes[field] = value if isinstance(FIELD_WORDS[field], dict) else int(value)
    return changes


def model_name(changes):
    """The name of a model in words, from the fields it changes, in the order of FIELD_WORDS."""
    # before: {"ls_iterations": 4, "max_contact_points": 16}
    # after:  "contact points 16, line-search iterations 4"
    words = []
    for field, form in FIELD_WORDS.items():
        if field in changes:
            words.append(form[changes[field]] if isinstance(form, dict) else form.format(changes[field]))
    return ", ".join(words) or DEFAULT_MODEL


def per_million(count, env_steps):
    """A count of environment steps as a rate per million environment steps."""
    return 1e6 * count / env_steps


def mean_and_spread(values):
    """(mean, sample standard deviation) of a list of numbers."""
    return statistics.fmean(values), statistics.stdev(values)


def emphasize(texts, values, best):
    """Bold the best value of a column and underline the second best; ties share the mark."""
    # before: texts ["3.0", "1.0", "2.0"], values [3.0, 1.0, 2.0], best "smallest"
    # after:  ["3.0", "**1.0**", "<u>2.0</u>"]
    ranked = sorted({v for v in values if v is not None}, reverse=(best == "largest"))
    marks = {ranked[0]: "**{}**"} if ranked else {}
    if len(ranked) > 1:
        marks[ranked[1]] = "<u>{}</u>"
    return [marks.get(value, "{}").format(text) for text, value in zip(texts, values)]


def map_words(map_name):
    """How a map is written in cell names: nothing for the U-maze, else ', <map> map'."""
    return "" if map_name == "umaze" else f", {map_name} map"


def speed_name(group, n_envs, map_name="umaze"):
    """Name of the cell that times one group of models at one batch size."""
    return f"speed: {group}, {n_envs:,} environments{map_words(map_name)}"


def checks_prefix(map_name="umaze"):
    """What the names of the check cells of one map start with."""
    return f"physics checks{map_words(map_name)}: "


def checks_name(model, map_name="umaze"):
    """Name of the cell that holds the physics checks of one model on one map."""
    return checks_prefix(map_name) + model


PREVIEW = "processor preview, "   # starts the name of a check cell that was measured on processors


def usable_cells(record):
    """The latest usable cell per name, in the order of first appearance; processor trials are left out."""
    chosen = {}
    for run in record["runs"]:
        if run.get("discarded") or run["card"]["name"] == "processors":
            continue
        for cell in run["cells"]:
            if cell["status"] == "measured" and not cell["other_processes_on_card"]:
                chosen[cell["name"]] = dict(cell, machine=run["machine"], unit=run["unit"])
    return chosen


def unusable_cells(record):
    """Every cell that is not in the tables, with the reason, in the order it was recorded."""
    rows, chosen = [], usable_cells(record)
    for run in record["runs"]:
        if run["card"]["name"] == "processors":
            continue
        for cell in run["cells"]:
            if run.get("discarded"):
                reason = f"run discarded: {run['discarded']}"
            elif cell["status"] == "not measured":
                reason = cell["reason"]
            elif cell["status"] != "measured":
                reason = "the unit ended during this cell"
            elif cell["other_processes_on_card"]:
                reason = f"another process was on the card: {cell['other_processes_on_card']}"
            elif chosen[cell["name"]]["unit"] != run["unit"]:
                reason = f"replaced by the later measurement in unit `{chosen[cell['name']]['unit']}`"
            else:
                continue
            rows.append((cell["name"], run["unit"], reason))
    return rows


# ----------------------------------------------------------------------------------------------
# the Markdown tables (standard library only)
# ----------------------------------------------------------------------------------------------

def render_speed(cell):
    """One speed cell as a throughput table, fastest model first."""
    lines = [f"### {cell['name'][len('speed: '):]}", "",
             f"Measured on {cell['machine']}, {cell['measured_pacific']}, {cell['rounds_done']} rounds of"
             f" {cell['env_steps_per_call']} environment steps, the models timed in turn.", "",
             "| model | copies<br>(environments) | seconds per iteration<br>(per environment step) ↓ |"
             " total steps per<br>second (millions) | steps per second<br>per environment |"
             " hours per million steps<br>per environment | time against<br>the default model |"
             " first<br>call (s) | environments reset for a<br>non-finite state in<br>the 100 start steps |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = sorted(cell["functions"], key=lambda name: split.median_ms(cell, name))
    rows = [split.throughput_row(cell["functions"][name]["environments"], split.median_ms(cell, name) / 1000.0)
            for name in names]
    default_ms = split.median_ms(cell, DEFAULT_MODEL) if DEFAULT_MODEL in cell["functions"] else None
    columns = [emphasize([f"{row[key]:{form}}" for row in rows], [row[key] for row in rows], best)
               for key, form, best in (("seconds_per_environment_step", ".5f", "smallest"),
                                       ("total_steps_per_second_millions", ".3f", "largest"),
                                       ("steps_per_second_per_environment", ".1f", "largest"),
                                       ("hours_per_million_steps_per_environment", ".2f", "smallest"))]
    for k, name in enumerate(names):
        function = cell["functions"][name]
        against = "" if default_ms is None else f"{100 * (split.median_ms(cell, name) / default_ms - 1):+.1f}%"
        lines.append(f"| {name} | {function['environments']:,} | " + " | ".join(column[k] for column in columns)
                     + f" | {against} | {function['first_call_seconds']:.1f} |"
                       f" {function['start_state']['environments_reset_for_a_non_finite_state']:,} |")
    return lines + [""]


def render_equality(cell):
    """The equality cell: each comparison beside the same comparison between two calls of one program."""
    lines = ["## Do the two speed options change the numbers?", ""]
    if cell is None:
        return lines + ["Not measured.", ""]
    lines += [
        f"Measured on {cell['machine']}, {cell['measured_pacific']}, {cell['environments']:,} environments."
        " Two results are called equal when every number has the same bits. Define the relative"
        " difference of two numbers $x$ and $y$ as $\\dfrac{|x - y|}{1 + |y|}$; the tables give the"
        " largest one over all environments and all arrays compared, and the share of environment-steps"
        f" (one environment at one step) whose largest relative difference is above {NOTABLE_DIFFERENCE:g}."
        " Each comparison is made twice:"
        " between the option and the stepper as shipped, and between two calls of the same compiled"
        " program of the stepper as shipped, which shows what the card itself changes from run to run.", "",
        f"| comparison | steps | environments<br>not equal in at<br>least one step | environment-steps<br>over {NOTABLE_DIFFERENCE:g} |"
        " largest relative<br>difference | same program twice:<br>environments<br>not equal |"
        f" same program twice:<br>environment-steps<br>over {NOTABLE_DIFFERENCE:g} | same program twice:<br>largest relative<br>difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in cell["comparisons"]:
        lines.append(f"| {row['comparison']} | {row['steps']} | {row['environments_not_equal']:,} |"
                     f" {100 * row['share_over']:.2f}% | {row['largest_relative_difference']:.2e} |"
                     f" {row['twice_environments_not_equal']:,} | {100 * row['twice_share_over']:.2f}% |"
                     f" {row['twice_largest_relative_difference']:.2e} |")
    return lines + [""]


def rate_cells(cells_by_model, rollout, key):
    """One column of the check tables: the per-million rate of one count in one rollout, per model."""
    values = []
    for cell in cells_by_model:
        facts = cell["rollouts"].get(rollout)
        values.append(None if facts is None or key not in facts["counts"]
                      else per_million(facts["counts"][key], facts["environment_steps"]))
    return emphasize(["not measured" if v is None else f"{v:,.1f}" for v in values], values, "smallest")


def render_checks(cells, gates, map_name="umaze", preview=False):
    """The physics checks of every model on one map: non-finite states, contact limits, solver, rest, travel, gates.

    preview=True takes the cells measured on processors instead of those measured on the card.
    """
    where = "U-maze" if map_name == "umaze" else map_name.capitalize() + " map"
    lines = [f"## Physics checks ({where}{', on processors: a preview with fewer environments' if preview else ''})", ""]
    prefix = (PREVIEW if preview else "") + checks_prefix(map_name)
    models = [name[len(prefix):] for name in cells if name.startswith(prefix)]
    if not models:
        return lines + ["Not measured.", ""]
    checked = [cells[prefix + model] for model in models]
    if preview:
        lines += [f"These rows were measured on {checked[0]['machine']} while the card was in use by other work."
                  " They hold the same three rollouts with fewer environments, so rare events are resolved less"
                  " well than on the card.", ""]
    first = checked[0]
    lines += [
        f"Each model was stepped at {first['environments']:,} environments with the episode limit switched off,"
        " so an environment goes back to the spawn only after a non-finite state. Define $k$ as a count"
        " of environment steps with some property, $n$ as the number of environments and $T$ as the"
        " number of steps of the rollout. A rate per million environment steps is"
        " $\\dfrac{10^{6} \\times k}{n \\times T}$. The best value of a column is bold and the second"
        " best underlined (smallest for every rate). Rows are in the order the models were given.", "",
        "The three rollouts:", "",
        f"1. **{RANDOM_ROLLOUT}**: {TRAVEL_STEPS} steps from the spawn, each torque uniform in [-1, 1].",
        f"2. **{WALL_ROLLOUT}**: {TRAVEL_STEPS} steps from starts spread uniformly over"
        f" {START_SPREAD} m each way around the centre of the start cell (walls on three sides, 2 m from"
        " the centre). Even environments hold one torque vector, uniform in [-1, 1], for the whole"
        f" rollout; odd environments apply full torques whose signs switch every {SWITCH_STEPS} steps"
        " with a phase drawn per joint.",
        f"3. **{REST_ROLLOUT}**: {REST_STEPS} steps from the spawn.", ""]

    # GOAL: table 1, non-finite states and motion no ant should show.
    lines += ["### Non-finite states and implausible motion", "",
              "A non-finite respawn is an environment step whose state came out non-finite (the environment"
              f" then goes back to the spawn). A step above the walls is one whose torso ends above {WALL_HEIGHT:.0f} m."
              " A step with the torso inside a wall is one whose torso centre ends inside the square of a wall"
              " cell and below the top of the wall.", "",
              "| model | non-finite respawns<br>per million steps:<br>random torques | into walls |"
              " torso inside a wall<br>per million:<br>random torques | into walls |"
              " steps above the walls<br>per million:<br>random torques | into walls |"
              " largest torso<br>height (m):<br>random; walls | largest speed of any<br>coordinate:<br>random; walls |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    columns = [rate_cells(checked, rollout, key)
               for key in ("non_finite_respawns", "steps_torso_inside_wall", "steps_above_walls")
               for rollout in (RANDOM_ROLLOUT, WALL_ROLLOUT)]
    for k, (model, cell) in enumerate(zip(models, checked)):
        pair = lambda key, form: "; ".join(
            f"{cell['rollouts'][r]['counts'][key]:{form}}" if r in cell["rollouts"] else "not measured"
            for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
        lines.append(f"| {model} | " + " | ".join(column[k] for column in columns)
                     + f" | {pair('height_max', '.3g')} | {pair('speed_max', '.3g')} |")
    lines += ["", "- Counts behind the rates (non-finite respawns in random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  str(cell["rollouts"][r]["counts"]["non_finite_respawns"]) if r in cell["rollouts"] else "not measured"
                  for r in (RANDOM_ROLLOUT, WALL_ROLLOUT)) for model, cell in zip(models, checked)) + ".",
              "- Share of steps with the torso upside down (random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  f"{100 * cell['rollouts'][r]['counts']['steps_upside_down'] / cell['rollouts'][r]['environment_steps']:.1f}%"
                  if r in cell["rollouts"] else "not measured" for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
                  for model, cell in zip(models, checked)) + ".",
              f"- Each torque rollout is {first['environments'] * TRAVEL_STEPS:,} environment steps, so a count"
              " of zero bounds the rate below about 1 per million (3 divided by the steps, at 95%).", ""]

    # GOAL: table 2, how often the contact limits change what the solver sees.
    lines += ["### Contact limits", "",
              "After every step the contacts within their margin are counted twice at the new state: in the"
              " model's own contact list, and in the list of the same model with no limit on geometry pairs"
              " or contact points (the reference list). Over the limit: the reference list holds more contacts"
              " than the model has slots. Missed: the reference list holds more contacts than the model's own"
              " list, for either reason (too few slots, or the pair was not among the geometry pairs kept)."
              f" Deep in a wall: a reference contact with a wall box deeper than {DEEP_PENETRATION} m; deep in the"
              " floor: the same for the floor.", "",
              "| model | contact<br>slots | over the limit<br>per million steps:<br>random torques | into walls |"
              " missed<br>per million:<br>random torques | into walls | deep in a wall<br>per million:<br>random torques |"
              " into walls | most contacts<br>at once:<br>random; walls | steps with a wall<br>contact:<br>random; walls |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    columns = [rate_cells(checked, rollout, key)
               for key in ("steps_over_slots", "steps_missed", "steps_deep_in_wall")
               for rollout in (RANDOM_ROLLOUT, WALL_ROLLOUT)]
    for k, (model, cell) in enumerate(zip(models, checked)):
        rollouts = [cell["rollouts"].get(r) for r in (RANDOM_ROLLOUT, WALL_ROLLOUT)]
        most = "; ".join("not measured" if f is None else str(f["counts"]["contacts_max"]) for f in rollouts)
        wall = "; ".join("not measured" if f is None else
                         f"{100 * f['counts']['steps_with_wall_contact'] / f['environment_steps']:.1f}%"
                         for f in rollouts)
        lines.append(f"| {model} | {cell['contact_slots']} | " + " | ".join(column[k] for column in columns)
                     + f" | {most} | {wall} |")
    lines += ["", "- Missed contacts while slots were free, per million steps (random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  f"{per_million(cell['rollouts'][r]['counts']['steps_missed_with_free_slots'], cell['rollouts'][r]['environment_steps']):,.1f}"
                  if r in cell["rollouts"] else "not measured" for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
                  for model, cell in zip(models, checked)) + ".",
              "- Deepest reference wall contact in metres, negative is inside the wall (random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  f"{cell['rollouts'][r]['counts']['deepest_wall_min']:.3f}" if r in cell["rollouts"] else "not measured"
                  for r in (RANDOM_ROLLOUT, WALL_ROLLOUT)) for model, cell in zip(models, checked)) + ".",
              "- Deep in the floor, steps per million, and the deepest floor contact in metres (random torques; into"
              " walls): " + "; ".join(f"{model}: " + ", ".join(
                  f"{per_million(cell['rollouts'][r]['counts']['steps_deep_in_floor'], cell['rollouts'][r]['environment_steps']):,.1f}"
                  f" ({cell['rollouts'][r]['counts']['deepest_floor_min']:.3f})" if r in cell["rollouts"] else "not measured"
                  for r in (RANDOM_ROLLOUT, WALL_ROLLOUT)) for model, cell in zip(models, checked)) + ".", ""]

    # GOAL: table 3, how far the allowed solver passes are from a converged solve.
    lines += ["### Solver accuracy", "",
              f"After every step the solver is run once more at the new state of the first {first['solver_sample']:,}"
              " environments with the torque of that step, twice: with the model's own limits, and with"
              f" {REFERENCE_PASSES} passes of {REFERENCE_LINE_SEARCH} line-search iterations on the same contacts (the"
              " converged solve). Define $a$ as the vector of accelerations the model's solve returns and $b$ as"
              " the one the converged solve returns. The solver error of a step is"
              " $\\dfrac{\\max_i |a_i - b_i|}{1 + \\max_i |b_i|}$. The table gives the share of the sampled steps whose"
              " error is above 1% and above 10%, and the share in which MJX's own stop test still asks for"
              " another pass after the last allowed one. Smaller is better in every column.", "",
              "| model | passes;<br>line-search<br>iterations | error above 1%:<br>random torques | into walls |"
              " error above 10%:<br>random torques | into walls | stop test not met:<br>random torques | into walls |"
              " largest error:<br>random; walls |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    share_columns = []
    for key in ("sampled_steps_error_over_1_percent", "sampled_steps_error_over_10_percent", "sampled_steps_unconverged"):
        for rollout in (RANDOM_ROLLOUT, WALL_ROLLOUT):
            values = [None if rollout not in cell["rollouts"] else
                      100 * cell["rollouts"][rollout]["counts"][key] / cell["rollouts"][rollout]["sampled_steps"]
                      for cell in checked]
            share_columns.append(emphasize(["not measured" if v is None else f"{v:.2f}%" for v in values],
                                           values, "smallest"))
    for k, (model, cell) in enumerate(zip(models, checked)):
        largest = "; ".join(f"{cell['rollouts'][r]['counts']['solver_error_max']:.2g}" if r in cell["rollouts"]
                            else "not measured" for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
        lines.append(f"| {model} | {cell['solver_iterations']}; {cell['ls_iterations']} | "
                     + " | ".join(column[k] for column in share_columns) + f" | {largest} |")
    lines += ["", "- Share of the sampled steps that used every allowed pass (random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  f"{100 * cell['rollouts'][r]['counts']['passes_histogram'][cell['solver_iterations']] / cell['rollouts'][r]['sampled_steps']:.1f}%"
                  if r in cell["rollouts"] else "not measured" for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
                  for model, cell in zip(models, checked)) + ".",
              "- Share of the sampled steps with an error above 0.1% (random torques; into walls): "
              + "; ".join(f"{model}: " + ", ".join(
                  f"{100 * cell['rollouts'][r]['counts']['sampled_steps_error_over_0.1_percent'] / cell['rollouts'][r]['sampled_steps']:.2f}%"
                  if r in cell["rollouts"] else "not measured" for r in (RANDOM_ROLLOUT, WALL_ROLLOUT))
                  for model, cell in zip(models, checked)) + ".", ""]

    # GOAL: table 4, the resting height and the distance travelled.
    lines += ["### Resting height and distance travelled", "",
              f"Resting height: the torso height after {REST_STEPS} steps without torques (reference"
              f" {RESTING_HEIGHT} m). Distance travelled: the straight-line distance in the plane between the"
              f" torso at the spawn and after {TRAVEL_STEPS} steps of random torques, for the first"
              f" {TRAVEL_SEQUENCES} environments (the same {TRAVEL_SEQUENCES} torque sequences for every model) and"
              " for all environments; environments that were respawned are left out. Spread is the sample"
              " standard deviation.", "",
              "| model | resting<br>height (m) | height minus<br>the reference (m) | largest speed<br>at rest |"
              f" distance, {TRAVEL_SEQUENCES} sequences:<br>mean (m) | spread (m) | distance, all environments:<br>mean (m) |"
              " spread (m) | median (m) | largest (m) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, cell in zip(models, checked):
        rest, travel = cell["rollouts"].get(REST_ROLLOUT), cell["rollouts"].get(RANDOM_ROLLOUT)
        row = f"| {model} |"
        row += " not measured | | |" if rest is None else (
            f" {rest['resting_height_mean']:.5f} | {rest['resting_height_mean'] - RESTING_HEIGHT:+.5f} |"
            f" {rest['resting_speed_max']:.1e} |")
        if travel is None:
            row += " not measured | | | | | |"
        else:
            few, every = travel["distance_first_sequences"], travel["distance_all"]
            mean, spread = mean_and_spread(few)
            row += (f" {mean:.3f} | {spread:.3f} | {every['mean']:.3f} | {every['spread']:.3f} |"
                    f" {every['median']:.3f} | {every['largest']:.3f} |")
        lines.append(row)
    rest_spans = [f"{model}: {cell['rollouts'][REST_ROLLOUT]['resting_height_min']:.5f} to"
                  f" {cell['rollouts'][REST_ROLLOUT]['resting_height_max']:.5f}"
                  for model, cell in zip(models, checked) if REST_ROLLOUT in cell["rollouts"]]
    lines += ["", "- Resting height, smallest to largest over the environments (all start from the same state): "
              + "; ".join(rest_spans) + ".", ""]

    # GOAL: table 5, the processor gates (they are written for the U-maze).
    if map_name == "umaze" and not preview:
        lines += render_gates(gates)
    return lines


def render_gates(gates):
    """The three gates of test_antmaze_parity_cpu_mujoco.py per model, with every number they print."""
    lines = ["### Gates against C MuJoCo (processors)", ""]
    if not gates:
        return lines + ["Not measured.", ""]
    lines += [
        "The three gates of `tests/envs/test_antmaze_parity_cpu_mujoco.py`, run unchanged with the model in"
        " place of the default (C MuJoCo then runs the model's own solver settings and ignores its contact"
        " limits). Bounds: one step without contact under 1e-4 m and in contact under 5e-2 m; 20 steps"
        " agree under 1e-4 m before the first contact and stay under 4 m; a state settled under the"
        " reference integrator (RK4, 100 solver iterations) or under the model moves less than 1e-6 m in"
        " 200 steps of the other, and the two resting heights are within 1e-4 m. A number over its bound"
        " is marked.", "",
        "| model | one step,<br>no contact (m) | one step,<br>in contact (m) | 20 steps: before<br>the first contact (m) |"
        " 20 steps:<br>last step (m) | settled state handed<br>over, largest move (m) | resting heights<br>apart (m) |"
        " gates passed |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    keys = ("one step, no contact (m)", "one step, in contact (m)", "before the first contact (m)",
            "20 steps (m)", "handover drift (m)", "resting heights apart (m)")
    for row in gates["models"]:
        cells = []
        for key in keys:
            value = row["numbers"].get(key)
            cells.append("not printed" if value is None else
                         f"{value:.2e}" + (f" (over {GATE_BOUNDS[key]:g})" if value >= GATE_BOUNDS[key] else ""))
        passed = [name for name, gate in row["gates"].items() if gate["passed"]]
        failed = [name for name, gate in row["gates"].items() if not gate["passed"]]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + f" | {len(passed)} of {len(row['gates'])}"
                     + (f"; failed: {', '.join(failed)}" if failed else "") + " |")
    lines += ["", f"Run on {gates['host']} ({gates['measured_pacific']}), JAX on processors, MuJoCo {gates['mujoco']}.", ""]
    return lines


def render_wall_pairs(wall_pairs):
    """Where a body of the ant that touches a wall is not tested against it, per map and pair limit."""
    lines = ["## Which wall boxes the ant is tested against (all three maps)", ""]
    if not wall_pairs:
        return lines + ["Not measured.", ""]
    limits = sorted(int(k) for k in wall_pairs["maps"][0]["not_tested"])
    lines += [
        "MJX tests, for each kind of geometry pair, only the `max_geom_pairs` pairs whose bounding spheres"
        " overlap most. A wall box that is a long run of cells has a large bounding sphere and ranks ahead of"
        " a short box that the ant touches. The table is arithmetic on the model's positions and radii (no"
        f" physics): points are placed every {wall_pairs['spacing_m']} m along the open side of every wall face,"
        " and at each point a body is placed touching the face. Torso: the torso sphere. One leg part: one leg"
        " capsule lying against the face, the rest of the ant at its spawn pose 0.6 m further out. A point"
        " counts as not tested when the touched wall box is not among the pairs MJX keeps. The default model"
        " uses 8 geometry pairs.", "",
        "| map | wall<br>boxes | face<br>points | body | " + " | ".join(
            f"not tested at<br>{k} geometry pairs" for k in limits) + " |",
        "|---|---:|---:|---|" + "---:|" * len(limits)]
    for row in wall_pairs["maps"]:
        for body, counts in (("torso", "torso"), ("one leg part", "one leg part")):
            lines.append(f"| {row['map']} | {row['wall_boxes']} | {row['face_points']:,} | {body} | " + " | ".join(
                f"{row['not_tested'][str(k)][counts]:,} ({100 * row['not_tested'][str(k)][counts] / row['face_points']:.0f}%)"
                for k in limits) + " |")
    return lines + ["", f"Computed {wall_pairs['measured_pacific']} on processors by"
                    " `units/wall_pairs_kept_on_processors.py`.", ""]


def render(record, gates, wall_pairs=None, previews=()):
    """The whole Markdown report from the record, the gate record, the wall-pair record and the processor previews."""
    cells = usable_cells({"runs": list(record["runs"]) + list(previews)})
    kept = [run for run in record["runs"] if not run.get("discarded") and run["card"]["name"] != "processors"]
    lines = ["# AntMaze on MuJoCo MJX: settings, speed and physics checks", "",
             "Generated by `benchmarks/env/profile_antmaze_mjx_settings.py --render` from"
             " `antmaze_mjx_settings.json` and `gates_on_processors.json`; change the script, not this file."
             " Times are Pacific, converted from machines running on Eastern. Measurements are on the U-maze"
             " unless a heading names another map.", ""]
    units = {}
    for run in kept:
        units.setdefault(run["unit"], []).append(run)
    for unit, runs in units.items():
        v, first = runs[0]["versions"], runs[0]
        ended = [run["end_epoch"] for run in runs if run.get("finished")]
        wall = (f"{(max(ended) - first['unit_start_epoch']) / 60:.1f} minutes from the unit's start to the end of"
                " its last finished part" if ended else "no part finished")
        lines.append(f"- Unit `{unit}`: {first['machine']} ({first['card']['name']}, host {first['host']}), started"
                     f" {first['unit_started_pacific']}, {wall}; JAX {v['jax']}, jaxlib {v['jaxlib']}, MuJoCo"
                     f" {v['mujoco']}, MJX {v['mujoco-mjx']}, Python {first['python']}, driver"
                     f" {first['card']['driver']}.")
    lines += ["", "## Speed of the environment step", "",
              "Define $n$ as the number of environments and $s$ as the median seconds per environment step"
              " (one step of all $n$ environments, 5 physics steps each).", "",
              "- Total steps per second: $\\dfrac{n}{s}$, shown in millions.",
              "- Steps per second per environment: $r = \\dfrac{1}{s}$.",
              "- Hours per million steps per environment: $\\dfrac{10^{6}}{3600 \\times r}$.",
              "- Time against the default model: $\\dfrac{s}{s_{0}} - 1$, with $s_{0}$ the default model's"
              " $s$ in the same cell.", "",
              "Rows are sorted by seconds per environment step. The best value of a column is bold and the"
              " second best underlined.", ""]
    speed = [cell for name, cell in cells.items() if name.startswith("speed: ")]
    for cell in speed:
        lines += render_speed(cell)
    if not speed:
        lines += ["Not measured.", ""]
    lines += render_equality(cells.get(EQUALITY_NAME))
    lines += render_checks(cells, gates)
    for map_name in ("medium", "large"):
        if any(name.startswith(checks_prefix(map_name)) for name in cells):
            lines += render_checks(cells, gates, map_name)
    for map_name in ("umaze", "medium", "large"):
        if any(name.startswith(PREVIEW + checks_prefix(map_name)) for name in cells):
            lines += render_checks(cells, gates, map_name, preview=True)
    lines += render_wall_pairs(wall_pairs)
    lines += ["## Not measured or not used", ""]
    unused = unusable_cells(record)
    lines += [f"- {name} (unit `{unit}`): {reason}." for name, unit, reason in unused] or ["Nothing."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------------------------

def start_facts(state):
    """What a start state looks like, for either kind of carried physics state."""
    import jax.numpy as jnp
    return {"environments_reset_for_a_non_finite_state": int(state.nan_count.sum()),
            "state_is_finite": bool(jnp.isfinite(state.data.qpos).all() & jnp.isfinite(state.data.qvel).all())}


def build_env(changes, n_envs, map_name="umaze", **more):
    """The environment of one model on one map at one batch size (one environment per copy)."""
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    return JaxAntMaze(dataclasses.replace(preset(map_name), **changes, **more), n_envs, 1)


def measure_speed(cell, models, n_envs, rounds, save, map_name="umaze"):
    """Time the whole environment step of each model at one batch size, in turn."""
    functions = []
    for changes in models:
        # GOAL: build and compile this model, and bring it to its own start state.
        env = build_env(changes, n_envs, map_name)
        env_call = split.build_env_call(env)
        start, first_call = split.settle(env, env_call)
        torques = split.uniform_torques("timed torques", 0, env.B, env.act_dim)
        name = model_name(changes)
        cell["functions"][name] = {"environments": n_envs, "changes": changes, "start_state": start_facts(start)}
        functions.append((name, functools.partial(env_call, start, torques), first_call))
    split.time_functions(cell, functions, rounds, save)


# ----------------------------------------------------------------------------------------------
# equality
# ----------------------------------------------------------------------------------------------

def tree_difference(a, b):
    """Per environment: whether any array of two trees differs, and the largest relative difference."""
    import jax
    import jax.numpy as jnp
    n_envs = jax.tree.leaves(a)[0].shape[0]
    differs, worst = jnp.zeros((n_envs,), bool), jnp.zeros((n_envs,), jnp.float32)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if x.size == 0:
            continue
        # before: x, y of shape [n_envs, ...]; after: one row per environment, NaN on both sides counts as equal
        x, y = x.reshape(n_envs, -1).astype(jnp.float32), y.reshape(n_envs, -1).astype(jnp.float32)
        unequal = (x != y) & ~(jnp.isnan(x) & jnp.isnan(y))
        differs = differs | unequal.any(-1)
        gap = jnp.where(unequal, jnp.abs(x - y) / (1.0 + jnp.abs(y)), 0.0)
        worst = jnp.maximum(worst, jnp.nan_to_num(gap, nan=jnp.inf, posinf=jnp.inf).max(-1))
    return differs, worst


class Tally:
    """Over the steps of one comparison: which environments ever differed, and by how much."""

    def __init__(self):
        """Nothing compared yet."""
        self.differs, self.worst, self.pairs, self.pairs_over = None, 0.0, 0, 0

    def add(self, differs, worst):
        """Fold in the comparison of one step: differs [n_envs] bool, worst [n_envs] relative differences."""
        import numpy as np
        differs, worst = np.asarray(differs), np.asarray(worst)
        self.differs = differs if self.differs is None else (self.differs | differs)
        self.worst = max(self.worst, float(worst.max()))
        self.pairs += worst.size
        self.pairs_over += int((worst > NOTABLE_DIFFERENCE).sum())

    def row(self, comparison, steps, twice):
        """One row of the equality table, beside the same comparison between two calls of one program."""
        return {"comparison": comparison, "steps": steps,
                "environments_not_equal": int(self.differs.sum()),
                "share_over": self.pairs_over / self.pairs, "largest_relative_difference": self.worst,
                "twice_environments_not_equal": int(twice.differs.sum()),
                "twice_share_over": twice.pairs_over / twice.pairs,
                "twice_largest_relative_difference": twice.worst}


def compare_env_rollouts(calls, start, torque_name, chunks):
    """Run several compiled environment calls from their start states; compare each with the first, step by step.

    calls: {name: compiled call}; the first is the reference and is run twice. Returns {name: Tally}
    with the extra name "same program twice".
    """
    import jax
    import numpy as np
    difference = jax.jit(tree_difference)
    names = list(calls)
    states = {name: start[name] for name in names}
    states["same program twice"] = start[names[0]]
    tallies = {name: Tally() for name in list(names[1:]) + ["same program twice"]}
    n_envs = states[names[0]].step_count.size
    for chunk in range(chunks):
        torques = split.uniform_torques(torque_name, chunk, n_envs, 8)
        outputs = {}
        for name in states:
            call = calls[names[0]] if name == "same program twice" else calls[name]
            states[name], outputs[name] = call(states[name], torques)
        # GOAL: compare the observations, rewards and flags of every step of the chunk.
        # before: each output [steps, C, N, ...]; after: per step a list of arrays [n_envs, ...]
        for step in range(split.ENV_STEPS_PER_CALL):
            reference = [np.asarray(x[step]).reshape(n_envs, -1) for x in outputs[names[0]]]
            for name, tally in tallies.items():
                other = [np.asarray(x[step]).reshape(n_envs, -1) for x in outputs[name]]
                tally.add(*difference(other, reference))
    return tallies


def measure_equality(cell, n_envs, save):
    """Compare the two speed options with the stepper as shipped; fills cell['comparisons']."""
    import jax
    from mujoco import mjx
    from exploration_platform.envs.antmaze.mjx_antmaze import physics_step_fixed_count
    fixed, carried = {"solver_loop": "fixed_count"}, {"carried_state": "next_step_inputs"}
    envs = {"shipped": build_env({}, n_envs), "fixed": build_env(fixed, n_envs),
            "scan": build_env({"solver_loop": "fixed_count_scan"}, n_envs),
            "carried": build_env(carried, n_envs), "both": build_env({**fixed, **carried}, n_envs)}
    forms = {"fixed": "fixed-count solver loop", "scan": "fixed-count solver loop as one scan"}
    calls = {name: split.build_env_call(env) for name, env in envs.items()}
    cell["environments"], cell["comparisons"] = n_envs, []
    landed, _ = split.settle(envs["shipped"], calls["shipped"])
    cell["start_state"] = split.state_facts(landed)

    # GOAL: one physics step computed both ways from the same data, 50 steps along mjx.step's trajectory.
    def stepper(physics_step):
        """A compiled physics step of every environment."""
        def call(data):
            with jax.enable_x64(False):
                return jax.vmap(physics_step, in_axes=(None, 0))(envs["shipped"].model, data)
        return jax.jit(call)

    def next_inputs(data):
        """The arrays of the MJX data that the next physics step reads."""
        return [data.qpos, data.qvel, data.qacc_warmstart]
    step_shipped, difference = stepper(mjx.step), jax.jit(tree_difference)
    steppers = {"fixed": stepper(physics_step_fixed_count),
                "scan": stepper(functools.partial(physics_step_fixed_count, written_out=False))}
    data = landed.data
    inputs, every = {name: Tally() for name in (*forms, "twice")}, {name: Tally() for name in (*forms, "twice")}
    for k in range(50):
        data = data.replace(ctrl=split.uniform_torques("one-step torques", k, n_envs, 8)[0])
        reference = step_shipped(data)
        others = {"twice": step_shipped(data), **{name: step(data) for name, step in steppers.items()}}
        for name, other in others.items():
            inputs[name].add(*difference(next_inputs(other), next_inputs(reference)))
            every[name].add(*difference(other, reference))
        data = reference
    for name, words in forms.items():
        cell["comparisons"] += [
            inputs[name].row(f"{words}: one physics step from the same state, the arrays the next step reads"
                             " (positions, velocities, solver start values)", 50, inputs["twice"]),
            every[name].row(f"{words}: one physics step from the same state, every array of the MJX data"
                            " (constraint forces included)", 50, every["twice"])]
    save()

    # GOAL: 50 environment steps run freely from the landed state, each model on its own trajectory.
    tallies = compare_env_rollouts({name: calls[name] for name in ("shipped", *forms)},
                                   {name: landed for name in ("shipped", *forms)}, "free-run torques", 5)
    for name, words in forms.items():
        cell["comparisons"].append(tallies[name].row(
            f"{words}: 50 environment steps run freely (observations, rewards, flags)", 50,
            tallies["same program twice"]))
    save()

    # GOAL: 100 environment steps from the spawn with only the next step's inputs carried.
    tallies = compare_env_rollouts({"shipped": calls["shipped"], "carried": calls["carried"]},
                                   {"shipped": envs["shipped"].reset(), "carried": envs["carried"].reset()},
                                   "carried-state torques", 10)
    cell["comparisons"].append(tallies["carried"].row(
        "only the next step's inputs carried: 100 environment steps from the spawn", 100,
        tallies["same program twice"]))
    tallies = compare_env_rollouts({"fixed": calls["fixed"], "both": calls["both"]},
                                   {"fixed": envs["fixed"].reset(), "both": envs["both"].reset()},
                                   "carried-state torques", 10)
    cell["comparisons"].append(tallies["both"].row(
        "only the next step's inputs carried, on top of the fixed-count loop: 100 environment steps from the spawn"
        " (both sides of the row compare with the fixed-count loop)", 100, tallies["same program twice"]))
    save()
    for row in cell["comparisons"]:
        print(f"  {row['comparison']}: {row['environments_not_equal']} environments not equal,"
              f" {100 * row['share_over']:.2f}% of environment-steps over {NOTABLE_DIFFERENCE:g}, largest relative"
              f" difference {row['largest_relative_difference']:.2e}; same program twice:"
              f" {row['twice_environments_not_equal']}, {100 * row['twice_share_over']:.2f}%,"
              f" {row['twice_largest_relative_difference']:.2e}", flush=True)


# ----------------------------------------------------------------------------------------------
# physics checks
# ----------------------------------------------------------------------------------------------

def torque_block(name, chunk, steps, n_envs):
    """Uniform torques in [-1, 1], shape [steps, n_envs, 8], one random stream per environment."""
    import jax
    import jax.numpy as jnp
    key = split.substream_key(name, chunk)
    # before: per environment i a [steps, 8] block from fold_in(key, i); after: axes [steps, n_envs, 8]
    per_env = jax.vmap(lambda i: jax.random.uniform(
        jax.random.fold_in(key, i), (steps, 8), jnp.float32, -1.0, 1.0))(jnp.arange(n_envs))
    return jnp.swapaxes(per_env, 0, 1)


def wall_torque_block(chunk, steps, n_envs):
    """The torques of the wall rollout for one chunk, shape [steps, n_envs, 8]."""
    import jax
    import jax.numpy as jnp
    # GOAL: even environments hold one vector; odd ones switch the sign of full torques every SWITCH_STEPS.
    held = jax.random.uniform(split.substream_key("wall rollout held torques"), (n_envs, 8), jnp.float32, -1.0, 1.0)
    phase = jax.random.randint(split.substream_key("wall rollout switching phases"), (n_envs, 8), 0, 2 * SWITCH_STEPS)
    # before: t = step numbers [steps]; phase [n_envs, 8] in 0 .. 2 * SWITCH_STEPS - 1
    # after:  +1 for SWITCH_STEPS steps, then -1 for SWITCH_STEPS steps, shifted per joint by its phase
    t = chunk * steps + jnp.arange(steps)
    switching = jnp.where(((t[:, None, None] + phase[None]) // SWITCH_STEPS) % 2 == 0, 1.0, -1.0)
    even = (jnp.arange(n_envs) % 2 == 0)[None, :, None]
    return jnp.where(even, held[None], switching).astype(jnp.float32)


def build_check_call(env, reference_model, geom_is_wall):
    """Compiled function: CHECK_CHUNK environment steps, and after each step the facts of the new state.

    geom_is_wall: one flag per geometry of the model, True for the wall boxes. Returns the function, the
    model's contact slots and the number of environments whose solver is looked at.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    from exploration_platform.envs.antmaze.mjx_antmaze import newton_passes, wants_another_pass
    with jax.enable_x64(False):
        blank, reference_blank = mjx.make_data(env.model), mjx.make_data(reference_model)
    converged_model = env.model.replace(opt=env.model.opt.replace(
        iterations=REFERENCE_PASSES, ls_iterations=REFERENCE_LINE_SEARCH))
    slots, passes_allowed = int(blank._impl.contact.dist.shape[0]), env.cfg.solver_iterations
    sample, is_wall = min(SOLVER_SAMPLE, env.B), jnp.asarray(geom_is_wall)

    def near_contacts(model, data):
        """The contact list of one model at one state; which contacts are within their margin, which touch a wall."""
        contact = mjx.collision(model, mjx.kinematics(model, data))._impl.contact
        near = contact.dist < contact.includemargin
        # before: contact.geom [slots, 2] geometry numbers; after: [slots] True where either one is a wall box
        return contact, near, near & is_wall[contact.geom].any(-1)

    def contact_facts(qpos, qvel):
        """One environment: its own contacts beside the reference list, and where the torso is."""
        _, own_near, _ = near_contacts(env.model, blank.replace(qpos=qpos))
        reference, near, wall = near_contacts(reference_model, reference_blank.replace(qpos=qpos))
        contacts, deepest_wall = near.sum(), jnp.where(wall, reference.dist, 0.0).min()
        deepest_floor = jnp.where(near & ~wall, reference.dist, 0.0).min()
        torso_in_wall = ~env.open_cells[env.cell_index(qpos)] & (qpos[2] < WALL_HEIGHT)
        return {
            "steps_over_slots": contacts > slots,
            "steps_missed": contacts > own_near.sum(),
            "steps_missed_with_free_slots": (contacts > own_near.sum()) & (contacts <= slots),
            "steps_with_wall_contact": wall.any(),
            "steps_deep_in_wall": deepest_wall < -DEEP_PENETRATION,
            "steps_deep_in_floor": deepest_floor < -DEEP_PENETRATION,
            "steps_above_walls": qpos[2] > WALL_HEIGHT,
            "steps_torso_inside_wall": torso_in_wall,
            "steps_upside_down": 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2) < 0.0,
            "contacts_sum": contacts, "contacts_max": contacts,
            "deepest_wall_min": deepest_wall, "deepest_floor_min": deepest_floor,
            "height_max": qpos[2], "speed_max": jnp.abs(qvel).max(),
        }

    def solver_facts(qpos, qvel, warmstart, ctrl):
        """One environment: the model's solver at this state beside a converged solve of the same constraints."""
        data = blank.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warmstart, ctrl=ctrl)
        data = mjx.fwd_position(env.model, data)
        data = mjx.fwd_acceleration(env.model, mjx.fwd_actuation(env.model, mjx.fwd_velocity(env.model, data)))
        ctx, exact = newton_passes(env.model, data), newton_passes(converged_model, data)
        # GOAL: the largest difference of any acceleration, against the largest acceleration; non-finite counts as infinite.
        error = jnp.abs(ctx.qacc - exact.qacc).max() / (1.0 + jnp.abs(exact.qacc).max())
        error = jnp.nan_to_num(error, nan=jnp.inf, posinf=jnp.inf)
        return {
            "sampled_steps_unconverged": wants_another_pass(env.model, ctx),
            "sampled_steps_error_over_0.1_percent": error > 1e-3,
            "sampled_steps_error_over_1_percent": error > 1e-2,
            "sampled_steps_error_over_10_percent": error > 1e-1,
            "solver_error_max": error,
            "passes_histogram": ctx.solver_niter == jnp.arange(passes_allowed + 1),
        }

    def call(state, torques):
        """state: the environment state; torques [CHECK_CHUNK, B, 8]."""
        def body(st, torque):
            """One environment step, then the facts of the new state reduced over the environments."""
            st, *_ = env.step(st, torque.reshape(env.C, env.N, env.act_dim))
            with jax.enable_x64(False):
                facts = jax.vmap(contact_facts)(st.data.qpos, st.data.qvel)
                facts.update(jax.vmap(solver_facts)(
                    st.data.qpos[:sample], st.data.qvel[:sample], st.data.qacc_warmstart[:sample],
                    jnp.clip(torque[:sample], -1.0, 1.0)))
                # before: every fact [B] or [sample] (the histogram [sample, passes + 1]); after: one number per fact
                reduced = {name: (value.max(0) if name.endswith("_max") else value.min(0) if name.endswith("_min")
                                  else value.astype(jnp.int32).sum(0)) for name, value in facts.items()}
            return st, reduced
        state, per_step = jax.lax.scan(body, state, torques)
        # before: every fact [CHECK_CHUNK, ...]; after: reduced over the steps the same way
        return state, {name: (value.max(0) if name.endswith("_max") else value.min(0) if name.endswith("_min")
                              else value.sum(0)) for name, value in per_step.items()}
    return jax.jit(call), slots, sample


def run_rollout(check_call, state, torque_fn, steps):
    """Step a rollout in chunks; returns the last state, the summed facts and the seconds of the first call."""
    import jax
    import numpy as np
    counts, first_call_seconds = {}, None
    for chunk in range(steps // CHECK_CHUNK):
        torques = torque_fn(chunk)
        jax.block_until_ready(torques)
        t0 = time.perf_counter()
        state, facts = check_call(state, torques)
        jax.block_until_ready(facts)
        if first_call_seconds is None:
            first_call_seconds = time.perf_counter() - t0
        # GOAL: fold the chunk into the totals: largest, smallest or sum, by the fact's name.
        for name, value in facts.items():
            value = np.asarray(value).tolist()
            if name not in counts:
                counts[name] = value
            elif name.endswith("_max"):
                counts[name] = max(counts[name], value)
            elif name.endswith("_min"):
                counts[name] = min(counts[name], value)
            elif isinstance(value, list):
                counts[name] = [a + b for a, b in zip(counts[name], value)]
            else:
                counts[name] += value
    counts["non_finite_respawns"] = int(state.nan_count.sum())
    return state, counts, first_call_seconds


def measure_checks(cell, changes, n_envs, save, map_name="umaze", travel_steps=TRAVEL_STEPS, rest_steps=REST_STEPS):
    """The three rollouts of one model on one map; fills cell['rollouts'] and saves after each."""
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml
    # GOAL: the model with the episode limit switched off, and the same model without contact limits.
    env = build_env(changes, n_envs, map_name, max_episode_steps=10 ** 6)
    unlimited = mujoco.MjModel.from_xml_string(build_antmaze_xml(
        dataclasses.replace(env.cfg, max_geom_pairs=-1, max_contact_points=-1)))
    with jax.enable_x64(False):
        reference_model = mjx.put_model(unlimited)
    # the wall boxes are the geometries build_antmaze_xml names block_<k>; every other contact is with the floor
    geom_is_wall = [mujoco.mj_id2name(unlimited, mujoco.mjtObj.mjOBJ_GEOM, g).startswith("block_")
                    for g in range(unlimited.ngeom)]
    check_call, slots, sample = build_check_call(env, reference_model, geom_is_wall)
    cell.update(environments=n_envs, changes=changes, contact_slots=slots, solver_sample=sample,
                solver_iterations=env.cfg.solver_iterations, ls_iterations=env.cfg.ls_iterations, rollouts={})

    def finish(name, steps, counts, first_call_seconds, t0, **more):
        """Store one rollout in the cell."""
        cell["rollouts"][name] = dict(steps=steps, environment_steps=steps * n_envs,
                                      sampled_steps=steps * sample, counts=counts,
                                      first_call_seconds=first_call_seconds, seconds=time.time() - t0, **more)
        save()
        print(f"  {name}: {steps} steps in {time.time() - t0:.0f} s, non-finite respawns"
              f" {counts['non_finite_respawns']}, over the contact limit {counts['steps_over_slots']}, missed"
              f" {counts['steps_missed']}, solver error over 1 percent in"
              f" {counts['sampled_steps_error_over_1_percent']} of {steps * sample} sampled steps", flush=True)

    # GOAL: random torques from the spawn; the distance each torso travelled.
    t0, start = time.time(), env.reset()
    state, counts, first = run_rollout(check_call, start, functools.partial(
        torque_block, "check random torques", steps=CHECK_CHUNK, n_envs=n_envs), travel_steps)
    distance = np.linalg.norm(np.asarray(state.data.qpos[:, :2]) - np.asarray(start.data.qpos[:, :2]), axis=1)
    clean = np.asarray(state.nan_count).reshape(-1) == 0
    every = distance[clean]
    finish(RANDOM_ROLLOUT, travel_steps, counts, first, t0,
           distance_first_sequences=distance[:TRAVEL_SEQUENCES][clean[:TRAVEL_SEQUENCES]].tolist(),
           distance_all={"environments": int(clean.sum()), "mean": float(every.mean()),
                         "spread": float(every.std(ddof=1)), "median": float(np.median(every)),
                         "largest": float(every.max())})

    # GOAL: starts spread over the start cell, torques that push in one direction or in a fixed rhythm.
    t0 = time.time()
    offsets = jax.random.uniform(split.substream_key("wall rollout start offsets"), (n_envs, 2), jnp.float32,
                                 -START_SPREAD, START_SPREAD)
    start = env.reset()
    start = start._replace(data=start.data.replace(qpos=start.data.qpos.at[:, :2].add(offsets)))
    state, counts, first = run_rollout(check_call, start, functools.partial(
        wall_torque_block, steps=CHECK_CHUNK, n_envs=n_envs), travel_steps)
    finish(WALL_ROLLOUT, travel_steps, counts, first, t0)

    # GOAL: no torques from the spawn; where the ant comes to rest.
    t0 = time.time()
    zeros = jnp.zeros((CHECK_CHUNK, n_envs, 8), jnp.float32)
    state, counts, first = run_rollout(check_call, env.reset(), lambda chunk: zeros, rest_steps)
    height = np.asarray(state.data.qpos[:, 2])
    finish(REST_ROLLOUT, rest_steps, counts, first, t0,
           resting_height_mean=float(height.mean()), resting_height_min=float(height.min()),
           resting_height_max=float(height.max()),
           resting_speed_max=float(np.abs(np.asarray(state.data.qvel)).max()))


# ----------------------------------------------------------------------------------------------
# the gates on processors
# ----------------------------------------------------------------------------------------------

GATE_PATTERNS = {
    "test_one_step_parity": (
        ("one step, no contact (m)", r"contact-free(?: states)?, worst(?: \|dqpos\|)? ([0-9.e+-]+)"),
        ("one step, in contact (m)", r"in contact, worst ([0-9.e+-]+)")),
    "test_trajectory_divergence_structure": (
        ("before the first contact (m)", r"pre-contact worst ([0-9.e+-]+)"),
        ("20 steps (m)", r"final divergence ([0-9.e+-]+)")),
}


def gate_numbers(name, printed):
    """The numbers one gate printed before its assertions, by table column."""
    numbers = {}
    for column, pattern in GATE_PATTERNS.get(name, ()):
        found = re.search(pattern, printed)
        if found:
            numbers[column] = float(found.group(1).rstrip("."))
    if name == "test_equilibrium_preservation":
        # before: "reference -> tuned: settled z 0.38248, after handover z 0.38248, max |dqpos| 3.70e-11" (twice)
        # after:  the larger of the two moves, and the gap between the two settled heights
        rows = re.findall(r"settled z ([0-9.]+), after handover z [0-9.]+, max \|dqpos\| ([0-9.e+-]+)", printed)
        if rows:
            numbers["handover drift (m)"] = max(float(drift) for _, drift in rows)
        if len(rows) == 2:
            numbers["resting heights apart (m)"] = abs(float(rows[0][0]) - float(rows[1][0]))
    return numbers


def measure_gates(models, out):
    """Run the three gates of test_antmaze_parity_cpu_mujoco.py for each model; write the gate record.

    A model already in the record is replaced by its new row; the other rows are kept.
    """
    from exploration_platform.envs.antmaze.am_common import preset
    sys.path.insert(0, str(BASE / "tests" / "envs"))
    import test_antmaze_parity_cpu_mujoco as gates
    earlier = json.loads(out.read_text())["models"] if out.exists() else []
    rerun = {model_name(changes) for changes in models}
    record = {"host": socket.gethostname(), "measured_pacific": split.pacific(time.time()),
              "mujoco": importlib.metadata.version("mujoco"), "jax": importlib.metadata.version("jax"),
              "models": [row for row in earlier if row["model"] not in rerun]}
    names = ("test_one_step_parity", "test_trajectory_divergence_structure", "test_equilibrium_preservation")
    for changes in models:
        gates.CFG = dataclasses.replace(preset("umaze"), **changes)
        row = {"model": model_name(changes), "changes": changes, "gates": {}, "numbers": {}}
        for name in names:
            # GOAL: run the gate as written; keep what it printed and the assertion it failed, if any.
            printed, failure = io.StringIO(), None
            with contextlib.redirect_stdout(printed):
                try:
                    getattr(gates, name)()
                except AssertionError as error:
                    failure = str(error)
            row["gates"][name] = {"passed": failure is None, "printed": printed.getvalue(), "assertion": failure}
            row["numbers"].update(gate_numbers(name, printed.getvalue()))
            print(f"{row['model']}: {name} {'passed' if failure is None else 'FAILED: ' + failure}", flush=True)
        record["models"].append(row)
        split.flush(record, out)
    return record


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def main():
    """Measure one part inside the unit's time limit, saving as it goes; or render."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the JSON record file; runs are appended to it")
    ap.add_argument("--render", action="store_true", help="only rebuild the Markdown from the records")
    ap.add_argument("--machine", default=None, help='for example "serval02 card 1"')
    ap.add_argument("--unit", default=None, help="name of the unit this run belongs to")
    ap.add_argument("--part", choices=["speed", "equality", "checks"])
    ap.add_argument("--models", nargs="+", default=["default"], help="field=value,field=value or default")
    ap.add_argument("--group", default="models", help="speed: what this group of models is, in words")
    ap.add_argument("--batch", type=int, default=4096, help="environments (speed, equality, checks)")
    ap.add_argument("--map", default="umaze", choices=["umaze", "medium", "large"], help="speed and checks")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--unit-start-epoch", type=float, default=None, help="when the unit started (date +%%s)")
    ap.add_argument("--unit-minutes", type=float, default=15.0, help="wall-time limit of the whole unit")
    ap.add_argument("--processor-trial", action="store_true",
                    help="run on processors to try the code; the run is never used in the tables")
    ap.add_argument("--trial-steps", type=int, nargs=2, default=None,
                    help="processor trial only: steps of the torque rollouts and of the rest rollout")
    args = ap.parse_args()
    out = Path(args.out)
    gates_file, walls_file = out.with_name("gates_on_processors.json"), out.with_name("wall_pairs_kept_on_processors.json")
    side_records = lambda: [json.loads(path.read_text()) if path.exists() else None for path in (gates_file, walls_file)] + [
        [json.loads(path.read_text()) for path in sorted(out.parent.glob("processor_previews/*.json"))]]
    if args.render:
        out.with_suffix(".md").write_text(render(json.loads(out.read_text()), *side_records()))
        print(f"wrote {out.with_suffix('.md')}")
        return
    if args.machine is None or args.unit is None or args.part is None:
        ap.error("--machine, --unit and --part are required for a measurement")
    unit_start = args.unit_start_epoch or time.time()
    deadline = unit_start + 60.0 * args.unit_minutes - 45.0

    # GOAL: refuse to measure on the processor unless this is a trial of the code.
    import jax
    import exploration_platform  # noqa: F401  (64-bit statistics switched on before any array exists, as in training)
    on_card = jax.devices()[0].platform == "gpu"
    if not on_card and not args.processor_trial:
        print(f"JAX did not load the graphics card (device {jax.devices()[0]}); nothing measured")
        sys.exit(split.NO_GPU_EXIT_CODE)

    # GOAL: the run's identity: machine, card, versions, sources, time.
    card = split.card_record() if on_card else {"name": "processors", "uuid": "", "driver": ""}
    watch = (lambda: split.other_processes_on_card(card["uuid"])) if on_card else (lambda: [])
    here = Path(__file__).resolve()
    env_dir = BASE / "src" / "exploration_platform" / "envs" / "antmaze"
    run = {
        "unit": args.unit, "machine": args.machine, "host": socket.gethostname(), "card": card,
        "started_pacific": split.pacific(time.time()), "started_iso": datetime.now().astimezone().isoformat(),
        "unit_start_epoch": unit_start, "unit_started_pacific": split.pacific(unit_start),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "numpy")},
        "python": sys.version.split()[0],
        "source_sha256": {path.name: split.sha256_of(path)
                          for path in (here, env_dir / "mjx_antmaze.py", env_dir / "am_common.py")},
        "settings": {"part": args.part, "models": args.models, "batch": args.batch, "rounds": args.rounds,
                     "map": args.map,
                     "base_seed": split.BASE_SEED},
        "cells": [], "finished": False,
    }
    record = json.loads(out.read_text()) if out.exists() else {"runs": []}
    record["runs"].append(run)
    save = functools.partial(split.flush, record, out)
    save()
    print(f"{run['started_pacific']} unit {args.unit} on {args.machine}: {card['name']} {card['uuid']}, part"
          f" {args.part}, JAX {run['versions']['jax']}, MuJoCo {run['versions']['mujoco']}", flush=True)

    # GOAL: the cells of this part; a cell that is not expected to fit is recorded as not measured.
    models = [parse_changes(text) for text in args.models]
    if args.part == "speed":
        plan = [("speed", speed_name(args.group, args.batch, args.map), FIRST_GUESS_SECONDS["speed"] * len(models),
                 functools.partial(measure_speed, models=models, n_envs=args.batch, rounds=args.rounds,
                                   map_name=args.map))]
    elif args.part == "equality":
        plan = [("equality", EQUALITY_NAME, FIRST_GUESS_SECONDS["equality"],
                 functools.partial(measure_equality, n_envs=args.batch))]
    else:
        steps = {} if args.trial_steps is None else dict(travel_steps=args.trial_steps[0],
                                                         rest_steps=args.trial_steps[1])
        plan = [("checks", checks_name(model_name(changes), args.map), FIRST_GUESS_SECONDS["checks"],
                 functools.partial(measure_checks, changes=changes, n_envs=args.batch, map_name=args.map, **steps))
                for changes in models]
    pace = 1.0
    for kind, name, guess, measure in plan:
        needed, left = guess * pace, deadline - time.time()
        if needed > left:
            run["cells"].append({"kind": kind, "name": name, "status": "not measured",
                                 "reason": f"{left:.0f} s were left of the {args.unit_minutes:.0f}-minute unit"
                                           f" and the cell was expected to take {needed:.0f} s"})
            print(f"not measured: {name} ({left:.0f} s left, {needed:.0f} s expected)", flush=True)
            save()
            continue
        t0 = time.time()
        cell = {"kind": kind, "name": name, "status": "running", "measured_pacific": split.pacific(t0),
                "env_steps_per_call": split.ENV_STEPS_PER_CALL, "rounds_done": 0, "functions": {},
                "other_processes_on_card": watch()}
        run["cells"].append(cell)
        print(f"{split.pacific(t0)} cell: {name}", flush=True)
        measure(cell, save=save)
        cell["other_processes_on_card"] += watch()
        cell["wall_seconds"] = time.time() - t0
        cell["status"] = "measured"
        pace = min(3.0, max(0.5, cell["wall_seconds"] / guess))
        save()
        print(f"  cell took {cell['wall_seconds']:.0f} s; {deadline - time.time():.0f} s left", flush=True)

    # GOAL: close the run and rebuild the tables.
    run["finished"], run["end_epoch"] = True, time.time()
    save()
    if on_card:
        out.with_suffix(".md").write_text(render(record, *side_records()))
    print(f"{split.pacific(time.time())} part {args.part} finished; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
