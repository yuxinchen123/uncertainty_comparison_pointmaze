"""Can MuJoCo Warp (inside MuJoCo MJX) replace the JAX implementation for the AntMaze physics?

Measures, on one graphics card, the Warp AntMaze environment
(`exploration_platform/envs/antmaze/warp_antmaze.py`) beside the JAX one (`mjx_antmaze.py`):

  speed-warp, speed-jax  the environment step at a list of batch sizes, smallest first, each size
                         its own cell; a list stops when a doubling gains under 10% of total
                         steps per second or when the card's memory runs out
  settings               Warp with fewer solver iterations, fewer line-search iterations, other
                         pool sizes and other graph modes, all timed in turn at one batch size
  maps                   Warp on the U-maze, the Medium and the Large map, timed in turn
  trainer                the trainer's kind of program: a donated state carried through lax.scan
                         with resets inside
  platform-trainer-warp, platform-trainer-jax
                         the platform's own training iteration (rollout, bonus, advantages, update)
                         with each implementation of the physics
  second-process         a second process stepping its own Warp environment on the same card
  checks                 the correctness measurements of `antmaze_warp_checks.py`
  soak                   random torques at 4,096 environments for at least 2 million steps
  fewer-iterations       Warp with fewer solver iterations: one step against C MuJoCo of the specified
                         model, and the same soak

The timing is that of `profile_antmaze_physics_split.py`, whose functions are reused: the start
state is reached by 100 random-torque environment steps from the spawn, one timed call runs 10
environment steps from that state, each function is called once (tracing, compilation, kernel
loading, one execution), twice more to warm up, then in 15 timed rounds; the reported time is the
median. A Warp cell stops the unit with an error if MuJoCo Warp dropped a contact, a candidate
pair or a constraint row.

Usage (on a machine with a graphics card, never on a login node):
  WARP_CACHE_PATH=<node-local folder> PYTHONNOUSERSITE=1 PYTHONPATH=<platform>/src \
      <platform_jax_warp python> profile_antmaze_warp.py --out <folder>/antmaze_warp.json \
      --machine "serval02 card 1" --unit <unit name> --parts speed-warp
Rebuild the Markdown tables from the record (standard library only, any machine):
  python profile_antmaze_warp.py --out <folder>/antmaze_warp.json --render
"""
import argparse
import functools
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

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from profile_antmaze_physics_split import (  # noqa: E402  (standard library only at import)
    BASE, ENV_STEPS_PER_CALL, ENVIRONMENT_STEP, MIN_ROUNDS, NO_GPU_EXIT_CODE, build_env_call, card_record,
    flush, median_ms, other_processes_on_card, pacific, settle, sha256_of, throughput_row, time_functions,
    uniform_torques)

STOP_GAIN = 0.10             # a doubling that gains less than this share of total steps per second ends a list
SOAK_ENVIRONMENTS = 4096
SOAK_REQUIRED_STEPS = 2_000_000
TRAINER_ROLLOUT_STEPS, TRAINER_CALLS, TRAINER_EPISODE_STEPS = 128, 8, 50
SECOND_PROCESS_ENVIRONMENTS = (1024, 256)   # this process, the second process
DEFAULT_WARP = "Warp default: 4 solver iterations, 8 line-search iterations, 48 pool slots per environment, 160 rows"
# name -> (configuration changes, WarpAntMaze options, compiler options)
SETTINGS = (
    (DEFAULT_WARP, {}, {}, None),
    ("solver iterations 2", {"solver_iterations": 2}, {}, None),
    ("solver iterations 1", {"solver_iterations": 1}, {}, None),
    ("line-search iterations 4", {"ls_iterations": 4}, {}, None),
    ("solver iterations 2 and line-search iterations 4", {"solver_iterations": 2, "ls_iterations": 4}, {}, None),
    ("24 pool slots per environment", {}, {"contacts_per_env": 24}, None),
    ("96 constraint rows per environment", {}, {"njmax": 96}, None),
    ("graph mode WARP_STAGED", {}, {"graph_mode": "WARP_STAGED"}, None),
    ("compiler option xla_gpu_graph_min_graph_size=1", {}, {}, {"xla_gpu_graph_min_graph_size": 1}),
)
# seconds a cell is expected to take before any cell of its kind has been timed in this process
FIRST_GUESS_SECONDS = {"speed": 60.0, "settings": 240.0, "maps": 60.0, "trainer": 60.0, "second process": 90.0,
                       "check": 60.0, "soak": 150.0, "platform trainer": 200.0}
MAPS = ("umaze", "medium", "large")
DEFAULT_SOAK_SETTING = "4 solver iterations, 8 line-search iterations"
FEWER_ITERATIONS = (("2 solver iterations", {"solver_iterations": 2}),
                    ("1 solver iteration", {"solver_iterations": 1}),
                    ("2 solver iterations and 4 line-search iterations", {"solver_iterations": 2, "ls_iterations": 4}))


# ----------------------------------------------------------------------------------------------
# the Markdown tables (standard library only)
# ----------------------------------------------------------------------------------------------

def speed_name(implementation, n_envs):
    """Name of the cell that times the default environment step of one implementation at one batch size."""
    return f"speed, {implementation}, {n_envs:,} environments"


def usable_cells(record):
    """The latest usable cell per name: measured, the card not shared, the run kept."""
    chosen = {}
    for run in record["runs"]:
        if run.get("discarded"):
            continue
        for cell in run["cells"]:
            if cell["status"] == "measured" and not cell["other_processes_on_card"]:
                chosen[cell["name"]] = dict(cell, machine=run["machine"], unit=run["unit"])
    return chosen


def gib(n_bytes):
    """Bytes shown in GiB with two decimals."""
    return f"{n_bytes / 2 ** 30:.2f}"


def render_speed(cells, sizes):
    """The throughput table of both implementations, one row per batch size and implementation."""
    lines = ["## Speed of the environment step (U-maze)", "",
             "Define $n$ as the number of environments and $s$ as the median seconds per environment step"
             " (one step of all $n$ environments, 5 physics steps each, with reward, reset and observation).", "",
             "- Total steps per second: $\\dfrac{n}{s}$, shown in millions.",
             "- Steps per second per environment: $r = \\dfrac{1}{s}$.",
             "- Hours per million steps per environment: $\\dfrac{10^{6}}{3600 \\times r}$.",
             "- First call: seconds of the first call of the compiled function: tracing, compilation by JAX,"
             " for Warp also loading or compiling its kernels, and one execution of 10 environment steps.",
             "- Peak card memory: the memory the process held on the card after the cell (`nvidia-smi`),"
             " which includes every smaller batch size measured before it in the same process. JAX does not"
             " reserve memory in advance in these units.",
             "- Pool use: the largest number of contacts, and of candidate geom pairs, in Warp's shared pool"
             " at any physics step of the cell, each as a share of the pool (48 slots per environment); rows:"
             " the largest number of constraint rows of one environment, of 160.", "",
             "| implementation | copies<br>(environments) | seconds per iteration<br>(per environment step) |"
             " total steps per<br>second (millions) | steps per second<br>per environment |"
             " hours per million steps<br>per environment | first<br>call (s) | peak card<br>memory (GiB) |"
             " pool use: contacts,<br>candidate pairs; rows | measured |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    best = {}
    for implementation in ("warp", "jax"):
        previous = None
        for n in sizes:
            cell = cells.get(speed_name(implementation, n))
            if cell is None:
                continue
            row = throughput_row(n, median_ms(cell, ENVIRONMENT_STEP) / 1000.0)
            total = row["total_steps_per_second_millions"]
            gain = "" if previous is None else f" ({100 * (total / previous - 1):+.1f}%)"
            previous = total
            if total > best.get(implementation, (0.0, 0))[0]:
                best[implementation] = (total, n)
            pool = cell.get("pool")
            pool_text = "" if pool is None else (f"{100 * pool['nacon_peak'] / pool['naconmax']:.1f}%,"
                                                 f" {100 * pool['ncollision_peak'] / pool['naconmax']:.1f}%;"
                                                 f" {pool['nefc_peak']}")
            lines.append(
                f"| {'MuJoCo Warp' if implementation == 'warp' else 'MJX JAX'} | {n:,} |"
                f" {row['seconds_per_environment_step']:.5f} | {total:.3f}{gain} |"
                f" {row['steps_per_second_per_environment']:.1f} |"
                f" {row['hours_per_million_steps_per_environment']:.3f} |"
                f" {cell['functions'][ENVIRONMENT_STEP]['first_call_seconds']:.1f} |"
                f" {cell['memory']['process_mib_on_card'] / 1024:.2f} | {pool_text} |"
                f" {cell['machine']}, {cell['measured_pacific']} |")
    lines += ["", "The percentage beside a total is its change from the row above of the same implementation.", ""]
    for implementation, (total, n) in best.items():
        lines.append(f"- Best total of {'MuJoCo Warp' if implementation == 'warp' else 'MJX JAX'}:"
                     f" {total:.3f} million steps per second at {n:,} environments.")
    # GOAL: the ratio at every batch size both implementations have.
    both = [n for n in sizes if speed_name("warp", n) in cells and speed_name("jax", n) in cells]
    if both:
        lines += ["", "| environments | MJX JAX (ms per<br>environment step) | MuJoCo Warp (ms per<br>environment step) |"
                  " Warp time as a share<br>of the JAX time | total steps per second,<br>Warp against JAX |",
                  "|---:|---:|---:|---:|---:|"]
        for n in both:
            jax_ms = median_ms(cells[speed_name("jax", n)], ENVIRONMENT_STEP)
            warp_ms = median_ms(cells[speed_name("warp", n)], ENVIRONMENT_STEP)
            lines.append(f"| {n:,} | {jax_ms:.3f} | {warp_ms:.3f} | {100 * warp_ms / jax_ms:.1f}% |"
                         f" {jax_ms / warp_ms:.2f} times |")
    return lines + [""]


def render_kernel_compilation(record):
    """The first Warp call of every process, beside how many compiled kernels the kernel folder already held."""
    lines = ["## Kernel compilation", "",
             "MuJoCo Warp compiles its kernels for the card the first time a process needs them and keeps them in"
             " its kernel folder; a later process reads them from there. The first call below is the first call"
             " of the first Warp cell of each process: tracing, compilation by JAX, compiling or reading the"
             " kernels, and 10 environment steps.", "",
             "| unit | kernel folder | kernel folders<br>at the start | at the<br>end | first Warp cell | its first<br>call (s) |",
             "|---|---|---:|---:|---|---:|"]
    for run in record["runs"]:
        first = next((cell for cell in run["cells"] if cell["status"] == "measured" and cell.get("functions")
                      and cell.get("implementation") != "jax" and cell["kind"] in ("speed", "settings", "maps")), None)
        if run.get("discarded") or not run.get("warp") or first is None:
            continue
        w = run["warp"]
        seconds = next(iter(first["functions"].values()))["first_call_seconds"]
        lines.append(f"| {run['unit']} | `{w['kernel_cache_dir']}` | {w['kernel_folders_at_start']} |"
                     f" {w.get('kernel_folders_at_end', '?')} | {first['name']} | {seconds:.1f} |")
    return lines + [""]


def render_settings(cells):
    """One table per batch size of the Warp settings timed in turn."""
    lines = ["## Warp settings (U-maze, whole environment step)", ""]
    found = sorted((cell["environments"], name) for name, cell in cells.items() if cell["kind"] == "settings")
    if not found:
        return lines + ["Not measured.", ""]
    lines += ["Each row changes one thing of the first row. Solver iterations and line-search iterations are"
              " the model's `iterations` and `ls_iterations`, which MuJoCo Warp reads as the largest number of"
              " Newton iterations per physics step and of line-search iterations per Newton iteration; an"
              " environment whose solve has converged stops earlier. Collision settings are MuJoCo Warp's"
              " defaults in every row. Rows of one table were timed in turn in one cell.", ""]
    for n, name in found:
        cell = cells[name]
        base_ms = median_ms(cell, DEFAULT_WARP)
        lines += [f"### {n:,} environments ({cell['machine']}, {cell['measured_pacific']},"
                  f" {cell['rounds_done']} rounds of {cell['env_steps_per_call']} environment steps)", "",
                  "| setting | environment<br>step (ms) | time against<br>the first row |"
                  " total steps per<br>second (millions) | first<br>call (s) | non-finite resets<br>in the"
                  " 100 start steps | pool use: contacts,<br>candidate pairs; rows | mean solver<br>iterations used |",
                  "|---|---:|---:|---:|---:|---:|---|---:|"]
        for setting, function in cell["functions"].items():
            ms, pool = median_ms(cell, setting), function["pool"]
            lines.append(
                f"| {setting} | {ms:.3f} | {100 * ms / base_ms:.1f}% |"
                f" {throughput_row(n, ms / 1000.0)['total_steps_per_second_millions']:.3f} |"
                f" {function['first_call_seconds']:.1f} | {function['non_finite_resets']:,} |"
                f" {100 * pool['nacon_peak'] / pool['naconmax']:.1f}%,"
                f" {100 * pool['ncollision_peak'] / pool['naconmax']:.1f}%; {pool['nefc_peak']} |"
                f" {function['mean_solver_iterations']:.2f} |")
        lines.append("")
    return lines


def render_maps(cells):
    """The Warp environment step on the three maps, timed in turn, per batch size."""
    lines = ["## Warp on the three maps (whole environment step)", ""]
    found = sorted((cell["environments"], name) for name, cell in cells.items() if cell["kind"] == "maps")
    if not found:
        return lines + ["Not measured.", ""]
    lines += ["The U-maze has 8 wall boxes, the Medium map 20, the Large map 31; MuJoCo Warp tests every ant geom"
              " against every wall box in its broad phase. Default Warp settings; rows of one batch size were"
              " timed in turn in one cell.", "",
              "| environments | map | environment<br>step (ms) | time against<br>the U-maze |"
              " total steps per<br>second (millions) | first<br>call (s) | pool use: contacts,<br>candidate pairs; rows |"
              " measured |", "|---:|---|---:|---:|---:|---:|---|---|"]
    for n, name in found:
        cell = cells[name]
        base_ms = median_ms(cell, "umaze map")
        for map_name, function in cell["functions"].items():
            ms, pool = median_ms(cell, map_name), function["pool"]
            lines.append(
                f"| {n:,} | {map_name} | {ms:.3f} | {100 * ms / base_ms:.1f}% |"
                f" {throughput_row(n, ms / 1000.0)['total_steps_per_second_millions']:.3f} |"
                f" {function['first_call_seconds']:.1f} | {100 * pool['nacon_peak'] / pool['naconmax']:.1f}%,"
                f" {100 * pool['ncollision_peak'] / pool['naconmax']:.1f}%; {pool['nefc_peak']} |"
                f" {cell['machine']}, {cell['measured_pacific']} |")
    return lines + [""]


def render_trainer(cells):
    """The donated, carried scan beside the plain timing at the same batch size."""
    lines = ["## The trainer's kind of program", ""]
    found = [cell for cell in cells.values() if cell["kind"] == "trainer"]
    if not found:
        return lines + ["Not measured.", ""]
    lines += ["One compiled function with the environment state and the observation donated, a `lax.scan` over"
              f" {TRAINER_ROLLOUT_STEPS} environment steps, the returned state fed to the next call, the step"
              f" limit lowered to {TRAINER_EPISODE_STEPS} so"
              " that every environment is reset inside every scan, and a 64-bit sum beside the 32-bit physics."
              " The time per environment step is the median over the calls after the first two. The graphs"
              " captured are counted over the whole process (cells measured earlier in the same process"
              " captured graphs too), so the column shows whether the count still grows from call to call: it"
              " grows in the first two calls, by 1 per call with the scan not unrolled and by 16 per call with"
              " it unrolled 16 steps, and then stays. In graph mode `WARP_STAGED` the counters MuJoCo Warp"
              " returns (contacts, candidate pairs, rows, solver iterations) read 0 at every step, so that mode"
              " cannot report an overflow.", "",
              "| copies x environments<br>per copy | graph<br>mode | scan<br>unroll | calls | environment step<br>(ms), carried scan |"
              " total steps per<br>second (millions) | first<br>call (s) | resets counted /<br>expected |"
              " graphs captured<br>after each call | physics / statistic<br>data type | measured |",
              "|---|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    for cell in found:
        r = cell["result"]
        steady = statistics.median(r["seconds_per_call"][2:]) / r["rollout_steps"]
        n = r["copies"] * r["environments_per_copy"]
        lines.append(
            f"| {r['copies']:,} x {r['environments_per_copy']} | {cell['graph_mode']} | {r.get('scan_unroll', 1)} | {r['calls']} |"
            f" {1000 * steady:.3f} | {n / steady / 1e6:.3f} | {r['seconds_per_call'][0]:.1f} |"
            f" {r['resets_counted']:,} / {r['resets_expected_from_the_step_limit']:,} |"
            f" {', '.join(str(g) for g in r['graphs_captured_after_each_call'])} |"
            f" {r['dtypes']['qpos']} / {r['dtypes']['statistic']} | {cell['machine']}, {cell['measured_pacific']} |")
    return lines + [""]


def check_lines(name, r):
    """The numbers of one correctness measurement as Markdown lines, with the gate where spec.md has one."""
    sci = lambda x: "none" if x is None else f"{x:.2e}"
    pool = lambda q: (f"{q['nacon_peak']:,} contacts and {q['ncollision_peak']:,} candidate pairs at most in a pool"
                      f" of {q['naconmax']:,}, {q['nefc_peak']} rows at most of {q['njmax']},"
                      f" {'DROPPED: ' + ', '.join(q['reasons']) if q['dropped'] else 'nothing dropped'}")
    if name.startswith("check a: 20"):
        lines = [f"First C-MuJoCo state with a contact: state {r['first_state_with_a_contact']} of {r['steps']}."
                 " Gate of `spec.md`: below 1e-4 m before the first contact, below 4 m (one maze cell) over all steps.", "",
                 "| implementation | largest position difference from C<br>before the first contact (m) |"
                 " largest over<br>all steps (m) | at the last<br>step (m) |", "|---|---:|---:|---:|"]
        return lines + [f"| {'MuJoCo Warp' if key == 'warp' else 'MJX JAX'} | {sci(r[key]['largest_before_the_first_contact'])} |"
                        f" {sci(r[key]['largest_over_all_steps'])} | {sci(r[key]['at_the_last_step'])} |"
                        for key in ("warp", "jax")]
    if name.startswith("check a"):
        # GOAL: one row per regime and pair of implementations.
        lines = ["| kind of state | pair | states | largest position<br>difference (m) | median (m) | gate of `spec.md` |",
                 "|---|---|---:|---:|---:|---|"]
        gates = {("contact_free", "warp_against_c"): "below 1e-4", ("in_contact", "warp_against_c"): "below 5e-2",
                 ("contact_free", "jax_against_c"): "below 1e-4 (the existing gate)",
                 ("in_contact", "jax_against_c"): "below 5e-2 (the existing gate)"}
        for regime in ("contact_free", "in_contact"):
            for pair, v in r[regime].items():
                lines.append(f"| {regime.replace('_', ' ')} | {pair.replace('_', ' ')} | {v['count']} |"
                             f" {sci(v['largest'])} | {sci(v['median'])} | {gates.get((regime, pair), '')} |")
        return lines + ["", f"Pool: {pool(r['pool'])}."
                        + (f" Non-finite resets in the step: {r['non_finite_resets']}." if "non_finite_resets" in r else "")]
    if name.startswith("check b"):
        lines = [f"Reference resting height {r['reference_height']} m (gate: within 1e-4 m), {r['steps']:,} passive"
                 " environment steps from the spawn.", "",
                 "| implementation | height (m) | difference from<br>the reference (m) | largest speed<br>at the end |"
                 " a state settled in C MuJoCo<br>moved by, in 200 more steps (m) |", "|---|---:|---:|---:|---:|"]
        for key, label in (("c_mujoco", "C MuJoCo, float64"), ("warp", "MuJoCo Warp, float32"), ("jax", "MJX JAX, float32")):
            if key not in r:
                continue
            v = r[key]
            lines.append(f"| {label} | {v['height']:.6f} | {sci(abs(v['height'] - r['reference_height']))} |"
                         f" {sci(v['largest_speed'])} | {sci(v.get('c_settled_state_moved_by'))} |")
        return lines + (["", f"Warp pool: {pool(r['warp_pool'])}."] if "warp_pool" in r else [])
    if name.startswith("check c: copy isolation"):
        def row(label, v):
            """One comparison of two runs as a table row."""
            by = v["largest_difference_after_steps"]
            first = v["first_step_with_a_difference"]
            return (f"| {label} | {v['bitwise_equal']:,} of {r['watched']:,} | {'none' if first is None else first} | "
                    + " | ".join(sci(by[m]) for m in by) + " |")
        marks = list(r["others_changed"]["largest_difference_after_steps"])
        lines = [f"{r['environments']:,} environments, {r['steps']} environment steps from the spawn,"
                 f" {r['watched']:,} watched environments (every 8th);"
                 f" {r['unwatched_environments_that_differ']:,} of the other environments ended in a different"
                 f" state when their torques were changed. All states finite: {r['all_finite']}.", "",
                 "| watched environments compared between | equal bit for bit<br>at every step | first step<br>with a difference | "
                 + " | ".join(f"largest difference<br>up to step {m}" for m in marks) + " |",
                 "|---|---|---:|" + "---:|" * len(marks),
                 row("run A and run B (the other environments' torques changed)", r["others_changed"]),
                 row("run A and run A repeated (nothing changed)", r["same_run_repeated"])]
        if "pool_fill" in r:
            lines += ["", f"Pool use {100 * r['pool_fill']:.1f}% of the pool"
                      + (f" (sized from a largest use of {r['pool_sized_from_peak']:,})" if "pool_sized_from_peak" in r else "")
                      + f". Run A: {pool(r['pool_run_a'])}. Run B: {pool(r['pool_run_b'])}."]
        return lines
    if name.startswith("check c: a pool below"):
        return [f"- Pool of 2 slots per environment: {pool(r)}."]
    if name.startswith("check e: semantics"):
        return [f"- Fields that differ from the JAX environment over two steps of five prepared environments:"
                f" {len(r['mismatches'])}" + ("" if not r["mismatches"] else ": " + "; ".join(r["mismatches"])) + ".",
                f"- Largest difference of an observation (physics of the two implementations):"
                f" {sci(r['largest_observation_difference'])}."]
    if name.startswith("check e: non-finite"):
        lines = ["| value | written into | Warp environment<br>respawned | JAX environment<br>respawned |", "|---|---|---|---|"]
        lines += [f"| {c['value']} | {c['coordinate']} | {'yes' if c['warp_respawned'] else 'NO'} |"
                  f" {'yes' if c['jax_respawned'] else 'NO'} |" for c in r["cases"]]
        # the Warp environment's rule changed after the first measurements; a table measured before says so
        swallowed = [c for c in r["cases"] if not c["warp_respawned"]]
        note = ([] if not swallowed else [
            "", "This table was measured before the Warp environment tested the state it is handed: MuJoCo Warp turns"
            " a torso quaternion that holds a NaN into a valid one at the start of a physics step, so the state after"
            " the step was finite. Since 2026-11-12 12:26 PT `WarpAntMaze.step` also tests the positions and velocities"
            " it is handed and those after each physics step, and that case is respawned (the table of the same check"
            " on processors, and `tests/envs/test_antmaze_warp.py::test_non_finite_inputs_are_respawned`)."])
        return lines + ["", f"The healthy environment beside them: respawned {r['healthy_environment_respawned']},"
                        f" numbers unchanged {r['healthy_environment_unchanged']}."] + note
    return ["```json", json.dumps(r, indent=1), "```"]


def render_platform_trainer(cells):
    """The platform's own training iteration with each implementation, with the five throughput columns."""
    lines = ["## The platform's own training iteration", ""]
    found = [cell for cell in cells.values() if cell["kind"] == "platform trainer"]
    if not found:
        return lines + ["Not measured.", ""]
    lines += ["`Runner.iterate` of the platform with the study's program (`antmaze_current` rules, clipped update in"
              " 4 epochs of 4 minibatches, random network distillation on the next state, the episode ending at"
              " the goal, the episode log, the trainer's default scan unroll), U-maze, 128 steps per rollout. For"
              " Warp, `WarpAntMaze` was put into the composer's table of environment classes; no file of the"
              " trainer was edited. Define $c$ as the number of copies (policies), $e$ as the environments per"
              " copy and $s$ as the median seconds per iteration over the iterations after the first two.", "",
              "- Total steps per second: $\\dfrac{128 \\times c \\times e}{s}$, shown in millions.",
              "- Steps per second per copy: $r = \\dfrac{128 \\times e}{s}$.",
              "- Hours per million steps per copy: $\\dfrac{10^{6}}{3600 \\times r}$.", "",
              "| implementation | copies x environments<br>per copy | seconds per<br>iteration |"
              " total steps per<br>second (millions) | steps per second<br>per copy |"
              " hours per million<br>steps per copy | first two<br>iterations (s) | loss<br>finite | episodes ended;<br>"
              "non-finite resets | peak card<br>memory (GiB) | measured |",
              "|---|---|---:|---:|---:|---:|---|---|---|---:|---|"]
    for cell in sorted(found, key=lambda c: (c["result"]["copies"] * c["result"]["environments_per_copy"],
                                             c["result"]["implementation"])):
        r = cell["result"]
        s = statistics.median(r["seconds_per_iteration"][2:])
        per_copy = r["rollout_steps"] * r["environments_per_copy"] / s
        lines.append(
            f"| {'MuJoCo Warp' if r['implementation'] == 'warp' else 'MJX JAX'} |"
            f" {r['copies']:,} x {r['environments_per_copy']} | {s:.4f} | {r['copies'] * per_copy / 1e6:.3f} |"
            f" {per_copy:.1f} | {1e6 / (3600.0 * per_copy):.3f} |"
            f" {r['seconds_per_iteration'][0]:.0f}, {r['seconds_per_iteration'][1]:.0f} | {r['loss_finite']} |"
            f" {r['episodes_ended']:,}; {r['non_finite_resets']:,} | {cell['memory']['process_mib_on_card'] / 1024:.2f} |"
            f" {cell['machine']}, {cell['measured_pacific']} |")
    return lines + [""]


def render_checks(cells, processor_checks):
    """Every correctness measurement with its numbers: from the card, then from processors."""
    lines = ["## Correctness measurements on the card", ""]
    found = {name: cell for name, cell in cells.items() if cell["kind"] == "check"}
    if not found:
        lines += ["Not measured on the card.", ""]
    for name, cell in found.items():
        lines += [f"### {name} ({cell['machine']}, {cell['measured_pacific']})", ""] + check_lines(name, cell["result"]) + [""]
    lines += ["## Correctness measurements on processors (MuJoCo Warp on the processor device)", ""]
    if processor_checks is None:
        return lines + ["No record.", ""]
    lines += [f"Host {processor_checks['host']}, {processor_checks['measured_pacific']}, Slurm job"
              f" {processor_checks['slurm_job']}; the same functions as on the card.", ""]
    for name, result in processor_checks["checks"].items():
        lines += [f"### {name}", ""] + check_lines(name, result) + [""]
    return lines


def render_soak(cells):
    """The random-torque soak: steps, non-finite resets, pool use."""
    lines = ["## Soak with random torques", ""]
    found = [cell for cell in cells.values() if cell["kind"] == "soak"]
    if not found:
        return lines + ["Not measured.", ""]
    lines += ["Uniform random torques, fresh every 100 steps, the state carried from call to call; episodes end at"
              " the 700-step limit. Each soak has a row at its first mark past 2 million steps and a row at its end.", "",
              "| Warp setting | environments | environment steps<br>(all environments) | non-finite<br>resets | largest pool use<br>"
              "(contacts; candidate pairs; slots) | largest rows of one<br>environment (limit) |"
              " anything<br>dropped | seconds | measured |", "|---|---:|---:|---:|---|---|---|---:|---|"]
    for cell in found:
        for mark in cell["marks"]:
            pool = mark["pool"]
            lines.append(f"| {cell.get('setting', DEFAULT_SOAK_SETTING)} | {cell['environments']:,} |"
                         f" {mark['environment_steps']:,} | {mark['non_finite_resets']:,} |"
                         f" {pool['nacon_peak']:,}; {pool['ncollision_peak']:,}; {pool['naconmax']:,} |"
                         f" {pool['nefc_peak']} ({pool['njmax']}) | {'yes' if pool['dropped'] else 'no'} |"
                         f" {mark['seconds']:.0f} | {cell['machine']}, {cell['measured_pacific']} |")
    return lines + [""]


def render(record, processor_checks=None, sizes=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)):
    """The whole Markdown report from the record of the card and, when given, the record from processors."""
    cells = usable_cells(record)
    lines = ["# AntMaze physics through MuJoCo Warp, beside the JAX implementation", "",
             "Generated by `benchmarks/env/profile_antmaze_warp.py --render` from `antmaze_warp.json`; change"
             " the script, not this file. Times are Pacific, converted from machines running on Eastern.", ""]
    # GOAL: one bullet per unit with the machine, the card, the versions and the wall time.
    for run in record["runs"]:
        v, w = run["versions"], run.get("warp", {})
        lines.append(
            f"- Unit `{run['unit']}`{' (DISCARDED: ' + run['discarded'] + ')' if run.get('discarded') else ''}:"
            f" {run['machine']} ({run['card']['name']}, host {run['host']}), started {run['started_pacific']}, "
            + (f"its process ended {run['wall_seconds'] / 60:.1f} minutes after the unit started" if run.get("finished")
               else "ended before its last cell finished")
            + f"; parts {', '.join(run['parts'])}; JAX {v['jax']}, jaxlib {v['jaxlib']}, MuJoCo {v['mujoco']},"
              f" MJX {v['mujoco-mjx']}, warp-lang {v['warp-lang']}, Python {run['python']},"
              f" driver {run['card']['driver']}"
            + (f"; Warp kernel folder `{w['kernel_cache_dir']}` holding {w['kernel_folders_at_start']} kernel"
               f" folders at the start and {w.get('kernel_folders_at_end', '?')} at the end; Warp reports CUDA"
               f" toolkit {w['cuda_toolkit']} and driver {w['cuda_driver']}" if w else "") + ".")
    lines += ["", "## What the Warp environment is", "",
              "`exploration_platform/envs/antmaze/warp_antmaze.py` (`WarpAntMaze`), with the interface of"
              " `mjx_antmaze.JaxAntMaze`. Unless a row says otherwise:", "",
              "1. **Model**: the XML of `am_common.build_antmaze_xml`, the same string the JAX implementation"
              " loads (Gymnasium Ant plus the merged wall boxes), 5 physics steps of 0.01 s per environment step.",
              "2. **Integrator and solver**: `implicitfast`, Newton solver, pyramidal friction cones, at most 4"
              " solver iterations and 8 line-search iterations per physics step (the model's `iterations` and"
              " `ls_iterations`). MuJoCo Warp ends the solver loop of a physics step when every environment has"
              " converged or reached the limit. Float32.",
              "3. **Collision**: MuJoCo Warp's defaults (all-pairs broad phase with its plane, sphere and"
              " oriented-box filters). The two contact caps of the model (`max_geom_pairs`,"
              " `max_contact_points`) are read only by MJX's JAX implementation.",
              "4. **Sizes**: one contact pool of 48 slots per environment shared by all environments"
              " (`naconmax` = 48 x batch; candidate geom pairs use a pool of the same size), and at most 160"
              " constraint rows per environment (`njmax`).",
              "5. **Overflow**: when a pool or the row limit is exceeded MuJoCo Warp drops the excess and goes on."
              " The environment state keeps the largest number of contacts and of candidate pairs in the pool,"
              " the largest number of rows of one environment, and MuJoCo Warp's overflow flags per"
              " environment; `WarpAntMaze.raise_on_overflow` raises when any of them shows a drop. Every Warp"
              " cell below calls it, so a unit with dropped contacts ends with an error.",
              "6. **Foreign call**: `mujoco.mjx.step` on a model put with `impl=\"warp\"`, batched with"
              " `jax.vmap` over one axis, inside `jax.jit` and `jax.lax.scan`, graph mode `WARP` (Warp captures a"
              " CUDA graph per set of buffer addresses and replays it).", ""]
    lines += render_speed(cells, sizes) + render_kernel_compilation(record) + render_settings(cells)
    lines += render_maps(cells) + render_trainer(cells)
    lines += render_platform_trainer(cells)
    # GOAL: the second process, when it was tried.
    lines += ["## A second process on the same card", ""]
    second = [cell for cell in cells.values() if cell["kind"] == "second process"]
    for cell in second:
        r = cell["result"]
        child = r["child"]
        lines.append(
            f"- {cell['machine']}, {cell['measured_pacific']}: while this process kept stepping its"
            f" {SECOND_PROCESS_ENVIRONMENTS[0]:,} environments ({r['parent_calls_of_10_steps_while_the_child_ran']:,}"
            f" calls of 10 steps in {r['seconds']:.1f} s, state finite: {r['parent_state_is_finite']}), a second"
            f" process built its own Warp environment of {child['environments']:,} environments and stepped it"
            f" {child['environment_steps']} times: exit code {r['child_exit_code']}, {child['seconds']:.1f} s in all,"
            f" first call {child['first_call_seconds']:.1f} s (kernels read from the kernel folder), state finite:"
            f" {child['state_is_finite']}, nothing dropped in either process:"
            f" {not child['pool']['dropped'] and not r['parent_pool']['dropped']}.")
    lines += [] if second else ["Not measured."]
    lines.append("")
    lines += render_soak(cells) + render_checks(cells, processor_checks)
    # GOAL: everything that was planned or started and is not in the tables.
    lines += ["## Not measured or not used", ""]
    unused = []
    for run in record["runs"]:
        for cell in run["cells"]:
            if run.get("discarded"):
                reason = f"run discarded: {run['discarded']}"
            elif cell["status"] != "measured":
                reason = cell.get("reason", f"the unit ended during this cell (status: {cell['status']})")
            elif cell["other_processes_on_card"]:
                reason = f"another process was on the card: {cell['other_processes_on_card']}"
            elif cells[cell["name"]]["unit"] != run["unit"]:
                reason = f"replaced by the later measurement of this cell in unit `{cells[cell['name']]['unit']}`"
            else:
                continue
            unused.append(f"- {cell['name']} (unit `{run['unit']}`): {reason}.")
    return "\n".join(lines + (unused or ["Nothing."])) + "\n"


# ----------------------------------------------------------------------------------------------
# facts about the process and the card
# ----------------------------------------------------------------------------------------------

def memory_record(card_uuid, uses_warp):
    """What this process holds on the card: JAX's peak, Warp's pool peak, and nvidia-smi's figure."""
    import jax
    import warp as wp
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    # before: lines "GPU-fed6..., 12345, 2311"; after: 2311 for this process on this card
    mib = [int(row.split(",")[2]) for row in rows.strip().splitlines()
           if row.split(",")[0].strip() == card_uuid and int(row.split(",")[1]) == os.getpid()]
    return {"jax_peak_bytes": jax.local_devices()[0].memory_stats()["peak_bytes_in_use"],
            "warp_pool_peak_bytes": wp.get_mempool_used_mem_high("cuda:0") if uses_warp else None,
            "process_mib_on_card": mib[0] if mib else None}


def kernel_folders():
    """Number of compiled-kernel folders in Warp's kernel cache folder."""
    import warp as wp
    return sum(1 for entry in os.scandir(wp.config.kernel_cache_dir) if entry.is_dir())


def is_out_of_memory(error):
    """Whether an exception says the card's memory ran out."""
    text = str(error).lower()
    return "resource_exhausted" in text or "out of memory" in text or "failed to allocate" in text


# ----------------------------------------------------------------------------------------------
# measuring
# ----------------------------------------------------------------------------------------------

def build_environment(implementation, n_envs, changes=None, warp_options=None, map_name="umaze"):
    """The environment of one implementation at one batch size, and the seconds building it took."""
    import dataclasses
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze
    cfg = dataclasses.replace(preset(map_name), **(changes or {}))
    t0 = time.perf_counter()
    env = (WarpAntMaze(cfg, n_envs, 1, **(warp_options or {})) if implementation == "warp"
           else JaxAntMaze(cfg, n_envs, 1))
    return env, time.perf_counter() - t0


def warp_facts(env, state):
    """Pool use, non-finite resets and solver iterations of a Warp state; raises if anything was dropped."""
    import numpy as np
    return {"pool": env.raise_on_overflow(state),
            "non_finite_resets": int(np.asarray(state.nan_count).sum()),
            "mean_solver_iterations": float(np.asarray(state.data._impl.solver_niter).mean()),
            "state_is_finite": bool(np.isfinite(np.asarray(state.data.qpos)).all())}


def measure_speed(cell, implementation, n_envs, rounds, save):
    """Time the default environment step of one implementation at one batch size."""
    import numpy as np
    from antmaze_warp_checks import captured_graphs
    # GOAL: build, compile, and reach the start state by 100 random-torque steps.
    env, cell["build_seconds"] = build_environment(implementation, n_envs)
    env_call = build_env_call(env)
    start, first_call = settle(env, env_call)
    torques = uniform_torques("timed torques", 0, env.B, env.act_dim)
    cell["non_finite_resets_in_the_start_steps"] = int(np.asarray(start.nan_count).sum())
    time_functions(cell, [(ENVIRONMENT_STEP, functools.partial(env_call, start, torques), first_call)], rounds, save)
    # GOAL: for Warp, the pool use over the start steps and the timed steps; dropped contacts end the unit.
    if implementation == "warp":
        after, _ = env_call(start, torques)
        cell.update(warp_facts(env, after))
        cell["graphs_captured"] = captured_graphs()


def measure_settings(cell, n_envs, rounds, save):
    """Time the Warp settings in turn at one batch size, each with its own model and start state."""
    functions, checks = [], []
    for name, changes, options, compiler_options in SETTINGS:
        env, _ = build_environment("warp", n_envs, changes, options)
        env_call = build_env_call(env, compiler_options=compiler_options)
        start, first_call = settle(env, env_call)
        torques = uniform_torques("timed torques", 0, env.B, env.act_dim)
        cell["functions"][name] = {"changes": changes, "warp_options": options, **warp_facts(env, start)}
        functions.append((name, functools.partial(env_call, start, torques), first_call))
        checks.append((name, env, env_call, start, torques))
    time_functions(cell, functions, rounds, save)
    # GOAL: the timed steps must not have dropped anything either.
    for name, env, env_call, start, torques in checks:
        env.raise_on_overflow(env_call(start, torques)[0])


def measure_maps(cell, n_envs, rounds, save):
    """Time the default Warp environment step on the three maps in turn at one batch size."""
    functions, checks = [], []
    for map_name in MAPS:
        env, _ = build_environment("warp", n_envs, map_name=map_name)
        env_call = build_env_call(env)
        start, first_call = settle(env, env_call)
        torques = uniform_torques("timed torques", 0, env.B, env.act_dim)
        cell["functions"][f"{map_name} map"] = warp_facts(env, start)
        functions.append((f"{map_name} map", functools.partial(env_call, start, torques), first_call))
        checks.append((env, env_call, start, torques))
    time_functions(cell, functions, rounds, save)
    # GOAL: the timed steps must not have dropped anything either.
    for env, env_call, start, torques in checks:
        env.raise_on_overflow(env_call(start, torques)[0])


def measure_trainer(cell, n_copies, n_envs, graph_mode, scan_unroll, rounds, save):
    """The donated, carried scan of 128 steps with resets inside, at one shape, graph mode and unroll."""
    from antmaze_warp_checks import trainer_like_rollout
    cell["graph_mode"] = graph_mode
    cell["result"] = trainer_like_rollout(n_copies, n_envs, rollout_steps=TRAINER_ROLLOUT_STEPS,
                                          calls=TRAINER_CALLS, episode_steps=TRAINER_EPISODE_STEPS,
                                          warp_options={"graph_mode": graph_mode}, scan_unroll=scan_unroll)
    if cell["result"]["pool"]["dropped"]:
        raise RuntimeError(f"MuJoCo Warp dropped contacts or constraint rows: {cell['result']['pool']}")


def measure_platform_trainer(cell, implementation, n_copies, n_envs, rounds, save):
    """The platform's own training iteration with one implementation at one shape."""
    from antmaze_warp_checks import platform_training_iterations
    cell["result"] = platform_training_iterations(implementation, n_copies, n_envs, num_steps=TRAINER_ROLLOUT_STEPS,
                                                  timed_iterations=7)
    if cell["result"].get("pool", {}).get("dropped"):
        raise RuntimeError(f"MuJoCo Warp dropped contacts or constraint rows: {cell['result']['pool']}")


def measure_second_process(cell, rounds, save):
    """A child process steps its own Warp environment while this process keeps stepping its own."""
    import numpy as np
    env, _ = build_environment("warp", SECOND_PROCESS_ENVIRONMENTS[0])
    env_call = build_env_call(env)
    state, _ = settle(env, env_call)
    torques = uniform_torques("timed torques", 0, env.B, env.act_dim)
    # GOAL: start the child, and call our own step again and again until the child has ended.
    child = subprocess.Popen([sys.executable, str(HERE), "--child", str(SECOND_PROCESS_ENVIRONMENTS[1])],
                             stdout=subprocess.PIPE, text=True)
    calls, t0 = 0, time.perf_counter()
    while child.poll() is None:
        state, _ = env_call(state, torques)
        calls += 1
    output = child.stdout.read()
    cell["result"] = {"child_exit_code": child.returncode, "child": json.loads(output.strip().splitlines()[-1]),
                      "parent_calls_of_10_steps_while_the_child_ran": calls,
                      "seconds": time.perf_counter() - t0,
                      "parent_state_is_finite": bool(np.isfinite(np.asarray(state.data.qpos)).all()),
                      "parent_pool": env.raise_on_overflow(state)}
    if child.returncode != 0:
        raise RuntimeError(f"the second process failed: {output[-2000:]}")


def child_main(n_envs):
    """The second process: build a small Warp environment, step it 100 times, print one JSON line."""
    import exploration_platform  # noqa: F401
    import numpy as np
    t0 = time.perf_counter()
    env, _ = build_environment("warp", n_envs)
    state, first_call = settle(env, build_env_call(env))
    print(json.dumps({"pid": os.getpid(), "environments": n_envs, "environment_steps": 100,
                      "first_call_seconds": first_call, "seconds": time.perf_counter() - t0,
                      "state_is_finite": bool(np.isfinite(np.asarray(state.data.qpos)).all()),
                      "pool": env.raise_on_overflow(state)}))


def measure_check(cell, check, rounds, save):
    """One correctness measurement of antmaze_warp_checks.py; its numbers go into the cell."""
    cell["result"] = check()


def measure_soak(cell, deadline, soak_seconds, rounds, save, setting=DEFAULT_SOAK_SETTING, changes=None):
    """Random torques at 4,096 environments: at least 2 million steps, then on until the cell's time is used."""
    import numpy as np
    from antmaze_warp_checks import rollout_call, torques
    deadline = min(deadline, time.time() + soak_seconds)
    env, _ = build_environment("warp", SOAK_ENVIRONMENTS, changes)
    call = rollout_call(env)
    cell.update(environments=SOAK_ENVIRONMENTS, setting=setting, marks=[])
    state, steps, t0, required_done = env.reset(), 0, time.perf_counter(), False
    chunk = 0
    while True:
        # GOAL: 100 environment steps of fresh torques; the state is carried from call to call.
        state, _ = call(state, torques(f"soak torques {chunk}", 100, env.B))
        chunk, steps = chunk + 1, steps + 100 * env.B
        mark = {"environment_steps": steps, "seconds": time.perf_counter() - t0,
                "non_finite_resets": int(np.asarray(state.nan_count).sum()),
                "episodes_ended": int(np.asarray(state.reset_count).sum()),
                "pool": env.overflow_report(state)}
        if mark["pool"]["dropped"]:
            cell["marks"].append(mark)
            save()
            raise RuntimeError(f"MuJoCo Warp dropped contacts or constraint rows in the soak: {mark['pool']}")
        # GOAL: keep the mark at the required 2 million steps and the latest one.
        if not required_done and steps >= SOAK_REQUIRED_STEPS:
            required_done = True
            cell["marks"].append(mark)
        elif required_done:
            cell["marks"] = [cell["marks"][0], mark]
        if chunk % 10 == 0:
            save()
        if required_done and time.time() > deadline:
            break
    cell["marks"][-1]["environment_step_ms_carried"] = 1000 * mark["seconds"] / chunk / 100


def named_checks():
    """The correctness measurements by name, in the order they are run."""
    import antmaze_warp_checks as checks
    return [
        ("check a: one step against C MuJoCo and against JAX, 50 states", checks.one_step_parity),
        ("check a: 20 shared-torque steps from the spawn against C MuJoCo", checks.trajectory_divergence),
        ("check b: resting height after 1,200 passive steps", checks.resting_height),
        ("check c: copy isolation, 16 environments", functools.partial(checks.copy_isolation, 16)),
        ("check c: copy isolation, 4,096 environments, pool 3% above its largest use",
         functools.partial(checks.copy_isolation, 4096, pool_margin=1.03)),
        ("check c: copy isolation, 32,768 environments, pool 3% above its largest use",
         functools.partial(checks.copy_isolation, 32768, pool_margin=1.03)),
        ("check c: copy isolation of the JAX implementation, 4,096 environments",
         functools.partial(checks.copy_isolation, 4096, implementation="jax")),
        ("check c: a pool below the need is reported", checks.small_pool_is_reported),
        ("check e: semantics against JAX, continuing task", functools.partial(checks.semantics_against_jax, True)),
        ("check e: semantics against JAX, episode ends at the goal",
         functools.partial(checks.semantics_against_jax, False)),
        ("check e: non-finite inputs, which are respawned", checks.non_finite_inputs),
        ("check b: resting height in graph mode WARP_STAGED",
         functools.partial(checks.resting_height, warp_options={"graph_mode": "WARP_STAGED"}, with_jax=False)),
        ("check b: resting height with 2 solver iterations",
         functools.partial(checks.resting_height, changes={"solver_iterations": 2}, with_jax=False)),
        ("check b: resting height with 1 solver iteration",
         functools.partial(checks.resting_height, changes={"solver_iterations": 1}, with_jax=False)),
        ("check b: resting height with 4 line-search iterations",
         functools.partial(checks.resting_height, changes={"ls_iterations": 4}, with_jax=False)),
    ]


def processor_checks_main(out):
    """Run the correctness measurements with MuJoCo Warp on processors and write them beside the card's record.

    The two large copy-isolation runs are left to the card: on processors they would take hours.
    """
    record = {"host": socket.gethostname(), "measured_pacific": pacific(time.time()),
              "slurm_job": os.environ.get("SLURM_JOB_ID", "none"), "checks": {}}
    for name, check in named_checks():
        if "4,096" in name or "32,768" in name or "WARP_STAGED" in name:
            continue
        record["checks"][name] = check()
        print(f"{name}: {json.dumps(record['checks'][name])}", flush=True)
    # GOAL: the Warp stepper with fewer solver iterations against the specified model.
    import antmaze_warp_checks as checks
    for setting, changes in FEWER_ITERATIONS:
        name = f"check a: one step against C MuJoCo of the specified model, Warp with {setting}"
        record["checks"][name] = checks.one_step_parity(changes=changes, with_jax=False)
        print(f"{name}: {json.dumps(record['checks'][name])}", flush=True)
    flush(record, processor_record_path(out))


def processor_record_path(out):
    """The file of the processor checks that belongs to a record file of the card."""
    return out.with_name(out.stem + "_checks_processors.json")


def planned_cells(args, deadline):
    """The cells of this run in the order they are measured: (kind, name, extra fields, function)."""
    plan = []
    for implementation in ("warp", "jax"):
        if f"speed-{implementation}" in args.parts:
            plan += [("speed", speed_name(implementation, n), {"implementation": implementation, "environments": n},
                      functools.partial(measure_speed, implementation=implementation, n_envs=n))
                     for n in sorted(args.batches)]
    if "settings" in args.parts:
        plan += [("settings", f"settings, warp, {n:,} environments", {"environments": n},
                  functools.partial(measure_settings, n_envs=n)) for n in args.settings_batches]
    if "maps" in args.parts:
        plan += [("maps", f"maps, warp, {n:,} environments", {"environments": n},
                  functools.partial(measure_maps, n_envs=n)) for n in args.settings_batches]
    if "trainer" in args.parts:
        for shape in args.trainer_shapes:
            copies, envs = (int(x) for x in shape.split("x"))
            plan += [("trainer", f"trainer-like scan, {copies:,} x {envs}, graph mode {mode}, unroll {unroll}", {},
                      functools.partial(measure_trainer, n_copies=copies, n_envs=envs, graph_mode=mode,
                                        scan_unroll=unroll))
                     for mode in args.graph_modes for unroll in args.trainer_unrolls]
    for implementation in ("warp", "jax"):
        if f"platform-trainer-{implementation}" in args.parts:
            for shape in args.trainer_shapes:
                copies, envs = (int(x) for x in shape.split("x"))
                plan.append(("platform trainer", f"platform training iteration, {implementation}, {copies:,} x {envs}",
                             {}, functools.partial(measure_platform_trainer, implementation=implementation,
                                                   n_copies=copies, n_envs=envs)))
    if "second-process" in args.parts:
        plan.append(("second process", "second process on the same card", {}, measure_second_process))
    if "checks" in args.parts:
        plan += [("check", name, {}, functools.partial(measure_check, check=check)) for name, check in named_checks()]
    if "fewer-iterations" in args.parts:
        import antmaze_warp_checks as checks
        for setting, changes in FEWER_ITERATIONS:
            plan.append(("check", f"check a: one step against C MuJoCo of the specified model, Warp with {setting}",
                         {}, functools.partial(measure_check, check=functools.partial(
                             checks.one_step_parity, changes=changes, with_jax=False))))
            plan.append(("soak", f"soak, warp, {SOAK_ENVIRONMENTS:,} environments, {setting}", {},
                         functools.partial(measure_soak, deadline=deadline, soak_seconds=args.soak_seconds,
                                           setting=setting, changes=changes)))
    if "soak" in args.parts:
        plan.append(("soak", f"soak, warp, {SOAK_ENVIRONMENTS:,} environments", {},
                     functools.partial(measure_soak, deadline=deadline, soak_seconds=args.soak_seconds)))
    return plan


def main():
    """Measure the planned cells inside the unit's time limit, saving as it goes; or render."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="the JSON record file; runs are appended to it")
    ap.add_argument("--render", action="store_true", help="only rebuild the Markdown from the record")
    ap.add_argument("--processor-checks", action="store_true",
                    help="run the correctness measurements on processors (a compute node) and record them")
    ap.add_argument("--child", type=int, default=None, metavar="ENVIRONMENTS",
                    help="run as the second process of the second-process cell, with this many environments")
    ap.add_argument("--machine", default=None, help='for example "serval02 card 1"')
    ap.add_argument("--unit", default=None, help="name of the unit this run belongs to")
    ap.add_argument("--parts", nargs="+", default=["speed-warp"],
                    choices=["speed-warp", "speed-jax", "settings", "maps", "trainer", "platform-trainer-warp",
                             "platform-trainer-jax", "second-process", "checks", "soak", "fewer-iterations"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--settings-batches", nargs="+", type=int, default=[4096],
                    help="batch sizes of the settings cells and of the maps cells")
    ap.add_argument("--trainer-shapes", nargs="+", default=["1024x4"], help="copies x environments per copy")
    ap.add_argument("--graph-modes", nargs="+", default=["WARP"], choices=["WARP", "WARP_STAGED", "WARP_STAGED_EX"])
    ap.add_argument("--trainer-unrolls", nargs="+", type=int, default=[1],
                    help="unroll of the trainer-like scan (the trainer's default is 16)")
    ap.add_argument("--soak-seconds", type=float, default=240.0, help="the soak goes on for this long after its start")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--unit-start-epoch", type=float, default=None, help="when the unit started (date +%%s)")
    ap.add_argument("--unit-minutes", type=float, default=15.0, help="wall-time limit of the whole unit")
    args = ap.parse_args()
    if args.child is not None:
        child_main(args.child)
        return
    out = Path(args.out)
    markdown = out.with_suffix(".md")
    processors = processor_record_path(out)
    read_processors = lambda: json.loads(processors.read_text()) if processors.exists() else None
    if args.processor_checks:
        processor_checks_main(out)
        return
    if args.render:
        markdown.write_text(render(json.loads(out.read_text()), read_processors()))
        print(f"wrote {markdown}")
        return
    if args.machine is None or args.unit is None:
        ap.error("--machine and --unit are required for a measurement")
    if args.rounds < MIN_ROUNDS:
        ap.error(f"--rounds must be at least {MIN_ROUNDS}")
    unit_start = args.unit_start_epoch or time.time()
    # no new cell starts unless it is expected to end 30 seconds before the unit's limit
    deadline = unit_start + 60.0 * args.unit_minutes - 30.0

    # GOAL: refuse to measure on the processor, with an exit code the unit script recognises.
    import jax
    import exploration_platform  # noqa: F401  (switches on 64-bit statistics before any array exists, as in training)
    import warp as wp
    if jax.devices()[0].platform != "gpu":
        print(f"JAX did not load the graphics card (device {jax.devices()[0]}); nothing measured")
        sys.exit(NO_GPU_EXIT_CODE)
    # a process that measures only the JAX implementation never starts Warp, so its memory is JAX's alone
    uses_warp = not set(args.parts) <= {"speed-jax", "platform-trainer-jax"}
    if uses_warp:
        wp.init()

    # GOAL: the run's identity: machine, card, versions, sources, kernel folder, time.
    card = card_record()
    env_dir = BASE / "src" / "exploration_platform" / "envs" / "antmaze"
    sources = (HERE, HERE.parent / "antmaze_warp_checks.py", HERE.parent / "profile_antmaze_physics_split.py",
               env_dir / "warp_antmaze.py", env_dir / "mjx_antmaze.py", env_dir / "am_common.py")
    run = {
        "unit": args.unit, "machine": args.machine, "host": socket.gethostname(), "card": card,
        "parts": args.parts, "process_id": os.getpid(),
        "started_pacific": pacific(time.time()), "started_iso": datetime.now().astimezone().isoformat(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang", "numpy")},
        "python": sys.version.split()[0],
        "warp": {"kernel_cache_dir": wp.config.kernel_cache_dir, "kernel_folders_at_start": kernel_folders(),
                 "cuda_toolkit": ".".join(str(x) for x in wp.get_cuda_toolkit_version()),
                 "cuda_driver": ".".join(str(x) for x in wp.get_cuda_driver_version())} if uses_warp else {},
        "source_sha256": {path.name: sha256_of(path) for path in sources},
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "settings": {"rounds": args.rounds, "env_steps_per_call": ENV_STEPS_PER_CALL},
        "cells": [], "finished": False,
    }
    record = json.loads(out.read_text()) if out.exists() else {"runs": []}
    record["runs"].append(run)
    save = functools.partial(flush, record, out)
    save()
    print(f"{run['started_pacific']} unit {args.unit} on {args.machine}: {card['name']} {card['uuid']}, parts"
          f" {args.parts}, Warp {run['warp']}", flush=True)

    # GOAL: measure cell after cell; a cell that is not expected to fit is recorded as not measured.
    pace, stopped, last_total = {}, {}, {}
    for kind, name, extra, measure in planned_cells(args, deadline):
        cell = {"kind": kind, "name": name, **extra}
        run["cells"].append(cell)
        implementation = extra.get("implementation")
        needed, left = FIRST_GUESS_SECONDS[kind] * pace.get(kind, 1.0), deadline - time.time()
        if implementation in stopped or needed > left:
            cell.update(status="not measured", reason=stopped.get(
                implementation, f"{left:.0f} s were left of the {args.unit_minutes:.0f}-minute unit and the cell"
                                f" was expected to take {needed:.0f} s"))
            print(f"not measured: {name} ({cell['reason']})", flush=True)
            save()
            continue
        t0 = time.time()
        cell.update(status="running", measured_pacific=pacific(t0), env_steps_per_call=ENV_STEPS_PER_CALL,
                    rounds_done=0, functions={}, other_processes_on_card=other_processes_on_card(card["uuid"]))
        print(f"{pacific(t0)} cell: {name}", flush=True)
        try:
            measure(cell, rounds=args.rounds, save=save)
        except Exception as error:
            # the only error a list of batch sizes survives is the card's memory running out
            if kind != "speed" or not is_out_of_memory(error):
                cell.update(status="failed", reason=f"{type(error).__name__}: {str(error)[:500]}")
                save()
                raise
            cell.update(status="not measured", reason=f"the card's memory ran out: {str(error)[:300]}")
            stopped[implementation] = f"the card's memory ran out at {extra['environments']:,} environments"
            print(f"  memory ran out: {str(error)[:300]}", flush=True)
            save()
            continue
        # the child of the second-process cell is ours, so that cell is not checked for other processes
        if kind != "second process":
            cell["other_processes_on_card"] += other_processes_on_card(card["uuid"])
        cell.update(wall_seconds=time.time() - t0, memory=memory_record(card["uuid"], uses_warp), status="measured")
        pace[kind] = min(4.0, max(0.25, cell["wall_seconds"] / FIRST_GUESS_SECONDS[kind]))
        # GOAL: a list of batch sizes ends when a doubling gains under 10% of total steps per second.
        if kind == "speed":
            total = extra["environments"] / median_ms(cell, ENVIRONMENT_STEP)
            previous = last_total.get(implementation)
            if previous is not None and total < (1.0 + STOP_GAIN) * previous[1] and extra["environments"] <= 2 * previous[0]:
                stopped[implementation] = (f"the doubling from {previous[0]:,} to {extra['environments']:,} environments"
                                           f" gained {100 * (total / previous[1] - 1):.1f}% of total steps per second")
            last_total[implementation] = (extra["environments"], total)
        save()
        print(f"  cell took {cell['wall_seconds']:.0f} s; {deadline - time.time():.0f} s left; memory"
              f" {cell['memory']}", flush=True)

    # GOAL: close the run and write the tables.
    if uses_warp:
        run["warp"]["kernel_folders_at_end"] = kernel_folders()
    run["finished"], run["wall_seconds"] = True, time.time() - unit_start
    save()
    markdown.write_text(render(record, read_processors()))
    print(f"{pacific(time.time())} unit finished in {run['wall_seconds'] / 60:.1f} minutes; wrote {out} and"
          f" {markdown}", flush=True)


if __name__ == "__main__":
    main()
