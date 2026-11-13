"""Print the Markdown tables of notes.md from the records of this folder (the notes are written by hand
around them).

  /usr/bin/python3 tables.py --card h100-nvl_serval05 > /tmp/tables.md
"""
import argparse
import json
import statistics
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[1] / "records"
# the words for the names the scripts record (ideas.py variants, startup.py labels)
WORDS = {
    "unit's program": "the unit's program: rollout loop 16 steps per body, frame loop not unrolled",
    "frame-unroll-5": "frame loop unrolled (5 physics steps per body), rollout loop 16 steps per body",
    "rollout-unroll-1": "rollout loop not unrolled (1 step per body)",
    "rollout-unroll-4_frame-unroll-5": "rollout loop 4 steps per body, frame loop unrolled",
    "rollout-unroll-32_frame-unroll-5": "rollout loop 32 steps per body, frame loop unrolled",
    "solver-plain-loop": "MuJoCo Warp's solver loop as a plain loop of 4 iterations",
    "no-cache": "as the units start today: no cache, state as built",
    "no-cache_state-committed": "no cache, state put on the card",
    "cache-first-process-fills-it_state-committed": "cache, first process (fills it), state put on the card",
    "cache-second-process-reads-it_state-committed": "cache, second process (reads it), state put on the card",
}


def load(name: str) -> dict:
    """One JSON record of this folder."""
    return json.loads((RECORDS / name).read_text())


def piece(record: dict, name: str) -> float:
    """Median seconds of a piece of the iteration part, by the start of its name."""
    pieces = record["iteration"]["pieces"]
    matches = [key for key in pieces if key.startswith(name)]
    if len(matches) != 1:
        raise KeyError(f"{name!r} matches {matches}")
    return pieces[matches[0]]["median_seconds"]


def breakdown_rows(record: dict) -> list:
    """(name, seconds) of the iteration's parts; the last row is what the pieces do not explain.

    before: a record with the whole iteration 1.847 s and its pieces
    after:  [("whole iteration ...", 1.847), ("environment step ...", 1.80), ...]
    """
    whole = statistics.median(record["iteration"]["chained_rounds_seconds"])
    environment = piece(record, "environment alone, 128 steps, chained")
    physics = piece(record, "physics alone at the same torques")
    update = piece(record, "update (shuffle")
    gradients, clipped = piece(record, "sixteen gradients alone"), piece(record, "sixteen gradients with the clip")
    rollout_other = [("policy forward, 128 steps", piece(record, "policy forward alone")),
                     ("action noise draw", piece(record, "action noise draw"))]
    after = [("critic pass after the rollout", piece(record, "critic pass")),
             ("distillation forward, input statistics, whitening", piece(record, "distillation forward")),
             ("reward filter, its statistics, both advantage sums", piece(record, "reward filter")),
             ("rows put in per-policy order", piece(record, "rows put in")),
             ("coverage map", piece(record, "coverage map")),
             ("episode log sums", piece(record, "episode log sums"))]
    rows = [("whole iteration (chained calls, as the training loop calls it)", whole),
            ("environment step, 128 steps (chained, state donated)", environment),
            ("&nbsp;&nbsp;bare physics: 640 batched RK4 physics steps", physics),
            ("&nbsp;&nbsp;the environment's own work: fault tests, reward, reset, observation", environment - physics),
            *rollout_other, *after,
            ("update: shuffle, gather, 16 gradients, clip, Adam", update),
            ("&nbsp;&nbsp;16 gradients", gradients),
            ("&nbsp;&nbsp;clip", clipped - gradients),
            ("&nbsp;&nbsp;shuffle, gather and Adam", update - clipped)]
    explained = environment + sum(v for _, v in rollout_other) + sum(v for _, v in after) + update
    rows.append(("not explained by the pieces (joins between pieces, host work)", whole - explained))
    waited = statistics.median(record["iteration"]["one_call_with_wait_seconds"])
    rows.append(("host time between calls: one call with a wait minus chained", waited - whole))
    return rows


def breakdown_table(big: dict, small: dict) -> str:
    """The iteration at two policy counts, with a line through them: fixed part and part per policy."""
    a, b = breakdown_rows(small), breakdown_rows(big)
    ca, cb = small["copies"], big["copies"]
    lines = [f"| part | {cb} policies (s) | share | {ca} policies (s) | share | fixed part (s) | per policy (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for (name, sa), (_, sb) in zip(a, b):
        slope = (sb - sa) / (cb - ca)
        fixed = sa - ca * slope
        lines.append(f"| {name} | {sb:.4f} | {sb / b[0][1]:.1%} | {sa:.4f} | {sa / a[0][1]:.1%} | "
                     f"{fixed:.4f} | {1e3 * slope:.3f} |")
    return "\n".join(lines)


def native_table(record: dict) -> str:
    """MuJoCo Warp's stages of one RK4 physics step at two world counts, with a line through them."""
    big, small = (record["native"][key] for key in sorted(record["native"], key=int, reverse=True))
    wb, ws = big["worlds"], small["worlds"]
    step = big["stages"]["whole physics step (forward and the Runge-Kutta part)"]["median_seconds"]
    lines = [f"| stage (one call) | kernel launches | {wb:,} worlds (ms) | share of the step | {ws:,} worlds (ms) | "
             "fixed part (ms) | per 1,000 worlds (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in big["stages"].items():
        tb, ts = entry["median_seconds"], small["stages"][name]["median_seconds"]
        slope = (tb - ts) / (wb - ws)
        lines.append(f"| {name} | {entry['kernel_launches']} | {1e3 * tb:.3f} | {tb / step:.1%} | {1e3 * ts:.3f} | "
                     f"{1e3 * (ts - ws * slope):.3f} | {1e6 * slope:.3f} |")
    return "\n".join(lines)


def ideas_table(record: dict) -> str:
    """The five throughput columns, first call, captures and the paired gain of every variant."""
    lines = ["| program | copies | seconds per<br>iteration | total steps per<br>second (millions) | "
             "steps per second<br>per copy (thousands) | hours per million<br>steps per copy | first call (s) | "
             "graph captures<br>while timed | gain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in record["variants"].items():
        t = entry["throughput"]
        lines.append(f"| {WORDS[name]} | {t['copies']} | {t['seconds_per_iteration']:.4f} | "
                     f"{t['total_steps_per_second_millions']:.4f} | {t['steps_per_second_per_copy_thousands']:.3f} | "
                     f"{t['hours_per_million_steps_per_copy']:.4f} | {entry['first_call_seconds']:.1f} | "
                     f"{entry['graph_captures_during_timed_rounds']} | {entry['gain']:.3f} |")
    return "\n".join(lines)


def startup_table(path: Path) -> str:
    """Start-up seconds of the processes of one start-up test."""
    lines = ["| process | build (s) | iteration 1 (s) | iteration 2 (s) | iteration 3 (s) | process start to<br>"
             "end of iteration 3 (s) | iteration after<br>a window end (s) | programs compiled<br>after calls 1 to 4 | "
             "graph captures<br>in calls 1 to 3 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for line in path.read_text().splitlines():
        r = json.loads(line)
        calls = r["iteration_call_seconds"]
        lines.append(f"| {WORDS[r['label']]} | {r['build_seconds']:.1f} | {calls[0]:.1f} | {calls[1]:.1f} | {calls[2]:.2f} | "
                     f"{r['total_seconds_to_third_iteration']:.1f} | {r['iteration_after_a_window_end_seconds']:.2f} | "
                     f"{', '.join(map(str, r['iteration_programs_compiled']))} | "
                     f"{', '.join(map(str, r['iteration_graph_captures']))} |")
    return "\n".join(lines)


def main():
    """Print every table that has its records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="h100-nvl_serval05")
    arguments = parser.parse_args()
    card = arguments.card
    for title, make in (
            ("iteration breakdown", lambda: breakdown_table(load(f"breakdown_{card}_copies-128_second.json"),
                                                            load(f"breakdown_{card}_copies-32.json"))),
            ("native physics split", lambda: native_table(load(f"breakdown_{card}_copies-128.json"))),
            ("loop layout ideas", lambda: ideas_table(load(f"ideas_loop-layout_{card}_copies-128.json"))),
            ("solver loop", lambda: "\n".join([ideas_table(load(f"ideas_solver-plain-loop_{card}_copies-128.json")),
                                                 *ideas_table(load(f"ideas_solver-plain-loop_{card}_copies-32.json"))
                                                 .splitlines()[2:]])),
            ("start-up", lambda: startup_table(RECORDS / f"startup_{card}.jsonl"))):
        print(f"\n### {title}\n")
        print(make())


if __name__ == "__main__":
    main()
