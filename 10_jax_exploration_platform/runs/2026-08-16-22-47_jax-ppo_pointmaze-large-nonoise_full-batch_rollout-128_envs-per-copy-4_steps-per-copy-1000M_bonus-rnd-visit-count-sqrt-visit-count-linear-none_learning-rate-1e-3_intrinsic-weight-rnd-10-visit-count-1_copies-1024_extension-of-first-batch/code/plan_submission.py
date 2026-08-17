"""Decide, by arithmetic, how to spread this run's four work units over the cluster's cards.

The quantity being minimised is the MAKESPAN: the moment the LAST unit finishes, because that is
what a reader waiting for the run's result actually waits for. It is not the average finish time
of a unit and it is not the number of cards used.

The degree of freedom this run has that the 8,192-copy plan did not: a unit is 1,024 copies of one
configuration, the copies are independent seeds, and the platform can now cut a unit into CHUNKS
by copy index (`--copy-seed-offset`, tested in
`tests/agents/test_copy_seed_offset.py`). Chunks run on separate cards at the same time. That is
worth doing because the fused throughput tables show the aggregate rate FLATTENING as copies rise
while the rate one copy gets keeps falling: on an H100 NVL the card moves 45.67 million environment
steps per second at 1,024 copies but 36.21 million at 512, so two cards at 512 copies move 72.42
million against one card's 45.67 — the same 1,024 copies finish in 0.63 of the time. The cost is
one more build, one more queue entry and one more shard per chunk.

What the program reads:

1. **The fused throughput survey**, `<rnd-jax-submission skill>/gpu_node_throughput_survey/`: the
   measured total steps per second of every node class at 512 / 1,024 / 2,048 / 4,096 copies, each
   class at whichever processor count ran fastest, plus that cell's build and compile seconds and
   its peak card memory. Only cells whose `status` is `measured` are used. **A copy count with no
   measured cell is not costed and not chosen** — there is no measurement below 512 copies, so a
   split finer than two chunks cannot be priced from this survey and is reported as a bound
   instead of being planned on (see `optimistic_finer_split`).
2. **This run family's own measurements**, from the canary phase of the 8,192-copy attempt of the
   same four configurations, which fixes each arm's cost relative to the survey's trainer
   (`ARM_ANCHORS`). The survey was measured with PPO plus random network distillation; the two
   oracle arms and the arm with no bonus do about 2.3 times its work per second, and that ratio is
   measured here, not guessed.
3. **Live availability**, through `uva-submit-gpu-sweep`'s own `availability.py --json`, so the
   card list is what is free now rather than a briefing from earlier.

The H100 allowance does not bind: every job of this run occupies its whole card with one compiled
program, which is the FUSED case of the allowance amended on 2026-08-17, so the limit is
usefulness. The program takes a card only when the arithmetic says it moves the makespan, and the
proportionality clause of `rnd-jax-submission` §2 is enforced by the tie-break: among plans whose
makespan is within `PROPORTIONALITY_SECONDS` of the best, the one with the fewest chunks wins.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/plan_submission.py
Tests:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/test_plan_submission.py
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
GPU_SKILL = Path("/p/rlprojects/.claude/skills/uva-submit-gpu-sweep")
SURVEY = Path("/p/rlprojects/.claude/skills/rnd-jax-submission/gpu_node_throughput_survey")
CLASS_CATALOG = GPU_SKILL / "school_compute_resource" / "server_introduction.json"

# the run's shape
COPIES_PER_UNIT = 1024
ROLLOUT_STEPS = 128
ENVS_PER_COPY = 4
WINDOW_ITERATIONS = 200
EPISODE_STEPS = 400
# 10^9 / 512 = 1,953,125 iterations is not a whole 200-iteration window (9,765.625 of them). A
# window must span a whole number of turns of the 25-iteration episode clock or its reward sum
# carries the episode phase and cannot be scored, and 200 iterations is 8 whole turns. 9,766
# windows is 1,953,200 iterations and 1,000,038,400 steps per copy: 10^9 to within 0.0038 per
# cent, nearer than the 9,765 windows below it (0.0064 per cent short).
ITERATIONS = 1953200
STEPS_PER_COPY = ITERATIONS * ROLLOUT_STEPS * ENVS_PER_COPY

# the four configurations, each the arm's winner in the parent run
UNITS = [
    {"order": 1, "arm": "rnd_next_state", "intrinsic_weight": "10"},
    {"order": 2, "arm": "gt_position_velocity_sqrt", "intrinsic_weight": "1"},
    {"order": 3, "arm": "gt_position_velocity_linear", "intrinsic_weight": "1"},
    {"order": 4, "arm": "none", "intrinsic_weight": None},
]

# This run family's OWN measurements, from the canary phase of the 8,192-copy attempt of the same
# four configurations (its `canary_estimate_vs_actual.md`). Each is the steady seconds per
# iteration with the compiling first iteration excluded. They fix each arm's cost relative to the
# survey's trainer, and they are the only place a number about THIS platform enters.
ARM_ANCHORS = {
    "rnd_next_state": {"node_class": "serval06-09", "copies": 8192,
                       "seconds_per_iteration": 0.084422, "peak_device_memory_gib": 28.2,
                       "job": 6538637},
    "gt_position_velocity_sqrt": {"node_class": "serval06-09", "copies": 8192,
                                  "seconds_per_iteration": 0.034673,
                                  "peak_device_memory_gib": 7.8, "job": 6538638},
    "gt_position_velocity_linear": {"node_class": "serval06-09", "copies": 8192,
                                    "seconds_per_iteration": 0.034171,
                                    "peak_device_memory_gib": 7.8, "job": 6538639},
    "none": {"node_class": "cheetah01", "copies": 8192, "seconds_per_iteration": 0.068844,
             "peak_device_memory_gib": 7.8, "job": 6538640},
}
# the surveyed copy count the anchor is compared against: the survey stops at 4,096 and the
# anchors were taken at 8,192, so this is the nearest measured neighbour
ANCHOR_REFERENCE_COPIES = 4096

# how a unit may be cut. A chunk's copy count must be a copy count the survey measured, or the
# option cannot be priced; 1,024 / 4 = 256 has no measured cell anywhere in the survey.
SPLITS = (1, 2, 4, 8)

# setup a job pays before its first timed iteration, in seconds
PROCESS_START_SECONDS = 45.0      # interpreter, jax import, environment build, run-folder writes
WARM_CACHE_FRACTION = 0.45        # the canary writes the node-local compiled-program cache first;
                                  # the 8,192-copy attempt measured 226 -> 100 s and 397 -> 166 s
EXPECTED_QUEUE_SECONDS = 120.0    # a card the availability read says is free still costs a queue
                                  # entry; every job of the 8,192-copy attempt started inside 10 s,
                                  # so this is deliberately pessimistic
# two plans whose makespans are within this many seconds of each other count as equally good, and
# the one with fewer chunks wins — the proportionality clause of rnd-jax-submission section 2
PROPORTIONALITY_SECONDS = 300.0
MEMORY_INFLATION = 1.25           # uva-submit-gpu-sweep step 5, for a figure transferred between
                                  # cards


def load_survey() -> dict:
    """Every measured survey cell, keyed by (node class, copies), each class at its fastest count.

    before: 296 files under data/throughput/, several processor counts per class;
    after:  {("serval06-09", 512): {"total_steps_per_second": 3.62e7, "build_seconds": ...}, ...}
            holding, per (class, copies), the cell of whichever processor count ran fastest.
    """
    best = {}
    for path in glob.glob(str(SURVEY / "data" / "throughput" / "*__full_batch.json")):
        record = json.load(open(path))
        for cell in record["cells"]:
            if cell.get("status") != "measured":
                continue
            key = (record["node_class"], cell["n_copies"])
            current = best.get(key)
            if current is None or cell["total_steps_per_second"] > current["total_steps_per_second"]:
                best[key] = dict(cell, requested_cpus=record["requested_cpus"])
    if not best:
        raise SystemExit(f"no measured throughput cells under {SURVEY}")
    return best


def load_class_catalog() -> dict:
    """Card memory and card count per node class, from the cluster's own catalog."""
    classes = json.load(open(CLASS_CATALOG))["classes"]
    return {entry["name"]: entry for entry in classes}


def live_cards(availability_json: dict = None) -> list:
    """One entry per FREE card, tagged with its node and class, read live unless one is supplied.

    before: availability's nodes map, serval06 with free_gpus 2;
    after:  [{"node": "serval06", "node_class": "serval06-09"}, {same again}, ...] — one entry per
            free card, because a card is the unit a chunk is assigned to.
    """
    if availability_json is None:
        output = subprocess.run(
            [sys.executable if os.access(sys.executable, os.X_OK) else "python3",
             str(GPU_SKILL / "scripts" / "availability.py"), "--json"],
            capture_output=True, text=True, check=True).stdout
        availability_json = json.loads(output)
    cards = []
    for node, entry in availability_json["nodes"].items():
        live = entry["live"]
        if not live.get("usable"):
            continue
        for _ in range(live.get("free_gpus", 0)):
            cards.append({"node": node, "node_class": entry["class"]})
    return cards


def arm_factor(arm: str, survey: dict) -> float:
    """How much work this arm does per second, as a multiple of the survey's trainer.

    before: `none` measured at 0.068844 s per iteration on cheetah01 at 8,192 copies, and the
            survey's cheetah01 cell at 4,096 copies moving 2.558e7 steps per second;
    after:  8192 x 512 / 0.068844 / 2.558e7 = 2.382 — this arm does 2.38 times the survey
            trainer's work per second on the same card.
    """
    anchor = ARM_ANCHORS[arm]
    measured = (anchor["copies"] * ROLLOUT_STEPS * ENVS_PER_COPY
                / anchor["seconds_per_iteration"])
    reference = survey[(anchor["node_class"], ANCHOR_REFERENCE_COPIES)]["total_steps_per_second"]
    return measured / reference


def load_measured_cells() -> dict:
    """This run's OWN probe measurements, keyed by (arm, node class, copies); empty if none yet.

    A cell here is a direct 200-iteration measurement of this arm on this card at this copy count,
    written by `code/measured_cells.py` from the probe shards. It is preferred over the survey
    wherever it exists, because the survey measures a different trainer and reaches it only through
    a transfer factor, and because the survey has nothing below 512 copies at all.
    """
    path = RUN_DIR / "code" / "measured_cells.json"
    if not path.exists():
        return {}
    return {(cell["arm"], cell["node_class"], cell["copies"]): cell
            for cell in json.load(open(path))["cells"]}


def probed_reference(arm: str, copies: int, survey: dict, measured: dict):
    """A class where this arm WAS probed at this copy count and which the survey also covers.

    This is the bridge for a class nothing probed: the survey's business is ranking cards against
    each other, so its ratio between two classes at one copy count is the part of it worth
    transferring. Its ABSOLUTE rate for this arm at this copy count is not, which the probes showed
    outright — the arm factors below were measured at 8,192 copies, where the bonus dominates the
    iteration, and at 1,024 copies the four arms come within 1.7 times of each other instead of 2.4,
    so scaling a survey cell by an 8,192-copy factor credits a slow card with an impossible rate.
    """
    for (probed_arm, node_class, probed_copies), cell in (measured or {}).items():
        if probed_arm == arm and probed_copies == copies and (node_class, copies) in survey:
            return node_class, cell
    return None, None


def per_copy_rate(arm: str, node_class: str, copies: int, survey: dict, factors: dict,
                  measured: dict = None):
    """Environment steps per second ONE copy gets, or None when nothing measured covers the cell.

    Order of preference, each step named by `rate_source`:

    1. this run's own probe of exactly this arm, class and copy count — a measurement, used as is;
    2. this run's probe of the same arm at the same copy count on another class, carried across by
       the survey's ratio between the two classes AT THAT COPY COUNT — the arm and the copy count
       stay measured and only the card ratio is transferred;
    3. the survey's cell for the class scaled by the arm's factor measured at 8,192 copies — a last
       resort for a copy count no probe covers, and marked as such.

    Nothing is extrapolated beyond a copy count some table measured.
    """
    own = (measured or {}).get((arm, node_class, copies))
    if own is not None:
        return own["steps_per_second_per_copy"]
    cell = survey.get((node_class, copies))
    if cell is None:
        return None
    reference_class, reference_cell = probed_reference(arm, copies, survey, measured)
    if reference_class is not None:
        card_ratio = (cell["total_steps_per_second"]
                      / survey[(reference_class, copies)]["total_steps_per_second"])
        return reference_cell["steps_per_second_per_copy"] * card_ratio
    return factors[arm] * cell["total_steps_per_second"] / copies


def rate_source(arm: str, node_class: str, copies: int, survey: dict, measured: dict) -> str:
    """Which table a chunk's rate came from, so the plan records its own provenance."""
    if (arm, node_class, copies) in (measured or {}):
        return "probed on this card"
    if (node_class, copies) not in survey:
        return "not priceable"
    reference_class, _ = probed_reference(arm, copies, survey, measured)
    if reference_class is not None:
        return f"probed on {reference_class}, carried by the survey's card ratio at {copies} copies"
    return "survey cell scaled by the arm factor measured at 8,192 copies"


def fits(arm: str, node_class: str, copies: int, catalog: dict) -> bool:
    """Whether the class's card holds this arm at this copy count, with the 25 per cent margin.

    The peak is scaled from this arm's own anchor measurement rather than from the survey, because
    the arms differ in what they hold on the card: distillation carries a predictor network and the
    visit-count arms carry a count table.
    """
    anchor = ARM_ANCHORS[arm]
    peak_gib = anchor["peak_device_memory_gib"] * copies / anchor["copies"]
    card_gib = catalog[node_class]["gpu_mem_mb"] / 1024.0
    return card_gib >= MEMORY_INFLATION * peak_gib


def setup_seconds(arm: str, node_class: str, copies: int, survey: dict, measured: dict,
                  warm: bool = True) -> float:
    """Seconds a job spends before its first timed iteration, from whichever table priced it.

    This run's own probe measured the build, the prime and the compiling first iteration on the
    card itself, so where a probe cell exists that is the figure; otherwise the survey's build plus
    compile for the class. Either way the warm case scales it down, because the canary on the same
    node writes the node-local compiled-program cache before the science job runs.
    """
    own = (measured or {}).get((arm, node_class, copies))
    if own is not None:
        compile_cost = (own["build_and_prime_seconds"]
                        + own["first_iteration_with_compile_seconds"])
    else:
        cell = survey[(node_class, copies)]
        compile_cost = cell["build_seconds"] + cell["compile_seconds"] + PROCESS_START_SECONDS
    if warm:
        compile_cost *= WARM_CACHE_FRACTION
    return compile_cost + EXPECTED_QUEUE_SECONDS


def chunk_seconds(arm: str, node_class: str, copies: int, survey: dict, factors: dict,
                  measured: dict = None, warm: bool = True):
    """Total seconds for one chunk of `copies` copies of `arm` on one card of `node_class`."""
    rate = per_copy_rate(arm, node_class, copies, survey, factors, measured)
    if rate is None:
        return None
    return STEPS_PER_COPY / rate + setup_seconds(arm, node_class, copies, survey, measured, warm)


def schedule(chunks: list, cards: list, survey: dict, factors: dict, catalog: dict,
             measured: dict = None):
    """List-schedule chunks onto cards of unequal speed, longest chunk first, earliest finish.

    Each card runs its chunks one after another, so a card's finish time is the sum of what it was
    given; a chunk goes to whichever card would finish it soonest. This is the standard rule for
    machines of unequal speed: it keeps the long chunks on the fast cards without leaving a fast
    card idle, and it prefers a free slow card over a loaded fast one exactly when the arithmetic
    says the chunk lands sooner there.

    before: chunks [(rnd, 512), (rnd, 512), (sqrt, 1024)], cards [H100, H100, H100]
    after:  one chunk per card, each finishing at its own single chunk's time
    Returns (assignment, makespan_seconds) or (None, None) when some chunk fits no free card.
    """
    load = [0.0] * len(cards)
    assignment = []
    # longest first, costed on the fastest card that can hold the chunk at all
    def longest_key(chunk):
        times = [chunk_seconds(chunk["arm"], card["node_class"], chunk["copies"], survey, factors,
                              measured)
                 for card in cards
                 if fits(chunk["arm"], card["node_class"], chunk["copies"], catalog)]
        times = [t for t in times if t is not None]
        return -min(times) if times else 0.0

    for chunk in sorted(chunks, key=longest_key):
        best = None
        for index, card in enumerate(cards):
            if not fits(chunk["arm"], card["node_class"], chunk["copies"], catalog):
                continue
            duration = chunk_seconds(chunk["arm"], card["node_class"], chunk["copies"],
                                     survey, factors, measured)
            if duration is None:
                continue
            finish = load[index] + duration
            if best is None or finish < best[0]:
                best = (finish, index, duration)
        if best is None:
            return None, None
        finish, index, duration = best
        load[index] = finish
        assignment.append(dict(chunk, node=cards[index]["node"],
                               node_class=cards[index]["node_class"],
                               seconds=duration, finish_seconds=finish,
                               rate_source=rate_source(chunk["arm"],
                                                       cards[index]["node_class"],
                                                       chunk["copies"], survey, measured)))
    return assignment, max((row["finish_seconds"] for row in assignment), default=0.0)


def chunks_of(unit: dict, k: int) -> list:
    """A unit cut into k chunks, each holding a disjoint slice of the unit's copy indices.

    before: unit 1 (rnd_next_state, 1024 copies), k = 2
    after:  two chunks of 512 copies, the first holding copy indices 0..511 and the second
            512..1023, so their union is exactly the 1,024-copy run and neither repeats a seed
    """
    if COPIES_PER_UNIT % k:
        return []
    size = COPIES_PER_UNIT // k
    return [{"order": unit["order"], "arm": unit["arm"],
             "intrinsic_weight": unit["intrinsic_weight"], "chunk": j, "chunks": k,
             "copies": size, "copy_index_first": j * size,
             "copy_index_last": (j + 1) * size - 1} for j in range(k)]


def priceable(unit: dict, k: int, survey: dict, cards: list, catalog: dict,
              measured: dict = None) -> bool:
    """Whether a k-way split of this unit can be costed at all from measured data.

    A split is priceable when its chunk copy count has a cell — this run's own probe first, the
    shared survey second — on at least one free card's class that can also hold it. Nothing is
    extrapolated: a copy count neither table covers makes that split unavailable rather than
    guessed.
    """
    if COPIES_PER_UNIT % k:
        return False
    size = COPIES_PER_UNIT // k
    return any(
        ((unit["arm"], card["node_class"], size) in (measured or {})
         or (card["node_class"], size) in survey)
        and fits(unit["arm"], card["node_class"], size, catalog) for card in cards)


def plan(cards: list, survey: dict, factors: dict, catalog: dict, measured: dict = None) -> dict:
    """Every priceable combination of per-unit splits, costed; the makespan-minimising one wins."""
    per_unit_options = []
    for unit in UNITS:
        options = [k for k in SPLITS if priceable(unit, k, survey, cards, catalog, measured)]
        if not options:
            raise SystemExit(f"no priceable split for unit {unit['order']} ({unit['arm']}): no "
                             f"free card's class has a measured cell it can hold")
        per_unit_options.append(options)

    considered = []
    for combination in itertools.product(*per_unit_options):
        chunks = [chunk for unit, k in zip(UNITS, combination) for chunk in chunks_of(unit, k)]
        assignment, makespan = schedule(chunks, cards, survey, factors, catalog, measured)
        if assignment is None:
            continue
        considered.append({"splits": list(combination), "chunk_count": len(chunks),
                           "makespan_seconds": makespan, "assignment": assignment})
    if not considered:
        raise SystemExit("no combination of splits could be scheduled on the free cards")

    # the makespan decides; among plans that finish within PROPORTIONALITY_SECONDS of the best,
    # the one with fewer chunks wins, because the extra chunks buy no finish time and cost a build,
    # a queue entry, a shard and a line of the monitoring table each
    best_makespan = min(row["makespan_seconds"] for row in considered)
    chosen = min((row for row in considered
                  if row["makespan_seconds"] <= best_makespan + PROPORTIONALITY_SECONDS),
                 key=lambda row: (row["chunk_count"], row["makespan_seconds"]))
    considered.sort(key=lambda row: row["makespan_seconds"])
    return {"chosen": chosen, "considered": considered}


def unpriced_finer_splits(cards: list, survey: dict, catalog: dict, measured: dict) -> dict:
    """Which splits neither table can price, and how much a split one step finer might be worth.

    A split whose chunk copy count no table covers is unavailable, not guessed. This reports what
    is missing, and — only when something IS missing — a BOUND on what it could have been worth,
    obtained by assuming the per-copy rate improves at the next halving by the same factor it
    improves by at the last halving both tables do cover. The bound is never used to decide; it
    exists so that "we stopped splitting here" is answered by arithmetic instead of left open, and
    so a later session can see whether a measurement would pay for itself.
    """
    fastest = "serval06-09"
    missing = []
    for unit in UNITS:
        for k in SPLITS:
            if COPIES_PER_UNIT % k:
                continue
            if not priceable(unit, k, survey, cards, catalog, measured):
                missing.append({"unit": unit["order"], "arm": unit["arm"], "split": k,
                                "chunk_copies": COPIES_PER_UNIT // k})
    if not missing:
        return {"anything_unpriced": False,
                "note": "every split of every unit is priced from a measured cell"}
    # the last halving both tables cover on the fastest class, as the improvement factor to assume
    coarse = per_copy_rate("rnd_next_state", fastest, 1024, survey, {"rnd_next_state": 1.0},
                           measured)
    fine = per_copy_rate("rnd_next_state", fastest, 512, survey, {"rnd_next_state": 1.0}, measured)
    gain = (fine / coarse) if coarse and fine else None
    return {"anything_unpriced": True, "unpriced": missing,
            "assumed_gain_at_the_next_halving": gain,
            "note": "a bound only; nothing below the smallest measured copy count is planned on"}


def hours(seconds: float) -> str:
    """Seconds as hours and minutes, the unit a run is planned in."""
    return f"{seconds / 3600:.2f} h"


def main() -> None:
    """Compute the plan, print it as a table, and write code/submission_plan.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability-json", default="",
                        help="a saved availability.py --json file, for a replay; live by default")
    args = parser.parse_args()

    survey = load_survey()
    catalog = load_class_catalog()
    measured = load_measured_cells()
    factors = {unit["arm"]: arm_factor(unit["arm"], survey) for unit in UNITS}
    saved = json.load(open(args.availability_json)) if args.availability_json else None
    cards = live_cards(saved)
    result = plan(cards, survey, factors, catalog, measured)
    unpriced = unpriced_finer_splits(cards, survey, catalog, measured)

    print(f"free cards read live: {len(cards)}")
    by_class = {}
    for card in cards:
        by_class[card["node_class"]] = by_class.get(card["node_class"], 0) + 1
    for name, count in sorted(by_class.items(), key=lambda item: -item[1]):
        print(f"  {name:18} {count} free")
    print(f"\nthis run's own probe cells: {len(measured)}")
    for (arm, node_class, copies), cell in sorted(measured.items()):
        print(f"  {arm:30} {node_class:14} {copies:>5} copies  "
              f"{cell['seconds_per_iteration']:.5f} s/iteration  "
              f"{cell['steps_per_second_per_copy']:>8.0f} steps/s/copy")
    print("\narm cost relative to the survey's trainer, from this run family's own canaries "
          "(used only where no probe cell covers the case):")
    for arm, factor in factors.items():
        print(f"  {arm:30} x{factor:.4f}")

    chosen = result["chosen"]
    print(f"\nCHOSEN PLAN — makespan {hours(chosen['makespan_seconds'])}, "
          f"{chosen['chunk_count']} chunks")
    print(f"{'unit':>5} {'arm':30} {'chunks':>7} {'copies':>7} {'copy indices':>14} "
          f"{'node':>11} {'class':>14} {'est':>8} {'finish':>8}  rate from")
    for row in sorted(chosen["assignment"], key=lambda r: (r["order"], r["chunk"])):
        print(f"{row['order']:>5} {row['arm']:30} {row['chunk'] + 1}/{row['chunks']:<5} "
              f"{row['copies']:>7} {row['copy_index_first']:>6}-{row['copy_index_last']:<7} "
              f"{row['node']:>11} {row['node_class']:>14} "
              f"{hours(row['seconds']):>8} {hours(row['finish_seconds']):>8}  "
              f"{row['rate_source']}")

    print("\nalternatives considered, by makespan:")
    print(f"{'splits per unit':>20} {'chunks':>7} {'makespan':>10}   verdict")
    for row in result["considered"][:14]:
        verdict = ("chosen" if row is chosen else
                   "rejected: slower" if row["makespan_seconds"] > chosen["makespan_seconds"]
                   else "rejected: same finish within the proportionality margin, more chunks")
        print(f"{str(row['splits']):>20} {row['chunk_count']:>7} "
              f"{hours(row['makespan_seconds']):>10}   {verdict}")

    print("\nsplits no table can price:")
    if unpriced["anything_unpriced"]:
        for row in unpriced["unpriced"]:
            print(f"  unit {row['unit']} {row['arm']} at {row['split']} chunks of "
                  f"{row['chunk_copies']} copies — no measured cell")
        gain = unpriced["assumed_gain_at_the_next_halving"]
        print(f"  bound only: the last measured halving improves the per-copy rate by "
              f"{gain:.3f}x, so a further halving is worth at most that much again — not planned on")
    else:
        print(f"  none: {unpriced['note']}")

    output = {"generated_at_pacific": subprocess.run(
                  ["date", "+%Y-%m-%d %H:%M PT"], capture_output=True, text=True,
                  env=dict(os.environ, TZ="America/Los_Angeles")).stdout.strip(),
              "iterations": ITERATIONS, "steps_per_copy": STEPS_PER_COPY,
              "copies_per_unit": COPIES_PER_UNIT, "window_iterations": WINDOW_ITERATIONS,
              "arm_factors": factors, "free_cards": cards,
              "own_probe_cells": [dict(cell) for cell in measured.values()],
              "chosen": chosen, "considered": result["considered"],
              "unpriced_finer_splits": unpriced}
    (RUN_DIR / "code" / "submission_plan.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {RUN_DIR / 'code' / 'submission_plan.json'}")


if __name__ == "__main__":
    main()
