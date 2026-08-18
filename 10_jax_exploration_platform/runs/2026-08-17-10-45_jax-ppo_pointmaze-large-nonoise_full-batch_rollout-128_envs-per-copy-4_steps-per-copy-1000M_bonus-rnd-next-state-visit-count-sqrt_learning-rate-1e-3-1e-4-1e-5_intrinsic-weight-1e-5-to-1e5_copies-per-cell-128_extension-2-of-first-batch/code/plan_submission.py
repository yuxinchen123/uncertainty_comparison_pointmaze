"""Decide, by arithmetic, how to spread this run's two work units over the cluster's cards.

The quantity being minimised is the MAKESPAN: the moment the LAST chunk finishes, because that is
what a reader waiting for the run's result actually waits for. It is not the average finish time of
a chunk and it is not the number of cards used.

A work unit here is one algorithm arm's whole learning-rate by intrinsic-weight grid, fused into
one compiled program: 3 learning rates x 11 intrinsic weights = 33 cells x 128 copies = 4,224
copies, each copy advancing about 1,000M environment steps. A unit may be CUT INTO CHUNKS by copy
index, because the copies are independent seeds and a chunk of them runs at a much better per-copy
rate on its own card. Chunk j of k passes `--copy-seed-offset j x (128 / k)` and holds copy indices
j x (128/k) .. (j+1) x (128/k) - 1 of EVERY cell, so the k chunks partition each cell's 128 copies
and every chunk carries all 33 configurations.

What the program reads:

1. **This run's own rate probes**, `code/measured_cells.json`: both arms measured for 400 real
   iterations at 528 / 1,056 / 2,112 / 4,224 copies on an H100 NVL, an A100-SXM4 and an
   A100-PCIE. That is a direct measurement of this arm at this copy count, and it is used as is
   wherever it exists.
2. **The fused throughput survey**, `<rnd-jax-submission skill>/gpu_node_throughput_survey/`, for
   one thing only: the RATIO between two node classes at one copy count, which carries a probed
   rate onto a class nothing probed. Its absolute rates are never scaled into this run's arms.
   The previous extension's hard lesson is why: an arm's cost relative to the survey's trainer is
   not a constant of the arm, it is a function of the copy count (2.4 times at 8,192 copies, 1.7
   at 1,024), so a factor measured at one copy count credits a slow card with an impossible rate
   at another. Only card-to-card ratios at the SAME copy count are transferred, and the arms and
   the copy counts stay measured.
3. **Live availability**, through `uva-submit-gpu-sweep`'s own `availability.py --json`, so the
   card list is what is free now rather than a briefing from earlier.

The H100 allowance does not bind: every job of this run occupies its whole card with one compiled
program, which is the FUSED case of the allowance amended on 2026-08-17, so the limit is
usefulness. The proportionality clause of `rnd-jax-submission` section 2 is enforced by the
tie-break: among plans whose makespan is within `PROPORTIONALITY_SECONDS` of the best, the one with
the fewest chunks wins.

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
LEARNING_RATES = ("1e-3", "1e-4", "1e-5")
INTRINSIC_WEIGHTS = ("1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1", "10", "100", "1000", "10000",
                     "100000")
CELLS = len(LEARNING_RATES) * len(INTRINSIC_WEIGHTS)
COPIES_PER_CELL = 128
COPIES_PER_UNIT = CELLS * COPIES_PER_CELL
ROLLOUT_STEPS = 128
ENVS_PER_COPY = 4
WINDOW_ITERATIONS = 200
EPISODE_STEPS = 400
# 10^9 / 512 = 1,953,125 iterations is not a whole 200-iteration window (9,765.625 of them). A
# window must span a whole number of turns of the 25-iteration episode clock or its reward sum
# carries the episode phase and cannot be scored, and 200 iterations is 8 whole turns. 9,766
# windows is 1,953,200 iterations and 1,000,038,400 steps per copy: 10^9 to within 0.0038 per cent,
# nearer than the 9,765 windows below it. The document says 1000M steps; this is the exact count.
ITERATIONS = 1953200
STEPS_PER_COPY = ITERATIONS * ROLLOUT_STEPS * ENVS_PER_COPY

# the two arms, each its whole grid in one fused program
UNITS = [
    {"order": 1, "arm": "rnd_next_state"},
    {"order": 2, "arm": "gt_position_velocity_sqrt"},
]

# how a unit may be cut. The copies-per-cell of a chunk must be a whole number, and the chunk's
# total copy count must be one this run's probe measured, or the option cannot be priced.
SPLITS = (1, 2, 4, 8, 16, 32)

# peak card memory each arm holds, measured on this run family's 8,192-copy canaries and scaled by
# copies. At the copy counts this run actually considers the largest requirement is 18 GiB (the
# distillation arm whole), so the model's precision only ever decides whether a 24 GB card may hold
# an uncut unit — every card in the cluster clears the 2.3 GiB an eighth of one needs.
ARM_MEMORY = {
    "rnd_next_state": {"peak_device_memory_gib": 28.2, "at_copies": 8192, "job": 6538637},
    "gt_position_velocity_sqrt": {"peak_device_memory_gib": 7.8, "at_copies": 8192,
                                  "job": 6538638},
}
MEMORY_INFLATION = 1.25           # uva-submit-gpu-sweep step 5, for a figure transferred between
                                  # cards

# setup a job pays before its first timed iteration, in seconds
PROCESS_START_SECONDS = 45.0      # interpreter, jax import, environment build, run-folder writes
WARM_CACHE_FRACTION = 0.45        # the canary writes the node-local compiled-program cache first;
                                  # the 8,192-copy attempt measured 226 -> 100 s and 397 -> 166 s
EXPECTED_QUEUE_SECONDS = 120.0    # a card the availability read says is free still costs a queue
                                  # entry; every job of the previous extension started inside 10 s,
                                  # so this is deliberately pessimistic
# two plans whose makespans are within this many seconds of each other count as equally good, and
# the one with fewer chunks wins — the proportionality clause of rnd-jax-submission section 2
PROPORTIONALITY_SECONDS = 900.0

# the survey copy count nearest each of this run's chunk sizes, for the card-ratio transfer. The
# survey measured powers of two; this run's chunks are 33 cells wide, so they sit 3 per cent above
# them.
NEAREST_SURVEY_COPIES = {132: 512, 264: 512, 528: 512, 1056: 1024, 2112: 2048, 4224: 4096}
# 132 and 264 copies map to 512 because the survey measured nothing below it. What is transferred
# is only the ratio between two cards, and that ratio holds up at the finer counts: measured here,
# the RTX A4500 leads the Quadro RTX 6000 by 1.31 to 1.36 times at 264 and 132 copies against the
# survey's 1.39 at 512, and the RTX A4000 by 1.34 against 1.40 — within 6 per cent either way.

# a class that measures below this fraction of the fastest free class is not worth a chunk of this
# run: its chunk would set the makespan on its own. Read as "a card slower than a seventh of an
# H100 NVL cannot help", and it exists to keep the search from considering 130 cards.
SLOWEST_USEFUL_FRACTION = 0.10

# how many cards of each partition this user may actually hold at once. The GPU caps are 40 on gpu
# and 20 on gnolim; the processor caps bind first here, because a job asks for 8 processors and
# other sessions of this user already hold 84 of the gpu partition's 400. Recomputed at launch,
# not inherited: both numbers move.
CARD_CAP = {"gpu": 39, "gnolim": 10}


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


def load_measured_cells() -> dict:
    """This run's OWN probe measurements, keyed by (arm, node class, copies); empty if none yet.

    A cell here is a direct 400-iteration measurement of this arm on this card at this copy count,
    written by `code/measured_cells.py` from the probe shards.
    """
    path = RUN_DIR / "code" / "measured_cells.json"
    if not path.exists():
        return {}
    return {(cell["arm"], cell["node_class"], cell["copies"]): cell
            for cell in json.load(open(path))["cells"]}


def reservation_of(note: str) -> str:
    """Our reservation's name when the availability note says the node is under one, else "".

    before: "reserved sl5nw_156 (ours)" ; after: "sl5nw_156"
    before: "open"                      ; after: ""
    A node under a reservation refuses a job that does not name it, so this has to reach the
    submission script rather than being noticed by a human.
    """
    if not note.startswith("reserved ") or "(ours)" not in note:
        return ""
    return note.split()[1]


def live_cards(availability_json: dict = None) -> list:
    """One entry per FREE card, tagged with its node, class, partition and reservation.

    before: availability's nodes map, jaguar03 with free_gpus 8 and note "reserved sl5nw_156
            (ours)";
    after:  eight entries {"node": "jaguar03", "node_class": "jaguar03", "partition": "gpu",
            "reservation": "sl5nw_156", "free_in_seconds": 0.0} — one entry per card, because a
            card is the unit a chunk is assigned to.
    """
    if availability_json is None:
        output = subprocess.run(
            [sys.executable, str(GPU_SKILL / "scripts" / "availability.py"), "--json"],
            capture_output=True, text=True, check=True).stdout
        availability_json = json.loads(output)
    cards = []
    for node, entry in availability_json["nodes"].items():
        live = entry["live"]
        if not live.get("usable"):
            continue
        for _ in range(live.get("free_gpus", 0)):
            cards.append({"node": node, "node_class": entry["class"],
                          "partition": entry["partition"],
                          "reservation": reservation_of(live.get("note", "")),
                          "free_in_seconds": 0.0})
    return cards


def capped_cards(cards: list, survey: dict, measured: dict) -> list:
    """At most as many cards per partition as the per-user caps allow, fastest classes first.

    before: 130 free cards, 118 of them in the gpu partition;
    after:  the fastest CARD_CAP["gpu"] of those plus the fastest CARD_CAP["gnolim"] of the rest.
    The cap is the smaller of the partition's GPU cap and what its CPU cap allows at 8 processors
    a job, both read from `uva-submit-gpu-sweep`; a plan that ignored it would place chunks on
    cards the scheduler would never hand over.
    """
    smallest = COPIES_PER_UNIT // max(SPLITS)

    def speed(card):
        """How fast this card's class is for the slower arm at the smallest chunk, or 0."""
        return per_copy_rate(UNITS[0]["arm"], card["node_class"], smallest, survey, measured) or 0.0

    kept = []
    for partition, cap in CARD_CAP.items():
        pool = sorted([card for card in cards if card["partition"] == partition],
                      key=speed, reverse=True)
        kept += pool[:cap]
    return kept


def nearest_survey_cell(node_class: str, copies: int, survey: dict):
    """The survey cell this run's chunk size is carried by, or None when the survey has none."""
    surveyed = NEAREST_SURVEY_COPIES.get(copies)
    if surveyed is None:
        return None
    return survey.get((node_class, surveyed))


def probed_reference(arm: str, copies: int, survey: dict, measured: dict):
    """A class where this arm WAS probed at this copy count and which the survey also covers.

    This is the bridge for a class nothing probed: the survey's business is ranking cards against
    each other, so its RATIO between two classes at one copy count is the part of it worth
    transferring. Its absolute rate for this arm is not — that is the previous extension's lesson,
    recorded in this module's docstring. The fastest probed class is chosen, so the ratio is always
    read in the same direction.
    """
    candidates = [(node_class, cell) for (probed_arm, node_class, probed_copies), cell
                  in (measured or {}).items()
                  if probed_arm == arm and probed_copies == copies
                  and nearest_survey_cell(node_class, copies, survey) is not None]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1]["steps_per_second_per_copy"])


def per_copy_rate(arm: str, node_class: str, copies: int, survey: dict, measured: dict = None):
    """Environment steps per second ONE copy gets, or None when nothing measured covers the cell.

    Order of preference, each step named by `rate_source`:

    1. this run's own probe of exactly this arm, class and copy count — a measurement, used as is;
    2. this run's probe of the same arm at the same copy count on another class, carried across by
       the survey's ratio between the two classes at the nearest surveyed copy count — the arm and
       the copy count stay measured and only the card ratio is transferred.

    Nothing else is offered. A class the survey never measured, or a chunk size the probe never
    measured, has no price and is not chosen.
    """
    own = (measured or {}).get((arm, node_class, copies))
    if own is not None:
        return own["steps_per_second_per_copy"]
    cell = nearest_survey_cell(node_class, copies, survey)
    if cell is None:
        return None
    reference_class, reference_cell = probed_reference(arm, copies, survey, measured)
    if reference_class is None:
        return None
    reference_survey = nearest_survey_cell(reference_class, copies, survey)
    card_ratio = cell["total_steps_per_second"] / reference_survey["total_steps_per_second"]
    return reference_cell["steps_per_second_per_copy"] * card_ratio


def rate_source(arm: str, node_class: str, copies: int, survey: dict, measured: dict) -> str:
    """Which table a chunk's rate came from, so the plan records its own provenance."""
    if (arm, node_class, copies) in (measured or {}):
        return "probed on this card at this copy count"
    reference_class, _ = probed_reference(arm, copies, survey, measured)
    if nearest_survey_cell(node_class, copies, survey) is None or reference_class is None:
        return "not priceable"
    surveyed = NEAREST_SURVEY_COPIES[copies]
    return (f"probed on {reference_class}, carried by the survey's card ratio at "
            f"{surveyed} copies")


def fits(arm: str, node_class: str, copies: int, catalog: dict) -> bool:
    """Whether the class's card holds this arm at this copy count, with the 25 per cent margin."""
    anchor = ARM_MEMORY[arm]
    peak_gib = anchor["peak_device_memory_gib"] * copies / anchor["at_copies"]
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
        cell = nearest_survey_cell(node_class, copies, survey)
        compile_cost = cell["build_seconds"] + cell["compile_seconds"] + PROCESS_START_SECONDS
    if warm:
        compile_cost *= WARM_CACHE_FRACTION
    return compile_cost + EXPECTED_QUEUE_SECONDS


def chunk_seconds(arm: str, node_class: str, copies: int, survey: dict, measured: dict = None,
                  warm: bool = True):
    """Total seconds for one chunk of `copies` copies of `arm` on one card of `node_class`."""
    rate = per_copy_rate(arm, node_class, copies, survey, measured)
    if rate is None:
        return None
    return STEPS_PER_COPY / rate + setup_seconds(arm, node_class, copies, survey, measured, warm)


def usable_cards(cards: list, survey: dict, measured: dict, catalog: dict) -> list:
    """The cards worth considering: priceable for some chunk size, and not hopelessly slow.

    before: 130 free cards, most of them GTX 1080-class;
    after:  the ones whose class can be priced for a chunk of this run AND reaches at least
            SLOWEST_USEFUL_FRACTION of the fastest priceable class, because a chunk on a card
            slower than that sets the makespan by itself and the plan would never choose it.
    """
    smallest = COPIES_PER_UNIT // max(SPLITS)
    rates = {}
    for card in cards:
        rate = per_copy_rate(UNITS[0]["arm"], card["node_class"], smallest, survey, measured)
        if rate is not None and fits(UNITS[0]["arm"], card["node_class"], smallest, catalog):
            rates[card["node_class"]] = rate
    if not rates:
        raise SystemExit("no free card's class can be priced for any chunk size of this run")
    fastest = max(rates.values())
    keep = {name for name, rate in rates.items()
            if rate >= SLOWEST_USEFUL_FRACTION * fastest}
    return [card for card in cards if card["node_class"] in keep]


def schedule(chunks: list, cards: list, survey: dict, catalog: dict, measured: dict = None):
    """List-schedule chunks onto cards of unequal speed, longest chunk first, earliest finish.

    Each card runs its chunks one after another, so a card's finish time is when it comes free plus
    the sum of what it was given; a chunk goes to whichever card would finish it soonest. This is
    the standard rule for machines of unequal speed: it keeps the long chunks on the fast cards
    without leaving a fast card idle, and it prefers a free slow card over a loaded fast one exactly
    when the arithmetic says the chunk lands sooner there.

    before: chunks [(rnd, 528), (rnd, 528), (sqrt, 1056)], cards [H100, A40, A40]
    after:  one chunk per card, each finishing at its own single chunk's time
    Returns (assignment, makespan_seconds) or (None, None) when some chunk fits no card.
    """
    load = [card["free_in_seconds"] for card in cards]
    assignment = []

    def longest_key(chunk):
        """A chunk's time on the fastest card that can hold it, negated so the longest sorts first."""
        times = [chunk_seconds(chunk["arm"], card["node_class"], chunk["copies"], survey, measured)
                 for card in cards
                 if fits(chunk["arm"], card["node_class"], chunk["copies"], catalog)]
        times = [t for t in times if t is not None]
        return -min(times) if times else 0.0

    for chunk in sorted(chunks, key=longest_key):
        best = None
        for index, card in enumerate(cards):
            if not fits(chunk["arm"], card["node_class"], chunk["copies"], catalog):
                continue
            duration = chunk_seconds(chunk["arm"], card["node_class"], chunk["copies"], survey,
                                     measured)
            if duration is None:
                continue
            finish = load[index] + duration
            if best is None or finish < best[0]:
                best = (finish, index, duration)
        if best is None:
            return None, None
        finish, index, duration = best
        load[index] = finish
        assignment.append(dict(chunk, card_index=index, node=cards[index]["node"],
                               node_class=cards[index]["node_class"],
                               reservation=cards[index]["reservation"],
                               starts_after_seconds=cards[index]["free_in_seconds"],
                               seconds=duration, finish_seconds=finish,
                               rate_source=rate_source(chunk["arm"], cards[index]["node_class"],
                                                       chunk["copies"], survey, measured)))
    return assignment, max((row["finish_seconds"] for row in assignment), default=0.0)


def chunks_of(unit: dict, k: int) -> list:
    """A unit cut into k chunks, each holding a disjoint slice of every cell's copy indices.

    before: unit 1 (rnd_next_state, 33 cells x 128 copies), k = 8
    after:  eight chunks of 16 copies per cell (528 copies each), the first holding copy indices
            0..15 of every cell and the last 112..127, so their union is exactly the 128 copies per
            cell of the whole unit and no chunk repeats a seed
    """
    if COPIES_PER_CELL % k:
        return []
    per_cell = COPIES_PER_CELL // k
    return [{"order": unit["order"], "arm": unit["arm"], "chunk": j, "chunks": k,
             "copies_per_cell": per_cell, "copies": per_cell * CELLS,
             "copy_index_first": j * per_cell,
             "copy_index_last": (j + 1) * per_cell - 1} for j in range(k)]


def priceable(unit: dict, k: int, survey: dict, cards: list, catalog: dict,
              measured: dict = None) -> bool:
    """Whether a k-way split of this unit can be costed at all from measured data.

    A split is priceable when its chunk copy count has a price on at least one card's class that
    can also hold it. Nothing is extrapolated: a copy count no probe covers makes that split
    unavailable rather than guessed.
    """
    if COPIES_PER_CELL % k:
        return False
    size = (COPIES_PER_CELL // k) * CELLS
    return any(per_copy_rate(unit["arm"], card["node_class"], size, survey, measured) is not None
               and fits(unit["arm"], card["node_class"], size, catalog) for card in cards)


def plan(cards: list, survey: dict, catalog: dict, measured: dict = None) -> dict:
    """Every priceable combination of per-unit splits, costed; the makespan-minimising one wins."""
    per_unit_options = []
    for unit in UNITS:
        options = [k for k in SPLITS if priceable(unit, k, survey, cards, catalog, measured)]
        if not options:
            raise SystemExit(f"no priceable split for unit {unit['order']} ({unit['arm']}): no "
                             f"card's class has a measured cell it can hold")
        per_unit_options.append(options)

    considered = []
    for combination in itertools.product(*per_unit_options):
        chunks = [chunk for unit, k in zip(UNITS, combination) for chunk in chunks_of(unit, k)]
        assignment, makespan = schedule(chunks, cards, survey, catalog, measured)
        if assignment is None:
            continue
        considered.append({"splits": list(combination), "chunk_count": len(chunks),
                           "makespan_seconds": makespan, "assignment": assignment,
                           "cards_used": len({row["card_index"] for row in assignment})})
    if not considered:
        raise SystemExit("no combination of splits could be scheduled on the free cards")

    # the makespan decides; among plans that finish within PROPORTIONALITY_SECONDS of the best, the
    # one with fewer chunks wins, because the extra chunks buy no finish time and cost a build, a
    # queue entry, a shard and a line of the monitoring table each
    best_makespan = min(row["makespan_seconds"] for row in considered)
    chosen = min((row for row in considered
                  if row["makespan_seconds"] <= best_makespan + PROPORTIONALITY_SECONDS),
                 key=lambda row: (row["chunk_count"], row["makespan_seconds"]))
    considered.sort(key=lambda row: row["makespan_seconds"])
    return {"chosen": chosen, "considered": considered}


def hours(seconds: float) -> str:
    """Seconds as hours, the unit a run is planned in."""
    return f"{seconds / 3600:.2f} h"


def print_plan(result: dict, cards: list, measured: dict) -> None:
    """Print the free-card summary, the chosen assignment and the alternatives, as tables."""
    print(f"cards the planner may use: {len(cards)}")
    by_class = {}
    for card in cards:
        by_class[card["node_class"]] = by_class.get(card["node_class"], 0) + 1
    for name, count in sorted(by_class.items(), key=lambda item: -item[1]):
        print(f"  {name:18} {count}")
    print(f"\nthis run's own probe cells: {len(measured)}")

    chosen = result["chosen"]
    print(f"\nCHOSEN PLAN — makespan {hours(chosen['makespan_seconds'])}, "
          f"{chosen['chunk_count']} chunks on {chosen['cards_used']} cards")
    print(f"{'unit':>5} {'arm':28} {'chunk':>7} {'copies':>7} {'per cell':>9} "
          f"{'copy idx':>10} {'node':>11} {'class':>14} {'est':>8} {'finish':>8}  rate from")
    for row in sorted(chosen["assignment"], key=lambda r: (r["order"], r["chunk"])):
        print(f"{row['order']:>5} {row['arm']:28} {row['chunk'] + 1}/{row['chunks']:<5} "
              f"{row['copies']:>7} {row['copies_per_cell']:>9} "
              f"{row['copy_index_first']:>4}-{row['copy_index_last']:<5} "
              f"{row['node']:>11} {row['node_class']:>14} "
              f"{hours(row['seconds']):>8} {hours(row['finish_seconds']):>8}  "
              f"{row['rate_source']}")

    print("\nalternatives considered, by makespan:")
    print(f"{'splits per unit':>18} {'chunks':>7} {'cards':>6} {'makespan':>10}   verdict")
    for row in result["considered"]:
        verdict = ("chosen" if row is chosen else
                   "rejected: slower" if row["makespan_seconds"] > chosen["makespan_seconds"]
                   else "rejected: same finish within the proportionality margin, more chunks")
        print(f"{str(row['splits']):>18} {row['chunk_count']:>7} {row['cards_used']:>6} "
              f"{hours(row['makespan_seconds']):>10}   {verdict}")


def main() -> None:
    """Compute the plan, print it as a table, and write code/submission_plan.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability-json", default="",
                        help="a saved availability.py --json file, for a replay; live by default")
    parser.add_argument("--output", default="submission_plan.json",
                        help="the file under code/ the plan is written to")
    parser.add_argument("--extra-cards", default="",
                        help="a JSON list of cards that are NOT free now, each with the seconds "
                             "until the scheduler projects it free, appended to the card list. "
                             "Used to price waiting for a fast card against starting on a slow "
                             "one; the submitted plan is computed without it.")
    args = parser.parse_args()

    survey = load_survey()
    catalog = load_class_catalog()
    measured = load_measured_cells()
    saved = json.load(open(args.availability_json)) if args.availability_json else None
    cards = capped_cards(usable_cards(live_cards(saved), survey, measured, catalog),
                         survey, measured)
    if args.extra_cards:
        cards += json.load(open(args.extra_cards))
    result = plan(cards, survey, catalog, measured)
    print_plan(result, cards, measured)

    output = {"generated_at_pacific": subprocess.run(
                  ["date", "+%Y-%m-%d %H:%M PT"], capture_output=True, text=True,
                  env=dict(os.environ, TZ="America/Los_Angeles")).stdout.strip(),
              "iterations": ITERATIONS, "steps_per_copy": STEPS_PER_COPY,
              "cells": CELLS, "copies_per_cell": COPIES_PER_CELL,
              "copies_per_unit": COPIES_PER_UNIT, "window_iterations": WINDOW_ITERATIONS,
              "cards": cards,
              "own_probe_cells": [dict(cell) for cell in measured.values()],
              "chosen": result["chosen"], "considered": result["considered"]}
    path = RUN_DIR / "code" / args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
