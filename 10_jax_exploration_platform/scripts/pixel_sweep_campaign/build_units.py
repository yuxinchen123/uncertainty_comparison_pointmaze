"""Turn a study description (JSON) into queue units of the pixel Montezuma sweep, and pack or continue them.

    python build_units.py --run-dir RUN --study description.json [--dry-run] [--list]
    python build_units.py --run-dir RUN --pack-pending --packs N [--match TEXT] [--from-end] [--dry-run]
    python build_units.py --run-dir RUN --unpack-pending [--match TEXT] [--dry-run]
    python build_units.py --run-dir RUN --study description.json --continue-units [--dry-run]

A run is one policy: one method, one value of each swept setting, one seed. Every run is built as a
single unit (one train_final.py process with one copy), named in words:

    grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-13
    grid_rnd_intrinsic-weight-1_seed-13

Order of claims: seed by seed. Every configuration at the first seed stands before any
configuration at the second seed; inside a seed the methods stand in the description's order, and a
method's configurations with the setting it packs over innermost (the four step sizes of one
sharing group and one weight are neighbours).

--pack-pending turns four singles that never started into one four-copy unit, for a card of the
"four copies" profile: the four values of the method's `pack_over` setting at one seed (the four
step sizes of one sharing group and one weight; four of RND's weights), or four seeds of one
configuration (`"pack_over": "seed"`). The singles move to RUN/queue/packed/, where no worker claims
them. Fewer than four of a kind stay singles (RND's six weights at one seed give one pack and two
singles). --packs says how many packs to make: make only as many as there are cards for them.
--unpack-pending takes packs that never started apart again.

--continue-units raises the target of units that exist (stage 3 continues the run folders of stage
2): a description with the same phase, methods and seeds and a higher `target_updates` moves each
finished unit back to pending with the new target; its run folder, its profile and its history stay.

Standard library only. The description's keys are in README.md of this folder.
"""
import argparse
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from campaign_imports import PLATFORM, common, unit_queue
import pixel_queue

WORDS = re.compile(r"[a-z0-9][a-z0-9.-]*")
PACK_WORD = "four-copies"      # stands in the name of every pack, and in no other name
PACK_COPIES = 4
# the swept settings: the word of a description and of a unit name -> the learner's own name
LEARNER_KEYS = {"step_size": "fast_last_layer_step", "sharing_group": "fast_last_layer_sharing_group",
                "intrinsic_weight": "intrinsic_coefficient"}
NAME_ORDER = ("step_size", "sharing_group", "intrinsic_weight")       # the order of settings in a name
GRID_ORDER = ("sharing_group", "intrinsic_weight", "step_size")       # slowest first when a grid is expanded
PLURALS = {"step_size": "step-sizes", "intrinsic_weight": "intrinsic-weights"}
# the settings a pack may differ in between its copies: per-copy values of the learner, or the seed
PACK_OVER = ("step_size", "intrinsic_weight", "seed")
DEFAULT_TOTAL_TIMESTEPS = 2_000_000_000


def number_words(value):
    """Write a number the way a unit name holds it."""
    # before: 0.03 ; after: "0.03"      before: 1.0 ; after: "1"      before: 128 ; after: "128"
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def configuration_words(method, settings):
    """Name a configuration in words: the method, then each of its settings."""
    # before: "fast-last-layer", {"sharing_group": 16, "intrinsic_weight": 1.0, "step_size": 0.03}
    # after:  "fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1"
    parts = [f"{word.replace('_', '-')}-{number_words(settings[word])}" for word in NAME_ORDER if word in settings]
    return "_".join([method, *parts])


def expand_configurations(entry):
    """List the configurations of one method entry: a grid's cross product, or the listed ones.

    before: {"method": "rnd", "grid": {"intrinsic_weight": [0.25, 0.5]}}
    after:  [{"intrinsic_weight": 0.25}, {"intrinsic_weight": 0.5}]
    before: {"method": "fast-last-layer", "pack_over": "step_size",
             "grid": {"step_size": [0.01, 0.03], "sharing_group": [128, 16], "intrinsic_weight": [1]}}
    after:  [{"sharing_group": 128, "intrinsic_weight": 1.0, "step_size": 0.01},
             {"sharing_group": 128, "intrinsic_weight": 1.0, "step_size": 0.03},
             {"sharing_group": 16, "intrinsic_weight": 1.0, "step_size": 0.01},
             {"sharing_group": 16, "intrinsic_weight": 1.0, "step_size": 0.03}]
    """
    if ("grid" in entry) == ("configurations" in entry):
        raise ValueError(f"method {entry.get('method')!r} needs exactly one of grid and configurations")
    if "grid" in entry:
        unknown = set(entry["grid"]) - set(LEARNER_KEYS)
        if unknown:
            raise ValueError(f"a grid may name {sorted(LEARNER_KEYS)}; got {sorted(unknown)}")
        # the setting the method packs over runs fastest, so its values are neighbours in the order
        over = entry.get("pack_over")
        words = [word for word in GRID_ORDER if word in entry["grid"] and word != over]
        words += [over] if over in entry["grid"] else []
        rows = [dict(zip(words, values)) for values in product(*(entry["grid"][word] for word in words))]
    else:
        rows = [dict(row) for row in entry["configurations"]]
    configurations = []
    for row in rows:
        unknown = set(row) - set(LEARNER_KEYS)
        if unknown or "intrinsic_weight" not in row:
            raise ValueError(f"a configuration names intrinsic_weight and may name step_size and sharing_group; got {row}")
        # the sharing group is a whole number of environments; the other two are plain numbers
        settings = {word: int(value) if word == "sharing_group" else float(value) for word, value in row.items()}
        fast = settings.get("step_size", 0.0) > 0
        if fast != ("sharing_group" in settings) or ("step_size" in settings and not fast):
            raise ValueError(f"a positive step_size needs a sharing_group, and RND names neither; got {row}")
        configurations.append(settings)
    names = [json.dumps(settings, sort_keys=True) for settings in configurations]
    if len(set(names)) != len(names):
        raise ValueError(f"method {entry['method']!r} lists a configuration twice")
    return configurations


def study_seeds(seeds):
    """The seeds of a description, in order."""
    # before: {"first": 13, "count": 8} ; after: [13, 14, 15, 16, 17, 18, 19, 20]
    # before: {"list": [1, 2, 5]} ; after: [1, 2, 5]
    values = list(seeds["list"]) if "list" in seeds else list(range(seeds["first"], seeds["first"] + seeds["count"]))
    if not values or len(set(values)) != len(values):
        raise ValueError(f"the seeds must be a non-empty list without repeats; got {seeds}")
    return values


def single_unit(description, entry, settings, seed, built_against):
    """Build the specification of one single unit: one run of one configuration at one seed."""
    method, phase = entry["method"], description["phase"]
    configuration = configuration_words(method, settings)
    # the learner settings of this configuration; the card's settings are added when the unit first starts
    learner = {"total_timesteps": int(description.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS))}
    learner.update({LEARNER_KEYS[word]: settings[word] for word in NAME_ORDER if word in settings})
    environment = {}
    # the trainer refuses the learner's event flag without the environment's
    if description.get("record_game_events"):
        learner["record_game_events"], environment["report_game_events"] = True, True
    if description.get("record_bonus_profile"):
        learner["record_bonus_profile"] = True
    learner.update(description.get("learner_changes", {}))
    environment.update(description.get("environment_changes", {}))
    target = int(description["target_updates"])
    return {
        "unit_id": f"{phase}_{configuration}_seed-{seed}", "study": description["study"], "phase": phase,
        "method": method, "configuration": configuration, "pack_over": entry.get("pack_over"), "copies": 1,
        "runs": [{"method": method, "configuration": configuration, **settings, "seed": seed}],
        "copy_seeds": [seed], "members": [],
        "target_updates": target, "steps_per_update": pixel_queue.STEPS_PER_UPDATE,
        "steps_per_run": target * pixel_queue.STEPS_PER_UPDATE,
        "learner": learner, "per_copy_settings": {}, "environment": environment,
        "observation_warmup_iterations": int(description.get("observation_warmup_iterations", 50)),
        "checkpoint_seconds": int(description.get("checkpoint_seconds", 1200)),
        "built_against": built_against,
    }


def study_units(description, platform=PLATFORM):
    """Build the single units of a description, in the order workers claim them: seed by seed.

    before: methods rnd (weights 0.25, 0.5) and fast-last-layer (one configuration), seeds 13 and 14
    after:  [rnd 0.25 seed 13, rnd 0.5 seed 13, fast-last-layer seed 13,
             rnd 0.25 seed 14, rnd 0.5 seed 14, fast-last-layer seed 14]
    """
    for key in ("study", "phase"):
        if not WORDS.fullmatch(description[key].replace("_", "-")):
            raise ValueError(f"{key} {description[key]!r} must be lower-case words joined by hyphens")
    if "_" in description["phase"]:
        raise ValueError(f"the phase {description['phase']!r} is one part of every unit name and holds no underscore")
    marker = Path(platform) / "SOURCE_COMMIT"
    built_against = {"platform": str(platform),
                     "source_commit": marker.read_text().strip() if marker.exists() else "working tree"}
    methods = []
    for entry in description["methods"]:
        if not WORDS.fullmatch(entry["method"]) or PACK_WORD in entry["method"]:
            raise ValueError(f"method {entry['method']!r} must be lower-case words joined by hyphens, "
                             f"without {PACK_WORD!r}")
        if entry.get("pack_over") not in (None, *PACK_OVER):
            raise ValueError(f"pack_over must be one of {list(PACK_OVER)}; got {entry.get('pack_over')!r}")
        methods.append((entry, expand_configurations(entry)))
    # the seed is the outermost loop, so early results cover every configuration
    specifications = [single_unit(description, entry, settings, seed, built_against)
                      for seed in study_seeds(description["seeds"])
                      for entry, configurations in methods for settings in configurations]
    names = [specification["unit_id"] for specification in specifications]
    if len(set(names)) != len(names):
        raise ValueError("two units of this description have the same name; a method is listed twice")
    # a unit's files are named by its name; 255 bytes is the file system's limit for one name
    if max(len(name) for name in names) > 200:
        raise ValueError(f"a unit name is longer than 200 characters: {max(names, key=len)}")
    return specifications


def queue_record(specification, order, copy_number=0):
    """The small record of a unit that moves between the queue folders."""
    # minimum_device_memory_mb starts at 0: the profile's own memory range decides which card may take a
    # unit, and the field is raised only after the unit ran out of memory on a card
    return {"name": specification["unit_id"], "order": order, "minimum_device_memory_mb": 0,
            "minimum_copy_number": copy_number, "specification": f"specifications/{specification['unit_id']}.json",
            "phase": specification["phase"], "method": specification["method"], "copies": specification["copies"],
            "copy_seeds": specification["copy_seeds"], "target_updates": specification["target_updates"],
            "execution_profile": None, "attempts": 0, "failures": 0, "recoveries": 0, "history": []}


def next_order(run_dir):
    """The first free place in the queue's order: one after the largest order any unit holds."""
    # copied from scripts/maze_campaign/build_units.py, next_order (that file needs Python 3.11 to import)
    queue = Path(run_dir) / "queue"
    folders = [folder for folder in queue.iterdir() if folder.is_dir()] if queue.exists() else []
    orders = [common.read_json(path)["order"] for folder in folders for path in folder.glob("*.json")]
    return max(orders, default=0) + 1


def never_started_singles(run_dir):
    """Read the pending singles no worker ever started, with their specifications, in claim order."""
    singles = []
    for path in sorted((Path(run_dir) / "queue/pending").glob("*.json")):
        record = common.read_json(path)
        started = (record["attempts"] > 0 or record.get("execution_profile") is not None
                   or pixel_queue.run_folder(run_dir, record["name"]).exists())
        if record["copies"] == 1 and not started:
            singles.append((record, common.read_json(Path(run_dir) / record["specification"])))
    return sorted(singles, key=lambda single: (single[0]["order"], single[0]["name"]))


def pack_key(specification):
    """Name what four singles must share to run as one pack: everything but the setting they are packed over."""
    # before: a single of fast-last-layer, step size 0.03, group 16, weight 1, seed 13, packed over step_size
    # after:  the same text for its three neighbours with step sizes 0.01, 0.1 and 0.3, and another text
    #         for every other group, weight, seed, target or learner setting
    over = specification["pack_over"]
    run = {key: value for key, value in specification["runs"][0].items() if key not in (over, "configuration")}
    learner = {key: value for key, value in specification["learner"].items() if key != LEARNER_KEYS.get(over)}
    return json.dumps([specification["phase"], over, specification["target_updates"], run, learner,
                       specification["environment"], specification["observation_warmup_iterations"],
                       specification["checkpoint_seconds"]], sort_keys=True)


def pack_unit(members):
    """Build the specification of one pack from its four singles' specifications (in the order of the packed values)."""
    first = members[0]
    over, runs = first["pack_over"], [member["runs"][0] for member in members]
    # the name: the packed setting in the plural with its four values, the other settings as in a single
    # after: "grid_fast-last-layer_four-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-16_intrinsic-weight-1_seed-13"
    # after: "grid_rnd_four-copies_intrinsic-weights-0.25-0.5-1-2_seed-13"
    # after: "long_rnd_four-copies_intrinsic-weight-2_seeds-1-2-3-4"
    parts = []
    for word in NAME_ORDER:
        if word == over:
            parts.append(f"{PLURALS[word]}-" + "-".join(number_words(run[word]) for run in runs))
        elif word in runs[0]:
            parts.append(f"{word.replace('_', '-')}-{number_words(runs[0][word])}")
    seeds = [run["seed"] for run in runs]
    parts.append("seeds-" + "-".join(map(str, seeds)) if over == "seed" else f"seed-{seeds[0]}")
    name = "_".join([first["phase"], first["method"], PACK_WORD, *parts])
    # the shared learner settings are the first copy's; a learner setting that differs goes in per copy
    per_copy = {} if over == "seed" else {LEARNER_KEYS[over]: [run[over] for run in runs]}
    return dict(first, unit_id=name, copies=len(members), runs=runs, copy_seeds=seeds,
                members=[member["unit_id"] for member in members], per_copy_settings=per_copy,
                configuration=None)


def plan_packs(singles, packs, match=None, from_end=False):
    """Choose the packs to make from the never-started singles; returns lists of four (record, specification).

    before: the six RND singles of seed 13 (weights 0.25 to 8, packed over intrinsic_weight), packs 5
    after:  one pack, of the weights 0.25, 0.5, 1 and 2; the weights 4 and 8 stay singles
    """
    groups = {}
    for record, specification in singles:
        if specification["pack_over"] is not None:
            groups.setdefault(pack_key(specification), []).append((record, specification))
    candidates = []
    for members in groups.values():
        # in the order of the packed values, four at a time; fewer than four stay singles
        over = members[0][1]["pack_over"]
        members.sort(key=lambda member: member[1]["runs"][0][over])
        candidates += [members[index:index + PACK_COPIES]
                       for index in range(0, len(members) - PACK_COPIES + 1, PACK_COPIES)]
    # a pack stands where its first single stood in the claim order
    candidates.sort(key=lambda members: min(record["order"] for record, _ in members), reverse=from_end)
    if match is not None:
        candidates = [members for members in candidates if any(match in record["name"] for record, _ in members)]
    return candidates[:max(0, packs)]


def pack_pending(run_dir, packs, match=None, from_end=False, dry_run=False):
    """Turn never-started singles into four-copy units, under the queue lock; returns the new packs' names."""
    run_dir, made = Path(run_dir), []
    with common.queue_lock(run_dir):
        for members in plan_packs(never_started_singles(run_dir), packs, match, from_end):
            specification = pack_unit([member for _, member in members])
            name = specification["unit_id"]
            print(f"{name}\n    from: " + "\n          ".join(record["name"] for record, _ in members))
            made.append(name)
            if dry_run:
                continue
            common.write_json(run_dir / "specifications" / f"{name}.json", specification)
            # the singles leave pending first, so no worker can start one while its pack exists
            for record, _ in members:
                unit_queue.move(run_dir, dict(record, packed_into=name, packed_at=common.now()), "pending", "packed")
            record = queue_record(specification, min(record["order"] for record, _ in members),
                                  max(record["minimum_copy_number"] for record, _ in members))
            common.write_json(unit_queue.unit_path(run_dir, "pending", name), dict(record, created_at=common.now()))
    print(f"{'dry run: ' if dry_run else ''}{len(made)} pack(s) of {PACK_COPIES} singles"
          + ("" if dry_run else f" now pending in {run_dir / 'queue/pending'}"))
    return made


def unpack_pending(run_dir, match=None, dry_run=False):
    """Take pending packs that never started apart again, under the queue lock; returns their names."""
    run_dir, undone = Path(run_dir), []
    with common.queue_lock(run_dir):
        for path in sorted((run_dir / "queue/pending").glob("*.json")):
            record = common.read_json(path)
            started = record["attempts"] > 0 or pixel_queue.run_folder(run_dir, record["name"]).exists()
            if record["copies"] == 1 or started or (match is not None and match not in record["name"]):
                continue
            specification = common.read_json(run_dir / record["specification"])
            print(f"{record['name']}\n    back to: " + "\n             ".join(specification["members"]))
            undone.append(record["name"])
            if dry_run:
                continue
            # the pack's record goes first, so no worker can start it while its singles come back
            path.unlink()
            (run_dir / record["specification"]).unlink()
            for member in specification["members"]:
                single = common.read_json(unit_queue.unit_path(run_dir, "packed", member))
                single = {key: value for key, value in single.items() if key not in ("packed_into", "packed_at")}
                unit_queue.move(run_dir, single, "packed", "pending")
            common.append_json(run_dir / "infrastructure_events.jsonl", {
                "at": common.now(), "event": "pack_taken_apart", "unit": record["name"],
                "members": specification["members"]})
    print(f"{'dry run: ' if dry_run else ''}{len(undone)} pack(s) taken apart")
    return undone


def find_unit(run_dir, name):
    """Find the queue folder and the record of a unit; a single inside a pack gives the pack's."""
    # before: a single in queue/packed with {"packed_into": "<pack>"}, the pack in queue/done
    # after:  ("done", the pack's record)
    for folder in sorted(folder for folder in (Path(run_dir) / "queue").iterdir() if folder.is_dir()):
        path = folder / f"{name}.json"
        if path.exists():
            record = common.read_json(path)
            return find_unit(run_dir, record["packed_into"]) if folder.name == "packed" else (folder.name, record)
    return None, None


def continue_units(run_dir, specifications, dry_run=False):
    """Raise the target of the units a description names; finished ones go back to pending. Returns counts."""
    run_dir, counts, seen = Path(run_dir), {"raised": 0, "left": 0, "missing": 0}, set()
    with common.queue_lock(run_dir):
        for specification in specifications:
            folder, record = find_unit(run_dir, specification["unit_id"])
            if record is None:
                print(f"not in the queue (build it first): {specification['unit_id']}")
                counts["missing"] += 1
                continue
            # the four singles of a pack name the same unit: it is handled once
            if record["name"] in seen:
                continue
            seen.add(record["name"])
            target = specification["target_updates"]
            if record["target_updates"] >= target or folder not in ("pending", "done"):
                reason = (f"its target is {record['target_updates']} already" if record["target_updates"] >= target
                          else f"it is in queue/{folder}; raise it when it is pending or done")
                print(f"left as it is ({reason}): {record['name']}")
                counts["left"] += 1
                continue
            print(f"{record['name']}\n    target {record['target_updates']} -> {target} updates, from queue/{folder}")
            counts["raised"] += 1
            if dry_run:
                continue
            # before: {"target_updates": 61035, "result": {...}, "finished_at": "..."} in queue/done
            # after:  {"target_updates": 122070, "earlier_targets": [{"target_updates": 61035, "result": {...},
            #          "finished_at": "...", "raised_at": "..."}]} in queue/pending
            earlier = {"target_updates": record["target_updates"], "raised_at": common.now()}
            earlier.update({key: record.pop(key) for key in ("result", "finished_at") if key in record})
            record.setdefault("earlier_targets", []).append(earlier)
            record["target_updates"] = target
            if folder == "done":
                unit_queue.move(run_dir, record, "done", "pending")
            else:
                common.write_json(unit_queue.unit_path(run_dir, "pending", record["name"]), record)
    print(f"{'dry run: ' if dry_run else ''}{counts['raised']} unit(s) with a higher target, "
          f"{counts['left']} left as they are, {counts['missing']} not in the queue")
    return counts


def print_units(specifications, list_all):
    """Print what a description builds: counts by method and seed, and the first names in claim order."""
    methods, seeds = {}, {}
    for specification in specifications:
        methods[specification["method"]] = methods.get(specification["method"], 0) + 1
        seeds[specification["copy_seeds"][0]] = seeds.get(specification["copy_seeds"][0], 0) + 1
    first = specifications[0]
    print(f"{len(specifications)} single units, each one run of {first['target_updates']:,} updates "
          f"({first['steps_per_run']:,} steps)")
    print("    by method: " + ", ".join(f"{method} {count}" for method, count in methods.items()))
    print("    by seed, in claim order: " + ", ".join(f"seed {seed}: {count}" for seed, count in seeds.items()))
    shown = specifications if list_all else specifications[:6] + specifications[-2:]
    print("    names in claim order" + ("" if list_all else " (the first six and the last two; --list prints all)") + ":")
    for specification in shown:
        print(f"      {specification['unit_id']}")


def main(argv=None):
    """Build a description's units, or pack, unpack or continue units that exist."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--study", type=Path, help="the study description (JSON)")
    parser.add_argument("--list", action="store_true", help="print every unit name, not only the first and last")
    parser.add_argument("--pack-pending", action="store_true",
                        help="turn never-started singles into four-copy units (needs --packs)")
    parser.add_argument("--packs", type=int, help="with --pack-pending: make at most this many packs")
    parser.add_argument("--match", help="with --pack-pending or --unpack-pending: only units whose name holds this text")
    parser.add_argument("--from-end", action="store_true",
                        help="with --pack-pending: take the singles from the end of the claim order")
    parser.add_argument("--unpack-pending", action="store_true", help="take never-started packs apart again")
    parser.add_argument("--continue-units", action="store_true",
                        help="with --study: raise the target of the units it names; finished ones go back to pending")
    parser.add_argument("--dry-run", action="store_true", help="print what would be done and write nothing")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.pack_pending or args.unpack_pending:
        if args.study or args.pack_pending == args.unpack_pending or (args.pack_pending and args.packs is None):
            parser.error("give one of --pack-pending (with --packs N) and --unpack-pending, without --study")
        if args.pack_pending:
            pack_pending(run_dir, args.packs, args.match, args.from_end, args.dry_run)
        else:
            unpack_pending(run_dir, args.match, args.dry_run)
        return
    if not args.study:
        parser.error("give --study, --pack-pending or --unpack-pending")

    # units need the code copy the run folder is frozen at: an older copy may not know their options
    source = common.read_json(run_dir / "source.json") if (run_dir / "source.json").exists() else None
    platform = Path(source["snapshot"]) / "10_jax_exploration_platform" if source else PLATFORM
    copy_number = unit_queue.copy_number(platform)
    print(f"units are built against {platform} (code copy number {copy_number})")
    specifications = study_units(json.loads(args.study.read_text()), platform)
    if args.continue_units:
        continue_units(run_dir, specifications, args.dry_run)
        return
    print_units(specifications, args.list)
    if args.dry_run:
        print(f"dry run: {len(specifications)} units, nothing written")
        return
    # later builds queue behind earlier ones; a unit that already exists keeps its place and its state
    first = next_order(run_dir)
    added = [pixel_queue.enqueue(run_dir, specification, queue_record(specification, first + index, copy_number))
             for index, specification in enumerate(specifications)]
    stamp = datetime.now().astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
    copy = run_dir / "study_descriptions" / f"{stamp}_{args.study.name}"
    copy.parent.mkdir(exist_ok=True)
    copy.write_text(args.study.read_text())
    print(f"{added.count('added')} units added to {run_dir / 'queue/pending'}, {added.count('')} left as they are; "
          f"description copied to {copy}")


if __name__ == "__main__":
    main()
