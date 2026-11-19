"""The parts of the unit queue that differ for the pixel sweep: execution profiles, the claim, and progress.

A unit of the pixel sweep is one train_final.py process on one card: one copy (a single run), or
four or eight copies (a pack of four or eight runs). Its queue record moves between the folders of
RUN/queue exactly as a maze unit's does, and the maze queue's functions return it, finish it and
recover it (scripts/maze_campaign/unit_queue.py, imported). This file holds what the maze queue
cannot do:

  - Execution profiles (card_profiles.json): "one copy" for every card under 60 GB, "four copies"
    from 60 GB to under 140 GB, "eight copies" from 140 GB (the H200). A worker takes the profile of
    its card and starts only units with that profile's number of copies; it also continues units
    that started under a profile its own names in `continues` (an eight-copy worker continues
    four-copy packs). A unit's profile is fixed at its first start: the claim writes the profile's
    name and its learner and environment settings into the unit's record, and every later start
    uses those recorded settings. A unit may continue on another card type of the same profile.
  - Progress is read from the trainer's own files in RUN/data/<unit>/run/ (progress.json), not from
    a shard.
  - A unit is added without a configuration file (the trainer takes JSON on its command line).

Standard library only.
"""
import json
import os
from pathlib import Path

from campaign_imports import common, unit_queue

HERE = Path(__file__).resolve().parent
PROFILES_FILE = HERE / "card_profiles.json"
STEPS_PER_UPDATE = 128 * 128      # 128 environments per policy x 128 rollout steps
# the files of a unit's run folder that go with it to another server and back (train_final.py writes them)
RUN_FILES = ("run.json", "progress.json", "events.jsonl", "metrics.jsonl", "game_first_times.json",
             "game_windows.jsonl", "checkpoint.pkl")
# the small ones, enough for the live report while a unit trains elsewhere
RECORD_FILES = ("run.json", "progress.json", "events.jsonl", "game_first_times.json", "game_windows.jsonl")


def read_profiles(path=PROFILES_FILE):
    """Read the execution profiles and check that each one is complete; returns {name: profile}."""
    profiles = common.read_json(path)["profiles"]
    for name, profile in profiles.items():
        missing = {"copies", "minimum_card_memory_mb", "card_memory_below_mb", "continues", "learner", "environment",
                   "archived_runs"} - set(profile)
        if missing:
            raise ValueError(f"profile {name!r} of {path} lacks {sorted(missing)}")
        unknown = set(profile["continues"]) - set(profiles)
        if unknown:
            raise ValueError(f"profile {name!r} of {path} continues profiles that do not exist: {sorted(unknown)}")
    return profiles


def holds_card(profile, card_memory_mb):
    """Say whether a card of this memory belongs to a profile."""
    # before: the "one copy" profile (10,000 MB up to 60,000 MB) and a card of 20,470 MB ; after: True
    # before: the same profile and a card of 95,830 MB ; after: False
    below = profile["card_memory_below_mb"]
    return profile["minimum_card_memory_mb"] <= card_memory_mb and (below is None or card_memory_mb < below)


def profile_for_card(profiles, card_memory_mb):
    """Name the profile a card runs under; None when its memory is below every profile's range."""
    # before: 20470 (RTX A4500) ; after: "one copy"       before: 95830 (H100 NVL) ; after: "four copies"
    # before: 143771 (H200) ; after: "eight copies"        before: 8192 (GTX 1080) ; after: None
    return next((name for name, profile in profiles.items() if holds_card(profile, card_memory_mb)), None)


def run_folder(run_dir, name):
    """Name the folder train_final.py writes for one unit."""
    return Path(run_dir) / "data" / name / "run"


def records_folder(run_dir, state, name):
    """Name the folder a unit's records are read from: its run folder, or the copy of a remote server's records."""
    # before: state "running" ; after: RUN/data/<unit>/run
    # before: state "remote_gautschi" ; after: RUN/slurm/remote_live/gautschi/<unit>/run (remote_fleet.py copies
    #         the small record files of the units a server holds there at every tick)
    if state.startswith("remote_"):
        return Path(run_dir) / "slurm/remote_live" / state[len("remote_"):] / name / "run"
    return run_folder(run_dir, name)


def read_progress(folder):
    """Read a run folder's progress.json; the trainer rewrites it about once a minute.

    before (the file): {"update": 4120, "target_updates": 12207, "seconds_per_update": 1.41, "unix": 1795000000.0}
    after:  {"exists": True, "update": 4120, "target_updates": 12207, "completed": False,
             "seconds_per_update": 1.41, "unix": 1795000000.0}
    A folder without the file gives {"exists": False, "update": 0, ...}.
    """
    path = Path(folder) / "progress.json"
    if not path.exists():
        return {"exists": False, "update": 0, "target_updates": None, "completed": False,
                "seconds_per_update": None, "unix": None}
    record = json.loads(path.read_text())
    return {"exists": True, "update": int(record.get("update", 0)), "target_updates": record.get("target_updates"),
            "completed": record.get("completed") is True, "seconds_per_update": record.get("seconds_per_update"),
            "unix": record.get("unix")}


def is_complete(progress, target_updates):
    """Say whether a progress record means the unit reached its target."""
    # the update is compared too: a unit continued to a higher target still holds the completed
    # record of its earlier target until the trainer writes progress again
    # before: {"completed": True, "update": 61035}, target 122070 ; after: False
    return progress["completed"] and progress["update"] >= target_updates


def enqueue(run_dir, specification, record):
    """Add one unit: its specification first, then its pending record; returns "added" or "".

    A unit that is already in any queue folder (a remote server's and the packed singles' included)
    is left as it is and "" is returned.
    """
    # copied from scripts/maze_campaign/unit_queue.py, enqueue: without the configuration file a maze
    # unit has, and without replacing a pending unit
    run_dir = Path(run_dir)
    for state in unit_queue.STATES:
        (run_dir / "queue" / state).mkdir(parents=True, exist_ok=True)
    with common.queue_lock(run_dir):
        folders = [folder for folder in (run_dir / "queue").iterdir() if folder.is_dir()]
        if any((folder / f"{record['name']}.json").exists() for folder in folders):
            return ""
        # the pending record goes last: a worker may claim the unit the moment that record exists
        common.write_json(run_dir / record["specification"], specification)
        common.write_json(unit_queue.unit_path(run_dir, "pending", record["name"]),
                          dict(record, created_at=common.now()))
    return "added"


def of_profile(unit, profile_name, profile):
    """Say whether a unit is one a worker of this profile starts or continues, whatever its card and code copy."""
    # before: a single that never started ({"copies": 1, "execution_profile": None}), profile "one copy"; after: True
    # before: the same single, profile "four copies" ; after: False (the profile runs four copies)
    # before: a four-copy pack that never started, profile "eight copies" ; after: False (it starts eight-copy units)
    # before: a four-copy pack that started ({"execution_profile": "four copies"}), profile "eight copies" ; after: True
    fixed = unit.get("execution_profile")
    return (unit["copies"] == profile["copies"] and fixed in (None, profile_name)) or fixed in profile["continues"]


def accepts(unit, profile_name, profile, card_memory_mb, own_copy):
    """Say whether a worker of this profile, card and code copy may run a unit."""
    # before: a pack that started under "four copies" on an H100, a worker on an A100 with 80 GB ; after: True
    # before: an eight-copy pack built against code copy 3, a worker of copy 2 ; after: False
    return (of_profile(unit, profile_name, profile)
            and unit["minimum_device_memory_mb"] <= card_memory_mb
            and unit_queue.required_copy(unit) <= own_copy)


def claim(run_dir, card_memory_mb, claimant, profile_name, profile):
    """Claim the first pending unit this worker's profile, card and code copy may run.

    The rename into running/ is the claim. At a unit's first start the profile's name and its
    learner and environment settings are written into the record; a later start uses those.
    """
    # copied from scripts/maze_campaign/unit_queue.py, claim: the test for the card's memory is
    # replaced by `accepts`, and the profile is fixed at the first start
    run_dir = Path(run_dir)
    own_copy = claimant.get("copy_number", 0)
    with common.queue_lock(run_dir):
        # units are taken in the order the builder gave them (seed by seed), then by name
        units = sorted((common.read_json(path) for path in (run_dir / "queue/pending").glob("*.json")),
                       key=lambda unit: (unit["order"], unit["name"]))
        for unit in units:
            if not accepts(unit, profile_name, profile, card_memory_mb, own_copy):
                continue
            os.rename(unit_queue.unit_path(run_dir, "pending", unit["name"]),
                      unit_queue.unit_path(run_dir, "running", unit["name"]))
            # a unit put back by a remote server's tick still carries the claim of its ended job
            if "job_id" in unit:
                unit_queue.close_attempt(unit, "job_ended_without_a_result")
            unit.update(claimant, started_at=common.now(), attempts=unit["attempts"] + 1)
            if own_copy > unit_queue.required_copy(unit):
                unit["minimum_copy_number"] = own_copy
            # before: {"name": "u", "copies": 1, "execution_profile": None}
            # after:  {"name": "u", "copies": 1, "execution_profile": "one copy",
            #          "execution_profile_settings": {"learner": {...}, "environment": {...}}, ...}
            if unit.get("execution_profile") is None:
                unit.update(execution_profile=profile_name, execution_profile_fixed_at=common.now(),
                            execution_profile_settings={"learner": profile["learner"],
                                                        "environment": profile["environment"]})
            common.write_json(unit_queue.unit_path(run_dir, "running", unit["name"]), unit)
            return unit
    return None


def pending_for_others(run_dir, profile_name, profile, card_memory_mb, own_copy):
    """Count the pending units this worker left, by the reason it may not run them."""
    # after: {"another profile": 12, "more card memory": 0, "a newer code copy": 1}
    left = {"another profile": 0, "more card memory": 0, "a newer code copy": 0}
    for unit in unit_queue.read_units(run_dir, "pending"):
        if not of_profile(unit, profile_name, profile):
            left["another profile"] += 1
        elif unit["minimum_device_memory_mb"] > card_memory_mb:
            left["more card memory"] += 1
        elif unit_queue.required_copy(unit) > own_copy:
            left["a newer code copy"] += 1
    return left


def trainer_settings(unit, specification):
    """Join a unit's profile settings with its own: the learner and environment JSON of its trainer command.

    before: profile learner {"convolution_batching": "unrolled", ...}, the unit's learner
            {"intrinsic_coefficient": 2.0, "fast_last_layer_step": 0.03, ...}, per-copy settings {}
    after:  ({"convolution_batching": "unrolled", ..., "intrinsic_coefficient": 2.0, "fast_last_layer_step": 0.03, ...},
             {"separable_rectangle_drawing": true, ..., "report_game_events": true})
    A pack's values that differ between its copies go in as "per_copy_scalar_settings".
    """
    settings = unit["execution_profile_settings"]
    shared = set(settings["learner"]) & set(specification["learner"])
    if shared:
        raise ValueError(f"unit {unit['name']} and its profile {unit['execution_profile']!r} both set {sorted(shared)}")
    learner = {**settings["learner"], **specification["learner"]}
    if specification["per_copy_settings"]:
        learner["per_copy_scalar_settings"] = specification["per_copy_settings"]
    return learner, {**settings["environment"], **specification["environment"]}
