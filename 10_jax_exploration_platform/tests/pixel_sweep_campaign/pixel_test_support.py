"""Shared pieces of the pixel sweep's tests: paths, descriptions, cards, a stand-in trainer, and synthetic records.

Standard library only: these tests import no numerical library and start no trainer. They may run
on a login node, with the system's Python 3.10 or the project's Python 3.12:

    cd /p/rlprojects/RND/10_jax_exploration_platform
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/pixel_sweep_campaign -p "test_pixel_*.py"
"""
import json
from pathlib import Path
import sys

PLATFORM = Path(__file__).resolve().parents[2]
PIXEL = PLATFORM / "scripts/pixel_sweep_campaign"
# at the END of the path: files named worker.py, submit.py, ... exist in other campaign folders too, and
# these tests load this folder's only through own()
if str(PIXEL) not in sys.path:
    sys.path.append(str(PIXEL))

from campaign_imports import common, load_file, own, unit_queue  # noqa: E402,F401
import pixel_queue  # noqa: E402

build_units = own("build_units")
STAGE_ONE = next((PIXEL / "study_descriptions").glob("stage-1_grid_*.json"))
ARCHIVE_DATA = next(iter(sorted(Path(
    "/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs").glob("2026-11-06-01-14_*/data"))),
    None)
CARD_ONE_COPY = {"gpu": "NVIDIA RTX A4500", "memory_mb": 20470, "driver": "580.0", "hostname": "testnode",
                 "recorded_at": "2026-11-18T01:00:00-08:00", "cuda_visible_devices": "0"}
CARD_FOUR_COPIES = dict(CARD_ONE_COPY, gpu="NVIDIA H100 NVL", memory_mb=95830, hostname="bignode")
CARD_SECOND_TYPE = dict(CARD_ONE_COPY, gpu="NVIDIA GeForce RTX 5080", memory_mb=16303, hostname="othernode")
EVENT_NAMES = load_file("pixel_test_event_names",
                        PLATFORM / "src/exploration_platform/atari_pixels/game_event_names.py").GAME_EVENT_NAMES

# A stand-in for scripts/atari_pixel_campaign/train_final.py. It takes the same options, writes what a
# trainer leaves in its run folder (the command it was given, progress.json, a checkpoint file) and ends
# the way FAKE_TRAINER_MODE says.
FAKE_TRAINER = r'''
import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--copies", type=int, required=True)
parser.add_argument("--copy-seeds", required=True)
parser.add_argument("--learner", required=True)
parser.add_argument("--environment", required=True)
parser.add_argument("--target-updates", type=int, required=True)
parser.add_argument("--observation-warmup-iterations", type=int, required=True)
parser.add_argument("--checkpoint-seconds", type=float, required=True)
parser.add_argument("--time-limit-seconds", type=float, required=True)
args = parser.parse_args()
run = args.run_dir
run.mkdir(parents=True, exist_ok=True)
(run / "command.json").write_text(json.dumps(sys.argv[1:]))
(run / "environment.json").write_text(json.dumps({"JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR")}))
mode = os.environ["FAKE_TRAINER_MODE"]
if mode == "complete":
    (run / "checkpoint.pkl").write_bytes(b"state at the target")
    (run / "progress.json").write_text(json.dumps({"update": args.target_updates, "target_updates": args.target_updates,
                                                   "completed": True, "unix": 1.0}))
elif mode == "stop":
    # the time limit: the state is saved and the trainer ends with exit code 0
    (run / "checkpoint.pkl").write_bytes(b"state at update 5")
    (run / "progress.json").write_text(json.dumps({"update": 5, "target_updates": args.target_updates,
                                                   "seconds_per_update": 1.5, "unix": 1.0}))
elif mode == "exit_without_state":
    pass
elif mode == "fail":
    raise RuntimeError("the stand-in trainer was told to fail")
elif mode == "out_of_memory":
    print("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1000000 bytes", file=sys.stderr)
    sys.exit(1)
'''


def small_description(**changes):
    """A small study: RND at two weights and the fast last layer at two step sizes, on seeds 13 and 14."""
    study = {
        "study": "pixel-montezuma_test", "phase": "grid", "seeds": {"first": 13, "count": 2},
        "target_updates": 244, "total_timesteps": 2000000000, "record_game_events": True,
        "record_bonus_profile": True,
        "methods": [
            {"method": "rnd", "pack_over": "intrinsic_weight", "grid": {"intrinsic_weight": [1, 2]}},
            {"method": "fast-last-layer", "pack_over": "step_size",
             "grid": {"sharing_group": [16], "intrinsic_weight": [1], "step_size": [0.03, 0.1]}},
        ],
    }
    study.update(changes)
    return study


def queue_units(run_dir, specifications):
    """Put specifications in a run folder's queue, in order; returns their names."""
    for index, specification in enumerate(specifications, 1):
        assert pixel_queue.enqueue(run_dir, specification, build_units.queue_record(specification, index))
    return [specification["unit_id"] for specification in specifications]


def claimant(card, job_id="local"):
    """The claim fields a worker on a card writes into a unit's record."""
    return {"job_id": job_id, "restart_count": 0, "account": "tester", "hostname": card["hostname"], "card": card["gpu"],
            "card_memory_mb": card["memory_mb"], "source_commit": "working tree", "copy_number": 0}


def metric_lines(updates, copies=1, events=None, score=100.0, games=2, with_event_fields=True):
    """Synthetic metric lines of train_final.py: `updates` lines for `copies` copies.

    events: {update: {"key_pickup": rollout step, ...}} for copy 0; an event at an update is counted
    once there, and its first rollout step is the given one. Every update finishes `games` games per
    copy with `score` points each and loses three lives by a fall. Rooms: room 4 at every update,
    and room 5 from the update of "left_start_room" on.
    """
    events, lines, left = events or {}, [], False
    for update in range(1, updates + 1):
        here = events.get(update, {})
        left = left or "left_start_room" in here
        rooms = [[room == 4 or (room == 5 and left and copy == 0) for room in range(33)] for copy in range(copies)]
        line = {"update": update, "unix": 1795000000.0 + update, "episode_count": [games] * copies,
                "episode_return_sum": [score * games] * copies,
                "reward_ext_sum": [1.0 if "key_pickup" in here and copy == 0 else 0.0 for copy in range(copies)],
                "rooms_visited": rooms, "steps_per_policy": update * 16384}
        if with_event_fields:
            counts = [[0] * len(EVENT_NAMES) for _ in range(copies)]
            first = [[-1] * len(EVENT_NAMES) for _ in range(copies)]
            room_first = [[0 if room == 4 else -1 for room in range(33)] for _ in range(copies)]
            for copy in range(copies):
                counts[copy][EVENT_NAMES.index("life_lost_by_fall")] = 3
                first[copy][EVENT_NAMES.index("life_lost_by_fall")] = 7
            for name, step in here.items():
                counts[0][EVENT_NAMES.index(name)] = 1
                first[0][EVENT_NAMES.index(name)] = step
                if name == "left_start_room":
                    room_first[0][5] = step
            line.update(event_counts=counts, event_first_step=first, room_first_step=room_first,
                        games_with_key=[1 if copy == 0 and update >= min(
                            (at for at, names in events.items() if "key_pickup" in names), default=10 ** 9) else 0
                            for copy in range(copies)],
                        games_left_start_room=[1 if copy == 0 and left else 0 for copy in range(copies)])
        lines.append(line)
    return lines


def write_run_records(folder, lines, target_updates, completed=False):
    """Write a run folder's record files from metric lines, with the trainer's own fold, and its progress.json."""
    game_records = load_file("pixel_game_records", PLATFORM / "scripts/atari_pixel_campaign/game_records.py")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    game_records.rebuild(folder / "metrics.jsonl", folder)
    progress = {"update": len(lines), "target_updates": target_updates, "unix": 1795000000.0}
    progress.update({"completed": True} if completed else {"seconds_per_update": 1.4})
    (folder / "progress.json").write_text(json.dumps(progress))
