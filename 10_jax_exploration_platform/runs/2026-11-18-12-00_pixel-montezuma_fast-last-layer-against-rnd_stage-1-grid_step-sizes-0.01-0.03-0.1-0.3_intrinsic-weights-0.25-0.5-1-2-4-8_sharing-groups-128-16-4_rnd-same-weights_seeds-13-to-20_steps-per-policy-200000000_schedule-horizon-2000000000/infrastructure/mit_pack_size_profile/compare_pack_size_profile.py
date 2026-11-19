"""Read a pack-size profile copied back from MIT: the throughput table, and each copy's first updates against the archive.

    python3 compare_pack_size_profile.py --profile PROFILE_FOLDER [--max-rss-gb GB]

PROFILE_FOLDER is the job's folder (job-<id>/) with one sub-folder per setting (pack_size_profile.py). For the two
default settings every copy is compared with the archived baseline's run of its seed, with the rule of
scripts/pixel_sweep_campaign/smoke_check.py (the first update decides: whole numbers and true/false values equal,
every other number within |new - archive| <= 1e-3 |archive| + 1e-6). Seeds 1 to 4 come from the archived four-copy
run (H100 NVL, serval02 card 1), seeds 5 to 8 from the archived one-copy runs (RTX A4500, jaguar03). The copies of
seeds 1 to 4 of the eight-copy run are also compared field by field with the four-copy run of the same job (same
card): the count of differing fields over every update both wrote. Standard library only.
"""
import argparse
import json
from pathlib import Path
import sys

PIXEL = Path("/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign")
sys.path.insert(0, str(PIXEL))
from campaign_imports import load_file  # noqa: E402

smoke_check = load_file("pixel_smoke_check", PIXEL / "smoke_check.py")
ARCHIVE = next(Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs").glob(
    "2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_*/data"))
FOUR_COPY_RUN = "h100-serval02-card-1_final-training_seeds-1-to-4"
STEPS_PER_UPDATE = 128 * 128
REFERENCE_LINES = 20


def first_lines(path, count):
    """Read the first lines of a metrics file as parsed JSON."""
    with Path(path).open() as stream:
        return [json.loads(next(stream)) for _ in range(count)]


def copy_lines(lines, index):
    """Cut the lines of a multi-copy run down to one copy, in the shape of a one-copy run's lines."""
    # before: {"update": 1, "loss": [0.67, 0.54, 0.53, 0.64], "rooms_visited": [[...33], [...33], ...], "steps_per_policy": 16384}
    # after (index 1): {"update": 1, "loss": [0.54], "rooms_visited": [[...33]], "steps_per_policy": 16384}
    return [{field: [value[index]] if isinstance(value, list) else value for field, value in line.items()}
            for line in lines]


def archive_lines(seed):
    """The archived baseline's first lines for one seed, in one-copy shape."""
    if seed <= 4:
        seeds = json.loads((ARCHIVE / FOUR_COPY_RUN / "run/run.json").read_text())["copy_seeds"]
        return copy_lines(first_lines(ARCHIVE / FOUR_COPY_RUN / "run/metrics.jsonl", REFERENCE_LINES), seeds.index(seed))
    return first_lines(ARCHIVE / f"a4500-jaguar03_final-training_seed-{seed}" / "run/metrics.jsonl", REFERENCE_LINES)


def differing_fields(left, right):
    """Count the fields of two lists of lines that are not equal, the write time left out."""
    # before: two lines {"update": 1, "unix": 5.0, "loss": [0.5]} and {"update": 1, "unix": 9.0, "loss": [0.5]} ; after: 0
    return sum(left_line[field] != right_line.get(field) for left_line, right_line in zip(left, right)
               for field in left_line if field != "unix")


def throughput_row(summary, max_rss_gb):
    """One row of the five-column throughput table, with the peak card memory."""
    seconds = summary["seconds_per_update"]
    if not seconds:
        return f"| {summary['setting']} | {summary['copies']} | not measured ({summary['updates_written']} updates) | | | | |"
    per_copy = STEPS_PER_UPDATE / seconds
    return (f"| {summary['setting']} | {summary['copies']} | {seconds:.3f} | {summary['copies'] * per_copy / 1e6:.4f} | "
            f"{per_copy / 1e3:.2f} | {1e6 / (3600 * per_copy):.4f} | {summary['peak_card_memory_mb']:,} |")


def main(argv=None):
    """Print the throughput table and the comparisons."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--max-rss-gb", type=float, help="the job's sacct MaxRSS, printed beside the table")
    args = parser.parse_args(argv)
    summaries = {path.parent.name: json.loads(path.read_text()) for path in sorted(args.profile.glob("*/summary.json"))}
    print("| setting | copies | seconds per iteration | total steps per second (millions) | "
          "steps per second per copy (thousands) | hours per million steps per copy | peak card memory (MB) |")
    print("|---|---|---|---|---|---|---|")
    for summary in summaries.values():
        print(throughput_row(summary, args.max_rss_gb))
    if args.max_rss_gb:
        print(f"\nhost memory of the job (sacct MaxRSS, largest of the settings): {args.max_rss_gb} GB")
    # each copy of the default settings against the archive for its seed: the first update decides
    print("\nfirst updates against the archived baseline, per copy (rule: " + smoke_check.FIRST_LINE_RULE + "):")
    verdicts = {}
    for name in ("default_copies-4_seeds-1-to-4", "default_copies-8_seeds-1-to-8"):
        if name not in summaries:
            continue
        lines = [json.loads(line) for line in (args.profile / name / "run/metrics.jsonl").read_text().splitlines()
                 if line.strip()]
        for index, seed in enumerate(summaries[name]["copy_seeds"]):
            check = smoke_check.compare_with_archive(archive_lines(seed)[:len(lines)], copy_lines(lines, index))
            verdicts[(name, seed)] = check["passed"]
            print(f"  {name}, copy {index + 1} (seed {seed}): {'passed' if check['passed'] else 'NOT passed'}; "
                  f"{check['detail']}")
    # the eight-copy run's copies of seeds 1 to 4 against the four-copy run of the same job, field by field
    if {"default_copies-4_seeds-1-to-4", "default_copies-8_seeds-1-to-8"} <= set(summaries):
        four, eight = ([json.loads(line) for line in (args.profile / name / "run/metrics.jsonl").read_text().splitlines()
                        if line.strip()] for name in ("default_copies-4_seeds-1-to-4", "default_copies-8_seeds-1-to-8"))
        common = min(len(four), len(eight))
        for index in range(4):
            count = differing_fields(copy_lines(four[:common], index), copy_lines(eight[:common], index))
            print(f"  seed {index + 1}: eight copies against four copies on the same card, {common} updates: "
                  f"{count} differing field(s)")
    print("\nevery copy passed" if verdicts and all(verdicts.values()) else "\nNOT every copy passed")
    return 0 if verdicts and all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
