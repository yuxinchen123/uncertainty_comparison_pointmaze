"""Make the frozen code copy that every job of a run folder runs, and record it in RUN/source.json.

    python freeze_code.py --run-dir RUN

The copy is RUN/code/source_<commit>/10_jax_exploration_platform/{src,scripts,configs}, taken from
the commit itself with `git archive`, so nothing of the working tree can enter it. The command
refuses when one of those three folders holds a change that is not committed (a run whose code is
not in git must not be submitted). Jobs submitted afterwards run this copy; jobs submitted before
keep the copy they were submitted with. Run it again after a new commit to move new jobs to it.

Every new copy gets a copy number, one above the highest this run folder has had (the first
numbered copy is 1; a folder frozen before copy numbers existed counts as 0). The number is written
into the copy (<copy>/10_jax_exploration_platform/SOURCE_COPY_NUMBER, beside SOURCE_COMMIT, so it
travels with the copy to a remote server) and into source.json (`copy_number`, and
`highest_copy_number`). Units built afterwards need at least that number, and a worker of an older
copy leaves them alone (unit_queue.py, "Code copies"). So a folder may be frozen anew while jobs of
older copies still run. Freezing again at a commit that already has a copy keeps that copy's number.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from grid_campaign import common
import unit_queue

PLATFORM_NAME = "10_jax_exploration_platform"
FOLDERS = ("src", "scripts", "configs")


def git(repository, *arguments):
    """Run one git command in a repository and return its output."""
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True)


def freeze(run_dir, repository=common.PROJECT, leave_out_uncommitted=False):
    """Copy the committed code into the run folder and write source.json; returns the commit."""
    run_dir = Path(run_dir).resolve()
    paths = [f"{PLATFORM_NAME}/{folder}" for folder in FOLDERS]
    # every file the jobs run must be in the commit. With leave_out_uncommitted the copy is still
    # the commit alone (other work in progress in the tree is named and left out), so the jobs
    # never run a line that is not in git.
    changed = git(repository, "status", "--porcelain", "--", *paths).strip()
    if changed and not leave_out_uncommitted:
        raise SystemExit("these code files are not committed; commit and push them first:\n" + changed)
    if changed:
        print("left out of the copy, because they are not committed:\n" + changed)
    commit = git(repository, "rev-parse", "HEAD").strip()
    target = run_dir / "code" / f"source_{commit}"
    record_path = run_dir / "source.json"
    previous = common.read_json(record_path) if record_path.exists() else {}
    # the highest copy number this folder has had; a record written before the numbers counts as 0
    # before: {"commit": "2017e2b8...", ...} (no number) ; after: highest 0, so the next new copy is 1
    highest = previous.get("highest_copy_number", previous.get("copy_number", 0))
    if not target.exists():
        # unpacked under another name first, so a copy that was cut off is never taken for a whole one
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(repository), "archive", commit, "--", *paths],
                                 check=True, stdout=subprocess.PIPE).stdout
        subprocess.run(["tar", "-x", "-C", str(partial)], input=archive, check=True)
        # the workers read this file to record which copy ran each attempt of a unit
        (partial / PLATFORM_NAME / "SOURCE_COMMIT").write_text(commit + "\n")
        # and this one to know which units their copy may run: a new copy is one above every earlier one
        (partial / PLATFORM_NAME / unit_queue.COPY_NUMBER_FILE).write_text(f"{highest + 1}\n")
        os.rename(partial, target)
    # a copy that exists keeps the number it was made with (0 when it was made before the numbers)
    number = unit_queue.copy_number(target / PLATFORM_NAME)
    # new jobs take the copy named here; earlier copies stay where they are
    earlier = [value for value in [*previous.get("previous_commits", []), previous.get("commit")]
               if value and value != commit]
    common.write_json(record_path, {"commit": commit, "snapshot": str(target), "campaign": "maze",
                                    "frozen_at": common.now(), "previous_commits": earlier,
                                    "copy_number": number, "highest_copy_number": max(highest, number)})
    return commit


def main(argv=None):
    """Freeze the committed code for one run folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--leave-out-uncommitted", action="store_true",
                        help="freeze the last commit although the working tree holds uncommitted changes "
                             "to the code folders (for example other work in progress); those changes "
                             "are listed and are not in the copy")
    args = parser.parse_args(argv)
    record_path = args.run_dir.resolve() / "source.json"
    before = common.read_json(record_path).get("copy_number", 0) if record_path.exists() else None
    commit = freeze(args.run_dir, leave_out_uncommitted=args.leave_out_uncommitted)
    number = common.read_json(record_path)["copy_number"]
    print(f"jobs of {args.run_dir} now run the code of commit {commit} "
          f"({args.run_dir / 'code' / ('source_' + commit)}), copy number {number}")
    if before is not None and number != before:
        # said once at the change, because nothing else tells the reader that such jobs exist
        print(f"worker jobs submitted before this freeze run copy number {before} or lower: they take no unit built "
              "from now on, and jobs of theirs that still wait are counted as workers until they start and end "
              "(cancel those by their ledger ids to avoid that wait). Workers of a copy made before copy numbers "
              "existed do not check at all: keep units of new options away from them as before.")


if __name__ == "__main__":
    main()
