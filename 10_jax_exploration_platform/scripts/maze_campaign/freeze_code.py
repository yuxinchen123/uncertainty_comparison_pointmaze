"""Make the frozen code copy that every job of a run folder runs, and record it in RUN/source.json.

    python freeze_code.py --run-dir RUN

The copy is RUN/code/source_<commit>/10_jax_exploration_platform/{src,scripts,configs}, taken from
the commit itself with `git archive`, so nothing of the working tree can enter it. The command
refuses when one of those three folders holds a change that is not committed (a run whose code is
not in git must not be submitted). Jobs submitted afterwards run this copy; jobs submitted before
keep the copy they were submitted with. Run it again after a new commit to move new jobs to it.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from grid_campaign import common

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
        os.rename(partial, target)
    # new jobs take the copy named here; earlier copies stay where they are
    record_path = run_dir / "source.json"
    previous = common.read_json(record_path) if record_path.exists() else {}
    earlier = [value for value in [*previous.get("previous_commits", []), previous.get("commit")]
               if value and value != commit]
    common.write_json(record_path, {"commit": commit, "snapshot": str(target), "campaign": "maze",
                                    "frozen_at": common.now(), "previous_commits": earlier})
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
    commit = freeze(args.run_dir, leave_out_uncommitted=args.leave_out_uncommitted)
    print(f"jobs of {args.run_dir} now run the code of commit {commit} "
          f"({args.run_dir / 'code' / ('source_' + commit)})")


if __name__ == "__main__":
    main()
