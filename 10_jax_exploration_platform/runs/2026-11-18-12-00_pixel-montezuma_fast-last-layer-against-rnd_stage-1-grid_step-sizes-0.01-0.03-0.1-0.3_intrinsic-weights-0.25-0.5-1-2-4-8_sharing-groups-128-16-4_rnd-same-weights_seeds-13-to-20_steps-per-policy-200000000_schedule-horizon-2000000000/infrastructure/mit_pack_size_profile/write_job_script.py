"""Fill the pack-size profile's job script for MIT: the run folder there, the code copy, and the shared card guard.

    python3 write_job_script.py   # writes pack_size_profile_job.sh beside this file

The card guard is the grid campaign's text of /p/rlprojects/.claude/skills/mit-submit-gpu-sweep/scripts/card_guard.sh
(NEED_CARDS=1, GUARD_ACTION requeue, the card count read with jax), the one every MIT worker job of the fleet holds.
Standard library only.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path.insert(0, str(RUN.parents[1] / "scripts/pixel_sweep_campaign"))
from campaign_imports import grid  # noqa: E402


def main():
    """Write the filled job script from the template and the run folder's MIT table entry."""
    entry = json.loads((RUN / "infrastructure/remote_fleet_servers.json").read_text())["mit"]
    commit = json.loads((RUN / "source.json").read_text())["commit"]
    # before: "@REMOTE_RUN@", "@CODE@", "@CARD_GUARD@" in the template
    # after:  /home/shuzel/rnd_pixel/runs/<run>, /home/shuzel/rnd_pixel/code/source_<commit>/10_jax_exploration_platform,
    #         the card guard's lines
    text = ((HERE / "pack_size_profile_job.template.sh").read_text()
            .replace("@REMOTE_RUN@", entry["remote_run"])
            .replace("@CODE@", f"{entry['code_root']}/source_{commit}/10_jax_exploration_platform")
            .replace("@CARD_GUARD@", grid("remote_fleet").card_guard_text()))
    if "@" in text.replace("$@", ""):
        raise ValueError("a placeholder of the template is left unfilled")
    (HERE / "pack_size_profile_job.sh").write_text(text)
    print(f"wrote {HERE / 'pack_size_profile_job.sh'}")


if __name__ == "__main__":
    main()
