"""Load the maze campaign's and the grid campaign's files for the pixel sweep, under names that cannot clash.

This folder, `scripts/maze_campaign/` and `scripts/grid_initialization_campaign/` each have a
worker.py, a submit.py, a monitor.py and a remote_fleet.py. The other two folders are in use by a
running study and are imported, never edited:

    from campaign_imports import common, unit_queue, grid, maze, own

    common       scripts/grid_initialization_campaign/common.py (atomic JSON records, the queue lock,
                 Pacific time, this run's live job ids)
    unit_queue   scripts/maze_campaign/unit_queue.py (return, finish, recover, counts)
    grid(name)   a file of the grid campaign, loaded as the module grid_<name>
    maze(name)   a file of the maze campaign, loaded as the module maze_<name>
    own(name)    a file of this folder, loaded as the module pixel_<name>

Files whose names exist in more than one folder (worker, submit, monitor, remote_fleet, build_units)
are loaded only through grid(), maze() and own(), never by a bare `import`. Standard library only.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
PLATFORM = SCRIPTS.parent
MAZE = SCRIPTS / "maze_campaign"

# The maze folder goes to the END of the path, behind whatever the caller put first, and this folder goes
# behind the maze folder, wherever it stood before. A bare `import build_units` (the maze campaign's own
# tests do that) must keep finding the maze campaign's file when both folders are on the path; this
# folder's files with such shared names are loaded by their path (own), and the only names imported bare
# from this folder (campaign_imports, pixel_queue) exist nowhere else.
# before: sys.path = [<this folder>, ..., <site-packages>]           (a script of this folder was started)
# after:  sys.path = [..., <site-packages>, <maze folder>, <this folder>]
if str(MAZE) not in sys.path:
    sys.path.append(str(MAZE))
while str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.append(str(HERE))

import grid_campaign  # noqa: E402  (scripts/maze_campaign/grid_campaign.py)
from grid_campaign import common, load_file  # noqa: E402,F401
import unit_queue  # noqa: E402,F401  (scripts/maze_campaign/unit_queue.py)


def grid(name):
    """Load one file of the grid campaign as the module grid_<name>."""
    return grid_campaign.load(name)


def maze(name):
    """Load one file of the maze campaign as the module maze_<name>."""
    return grid_campaign.own(name)


def own(name):
    """Load one file of this folder as the module pixel_<name>."""
    return load_file("pixel_" + name, HERE / f"{name}.py")
