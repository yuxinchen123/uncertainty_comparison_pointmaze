"""Print how the bonus field looks along the path while a policy explores: start, last visited cells, frontier.

For chosen units and one intrinsic weight, at each count checkpoint: the median over copies of the furthest cell
reached, the bonus at the start cell, at the furthest visited cell, at the first never-visited cell, and the ratio of
the last two (the frontier contrast). Bonuses are the raw readout before reward normalization.
"""
import json
import sys
from pathlib import Path

import numpy as np


def profile(folder, weight):
    """Return rows (step, furthest cell, start bonus, furthest-visited bonus, frontier bonus, goal fraction)."""
    summary = json.loads((folder/'summary.json').read_text())
    settings = [row['configuration']['intrinsic_coefficient'] for row in summary['settings']]
    index = settings.index(weight)
    data = np.load(folder/'behavior.npz')
    selected = data['setting_indices'] == index
    history = np.concatenate([data['rnd_error_initial'][selected][None], data['rnd_error_history'][:, selected]])
    visits = np.concatenate([np.zeros_like(data['visit_count_history'][:1, selected]), data['visit_count_history'][:, selected]])
    steps = np.concatenate([[0], data['visit_count_history_steps']])
    first_goal = data['first_goal_steps'][selected]
    rows = []
    for point in range(len(steps)):
        # Before: [copies, cells] counts; after: per copy the furthest visited cell (the start counts as visited).
        reached = np.maximum((visits[point] > 0) * np.arange(visits.shape[2])[None], 0).max(axis=1)
        copies = np.arange(len(reached))
        frontier = np.minimum(reached + 1, visits.shape[2] - 1)
        rows.append((int(steps[point]), float(np.median(reached)), float(np.median(history[point][:, 0])),
                     float(np.median(history[point][copies, reached])), float(np.median(history[point][copies, frontier])),
                     float(np.median(history[point][copies, frontier] / history[point][copies, reached])),
                     float(np.mean((first_goal >= 0) & (first_goal <= steps[point])))))
    return rows


def main():
    """Print the profile of every unit folder named on the command line at the given weight."""
    weight = float(sys.argv[1])
    for name in sys.argv[2:]:
        folder = Path(name)
        print(folder.name.replace('screen-01_', '')[:110], 'weight', weight)
        print('      step  furthest  start-bonus  furthest-visited  frontier  frontier/furthest  goal')
        for row in profile(folder, weight):
            if row[0] in (0, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304):
                print(f'{row[0]:10d} {row[1]:9.0f} {row[2]:12.4g} {row[3]:17.4g} {row[4]:9.4g} {row[5]:18.3g} {row[6]:5.2f}')


if __name__ == '__main__':
    main()
