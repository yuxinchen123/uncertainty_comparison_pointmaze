"""Print the three-law bench results: slopes, start spread, and what happens to never-visited cells."""
import glob
import json
import sys

rows = []
for path in sorted(glob.glob(sys.argv[1] + '/*.json')):
    d = json.load(open(path))

    def g(case, key, digits=2):
        """Format one measure, or 'nan' when it is missing."""
        value = d[case].get(key)
        return 'nan' if value is None else f'{value:.{digits}f}'

    rows.append((d['idea'][:56], g('straight_uniform', 'start_max_over_min'), g('straight_uniform', 'slope_median'),
                 g('snake_uniform', 'slope_median'), g('straight_start-heavy', 'late_count_consistency_sd_log'),
                 g('snake_start-heavy', 'late_count_consistency_sd_log'),
                 g('straight_first-half-only', 'unvisited_final_over_start_median', 3),
                 g('snake_first-half-only', 'unvisited_final_over_start_median', 3),
                 g('straight_first-half-only', 'unvisited_over_visited_final_bonus', 1),
                 g('snake_first-half-only', 'unvisited_over_visited_final_bonus', 1),
                 f"{d['straight_uniform']['seconds']:.0f}s"))
print('idea'.ljust(56), 'start | slope st sn | consistency st sn | unvisited kept st sn | unvisited/visited st sn | time')
for r in rows:
    print(r[0].ljust(56), r[1], '|', r[2], r[3], '|', r[4], r[5], '|', r[6], r[7], '|', r[8], r[9], '|', r[10])
