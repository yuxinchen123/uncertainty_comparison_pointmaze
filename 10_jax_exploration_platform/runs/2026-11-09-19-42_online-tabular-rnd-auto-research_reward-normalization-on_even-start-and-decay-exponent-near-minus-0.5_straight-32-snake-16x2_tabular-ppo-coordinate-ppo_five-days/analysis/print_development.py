"""Print the development-stage rows of a batch: each idea's chosen configuration per case."""
import json
import math
import sys

rows = json.load(open(sys.argv[1]))['rows']
stage = sys.argv[2]
cases = [('straight', 'tabular'), ('straight', 'neural'), ('snake', 'tabular'), ('snake', 'neural')]
table = {}
for row in rows:
    if row['stage'] == stage:
        table.setdefault(row['idea'], {})[(row['kind'], row['policy'])] = row
print(f'{stage}: mean capped first-goal time in thousands, seeds per setting as recorded; settings finished of 52')
for idea, by_case in sorted(table.items()):
    for case in cases:
        row = by_case.get(case)
        if row is None:
            continue
        setting = row['configuration']
        print(f"  {idea[:58].ljust(58)} {case[0]:8s} {case[1]:8s} {row['mean'] / 1000:8.2f}±{(row['standard_error'] or 0) / 1000:5.2f} "
              f"found {100 * row['discovery']:3.0f}%  {row['settings_finished']:2d}/52  lr {setting['learning_rate']:g} "
              f"entropy {setting['entropy_coefficient']:g} weight 2^{round(math.log2(setting['intrinsic_coefficient']))}")
