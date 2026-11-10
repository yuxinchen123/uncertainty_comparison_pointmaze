"""Print every case's rows from best_rows.json, both batches together, sorted by the screen score."""
import json
import sys

rows = json.load(open(sys.argv[1]))['rows']
cases = [('straight', 'tabular'), ('straight', 'neural'), ('snake', 'tabular'), ('snake', 'neural')]
names = {'tabular': 'tabular-PPO', 'neural': 'coordinate-PPO'}
for kind, policy in cases:
    print(f'### {kind} grid, {names[policy]}')
    selected = sorted((row for row in rows if row['kind'] == kind and row['policy'] == policy), key=lambda row: row['mean'])
    for row in selected:
        slope = 'n/a' if row['slope_median'] is None else f"{row['slope_median']:.2f}"
        setting = row['configuration']
        error = float('nan') if row['standard_error'] is None else row['standard_error']
        print(f"   {row['idea'][:52]:52s} {row['mean']/1000:9.2f}K ± {error/1000:7.2f}K  found {100*row['discovery']:5.1f}%  "
              f"start {row['start_max_over_min']:.2f} slope {slope:>5s}  lr {setting['learning_rate']:g} ent {setting['entropy_coefficient']:g} "
              f"w {setting['intrinsic_coefficient']:g}  settings {row['settings_finished']:2d}  bad {row['nonfinite_copies']}/{row['all_copies']}")
