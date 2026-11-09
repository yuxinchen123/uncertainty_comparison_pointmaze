"""Look at the 128-seed confirmation distributions and how trimming k seeds from each end changes the ranking."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '/p/rlprojects/RND/10_jax_exploration_platform/scripts/grid_initialization_campaign')
from fixed_settings_report import collect, condition_label

run = Path(sys.argv[1])
B = 16777216
cases = defaultdict(list)
for row in collect(run):
    if row['wave'] == 'confirmation' and row['protocol'] == 'chosen':
        cases[(row['kind'], row['policy'])].append(row)
KS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 32]
for case, rows in cases.items():
    print('\n=====', case)
    ranks = {}
    for row in rows:
        v = np.sort([c['capped_first_goal_steps'] for c in row['copies']])
        q = np.quantile(v, [0, .05, .1, .25, .5, .75, .9, .95, 1])
        tm = [v[k:len(v)-k].mean() for k in KS]
        print(f"{condition_label(row['condition'])[:60]:60s} n={len(v)} capped={int((v>=B).sum())} "
              f"mean={v.mean():.0f} sd={v.std(ddof=1):.0f} skew={((v-v.mean())**3).mean()/v.std()**3:.1f}")
        print('   quantiles 0,5,10,25,50,75,90,95,100%:', ' '.join(f'{x:.0f}' for x in q))
        print('   lowest 12:', ' '.join(f'{x:.0f}' for x in v[:12]))
        print('   highest 12:', ' '.join(f'{x:.0f}' for x in v[-12:]))
        print('   trimmed mean k=' + ' '.join(f'{k}:{m:.0f}' for k, m in zip(KS, tm)))
        ranks[row['condition']] = tm
    for i, k in enumerate(KS):
        order = sorted(ranks, key=lambda c: ranks[c][i])
        print(f'  order k={k:2d}:', ' | '.join(condition_label(c)[:28] for c in order[:4]))
