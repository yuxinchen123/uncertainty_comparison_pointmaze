"""Compare the baseline's development runs trained twice: by the extension run and by the auto research (batch 6)."""
import glob
import json
import sys

import numpy as np

extension, research, case = sys.argv[1:4]


def scores(pattern):
    """Read {(learning rate, entropy, weight, policy seed): capped first-goal steps} from finished units."""
    table, hosts = {}, set()
    for path in glob.glob(pattern):
        summary = json.load(open(path))
        if not summary.get('completed'):
            continue
        settings = {row['setting_index']: row['configuration'] for row in summary['settings']}
        for record in summary['copy_records']:
            setting = settings[record['setting_index']]
            key = (setting['learning_rate'], setting['entropy_coefficient'], setting['intrinsic_coefficient'], record['policy_seed'])
            table[key] = record['capped_first_goal_steps']
    return table


first = scores(f'{extension}/data/research/fixed-development_{case}_online-tabular-rnd_state-normalization-off_reward-normalization-on_*/summary.json')
second = scores(f'{research}/data/research/development-06_{case}_control_*/summary.json')
common = sorted(set(first) & set(second))
a = np.asarray([first[key] for key in common], float)
b = np.asarray([second[key] for key in common], float)
print(case, '| pairs in both:', len(common), 'of', len(first), 'and', len(second))
print('identical pairs:', int((a == b).sum()), f'({100 * (a == b).mean():.1f}%)')
print('mean of the extension copy:', round(a.mean()), '| mean of the auto research copy:', round(b.mean()),
      '| mean absolute difference:', round(np.abs(a - b).mean()))
# Per configuration: the two means over the 16 seeds, for the configurations that differ most and the two chosen ones.
by_configuration = {}
for key, x, y in zip(common, a, b):
    by_configuration.setdefault(key[:3], []).append((x, y))
rows = sorted(((np.mean([x for x, _ in pairs]), np.mean([y for _, y in pairs]), key) for key, pairs in by_configuration.items()))
print('five best configurations by the extension copy (extension mean, auto research mean, learning rate, entropy, weight):')
for x, y, key in rows[:5]:
    print(f'   {x:10.0f} {y:10.0f}  {key[0]:g} {key[1]:g} 2^{int(round(np.log2(key[2])))}')
rows_b = sorted(rows, key=lambda row: row[1])
print('five best configurations by the auto research copy:')
for x, y, key in rows_b[:5]:
    print(f'   {x:10.0f} {y:10.0f}  {key[0]:g} {key[1]:g} 2^{int(round(np.log2(key[2])))}')
