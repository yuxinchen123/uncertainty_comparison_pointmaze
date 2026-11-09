"""Count far-out seeds per confirmation condition (beyond 3 interquartile ranges of the quartiles)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/p/rlprojects/RND/10_jax_exploration_platform/scripts/grid_initialization_campaign')
from fixed_settings_report import collect, condition_label

B = 16777216
for row in collect(Path(sys.argv[1])):
    if row['wave'] != 'confirmation' or row['protocol'] != 'chosen':
        continue
    v = np.asarray([c['capped_first_goal_steps'] for c in row['copies']], float)
    q1, q3 = np.quantile(v, [.25, .75]); iqr = q3 - q1
    high, low = int((v > q3 + 3*iqr).sum()), int((v < q1 - 3*iqr).sum())
    lv = np.log(v); l1, l3 = np.quantile(lv, [.25, .75]); li = l3 - l1
    print(f"{row['kind']:8s} {row['policy']:7s} capped={int((v>=B).sum()):3d} far_high={high:2d} far_low={low:2d} "
          f"log_far_high={int((lv > l3+3*li).sum()):2d} log_far_low={int((lv < l1-3*li).sum()):2d}  {condition_label(row['condition'])}")
