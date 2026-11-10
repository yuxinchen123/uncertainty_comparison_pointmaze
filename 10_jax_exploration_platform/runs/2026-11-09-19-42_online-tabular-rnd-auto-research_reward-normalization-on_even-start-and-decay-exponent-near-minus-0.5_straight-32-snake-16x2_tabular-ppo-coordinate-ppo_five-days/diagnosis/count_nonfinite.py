"""Count, per finished unit and intrinsic weight, the copies whose bonus became not-a-number or infinite."""
import json
import sys
from pathlib import Path

import numpy as np

for folder in sorted(Path(sys.argv[1]).glob('screen-*')):
    if not (folder/'summary.json').exists() or not (folder/'behavior.npz').exists():
        continue
    summary = json.loads((folder/'summary.json').read_text())
    data = np.load(folder/'behavior.npz')
    history, indices = data['rnd_error_history'], data['setting_indices']
    steps = data['visit_count_history_steps']
    # Before: [checkpoints, copies, cells]; after: per copy the first checkpoint with a non-finite bonus, or -1.
    bad = ~np.isfinite(history).all(axis=2)
    first = np.where(bad.any(axis=0), steps[bad.argmax(axis=0)], -1)
    weights = [row['configuration']['intrinsic_coefficient'] for row in summary['settings']]
    per_weight = [f"{int((first[indices == index] >= 0).sum())}" for index in range(len(weights))]
    total = int((first >= 0).sum())
    name = folder.name.replace('screen-01_', '').replace('snake-width-16-turns-2', 'snake').replace('straight-width-32-turns-0', 'straight').replace('_seeds-0-31', '').replace('policy-', '').replace('learning-rate-', 'lr-')
    earliest = int(first[first >= 0].min()) if total else -1
    print(f"{name[:92]:92s} bad copies {total:3d}/{len(first)}  per weight {'/'.join(per_weight)}  earliest step {earliest}")
