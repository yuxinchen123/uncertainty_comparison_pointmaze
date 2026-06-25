"""Count finished run-2 runs (one valid local JSON = one finished run) per algorithm and report which
main.tex milestones are met. No wandb modes now: 3 algorithms x 200 seeds, all logged locally under
data/local/. Usage: python count_progress.py [data_dir]"""
import json
import glob
import os
import sys
from collections import defaultdict

# data_dir/local/*.json is where train.py writes each finished run (z_logging_mode="local").
# Default = the run folder's data/ : code/ -> analysis/ -> <run>/ , then /data (three dirnames up). This
# matches common.py's DEFAULT_DATA_DIR; two dirnames (analysis/data) was wrong and counted nothing.
_RUN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_RUN_DIR, "data")
MODE = "local"
ALGOS = ["gt_position_velocity", "rnd_elliptical", "rnd_state"]

# Count one finished run per valid JSON (must have runtime_seconds = written only on success).
by = defaultdict(int)  # algorithm -> count of finished runs
for f in glob.glob(os.path.join(data_dir, MODE, "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue  # skip a half-written/corrupt file
    if "runtime_seconds" in d and d.get("algorithm") in ALGOS:
        by[d["algorithm"]] += 1

# Milestones are per algorithm (the smallest algorithm's count gates each one). Target is 200 per algorithm.
m50 = min(by[a] for a in ALGOS) >= 50
m100 = min(by[a] for a in ALGOS) >= 100
m150 = min(by[a] for a in ALGOS) >= 150
out = {
    "per_algo": {a: by[a] for a in ALGOS},
    "total_finished": sum(by[a] for a in ALGOS),
    "target_per_algo": 200,
    "milestone_50_per_algo": m50,
    "milestone_100_per_algo": m100,
    "milestone_150_per_algo": m150,
    "all_done": min(by[a] for a in ALGOS) >= 200,
}
print(json.dumps(out, indent=2))
