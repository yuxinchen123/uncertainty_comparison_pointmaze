"""Check scripts/summarize_episode_records.py on a hand-made shard whose answers are known."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from summarize_episode_records import return_offset, summarize_run


def window(index, episodes, ret, reached):
    """One window record with the tracker sums of two copies."""
    zeros = [0.0, 0.0]
    return {"record": "episode_window", "window_index": index,
            "env_steps_per_copy": 1000 * (index + 1),
            "diagnostics_per_copy": {"episode_episodes": episodes, "episode_ret": ret,
                                     "episode_reached": reached, "logstd_mean": zeros}}


def main():
    """Ten windows: return rises to 4 at window 3 and falls to 1; check peak, final, offset."""
    # offsets: a terminating -1-shift task moves to the lump-sum scale; others are untouched
    assert return_offset("pointmaze_medium_term250_nonoise_start6-1_goal1-6_shift-1@1") == 250
    assert return_offset("pointmaze_large_cont400_nonoise@1") == 0
    assert return_offset("pointmaze_umaze_term150_nonoise_start3-1_goal1-1@1") == 0

    # copy 0 returns per episode [1,2,3,4,3,2,1,1,1,1]; copy 1 the same; 2 episodes per window
    levels = [1, 2, 3, 4, 3, 2, 1, 1, 1, 1]
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "data").mkdir()
        start = {"record": "unit_start", "unit_id": "u", "bonus": "rnd_next_state",
                 "environment": "pointmaze_large_cont400_nonoise@1",
                 "cell_settings": [[1e-3, 10.0]], "copy_cell": [0, 0]}
        lines = [start] + [window(k, [2.0, 2.0], [2.0 * v, 2.0 * v], [2.0, 1.0])
                           for k, v in enumerate(levels)] + [{"record": "unit_complete"}]
        (run / "data" / "u.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        s = next(iter(summarize_run(run).values()))
    assert s["complete"] and s["peak"] == 4.0 and s["peak_step"] == 4000
    assert s["final"] == 1.0 and abs(s["holds_peak"] - 0.25) < 1e-12
    assert abs(s["whole_run"] - np.mean(levels)) < 1e-12
    assert s["reach_share_final"] == 0.75 and s["ever_reached"] == 1.0
    print("PASS summary: peak 4 at 4000 steps, final 1, holds 0.25, reach share 0.75")


if __name__ == "__main__":
    main()
