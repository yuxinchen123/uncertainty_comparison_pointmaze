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
    assert s["other_settings"] == {} and "diagnostics_final_and_at_peak" in s
    print("PASS summary: peak 4 at 4000 steps, final 1, holds 0.25, reach share 0.75")
    episode_log_shard()


def log_window(index, episodes, arrivals, returns, lengths, to_goal, first):
    """One window record with the episode log's sums of two copies (one per reward setting)."""
    return {"record": "episode_window", "window_index": index,
            "env_steps_per_copy": 1000 * (index + 1), "episodes_per_copy": episodes,
            "arrivals_per_copy": arrivals, "return_sum_per_copy": returns,
            "length_sum_per_copy": lengths, "steps_to_goal_sum_per_copy": to_goal,
            "first_goal_step_per_copy": first}


def episode_log_shard():
    """Two cells that differ only in the reward setting, recorded by the episode log.

    Episodes end at the goal or at 400 steps. Windows 0 and 1: two episodes per copy, none
    arrives. Window 2: four episodes per copy, three arrive after 100 steps each. Copy 0 is paid
    1 on arrival; copy 1 pays 1/400 per step, so an arrival after 100 steps returns -99/400.
    """
    none, some = [2, 2], [4, 4]
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "data").mkdir()
        start = {"record": "unit_start", "unit_id": "u", "bonus": "rnd_next_state",
                 "environment": "pointmaze_large_term400_nonoise_reward-arrival-plus-one-and-step-cost@1",
                 "cell_setting_names": ["int_coef", "ext_reward_scale", "ext_reward_offset"],
                 "cell_settings": [[4.0, 1.0, 0.0], [4.0, 0.0025, -0.0025]],
                 "copy_cell": [0, 1], "trainer_config": {"learning_rate": 1e-4, "int_coef": 1.0}}
        lines = [start,
                 log_window(0, none, [0, 0], [0.0, -2.0], [800, 800], [0, 0], [-1, -1]),
                 log_window(1, none, [0, 0], [0.0, -2.0], [800, 800], [0, 0], [-1, -1]),
                 log_window(2, some, [3, 3], [3.0, -1.0 - 3 * 99 / 400], [700, 700], [300, 300],
                            [2100, 2100]),
                 {"record": "unit_resume"}, {"record": "unit_complete"}]
        (run / "data" / "u.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        cells = summarize_run(run)
    # the key names the rate, the weight, then every other setting of the cell
    arrival = cells["u|lr=0.0001|beta=4|ext_reward_scale=1|ext_reward_offset=0"]
    cost = cells["u|lr=0.0001|beta=4|ext_reward_scale=0.0025|ext_reward_offset=-0.0025"]
    assert arrival["learning_rate"] == 1e-4 and arrival["intrinsic_weight"] == 4.0
    assert arrival["other_settings"] == {"ext_reward_scale": 1.0, "ext_reward_offset": 0.0}
    assert arrival["complete"] and arrival["windows"] == 3 and arrival["return_offset"] == 0.0
    # reward 1 on arrival: the curve is the share of episodes that arrive
    assert arrival["curve"] == [0.0, 0.0, 0.75] and arrival["peak"] == arrival["final"] == 0.75
    assert arrival["reach_share_final"] == 0.75 and arrival["ever_reached"] == 1.0
    assert arrival["episode_length_final"] == 175.0 and arrival["steps_to_goal_final"] == 100.0
    assert arrival["goal_discovery_rate"] == 1.0
    assert arrival["mean_capped_first_goal_step"] == 2100.0
    # the step cost: -1 while no episode arrives, then (-1 - 3 x 99/400) / 4
    np.testing.assert_allclose(cost["curve"], [-1.0, -1.0, (-1.0 - 3 * 99 / 400) / 4])
    assert abs(cost["final"] - (-1.0 - 3 * 99 / 400) / 4) < 1e-12
    print("PASS episode-log summary: two reward settings, arrivals, steps to the goal")


if __name__ == "__main__":
    main()
