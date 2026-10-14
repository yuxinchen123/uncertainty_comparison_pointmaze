"""Check selected-run records, retry exclusion, and explicit training arguments."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from aggregate_selected_run import aggregate
from run_training import parse_args, build_config
from exploration_platform.training.sweep import build_sweep


def main():
    """Exercise complete, partial, retried and invalid setting records without real training."""
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        (run / "data").mkdir()
        settings = run / "parameter_settings.json"
        settings.write_text(json.dumps({"rnd_next_state": [[1e-3, 100], [1e-4, 1]]}))
        sys.argv = ["run_training", "--run-dir", str(run), "--unit-id", "test",
                    "--parameter-settings", str(settings), "--copies-per-cell", "128"]
        cfg = build_config(parse_args())
        assert cfg.n_copies == 256
        assert build_sweep(cfg).group_settings == ((1e-3, 100.), (1e-4, 1.))
        sys.argv += ["--learning-rates", "1e-3"]
        try:
            build_config(parse_args())
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting selected and Cartesian settings accepted")

        # A no-bonus launch records zero weight and rejects nonzero sweep settings.
        sys.argv = ["run_training", "--run-dir", str(run), "--unit-id", "test",
                    "--bonus", "none"]
        assert build_config(parse_args()).int_coef == 0.
        sys.argv += ["--learning-rates", "1e-3,1e-4", "--copies-per-cell", "128"]
        assert build_sweep(build_config(parse_args())).group_settings == ((1e-3, 0.), (1e-4, 0.))
        sys.argv += ["--intrinsic-weights", "1"]
        try:
            build_config(parse_args())
        except ValueError:
            pass
        else:
            raise AssertionError("nonzero no-bonus sweep weight accepted")
        settings.write_text(json.dumps({"rnd_next_state": [[1e-3, 100], [1e-4, 1]],
                                        "none": [[1e-3, 1]]}))
        sys.argv = ["run_training", "--run-dir", str(run), "--unit-id", "test",
                    "--bonus", "none", "--parameter-settings", str(settings),
                    "--copies-per-cell", "128"]
        try:
            build_config(parse_args())
        except ValueError:
            pass
        else:
            raise AssertionError("nonzero no-bonus selected weight accepted")
        settings.write_text(json.dumps({"rnd_next_state": [[1e-3, 100], [1e-4, 1]],
                                        "none": [[1e-3, 0]]}))
        cfg = build_config(parse_args())
        assert cfg.int_coef == 0. and build_sweep(cfg).group_settings == ((1e-3, 0.),)

        # Before: old complete data plus a partial retry; after: no completed score until retry ends.
        start = {"record": "unit_start", "unit_id": "test", "bonus": "rnd_next_state",
                 "cell_settings": [[1e-3, 100]], "copy_seed_index": list(range(128)),
                 "iterations": 50}
        window = {"record": "episode_window", "phase_blocked": True,
                  "last_iteration": 25, "env_steps_per_copy": 12800,
                  "episodes_per_copy_in_window": 32,
                  "reward_ext_sum_per_copy": [64.] * 128, "coverage_per_copy": [.5] * 128}
        complete = {"record": "unit_complete", "iterations": 50}
        shard = run / "data/test.jsonl"
        records = [start, window, complete, start, window]
        shard.write_text("".join(json.dumps(record) + "\n" for record in records))
        result = aggregate(run)
        assert result["units_complete"] == 0 and result["cells"] == []
        assert len(next(iter(result["curves"].values()))) == 1
        with shard.open("a") as handle:
            handle.write(json.dumps(dict(window, phase_blocked=False, last_iteration=26)) + "\n")
            handle.write(json.dumps(complete) + "\n")
        result = aggregate(run)
        row = result["cells"][0]
        assert row["whole_run_reward"] == 2. and row["coverage_percent"] == 50.
        assert row["success_rate"] == 1. and row["whole_run_reward_standard_error"] == 0.
        assert len(next(iter(result["curves"].values()))) == 1
        print("PASS explicit driver settings; completed, partial and retried aggregate records",
              flush=True)


if __name__ == "__main__":
    main()
