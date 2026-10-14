"""Check table formatting and marker replacement without touching the real report."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from write_corrected_report import SETTINGS, draw, inject, table, verify_records
from aggregate_selected_run import best_per_algorithm


def main():
    """Verify the no-bonus zero weight, ranking marks, and preservation of the caption."""
    rows = []
    for index, bonus in enumerate(("rnd_next_state", "none")):
        rows.append({"bonus": bonus, "learning_rate": 1e-3,
                     "intrinsic_weight": 1. if index == 0 else 0., "copies": 128,
                     "whole_run_reward": 2. - index, "whole_run_reward_standard_error": .1,
                     "last_window_reward": 3. - index, "last_window_reward_standard_error": .2,
                     "coverage_percent": 99. - index, "coverage_percent_standard_error": .3,
                     "success_rate": 1. - index})
    output = table(rows)
    assert output.startswith("\\begin{tabular}") and output.endswith("\\end{tabular}")
    assert "no bonus" in output and "\\underline{" in output and "\\boldmath" in output
    assert output.count("128") == 2
    assert len(best_per_algorithm(rows)) == 2
    try:
        best_per_algorithm([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty result ranking accepted")
    begin = "% >>> AUTO-GENERATED TABLE START: test"
    end = "% <<< AUTO-GENERATED TABLE END: test"
    text = "prose\n" + begin + "\nold\n" + end + "\n\\caption{keep this caption}\n"
    result = inject(text, "test", output)
    assert "\nold\n" not in result and result.endswith("\\caption{keep this caption}\n")
    for invalid in ("no markers", text + "\n" + begin):
        try:
            inject(invalid, "test", output)
        except ValueError:
            pass
        else:
            raise AssertionError("missing or duplicate markers accepted")
    # The publication check must reject an incomplete budget or a non-finite recorded loss.
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        # Six corrected settings give six lines; extra trained settings are not plotted.
        import matplotlib.pyplot as plt
        curves = {f"{b}:{r:g}:{w:g}": [
            {"steps": 1e6, "mean": 1., "standard_error": .1},
            {"steps": 1e8, "mean": 2., "standard_error": .2}]
                  for b, r, w in SETTINGS}
        curves["none:0.001:0"] = next(iter(curves.values()))
        closed_figures = []
        original_close = plt.close
        plt.close = lambda figure: closed_figures.append(figure)
        try:
            draw(curves, run / "curves")
        finally:
            plt.close = original_close
        figure = closed_figures[-1]
        assert len(figure.axes[0].lines) == 6
        assert all("corrected" not in line.get_label() for line in figure.axes[0].lines)
        assert (run / "curves.pdf").exists() and (run / "curves.png").exists()
        original_close(figure)
        missing = dict(curves)
        missing.pop(next(iter(missing)))
        try:
            draw(missing, run / "missing")
        except KeyError:
            plt.close("all")
        else:
            raise AssertionError("missing corrected curve accepted")

        (run / "data").mkdir()
        window = {"record": "episode_window", "loss": 1.,
                  "reward_ext_sum_per_copy": [0.] * 128,
                  "rint_mean_per_copy": [0.] * 128, "coverage_per_copy": [.5] * 128}
        complete = {"record": "unit_complete", "iterations": 195325,
                    "env_steps_per_copy": 100006400}
        for index in range(11):
            (run / "data" / f"{index}.jsonl").write_text(
                json.dumps(window) + "\n" + json.dumps(complete) + "\n")
        verify_records(run)
        for invalid_window, invalid_complete in (
                (window, dict(complete, iterations=1)),
                (dict(window, loss=float("nan")), complete)):
            (run / "data/0.jsonl").write_text(json.dumps(invalid_window) + "\n"
                                               + json.dumps(invalid_complete) + "\n")
            try:
                verify_records(run)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid training measurements accepted")
    print("PASS corrected table, no-bonus zero weight, ranking marks and caption preservation")


if __name__ == "__main__":
    main()
