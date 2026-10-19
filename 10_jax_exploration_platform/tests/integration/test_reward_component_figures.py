"""Check units, covariance and missing-data errors in reward-component plots."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from write_reward_component_figures import SETTINGS, episode_components, pooled_summary, verify_extrinsic


def main():
    """Use independent arithmetic to verify the reward units and uncertainty calculation."""
    record = {"episodes_per_copy_in_window": 32,
              "reward_ext_sum_per_copy": [64., 128.],
              "rint_mean_per_copy": [.1, .095]}
    # The components are anticorrelated: (2,40) and (4,38) both sum to 42.
    rows = [episode_components(record, position, 400) for position in (0, 1)]
    assert rows == [(2., 40., 42.), (4., 38., 42.)]
    entry = {"n": 2, "sum": [sum(row[i] for row in rows) for i in range(3)],
             "square_sum": [sum(row[i]**2 for row in rows) for i in range(3)]}
    result = pooled_summary(entry, 2)
    assert result["raw_total"] == 42. and result["raw_total_standard_error"] == 0.
    assert result["extrinsic"] == 3. and result["extrinsic_standard_error"] == 1.
    assert result["raw_intrinsic_standard_error"] == 1.
    # A final shorter window changes the episode count, not the 400-step intrinsic conversion.
    short = dict(record, episodes_per_copy_in_window=16,
                 reward_ext_sum_per_copy=[32., 64.])
    assert episode_components(short, 0, 400) == rows[0]
    for invalid in (dict(record, episodes_per_copy_in_window=0),
                    dict(record, rint_mean_per_copy=[math.nan, 0])):
        try:
            episode_components(invalid, 0, 400)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reward data accepted")
    try:
        pooled_summary(entry, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete seed count accepted")
    # A publication check must detect a change even when the plotted steps still match.
    actual = {f"{b}:{r:g}:{w:g}": [{"steps": 1, "extrinsic": 2.,
                                     "extrinsic_standard_error": .1}] for b, r, w in SETTINGS}
    historical = {"curves": {name: {key: {"env_steps_per_copy": [1],
                       "mean_episode_return": [2.], "standard_error": [.1]} for key in actual}
                             for name in ("first", "second")}}
    cached = {"historical_first": actual, "historical_second": actual, "corrected": actual}
    corrected = {"curves": {key: [{"steps": 1, "mean": 2., "standard_error": .1}]
                            for key in actual}}
    assert verify_extrinsic(cached, historical, corrected)["largest_absolute_difference"] == 0.
    corrected["curves"][next(iter(actual))][0]["mean"] = 3.
    try:
        verify_extrinsic(cached, historical, corrected)
    except ValueError:
        pass
    else:
        raise AssertionError("changed published extrinsic reward accepted")
    print("PASS episode units, shortened windows, combined covariance, finite data and seed count")


if __name__ == "__main__":
    main()
