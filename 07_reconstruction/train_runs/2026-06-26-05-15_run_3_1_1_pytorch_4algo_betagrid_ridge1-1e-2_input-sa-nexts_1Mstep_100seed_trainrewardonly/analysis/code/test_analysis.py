"""Unit tests for the Train-run-3.1.1 analysis logic: method mapping, config grouping, per-method
best-config selection (training reward), and LaTeX formatting. Run: `pytest test_analysis.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import make_reward_table as T  # noqa: E402


def _rec(algorithm, beta, seed, reward, timing=None, inp=None, ridge=None):
    """Build a RunRecord with a one-point training curve at the final step (reward) for the tests."""
    return C.RunRecord(
        algorithm=algorithm, beta=beta, seed=seed, update_timing=timing, feature_input=inp,
        regularization=ridge, runtime_seconds=1.0, final_train_reward=reward,
        train_curve=[(1000000, reward)],
    )


def test_method_id_mapping():
    """Each algorithm/timing maps to the right method id; global splits on update_timing; gt is the oracle."""
    assert _rec("rnd_elliptical", 1.0, 0, 0.0).method_id() == "A1_batch"
    assert _rec("rnd_elliptical_global", 1.0, 0, 0.0, timing="sample").method_id() == "A2_global_sample"
    assert _rec("rnd_elliptical_global", 1.0, 0, 0.0, timing="add").method_id() == "A3_global_add"
    assert _rec("rnd_next_state", 1.0, 0, 0.0).method_id() == "A4_rnd_next_state"
    assert _rec("gt_position_velocity", 1.0, 0, 0.0).method_id() == C.GT_METHOD
    assert _rec("no_exploration", 1.0, 0, 0.0).method_id() is None


def test_config_id_distinguishes_axes():
    """A2 and A3 (same algorithm) get distinct config ids via timing; ridge/input also separate configs."""
    a2 = _rec("rnd_elliptical_global", 1.0, 0, 0.0, timing="sample", inp="state_action", ridge=1.0)
    a3 = _rec("rnd_elliptical_global", 1.0, 0, 0.0, timing="add", inp="state_action", ridge=1.0)
    assert a2.config_id() != a3.config_id()
    a2b = _rec("rnd_elliptical_global", 1.0, 0, 0.0, timing="sample", inp="next_state", ridge=1.0)
    assert a2.config_id() != a2b.config_id()  # different input -> different config
    # RND has no ridge/input: config_id renders them as the literal 'none' (not None/NaN)
    rnd = _rec("rnd_next_state", 1.0, 0, 0.0)
    assert rnd.config_id() == "A4_rnd_next_state|beta=1|ridge=none|input=none"


def test_best_config_per_method_picks_highest_mean_above_min_seeds():
    """Per method the highest-mean config with >= MIN_SEEDS seeds wins; under-seeded configs are excluded."""
    recs = []
    # A1 config X (beta=1, ridge=1, state_action): 30 seeds, mean reward 5.0  -> qualifies
    for s in range(30):
        recs.append(_rec("rnd_elliptical", 1.0, s, 5.0, timing="sample", inp="state_action", ridge=1.0))
    # A1 config Y (beta=10, ridge=1, state_action): only 5 seeds but mean 9.0 -> excluded (< MIN_SEEDS)
    for s in range(5):
        recs.append(_rec("rnd_elliptical", 10.0, s, 9.0, timing="sample", inp="state_action", ridge=1.0))
    # A4 RND (beta=100): 30 seeds, mean 3.0
    for s in range(30):
        recs.append(_rec("rnd_next_state", 100.0, s, 3.0))
    best = T.best_config_per_method(T.final_reward_frame(recs))
    by_method = {r["method"]: r for _, r in best.iterrows()}
    # A1 picks config X (the qualifying one), NOT the higher-mean-but-under-seeded Y
    assert by_method["A1_batch"]["beta"] == 1.0 and abs(by_method["A1_batch"]["Rbar"] - 5.0) < 1e-9
    assert by_method["A1_batch"]["n"] == 30
    # A4 present with its single config
    assert by_method["A4_rnd_next_state"]["beta"] == 100.0
    # rows are sorted by Rbar descending: A1 (5.0) before A4 (3.0)
    assert list(best["method"]) == ["A1_batch", "A4_rnd_next_state"]


def test_curve_config_id_matches_records_including_rnd():
    """The config_id rebuilt from a best-config row matches the records' config_id (RND's 'none' ridge/input
    included), so make_reward_curve finds a training curve for every method -- the bug the string id fixes."""
    import make_reward_curve as Cv
    recs = []
    for s in range(30):
        recs.append(_rec("rnd_elliptical", 1.0, s, 5.0, timing="sample", inp="state_action", ridge=1.0))
        recs.append(_rec("rnd_next_state", 100.0, s, 3.0))
    best = T.best_config_per_method(T.final_reward_frame(recs))
    ids_in_long = set(C.train_curve_frame(recs)["config_id"])
    for _, row in best.iterrows():
        assert Cv._row_config_id(row) in ids_in_long  # rebuilt id equals a real record's config_id
    # the curve builder produces a curve for BOTH methods, including RND (none/None match works)
    curves = Cv.method_curves(recs)
    assert set(curves) == {"A1_batch", "A4_rnd_next_state"}


def test_rank_marks_and_pow10_format():
    """rank_marks flags best+second (higher better); _fmt_pow10 renders clean powers of ten."""
    best, second = C.rank_marks([5.0, 3.0, 9.0], lower_is_better=False)
    assert best == {2} and second == {0}
    assert T._fmt_pow10(0.001) == "$10^{-3}$"
    assert T._fmt_pow10(1.0) == "$1$"
    assert T._fmt_pow10(1000.0) == "$10^{3}$"
    assert T._fmt_pow10(None) == "---"


def test_tabular_emits_four_rows_and_marks():
    """The tabular has one row per present method, with the best Rbar bold and second underlined."""
    recs = []
    for algo, beta, mean, timing in [("rnd_elliptical", 1.0, 5.0, "sample"),
                                     ("rnd_elliptical_global", 1.0, 8.0, "sample"),
                                     ("rnd_elliptical_global", 1.0, 2.0, "add"),
                                     ("rnd_next_state", 1.0, 4.0, None)]:
        for s in range(30):
            recs.append(_rec(algo, beta, s, mean, timing=timing,
                             inp=None if algo == "rnd_next_state" else "state_action",
                             ridge=None if algo == "rnd_next_state" else 1.0))
    summary = T.best_config_per_method(T.final_reward_frame(recs))
    tex = T.reward_tabular(summary)
    # all four method rows are present (each method's display string appears once)
    for disp in T.METHOD_DISPLAY.values():
        assert disp in tex
    assert "\\textbf{8.00}" in tex   # best mean bold (A2, 8.0)
    assert "\\underline{5.00}" in tex  # second-best underlined (A1, 5.0)
