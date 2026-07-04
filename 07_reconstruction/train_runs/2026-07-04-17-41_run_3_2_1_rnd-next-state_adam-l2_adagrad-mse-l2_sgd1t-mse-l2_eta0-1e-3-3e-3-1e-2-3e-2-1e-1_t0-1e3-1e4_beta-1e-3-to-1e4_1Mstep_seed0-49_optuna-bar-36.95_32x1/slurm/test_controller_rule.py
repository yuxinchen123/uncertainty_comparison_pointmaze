"""Unit tests for the controller's stopping rule (optuna_controller.compute_stats / decide).

Run with:  conda run -n exploration python -m pytest slurm/test_controller_rule.py  (from the run folder)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optuna_controller as ctl

K_ADAM = ctl.KEY_OF_INDEX[0]      # adam|l2|-|-|0.001
K_ADA = ctl.KEY_OF_INDEX[8]       # adagrad|mse|-|-|0.001
K_SGD = ctl.KEY_OF_INDEX[24]      # sgd1t|mse|0.001|1000|0.001


def counts(pending=None, running=None, done=None, failed=None, pruned=None):
    """Build a queue_counts dict with per-key counts (None -> empty)."""
    return {"pending": pending or {}, "running": running or {}, "done": done or {},
            "failed": failed or {}, "pruned": pruned or {}}


def test_compute_stats_formula():
    """n/mean/sd/U match the hand formula U = mean + 2.576*s/sqrt(n); n<2 gives an infinite U."""
    # golden path: rewards 0..9 -> mean 4.5, sd = sqrt(55/6) via the n-1 denominator
    n, mean, sd, upper = ctl.compute_stats(list(range(10)))
    assert (n, mean) == (10, 4.5)
    assert sd == math.sqrt(sum((r - 4.5) ** 2 for r in range(10)) / 9)
    assert upper == 4.5 + 2.576 * sd / math.sqrt(10)
    # edge: a single seed cannot bound anything -> U infinite (never stopped)
    assert ctl.compute_stats([3.0])[3] == float("inf")


def test_decide_bar_rule_floor_and_boundary():
    """Stops need n >= 10 AND U strictly below the bar; a 9-seed config is never stopped."""
    bar = 36.95
    # 10 identical zeros: sd 0, U 0 < bar -> stop. 9 zeros: below the floor -> no decision.
    rewards = {K_ADAM: [0.0] * 10, K_ADA: [0.0] * 9}
    qc = counts(pending={K_ADAM: 40, K_ADA: 41})
    decisions = dict((k, a) for k, a, _ in ctl.decide(rewards, qc, bar, set()))
    assert decisions.get(K_ADAM) == "stop"
    assert K_ADA not in decisions
    # boundary: U exactly == bar is NOT a stop (strict <); constant rewards at the bar stay racing
    rewards = {K_ADAM: [bar] * 10}
    assert ctl.decide(rewards, counts(pending={K_ADAM: 40}), bar, set()) == []


def test_decide_complete_and_cap():
    """A config at the 50-seed cap, or with no pending+running work left, is told complete."""
    bar = 36.95
    rewards = {K_ADAM: [50.0] * 50, K_ADA: [50.0] * 12}
    # K_ADAM hit the cap; K_ADA has no queue entries left (failures ate the rest)
    qc = counts(pending={}, running={}, done={K_ADA: 12}, failed={K_ADA: 38})
    decisions = dict((k, a) for k, a, _ in ctl.decide(rewards, qc, bar, set()))
    assert decisions.get(K_ADAM) == "complete"
    assert decisions.get(K_ADA) == "complete"


def test_decide_idempotent_and_exhausted():
    """Already-decided keys are skipped; all-entries-consumed with zero data is 'exhausted'."""
    bar = 36.95
    rewards = {K_ADAM: [0.0] * 10}
    qc = counts(pending={K_ADAM: 40}, failed={K_SGD: 50})
    # K_ADAM already decided -> nothing again (idempotence); K_SGD consumed all 50 with no data
    decisions = ctl.decide(rewards, qc, bar, already_decided={K_ADAM})
    actions = {k: a for k, a, _ in decisions}
    assert K_ADAM not in actions
    assert actions.get(K_SGD) == "exhausted"
    # a key with NO entries anywhere (queue not built / nothing consumed) is never 'exhausted'
    assert K_ADA not in actions
