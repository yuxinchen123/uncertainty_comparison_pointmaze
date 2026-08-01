"""Pytest wrapper for the truncation simulation (the launch gate). The quick default (60 configs)
runs in the suite; the full 240-config gate is exercised by running simulate_truncation.py --full
before launch (recorded in experiment_background.md)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulate_truncation  # noqa: E402


def test_truncation_simulation_60_configs():
    """The simulation of the REAL controller on 60 synthetic configs reports zero failures."""
    assert simulate_truncation.run_simulation(60, base_seed=0, verbose=False) == []


def test_truncation_simulation_other_seed():
    """Edge: a different arrival-order seed also passes (no order dependence)."""
    assert simulate_truncation.run_simulation(24, base_seed=7, verbose=False) == []
