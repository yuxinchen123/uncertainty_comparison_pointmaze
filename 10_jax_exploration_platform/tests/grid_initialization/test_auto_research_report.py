"""Tests of the auto research report's confirmation table: pooled seeds and the paired difference to the baseline."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/grid_initialization_campaign'))
import auto_research_report as report  # noqa: E402


def records(scores, start=20000):
    """Build per-seed confirmation records: four policy seeds per field seed."""
    return {start + index: {'policy_seed': start + index, 'field_seed': (start + index) // 4,
                            'capped_first_goal_steps': float(score), 'first_goal_step': int(score), 'coverage': 1.0,
                            'evaluation_goal_rate_sampled': 1.0, 'evaluation_goal_rate_greedy': 0.5}
            for index, score in enumerate(scores)}


def test_confirmation_row_pairs_the_idea_with_the_baseline_seed_by_seed():
    """The difference is the mean of per-seed differences on the seeds both have; its interval excludes zero here."""
    generator = np.random.default_rng(0)
    base_scores = generator.uniform(8000, 12000, 64)
    baseline = records(base_scores)
    idea = records(base_scores[:48] - 2000)
    row = report.confirmation_row(idea, baseline)
    assert row['seeds'] == 48 and row['paired']['seeds'] == 48
    assert row['paired']['mean'] == -2000.0 and row['paired']['interval'] == [-2000.0, -2000.0]
    assert abs(row['paired']['baseline_mean'] - base_scores[:48].mean()) < 1e-6
    assert row['discovery'] == 1.0 and row['greedy'] == 0.5 and row['standard_error'] > 0
    # The baseline's own row, and a row without a finished baseline, carry no difference.
    assert report.confirmation_row(baseline, baseline)['paired'] is None
    assert report.confirmation_row(idea, None)['paired'] is None


def test_confirmation_table_sorts_by_the_pooled_mean_and_names_the_baseline():
    """Rows are sorted by the pooled mean; the best is bold; the baseline row says so."""
    setting = {'learning_rate': 1e-3, 'entropy_coefficient': 0.01, 'intrinsic_coefficient': 0.25}
    scores = np.linspace(9000, 11000, 16)
    pooled = {('06', 'control', 'snake', 'neural'): records(scores),
              ('06', 'new-idea', 'snake', 'neural'): records(scores - 1500),
              ('06', 'new-idea', 'straight', 'neural'): records(scores)}
    configurations = {key: setting for key in pooled}
    table = report.confirmation_table(pooled, configurations, '06', 'snake', 'neural', {'new-idea': 'the new idea'})
    lines = table.strip().splitlines()
    assert len(lines) == 4 and lines[2].startswith('| the new idea | **8.50K**') and '| baseline |' in lines[3]
    assert '-1.50K [-1.50K, -1.50K], -15% (16 seeds)' in lines[2] and 'weight 2^-2' in lines[2]
    alone = report.confirmation_table(pooled, configurations, '06', 'straight', 'neural', {})
    assert 'baseline not finished' in alone
