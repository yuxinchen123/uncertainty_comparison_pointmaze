#!/usr/bin/env python
"""The score rule of train run 6 — ONE definition, imported by everything that scores.

This run has one environment, train run 5's PointMaze task, and uses train run 5's own score:

- initial_single_large_pointmaze_max_400 (writeup subsection "Train run 5")
  -> FINAL REWARD: the last train_history row's `train/mean_extrinsic_reward`, i.e. the mean
     extrinsic return over the last 100 training episodes at the 1,000,000-step evaluation. This
     is the number train run 5's tables report and the number its own bonus-weight race used.

compute_frozen_bars.py, truncation_controller.py, truncation_check.py and the analysis all import
`score_of_record` from here, so the bar and the runs raced against it can never be scored
differently.
"""

# environment -> score-rule name; every environment of this run must appear here
SCORE_RULE = {
    "initial_single_large_pointmaze_max_400": "final_reward",
}

# one-line human description per rule, printed in the bars file and in every decision line
RULE_DESCRIPTION = {
    "final_reward": ("last train_history row's train/mean_extrinsic_reward (the mean extrinsic "
                     "return over the last 100 training episodes at the final evaluation)"),
}


def final_reward(d):
    """Train run 5's score: the last train_history row's windowed training-episode reward."""
    rows = d.get("train_history") or []
    if not rows:
        return None
    return rows[-1].get("train/mean_extrinsic_reward")


_RULES = {"final_reward": final_reward}


def rule_for(env_setup):
    """The score-rule NAME for one environment. Hard-fails on an unknown environment rather than
    guessing — a silently wrong score rule would corrupt every truncation decision."""
    if env_setup not in SCORE_RULE:
        raise KeyError(f"no score rule registered for env_setup {env_setup!r}; "
                       f"known: {sorted(SCORE_RULE)}")
    return SCORE_RULE[env_setup]


def score_of_record(d, env_setup=None):
    """The score of ONE completed per-run record under its environment's rule.

    `env_setup` is taken from the record itself unless given explicitly (train run 5's own records
    predate the --env_setup flag and carry no env_setup field, so compute_frozen_bars.py passes it).
    Returns None when the record has no usable history rows yet.
    """
    env = env_setup if env_setup is not None else d.get("env_setup")
    return _RULES[rule_for(env)](d)
