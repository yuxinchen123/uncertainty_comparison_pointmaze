#!/usr/bin/env python
"""The per-environment score rule of this run — ONE definition, imported by everything that scores.

This run spans two writeup sections that score a run differently, and the user's instruction is that
each environment is judged against ITS OWN section's Adam 1e-4 number. So the score of a completed
per-run record depends on which environment it came from:

- initial_single_large_pointmaze_max_400 (train run 5, writeup subsection "Train run 5")
  -> FINAL REWARD: the last train_history row's `train/mean_extrinsic_reward`, i.e. the mean
     extrinsic return over the last 100 training episodes at the 1,000,000-step evaluation. This is
     the number train run 5's tables report and the number its own beta race used.
- AntMaze_UMaze-v5_start_bottom_left and AntMaze_Medium-v5_start_bottom_left (train run 1.2)
  -> WHOLE-RUN MEAN: the mean of `train/extrinsic_reward` over EVERY row of train_episode_history,
     i.e. the mean per-episode extrinsic return over all of the run's training episodes. This is the
     number train run 1.1 raced on and the number train run 1.2's frozen bars are computed from.

compute_frozen_bars.py, truncation_controller.py, truncation_check.py and the analysis all import
`score_of_record` from here, so the bar and the runs raced against it can never be scored differently.
"""

# environment -> score-rule name; every environment of this run must appear here
SCORE_RULE = {
    "initial_single_large_pointmaze_max_400": "final_reward",
    "AntMaze_UMaze-v5_start_bottom_left": "whole_run_mean",
    "AntMaze_Medium-v5_start_bottom_left": "whole_run_mean",
}

# one-line human description per rule, printed in the bars file and in every decision line
RULE_DESCRIPTION = {
    "final_reward": ("last train_history row's train/mean_extrinsic_reward (the mean extrinsic "
                     "return over the last 100 training episodes at the final evaluation)"),
    "whole_run_mean": ("mean of train/extrinsic_reward over all train_episode_history rows (the "
                       "mean per-episode extrinsic return over every training episode)"),
}


def final_reward(d):
    """Train run 5's score: the last train_history row's windowed training-episode reward."""
    rows = d.get("train_history") or []
    if not rows:
        return None
    last = rows[-1]
    return last.get("train/mean_extrinsic_reward")


def whole_run_mean(d):
    """Train run 1.1/1.2's score: the mean per-episode extrinsic return over all training episodes."""
    vals = [r["train/extrinsic_reward"] for r in (d.get("train_episode_history") or [])
            if "train/extrinsic_reward" in r]
    if not vals:
        return None
    return sum(vals) / len(vals)


_RULES = {"final_reward": final_reward, "whole_run_mean": whole_run_mean}


def rule_for(env_setup):
    """The score-rule NAME for one environment. Hard-fails on an unknown environment rather than
    guessing a rule — a silently wrong score rule would corrupt every truncation decision."""
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
