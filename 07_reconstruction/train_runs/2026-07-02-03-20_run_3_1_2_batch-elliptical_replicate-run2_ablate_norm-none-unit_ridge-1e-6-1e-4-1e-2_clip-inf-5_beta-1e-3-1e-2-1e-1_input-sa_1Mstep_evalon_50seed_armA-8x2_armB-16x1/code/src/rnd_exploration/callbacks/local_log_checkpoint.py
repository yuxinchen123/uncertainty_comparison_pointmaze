"""Checkpoint the per-run local JSON at the eval cadence, so a run killed mid-training (job walltime,
node failure, per-run timeout) leaves its partial curves instead of nothing. The flush callable is
provided by the driver (train.py) and atomically replaces the run's OWN file; this callback only decides
WHEN to fire. Run 3.1.1 lost 488 part-trained runs to walltime kills under the old write-once-at-end
convention — this callback is the fix (run-id-and-logging.md)."""
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


class LocalLogCheckpointCallback(BaseCallback):
    """Call `flush()` every `eval_freq` steps. Append LAST in the callback list so the flush sees the
    same step's freshly appended history rows from the callbacks that ran before it."""

    def __init__(self, eval_freq: int, flush: Callable[[], None], verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.flush = flush

    def _on_step(self) -> bool:
        # fire on the eval cadence only (eval_freq<=0 disables, mirroring the other callbacks)
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.flush()
        return True
