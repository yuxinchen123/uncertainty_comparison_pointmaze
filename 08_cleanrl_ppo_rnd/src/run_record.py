"""One JSON record per training run, flushed atomically, in the shape train run 8.1.2 uses.

The project logs a run as a single JSON file under `data/<sweep_id>/local/<run_id>_of_<total>.json`
holding the run's identity and configuration at the top level plus a small number of history lists.
The file is rewritten in full at every checkpoint, so a run that is killed still leaves a readable
partial record; `completed` says whether the run reached its end. This module carries that
convention for the CleanRL PPO + RND trainer, which has no Weights and Biases and no tensorboard.

The history lists here are the PPO analogues of the ones in run 8.1.2:

| run 8.1.2 (SAC)        | here (PPO + RND)                                                    |
|------------------------|---------------------------------------------------------------------|
| `train_episode_history`| one entry per finished Atari episode: extrinsic return, length, lives |
| `train_history`        | one entry per policy update: means over the update's rollout          |
| `eval_history`         | one entry per policy update: throughput and optimisation diagnostics  |
| `distance_history`     | not applicable — there is no oracle visit-count field on Atari        |

`train_episode_history` is capped, because a 2e9-step Atari run finishes millions of episodes and an
uncapped list would outgrow the filesystem. Past the cap the record keeps every `episode_stride`-th
episode, so the curve stays complete in shape while the file stays bounded.
"""

import json
import os
import time


class RunRecord:
    """Accumulate one training run's config and history, and flush it atomically to one JSON file."""

    def __init__(self, path, config, episode_history_cap=200_000, episode_stride=25):
        """Open a record at `path` for a run described by the `config` dict."""
        # `config` becomes the record's top-level fields verbatim, so the caller decides what
        # identifies a run. The history lists start empty and grow as the run reports.
        self.path = path
        self.config = dict(config)
        self.episode_history_cap = episode_history_cap
        self.episode_stride = episode_stride
        self.train_episode_history = []
        self.train_history = []
        self.eval_history = []
        # Counts every episode the run finished, including the ones the cap dropped, so an analysis
        # can tell "few episodes happened" from "most episodes were not kept".
        self.episodes_seen = 0
        self.episodes_dropped = 0
        self.start_time = time.time()
        # Wall-clock carried over from earlier segments of a resumed run, so `runtime_seconds`
        # measures the run and not the current Slurm job.
        self.prior_runtime_seconds = 0.0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add_episode(self, entry):
        """Record one finished Atari episode, applying the cap-and-stride rule past the cap."""
        # before: episodes_seen=200_000, cap reached, stride=25
        # after:  only every 25th further episode is appended; the rest bump episodes_dropped
        self.episodes_seen += 1
        if len(self.train_episode_history) < self.episode_history_cap:
            self.train_episode_history.append(entry)
        elif self.episodes_seen % self.episode_stride == 0:
            self.train_episode_history.append(entry)
        else:
            self.episodes_dropped += 1

    def add_update(self, train_entry, eval_entry):
        """Record one policy update: its rollout means and its throughput/optimisation diagnostics."""
        self.train_history.append(train_entry)
        self.eval_history.append(eval_entry)

    def as_dict(self, completed):
        """Build the full record dict, with the history lists last so a flush sees fresh rows."""
        # The top-level ordering mirrors run 8.1.2: identity and config first, then runtime, then
        # the history lists.
        out = dict(self.config)
        out["completed"] = completed
        out["episodes_seen"] = self.episodes_seen
        out["episodes_dropped_from_history"] = self.episodes_dropped
        out["episode_history_cap"] = self.episode_history_cap
        out["episode_history_stride_past_cap"] = self.episode_stride
        out["runtime_seconds"] = self.prior_runtime_seconds + (time.time() - self.start_time)
        out["eval_history"] = self.eval_history
        out["train_history"] = self.train_history
        out["train_episode_history"] = self.train_episode_history
        return out

    def flush(self, completed=False):
        """Write the whole record to a temp file and rename it over the target, never truncating it."""
        # A plain open() honours the directory's default ACL, which is what lets a collaborator's
        # analysis read the file; tempfile.mkstemp would force mode 600 and silently break that.
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.as_dict(completed), f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o660)
        # os.replace is atomic on the same filesystem: a reader sees either the old record or the
        # new one, never a half-written file.
        os.replace(tmp, self.path)

    def restore_from_disk(self):
        """Reload an existing record so a resumed run appends to it instead of starting over.

        Returns True when a record was found and loaded, False when this is a fresh run.
        """
        if not os.path.exists(self.path):
            return False
        with open(self.path) as f:
            old = json.load(f)
        # Take the history and the counters back; the config comes from the current invocation so a
        # changed argument is visible rather than silently overwritten by the old value.
        self.train_episode_history = old.get("train_episode_history", [])
        self.train_history = old.get("train_history", [])
        self.eval_history = old.get("eval_history", [])
        self.episodes_seen = old.get("episodes_seen", len(self.train_episode_history))
        self.episodes_dropped = old.get("episodes_dropped_from_history", 0)
        self.prior_runtime_seconds = old.get("runtime_seconds", 0.0)
        return True


def record_filename(run_id, run_total):
    """Build the per-run JSON filename the project's analysis loader expects."""
    return f"{run_id}_of_{run_total}.json"
