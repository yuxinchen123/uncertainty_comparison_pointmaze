"""One JSON record per training run, flushed atomically, in the shape train run 8.1.2 uses.

The project logs a run as a single JSON file under `data/<sweep_id>/local/<run_id>_of_<total>.json`
holding the run's identity and configuration at the top level plus a small number of history lists.
The file is rewritten in full at every checkpoint, so a run that is killed still leaves a readable
partial record; `completed` says whether the run reached its end.

Two departures from that shape, both forced by the size of this campaign (150 runs of 2e9 steps):

**The per-episode history lives in a sidecar, not in the record.** A full run finishes on the order
of 4.7 million Atari episodes. Keeping even the capped-and-strided 96,000 of them inside the record
means every flush rewrites 11 MB of episodes that have not changed. Measured over a run that is the
dominant cost: at a 25-update cadence the campaign would rewrite **7.9 TB** onto a 1 TB shared
filesystem to produce 2.4 GB of final data. The episodes now go to an append-only
`<run>.episodes.jsonl` beside the record, so a flush writes only what is new.

**The record is serialised with `json.dumps` and one `write`.** `json.dump` to a file object makes
one small write per token; measured 2.7 times slower at these sizes, and the cost is Python
serialisation rather than the disk.

The history lists here are the PPO analogues of the ones in run 8.1.2:

| run 8.1.2 (SAC)        | here (PPO + RND)                                                     |
|------------------------|----------------------------------------------------------------------|
| `train_episode_history`| the sidecar: one line per kept Atari episode                          |
| `train_history`        | one entry per logging interval: means over the whole interval         |
| `eval_history`         | one entry per logging interval: throughput, losses, gradient statistics |
| `distance_history`     | not applicable — there is no oracle visit-count field on Atari        |
"""

import json
import os
import time


class RunRecord:
    """Accumulate one training run's config and history, and flush it atomically to one JSON file."""

    def __init__(self, path, config, episode_history_cap=50_000, episode_stride=100):
        """Open a record at `path` for a run described by the `config` dict."""
        # `config` becomes the record's top-level fields verbatim, so the caller decides what
        # identifies a run. The history lists start empty and grow as the run reports.
        self.path = path
        self.episodes_path = path[: -len(".json")] + ".episodes.jsonl"
        self.config = dict(config)
        self.episode_history_cap = episode_history_cap
        self.episode_stride = episode_stride
        self.train_history = []
        self.eval_history = []
        # Episodes are buffered in memory and appended to the sidecar at each flush, so a flush
        # writes only the new lines rather than rewriting every episode ever recorded.
        self._episode_buffer = []
        self.episodes_kept = 0
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
        # before: episodes_seen=50_000, cap reached, stride=100
        # after:  only every 100th further episode is kept; the rest bump episodes_dropped
        self.episodes_seen += 1
        if self.episodes_kept < self.episode_history_cap or self.episodes_seen % self.episode_stride == 0:
            self._episode_buffer.append(entry)
            self.episodes_kept += 1
        else:
            self.episodes_dropped += 1

    def add_update(self, train_entry, eval_entry):
        """Record one logging interval: its training means and its diagnostics."""
        self.train_history.append(train_entry)
        self.eval_history.append(eval_entry)

    def as_dict(self, completed):
        """Build the record dict, with the history lists last so a flush sees fresh rows."""
        # The top-level ordering mirrors run 8.1.2: identity and config first, then runtime, then
        # the history lists.
        out = dict(self.config)
        out["completed"] = completed
        out["episodes_seen"] = self.episodes_seen
        out["episodes_kept"] = self.episodes_kept
        out["episodes_dropped_from_history"] = self.episodes_dropped
        out["episode_history_cap"] = self.episode_history_cap
        out["episode_history_stride_past_cap"] = self.episode_stride
        # The per-episode rows are in the sidecar; this names it so a reader is never left guessing.
        out["train_episode_history_file"] = os.path.basename(self.episodes_path)
        out["runtime_seconds"] = self.prior_runtime_seconds + (time.time() - self.start_time)
        out["eval_history"] = self.eval_history
        out["train_history"] = self.train_history
        return out

    def flush(self, completed=False):
        """Append any new episodes, then write the record over its target without truncating it."""
        # Episodes first: appending them before the record means the record's episode counts are
        # never ahead of the sidecar's contents.
        if self._episode_buffer:
            with open(self.episodes_path, "a") as f:
                f.write("".join(json.dumps(e) + "\n" for e in self._episode_buffer))
                f.flush()
                os.fsync(f.fileno())
            self._episode_buffer = []
            os.chmod(self.episodes_path, 0o660)

        # One dumps, one write. json.dump to a file object issues a small write per token and is
        # measured 2.7 times slower at these sizes.
        payload = json.dumps(self.as_dict(completed))
        tmp = self.path + ".tmp"
        # A plain open() honours the directory's default ACL, which is what lets a collaborator's
        # analysis read the file; tempfile.mkstemp would force mode 600 and silently break that.
        with open(tmp, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o660)
        # os.replace is atomic on the same filesystem: a reader sees either the old record or the
        # new one, never a half-written file.
        os.replace(tmp, self.path)

    def restore_for_resume(self, checkpoint_history, resume_update, resume_step):
        """Rebuild the history a resumed run continues from, truncated to the checkpoint's position.

        The run continues from the checkpoint, so the history must end at the checkpoint. Anything
        logged after it describes updates the resumed run is about to redo; keeping those rows would
        splice a discarded trajectory onto the kept one, duplicate every row in the overlap, and make
        the step axis fold backwards.

        Two sources, in this order of preference:

        1. **The history inside the checkpoint** (a format-2 checkpoint). It is exact by
           construction — it was captured at the very moment the training state was.
        2. **The JSON record on disk**, truncated to the checkpoint's position. Used when the
           checkpoint predates this mechanism, or when there is no checkpoint at all.

        The JSON is the weaker source precisely because it is FRESHER than the checkpoint: it is
        flushed every log step while the checkpoint is written on a slower cadence.

        Returns a one-line description of what happened, for the run's log.
        """
        # before: json has update rows [25, 50, 75, 100], checkpoint stopped at update 60
        # after:  rows [25, 50] are kept and [75, 100] dropped, because 75 and 100 will be redone
        if checkpoint_history is not None:
            self.train_history = checkpoint_history["train_history"]
            self.eval_history = checkpoint_history["eval_history"]
            self.episodes_seen = checkpoint_history["episodes_seen"]
            self.episodes_kept = checkpoint_history.get("episodes_kept", 0)
            self.episodes_dropped = checkpoint_history["episodes_dropped"]
            self.prior_runtime_seconds = checkpoint_history["runtime_seconds"]
            note = (f"history from the checkpoint: {len(self.train_history)} interval rows, "
                    f"{self.episodes_seen} episodes seen")
        elif os.path.exists(self.path):
            with open(self.path) as f:
                old = json.load(f)
            # The config comes from the current invocation, so a changed argument stays visible
            # rather than being silently overwritten by the old value.
            train_all, eval_all = old.get("train_history", []), old.get("eval_history", [])
            self.train_history = [r for r in train_all if r.get("update", 0) <= resume_update]
            self.eval_history = [r for r in eval_all if r.get("update", 0) <= resume_update]
            self.episodes_seen = old.get("episodes_seen", 0)
            self.episodes_kept = old.get("episodes_kept", 0)
            self.episodes_dropped = old.get("episodes_dropped_from_history", 0)
            self.prior_runtime_seconds = old.get("runtime_seconds", 0.0)
            note = (f"history from the record on disk, truncated to update {resume_update}: "
                    f"{len(self.train_history)} interval rows kept, "
                    f"{len(train_all) - len(self.train_history)} dropped past the checkpoint")
        else:
            return "no history to restore: neither the checkpoint nor the record carried any"

        # The sidecar is append-only, so a resume must cut it back to the checkpoint too or the
        # episodes of the discarded segment stay in front of the ones about to be re-recorded.
        note += "; " + self._truncate_episodes(resume_step)
        return note

    def _truncate_episodes(self, resume_step):
        """Cut the episode sidecar back to the checkpoint's step, rewriting it once."""
        if not os.path.exists(self.episodes_path):
            self.episodes_kept = 0
            return "no episode sidecar to truncate"
        kept, dropped = [], 0
        with open(self.episodes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if json.loads(line).get("step", 0) <= resume_step:
                    kept.append(line)
                else:
                    dropped += 1
        tmp = self.episodes_path + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(k + "\n" for k in kept))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o660)
        os.replace(tmp, self.episodes_path)
        self.episodes_kept = len(kept)
        return f"episode sidecar cut to {len(kept)} rows, {dropped} dropped past the checkpoint"


def record_filename(run_id, run_total):
    """Build the per-run JSON filename the project's analysis loader expects."""
    return f"{run_id}_of_{run_total}.json"
