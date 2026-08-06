"""Save and restore everything a PPO + RND run needs to continue after being killed.

The GPU partitions cap a job at 4 days, and a full CleanRL RND run is far longer than that, so a run
must survive being stopped and restarted. PPO is on-policy: there is no replay buffer to carry, and
the rollout buffer is rebuilt from scratch at the start of every update. What must be carried is the
learned state, the normalisation statistics, and the random-number streams.

What is saved
-------------
| item                     | why it must be saved                                                    |
|--------------------------|-------------------------------------------------------------------------|
| agent weights            | the policy and the two value heads                                       |
| RND predictor weights    | the network being distilled                                              |
| RND target weights       | randomly initialised and never trained, but it *defines* the bonus — a fresh target would silently restart exploration |
| optimizer state          | Adam's first and second moments over agent + predictor                   |
| observation normaliser   | mean, variance and count of the running statistics used to whiten the RND input |
| intrinsic reward normaliser | mean, variance and count used to scale the curiosity reward           |
| reward forward filter    | the discounted-return accumulator behind the intrinsic reward scale      |
| global step, update index| where the run is, and where the learning-rate schedule is                |
| recent-return window     | the deque behind the reported average episodic return                    |
| random streams           | python, numpy, torch cpu and torch cuda generator states                 |

What cannot be saved
--------------------
The envpool emulator state. envpool exposes no way to serialise the 128 Atari machines, so a resumed
run starts its environments from a fresh reset. The cost is bounded: at most one partial episode per
environment is lost per resume, and with a checkpoint every 8 hours that is a small fraction of a
run. This is a real gap, not a rounding error, and it is recorded in the run's background document.

The most recent checkpoint is written to a temp file and renamed into place, then the previous
checkpoint file is deleted — so there is exactly one checkpoint per run on disk, and a kill during a
write can never leave the run without a readable one.
"""

import os
import random
import time

import numpy as np
import torch


def _rms_state(rms):
    """Extract a gym RunningMeanStd into plain arrays that torch.save can hold."""
    return {"mean": np.asarray(rms.mean), "var": np.asarray(rms.var), "count": float(rms.count)}


def _load_rms(rms, state):
    """Write a saved RunningMeanStd state back into a live RunningMeanStd object."""
    rms.mean, rms.var, rms.count = state["mean"], state["var"], state["count"]


def save_checkpoint(path, *, agent, rnd_model, optimizer, obs_rms, reward_rms, discounted_reward,
                    global_step, update, avg_returns, device, record=None):
    """Write one checkpoint atomically and delete the previous one, keeping exactly one on disk.

    When `record` is given, the run's logged history rides along inside the checkpoint. That is not
    decoration: the sweep's requeue moves the JSON record out of the way when a worker dies, so
    without this the training state resumes correctly while the reward curve restarts empty. The
    history stored here is the history as of THIS checkpoint, which is exactly the point the run
    would continue from, so the two can never disagree.
    """
    # Collect every piece of state. The target network is included deliberately: it is frozen, so a
    # naive checkpoint that skips it would resume with a different random target and reset the
    # exploration signal.
    payload = {
        # 1 = no history carried. 2 = history carried. The reader accepts both, because checkpoints
        # written by an already-running process stay at 1 for that process's whole life.
        "format_version": 2 if record is not None else 1,
        "agent": agent.state_dict(),
        "rnd_model": rnd_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_rms": _rms_state(obs_rms),
        "reward_rms": _rms_state(reward_rms),
        "reward_forward_filter": None if discounted_reward.rewems is None else np.asarray(discounted_reward.rewems),
        "global_step": int(global_step),
        "update": int(update),
        "avg_returns": list(avg_returns),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    if record is not None:
        # The history as of this checkpoint, and the counters that go with it. The counters must
        # come from here rather than from the JSON: past the episode cap the kept rows are strided,
        # so the true episode count cannot be recovered by counting rows.
        # The per-episode rows are NOT carried here. They live in an append-only sidecar which is
        # not the run's completion marker, so the sweep's requeue leaves it in place; the resume
        # cuts it back to this checkpoint's step instead. Only the interval history and the counters
        # need carrying, which is why this costs kilobytes rather than megabytes.
        payload["record_history"] = {
            "train_history": record.train_history,
            "eval_history": record.eval_history,
            "episodes_seen": record.episodes_seen,
            "episodes_kept": record.episodes_kept,
            "episodes_dropped": record.episodes_dropped,
            "runtime_seconds": record.prior_runtime_seconds + (time.time() - record.start_time),
        }

    # Write beside the target, fsync, then rename: a reader (or a restart) sees either the previous
    # complete checkpoint or the new complete one, never a partial file.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o660)
    previous_exists = os.path.exists(path)
    os.replace(tmp, path)

    # "Delete the previous one" is satisfied by the rename itself — there is one checkpoint file per
    # run and it is overwritten in place. Any stale numbered checkpoint from an older scheme is
    # removed here so the run folder never accumulates them.
    stale = path + ".previous"
    if os.path.exists(stale):
        os.remove(stale)
    return {"path": path, "bytes": os.path.getsize(path), "replaced_previous": previous_exists}


def load_checkpoint(path, *, agent, rnd_model, optimizer, obs_rms, reward_rms, discounted_reward,
                    device):
    """Restore a checkpoint in place and return (global_step, update, avg_returns), or None."""
    # A missing file is the normal fresh-start case, not an error.
    if not os.path.exists(path):
        return None

    payload = torch.load(path, map_location=device, weights_only=False)
    # Accept both versions. A checkpoint written by a process that started before the history was
    # added is version 1 and carries none; refusing it would make every such checkpoint unreadable
    # and silently restart those runs from step 0 — far worse than the gap it was meant to close.
    if payload.get("format_version") not in (1, 2):
        raise ValueError(f"checkpoint {path} has format_version {payload.get('format_version')}, "
                         f"expected 1 or 2")

    agent.load_state_dict(payload["agent"])
    rnd_model.load_state_dict(payload["rnd_model"])
    optimizer.load_state_dict(payload["optimizer"])
    _load_rms(obs_rms, payload["obs_rms"])
    _load_rms(reward_rms, payload["reward_rms"])
    discounted_reward.rewems = payload["reward_forward_filter"]

    # Restoring the generator states makes the continuation follow the same stream it would have
    # followed had the run never stopped, for everything except the environments themselves.
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"].cpu() if torch.is_tensor(payload["rng_torch"]) else payload["rng_torch"])
    if payload["rng_torch_cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s for s in payload["rng_torch_cuda"]])

    return {
        "global_step": payload["global_step"],
        "update": payload["update"],
        "avg_returns": payload["avg_returns"],
        # None for a version-1 checkpoint, which carries no history.
        "record_history": payload.get("record_history"),
    }
