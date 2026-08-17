# Infrastructure history

Times are Pacific.

## 2026-08-16 22:22 PT — the parent run's missing-argument guards were re-tested before launch

The parent run's `infra_history.md` records three jobs that started with the runner's defaults
instead of their unit's arguments, because `run_unit.sh` claimed a unit by renaming its queue entry
and only then tried to read that entry. This run copies the fixed scripts, and the fix was
exercised before the first submission rather than assumed:

1. A job given a unit id with no queue entry anywhere printed `no queue entry for unit ...` and
   exited 2.
2. A job given a queue entry carrying two arguments printed `read no arguments from ...; refusing
   to start with the runner's defaults` and exited 3 — and the entry was still in `queue/pending/`
   afterwards, so no claim had happened.
3. `scripts/run_training.py` called without `--unit-id` exited 2 with
   `the following arguments are required: --unit-id`.

All three refuse rather than run defaults, which is what the fix was for.
