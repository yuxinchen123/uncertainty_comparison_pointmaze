# This sweep was stopped on 2026-08-05

Sweep `2026-08-01-02-03_run812` was stopped by the owner on 2026-08-05 at 15:45, before the stage-1
race reached any verdict. It is not complete and it is not failed — it was ended early so the cpu
and nolim pools could go to the follow-up run that tests the same original-small RND stack at Adam
learning rate 1e-3 (see `../2026-08-05-*_run_5_8_1_2_addendum_*`).

- No configuration reached the 30-seed decision floor, so nothing was pruned and nothing was
  declared a survivor.
- Every remaining queue marker is parked in `queue/2026-08-01-02-03_run812/stopped_2026-08-05/`
  (15,438 of them). Nothing was deleted.
- Runs already in flight at the stop were left to finish; their records are part of the data.
- The section 8.1.2 tables and figure in the development document were regenerated from the
  records on disk at the stop and are labelled as a stopped-state snapshot.

Full account, including how to resume: the `## 2026-08-05 15:45` entry of `infra_history.md`.
