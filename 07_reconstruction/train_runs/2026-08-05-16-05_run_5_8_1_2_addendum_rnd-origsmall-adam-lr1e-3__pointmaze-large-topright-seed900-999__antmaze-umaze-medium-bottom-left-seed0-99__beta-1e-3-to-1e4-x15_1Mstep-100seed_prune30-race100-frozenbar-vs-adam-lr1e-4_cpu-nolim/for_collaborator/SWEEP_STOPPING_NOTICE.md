# This sweep is stopping (2026-08-08) — nothing needed from you

The owner stopped the sweep on 2026-08-08 at 01:53: every unlaunched queue marker was
archived, so there is no more work to claim. Your jobs need NO action:

- runs currently in flight on your workers FINISH normally and their records are kept;
- when a worker finishes its run and finds the queue empty, it exits; when all workers of a
  job have exited, the job ends on its own;
- expect all your jobs to have drained within about 21 hours of the stop time.

Do not cancel anything — cancelling would kill in-flight runs whose data we keep. Thank you
for the workers; your jobs completed 479+ runs of this sweep.

A new sweep (train run 6, PointMaze, the four train-run-8.1.2 algorithms) launches in a new
run folder with its own for_collaborator/ packet; the owner will point you at it.
