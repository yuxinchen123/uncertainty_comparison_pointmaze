# H100 single-user lock protocol

Only ONE GPU-touching process may run on serval05 at a time (task requirement). Mechanism:

- The lock is the serval05-LOCAL file `/localtmp/sl5nw/locks/h100.lock`, taken with `flock`.
  Local, not NFS, because flock over NFS is unreliable.
- Every GPU-touching command (benchmark, profile, training, compile check that initializes
  CUDA) goes through `gpu_run.sh`: `bash locks/gpu_run.sh "<command>"`. A second caller blocks
  until the lock frees (up to 6 h, then fails loudly).
- CPU-only work on serval05 (editing, pip installs, CPU unit tests) does NOT take the lock.
- Long trainings holding the lock must write per-item progress to a file (live-progress rule)
  so a waiting agent can see the lock is legitimately held, not leaked. flock dies with its
  process, so a crashed job releases the lock automatically.
