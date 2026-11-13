# Infrastructure history: RND baseline on AntMaze

Times are Pacific, read from the clock or from scheduler records.

- 2026-11-12 15:57 PT: folder created, code frozen at commit 30b2c609, 8 units built (rollout-shape check).
  Placement: MIT cannot start the work (other sessions of this account hold 4 of 4 cards on `mit_preemptable` and
  have 5 jobs waiting on `mit_normal_gpu`); Gautschi's 2 H100 cards are in use by another session; so the units go
  to Gilbreth's standby cards with at least 24 GB (A100 80 GB, A30; not charged, 4-hour jobs, units continue from
  their checkpoints). ACES is next if Gilbreth is too slow.
- 2026-11-12 15:57 PT: the automatic first-unit check compares a server's first unit with a unit trained on UVA,
  and no AntMaze unit can run on UVA's cards today. The check was therefore set to passed by hand in the server
  table, with one unit at a time: the unit with 128 environments per policy and cells 41 to 48 went to Gilbreth
  (job 12346318). Its records are compared by hand with the UVA example run of the same trainer before more units
  are handed over.
- 2026-11-12 15:57 PT: monitor job 6891245 (UVA, no worker submissions).
- 2026-11-12 16:17 PT: first unit read by hand at window 36 of 256 (job 12346318 on an NVIDIA A30, started
  16:03 PT): the Warp kernels compiled on the node; record kinds and per-policy fields are complete, with no
  non-finite value; the contact pool is 8.2% used; one iteration at 16,384 environments takes 3.85 s, which is 0.545
  million steps per second (an H100 NVL: 1.67 million). The other 7 units were handed over (8 at once now).
  At 2.4 million steps per policy some policies of this unit already reach the goal. Its policies spend 0.7% to
  8.2% of their steps within 0.5 m of the ceiling (a random policy: about 0.33%), and their largest torso speeds
  are 7.8 to 12.8 m/s: policies trained with the bonus leave the floor far more often than random ones.
