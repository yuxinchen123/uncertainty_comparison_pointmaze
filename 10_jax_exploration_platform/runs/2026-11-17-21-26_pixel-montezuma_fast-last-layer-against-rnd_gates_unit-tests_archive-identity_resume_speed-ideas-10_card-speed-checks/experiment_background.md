# Gates and speed work for the fast last layer on pixel Montezuma

## Purpose

Checks that must pass before the 624-run sweep of the plan
`/u/sl5nw/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md` starts: unit tests of the fast
last layer and the game-event records, the default trainer against the archived 2-billion-step RND baseline, resume
bit for bit with the new options, ten speed ideas for the new parts (each profiled in under 10 minutes on serval05 or
serval02 card 1), and a speed and memory check per UVA card type (at most 15 minutes each). No result of this folder
is a scientific result; it decides whether and how the sweep runs.

## Code and data

- Code: commit 3d593552 on `main` (the fast last layer, the game-event records and the sweep tooling); the speed work
  adds options behind defaults that keep the RND program unchanged.
- Reference records: the archived baseline under
  `/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_*/data/`.
- Environment: `/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python`, the corrected asset pack through
  `XDG_DATA_HOME`, `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true` (the environment block in
  `implementation_design.md`).

## Results so far

| gate | result (2026-11-17 PT) |
|---|---|
| 1, unit tests | fast layer 18 passed; game events 13 passed; the default options trace to the same program as commit 19d50152 (8 programs) |
| 2, standard-library tests | 24 passed (fold, resume, archive lines, identity, metric comparison); sweep tooling 85 passed |
| 3, default path against the archive, RTX A4500, 200 updates of seed 5 | 3,000 archived fields equal with logging off and with logging on; final states of the two runs equal (194 leaves) |
| 5, resume with step 0.1, group 16, both record flags | 40 updates against 20 + 20: states, metrics and both record files equal |
| 4, default path against the archive, four copies, serval02 card 1 | 1,500 archived fields of 100 updates equal (`speed_ideas.md`) |
| 6, speed and memory per card type | `speed_ideas.md`; the RTX 4000 Ada is below the RTX A4000's speed and is left out |
| 5 again with the final code (accepted fast-layer forms as defaults), 2026-11-18 11:48 to 11:56 PT | RTX A4500, jaguar03; group 16 (jobs 6956638, 6956639) and group 4 (jobs 6956640, 6956641): 194 of 194 state leaves equal, 40 of 40 metric lines equal, both record files equal byte for byte, in both groups. Code: commit e0b6ab74 (pixel code as at 5aecde3b), clean. 1.824 s per update in both uninterrupted runs (median of updates 21 to 40) |
| 7, one real grid unit through the whole path | see below |

Speed on an RTX A4500 before the speed work (one copy, rough, other jobs on the node): default 1.730 s per update;
game events +1.5%; fast layer group 128 +5.1%, group 16 with logging +10.4%, group 4 +22.5%.

## Layout

`implementation_design.md` (the design), `speed_ideas.md` (the speed work), `slurm/jobs/<name>/` (job scripts and
outputs), `slurm/submitted_jobids_*.txt` (ledgers), `data/<check>/` (run folders of the checks), `logs/`.
