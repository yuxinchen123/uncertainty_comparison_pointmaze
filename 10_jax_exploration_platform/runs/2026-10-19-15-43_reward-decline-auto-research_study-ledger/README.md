# Reward-decline auto-research study: ledger

Plan: `~/.claude/plans/2026-10-19-15-27_rnd-reward-decline-fix-and-jaguar03-speed.md`.

Goal: the extrinsic reward of RND training on PointMaze rises, peaks, then falls. Find a change of
the environment's reward structure or of the algorithm after which it no longer falls; screen a
small map first, confirm on the large maze. Ten speed ideas on jaguar03 first.

| file | what it holds |
|---|---|
| `id_files.txt` | every Slurm id file of this study (one path per line); the monitor reads only these |
| `ideas.md` | every idea, its batch, its run folder and its outcome |
| `infra_history.md` | anomalies and infrastructure events, Pacific times |
| `slurm/` | code-gate test jobs and their logs |

Times are Pacific (machines run Eastern).
