# Upstream provenance — CleanRL

The folder `cleanrl/` next to this file is an **untouched clone** of the CleanRL repository. It is
gitignored in this repo (255 MB, and it carries its own `.git`), so this file is the record.

| field | value |
|---|---|
| upstream URL | `https://github.com/vwxyzjn/cleanrl.git` |
| commit at clone time | `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` |
| commit date | 2026-04-20 13:57:15 +0300 |
| commit subject | `docs: fix PPO title, variable typo, baselines URL, and PQN doc URL (#549)` |
| cloned on | 2026-08-05 |
| cloned into | `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl` |

To reproduce the clone:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl && git checkout fe8d8a03c41a7ef5b523e2e354bd01c363e786bb
```

## What we use from it

| path | use |
|---|---|
| `cleanrl/cleanrl/ppo_rnd_envpool.py` | the reference PPO + Random Network Distillation implementation |
| `cleanrl/docs/rl-algorithms/ppo-rnd.md` | the documentation page that records the envpool off-by-one bug |
| `cleanrl/benchmark/rnd.sh` | the benchmark command line used for the published result |

**The clone is never edited.** All of our changes live in the tracked folder `src/` next to this
file, which starts as a copy of `ppo_rnd_envpool.py` and is modified from there. Every difference
between `src/` and the upstream file is listed in the profiling document under `shuze_experiment/`.
