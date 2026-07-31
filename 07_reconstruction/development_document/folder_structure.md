# `07_reconstruction` — folder structure

Companion to `development_document/RND_development_document.tex`. This is the code base behind the development document: SAC on
`PointMaze_Large-v3` with a switchable intrinsic exploration bonus, plus a distance-to-ground-truth comparison of
each bonus against an oracle visit-count field.

After the 2026-06-24 reorganization the code is an installable package (`rnd_exploration`) with one entry point
(`train.py`) and a registry that is the single source of truth for the algorithms. Noise directories are omitted:
`__pycache__/`, `*.pyc`, and the git-ignored per-run `train_runs/**/data/` + `train_runs/**/logs/`.

```text
07_reconstruction/
│
├── train.py                       # the entry point (was 04_many_exploration_method.py): build_* helpers + run()
├── pyproject.toml                 # installable package; `pip install -e .` so `import rnd_exploration` resolves
├── note.md  SETUP.md  requirements.txt
├── ppo_rnd_envpool.py             # reference PPO+RND (Atari) implementation (vendored from CleanRL)
│
├── src/rnd_exploration/           # the importable package (layers import downward only)
│   ├── common/                    # leaf helpers: format.py, debug.py, heatmap_utils.py
│   ├── envs/                      # PointMaze Gymnasium wrappers: point_maze_wrappers.py, point_maze_utils.py
│   ├── methods/                   # intrinsic-reward methods + the registry
│   │   ├── base.py                #   IntrinsicRewardModel ABC: compute(samples) + update(samples)
│   │   ├── visit_count.py         #   VisitCount: count-based bonus (oracle ground truth)
│   │   ├── rnd.py                 #   RND: feature variants, ensemble, linear RND
│   │   ├── elliptical_bonus.py    #   EllipticalBonus: Mahalanobis / UCB bonus
│   │   └── __init__.py            #   AlgorithmSpec, REGISTRY, build_intrinsic_model, EnvContext,
│   │                              #     ALGORITHM_NAMES, ALGORITHMS_NO_ACTION  (single source of truth)
│   ├── buffers/                   # vector_intrinsic_replay_buffer.py (reward = ext + beta*int at sample())
│   ├── metrics/                   # distance-to-ground-truth: algorithm_vector.py, vector_distance.py,
│   │                              #   full_observation_list.py
│   └── callbacks/                 # wandb_eval_logging.py, train_episode_stats.py, distance_logging.py
│
├── tests/                         # hierarchical per-module pytest suite
│   ├── common/  envs/  buffers/  callbacks/
│   ├── methods/   (+ methods/rnd/ for the RND feature-mode / ensemble / linear tests)
│   ├── metrics/   (test_vector_distance.py, test_algorithm_vector.py, ...)
│   └── integration/  (test_algorithms.py runs train.py per algorithm; test_final_eval_n_episodes.py)
│
├── train_runs/                    # one self-contained folder per training run / sweep (replaces a configs/ dir)
│   └── <YYYY-MM-DD-HH-MM>_<purpose>/
│       ├── config/                #   the wandb sweep yaml + resolved config for this run
│       ├── experiment_background.md
│       ├── data/                  #   all data this run generated (wandb export, parquet, ...) — git-ignored
│       ├── logs/                  #   slurm + training logs — git-ignored
│       ├── slurm/                 #   the slurm submission script(s) used for this run
│       └── analysis/              #   optional dated analysis subfolders (analysis.md + code/ + plots/)
│
├── legacy/                        # frozen pre-reorganization reference (not runnable as-is; see legacy/README.md)
│   ├── 01_gt.py / 02_rnd_rlexplore.py / 03_rnd_my_implementation.py  (+ their sweep yamls)
│   └── debug/                     # diagnostic scripts + report.md write-ups
│
├── analysis/                      # cross-run W&B sweep analysis (analysis.md, plot/, pdf/, script/, data/)
│   └── 2026-06-24-cpu-parallelization-understanding/   # dated study (frozen; pinned to its commit)
│
└── development_document/          # THIS document (LaTeX)
    ├── RND_development_document.tex  # notation, method catalog, environments, train runs, distance metrics
    ├── folder_structure.md        # this file
    ├── bibliography.bib  neurips_2026.sty
    └── code/                      # figure-generating scripts for the writeup
```

## Reading order

1. `train.py` — the entry point. `parse_config()` builds the `Config`; `run()` builds the env stack, the
   intrinsic model (via the registry), SAC, and the callbacks, then trains.
2. `src/rnd_exploration/methods/__init__.py` — the registry: each `--algorithm` name maps (through one
   `AlgorithmSpec` table + `build_intrinsic_model`) to an intrinsic model.
3. `src/rnd_exploration/methods/` — the three model families (count, RND, elliptical) behind the
   `IntrinsicRewardModel` ABC.
4. `src/rnd_exploration/buffers/vector_intrinsic_replay_buffer.py` — where the bonus enters training: on
   `sample()`, reward becomes `extrinsic + beta * intrinsic`.
5. `src/rnd_exploration/metrics/` — the six metrics scoring each bonus field against the oracle
   `gt_position_velocity` field.
6. `train_runs/<run>/analysis/analysis.md` — results of a W&B sweep over `(algorithm, beta)`.
