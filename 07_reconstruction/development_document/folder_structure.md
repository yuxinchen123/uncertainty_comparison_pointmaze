# `07_reconstruction` — folder structure

Companion to `development_document/main.tex`. This is the code base behind the
development document: SAC on `PointMaze_Large-v3` with switchable intrinsic
exploration bonuses, plus a distance-to-ground-truth comparison of each bonus
against an oracle visit-count field.

Noise directories are omitted: `__pycache__/`, `*.pyc`, `wandb/` (2087 run
folders), `image/` (57 rendered figures), `slurm_output.log` (large), and the
per-run export trees under `analysis/data/`.

```text
07_reconstruction/
│
├── 01_gt.py                       # precursor: visit-count (oracle) bonus only
├── 02_rnd_rlexplore.py            # precursor: RND bonus via RLeXplore utilities
├── 03_rnd_my_implementation.py    # precursor: self-contained RND bonus
├── 04_many_exploration_method.py  # MAIN entrypoint: SAC + switchable bonus (--algorithm)
├── 0{1,2,3,4}_wandb_sweep.yaml     # W&B sweep configs for each script
├── ppo_rnd_envpool.py             # reference PPO+RND (Atari) implementation
├── note.md                        # coding notes (obs formats, replay buffer, RND knobs)
├── SETUP.md / requirements.txt    # environment setup
│
├── intrinsic/                     # intrinsic-reward (exploration bonus) library
│   ├── intrinsic_method/
│   │   ├── base.py                # IntrinsicRewardModel ABC: compute() + update()
│   │   ├── visit_count.py         # VisitCount: count-based bonus (oracle GT)
│   │   ├── rnd.py                 # RND: feature variants, ensemble, linear RND
│   │   └── elliptical_bonus.py    # EllipticalBonus: Mahalanobis / UCB bonus
│   ├── RLeXplore_utilities/       # adapters: batch / transition -> RND samples dict
│   ├── vector_intrinsic_replay_buffer.py   # ReplayBuffer: reward = ext + beta*int on sample()
│   └── intrinsic_replay_buffer.py
│
├── distance_to_GT/                # how close is a bonus field to the oracle field
│   ├── full_observation_list.py   # discretized (x,y,vx,vy) grid of valid maze cells
│   ├── vector_distance.py         # the six distance metrics (L1/L2 diff, inv, normalized)
│   ├── algorithm_vector.py        # builds bonus vector over the grid; GT = gt_position_velocity
│   └── test/
│
├── env_wrapper/                   # PointMaze Gymnasium wrappers
│   ├── point_maze_wrappers.py     # fixed goal/start, goal removal, visit-count, intrinsic-reward
│   └── point_maze_utils.py        # maze map, (x,y)->(row,col), (vx,vy)->bin, cell selection
│
├── utilities/
│   ├── callbacks/
│   │   ├── wandb_eval_logging.py      # eval/mean_extrinsic_reward + heatmaps
│   │   ├── train_episode_stats.py     # training episode stats
│   │   └── distance_logging.py        # logs distance_to_gt/* during training
│   ├── format.py                  # to_tensor / to_numpy helpers
│   ├── heatmap_utils.py           # visit-count heatmaps
│   └── debug.py
│
├── tests/                         # pytest: algorithms, distance metrics, final eval
│   ├── test_04_algorithms.py
│   ├── test_vector_distance.py
│   └── test_final_eval_n_episodes.py
│
├── debug/                         # diagnostics + reports (env, rnd, visit-count)
│
├── slurm/ , slurm_yuxin/          # cluster job scripts
│
├── analysis/                      # W&B sweep analysis
│   ├── analysis.md                # self-contained write-up (tables, formulas, correlation)
│   ├── analysis.html / pdf/       # rendered analysis
│   ├── plot/                      # best-beta bar chart, reward-vs-distance correlation
│   └── script/                    # download / combine / aggregate / plot scripts
│
└── development_document/          # THIS document (LaTeX)
    ├── main.tex                   # notation table + method catalog
    ├── folder_structure.md        # this file
    ├── bibliography.bib
    └── neurips_2026.sty
```

## Reading order

1. `04_many_exploration_method.py` — the driver. `_algorithm_to_config` maps each
   `--algorithm` name to an intrinsic model + feature.
2. `intrinsic/intrinsic_method/` — the three model families (count, RND, elliptical).
3. `intrinsic/vector_intrinsic_replay_buffer.py` — where the bonus enters training:
   on `sample()`, reward becomes `extrinsic + beta * intrinsic`.
4. `distance_to_GT/` — the six metrics that score each bonus field against the
   oracle `gt_position_velocity` field.
5. `analysis/analysis.md` — results of the W&B sweep over `(algorithm, beta)`.
