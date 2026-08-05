# Logging convention in RND train run 8.1.2 — full report

## 0. Where the code actually lives

Run 8.1.2 has **no `code/` snapshot folder**. Its workers execute the live tree via `PROJ_DIR`:

- Trainer: `/p/rlprojects/RND/07_reconstruction/train.py`
- Callbacks: `/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/callbacks/`
- Run folder: `/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/`

The run's `experiment_background.md` records launch commit `9dcb85ad1a9eb0b618849087b6d81e71848a6908`. I verified `git diff 9dcb85ad -- 07_reconstruction/train.py 07_reconstruction/src/rnd_exploration/callbacks/` is **empty**, so the files below are byte-identical to what the sweep ran.

`/p/rlprojects/RND/.claude/rules/run-id-and-logging.md` is the rule; it defers the generic work-queue part to `/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` and keeps the RND record format (five groups) and the "no final-eval special case" locally. Note the rule's "RND sweep order" paragraph (600 = 3 algorithms x 200 seeds) is stale relative to run 8.1.2, whose `run_total` is 24,200 — see §5.

---

## 1. `_write_local_log` — the record writer

`/p/rlprojects/RND/07_reconstruction/train.py:505-607` (verbatim):

```python
def _write_local_log(cfg: "Config", runtime_seconds: float, eval_history, distance_history, train_history,
                     train_episode_history, intrinsic_diagnostics=None, completed: bool = True) -> None:
    """Write THIS run's own JSON to <local_log_dir>/<z_logging_mode>/<run_name>.json (atomic replace).

    Logging design (see .claude/rules/run-id-and-logging.md):
    - One file per run (named by the run's id, run_name=NNN_of_TOTAL) -> no two runs ever write the same
      path, so there is no file-lock contention across the 512 concurrent workers.
    - Group-style, checkpointed: every metric is accumulated in memory by its callback during training and
      the whole record is flushed here -- at each eval cadence with completed=False (a checkpoint, so a
      run killed at the job walltime leaves its partial curves instead of nothing), and once at run end
      with completed=True. Each flush atomically REPLACES the run's own file (tmp + os.replace), so a
      reader never sees a half-written JSON. Never write incrementally per step.
    The history lists are the logging GROUPS; each row in a group is one eval-cadence snapshot."""
    # one file per run, under the logging-mode subdir; the dir is created lazily on first write
    out_dir = os.path.join(cfg.local_log_dir, cfg.z_logging_mode)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        # --- identity / config group ---
        "completed": completed,               # False on eval-cadence checkpoints, True on the final write
        "run_id": cfg.run_id,                 # this run's position in the sweep ("i")
        "run_total": cfg.run_total,           # sweep size ("total"); run_id/run_total is the run's identity
        "algorithm": cfg.algorithm,
        "beta": cfg.beta,
        "a_seed": cfg.a_seed,
        "z_logging_mode": cfg.z_logging_mode,
        # --- env identity (section-8 multi-env sweeps need it; older single-env records omit it) ---
        "env_setup": cfg.env_setup,
        "env_name": cfg.env_name,
        "start_cell": cfg.start_cell,
        "goal_cell": cfg.goal_cell,
        "continuing_task": cfg.continuing_task,
        "env_max_episode": cfg.env_max_episode,
        "position_noise_range": cfg.position_noise_range,
        "reward_shift": cfg.reward_shift,
        "discount_factor": cfg.discount_factor,
        "device": cfg.device,           # where this run actually computed ("cpu"/"cuda")
        "total_timesteps": cfg.total_timesteps,
        "eval_freq": cfg.eval_freq,
        "eval_standalone": cfg.eval_standalone,   # whether the standalone eval rollout ran (OFF by default)
        "runtime_seconds": runtime_seconds,
        # --- metric groups (each a list of per-eval-cadence snapshots, accumulated then flushed here) ---
        "eval_history": eval_history,         # eval group: eval/mean_extrinsic_reward, visit_counts/*, ... (all algos)
        "train_history": train_history,       # train group: train/mean_extrinsic_reward over the past
                                              # n_eval_episodes training episodes, ... (all algos)
        "train_episode_history": train_episode_history,  # episode-level: one row per completed training episode
                                              # (run-4 convention) -> full first-to-last training trajectory
        "distance_history": distance_history, # distance group: distance_to_gt/* (only ALGORITHMS_NO_ACTION algos)
    }
    # elliptical-only knobs: recorded so the analysis can group runs by covariance rule (batch/global),
    # update timing (sample/add), encoder input (s,a vs next-state), and ridge λ. Omitted for non-elliptical
    # algorithms (which never read them) so their JSON stays free of irrelevant defaults.
    # RND-only knobs (run 3.2.1): recorded so the analysis can group runs by optimizer method,
    # bonus readout, and the sgd1t schedule parameters. Omitted for non-RND algorithms (which never
    # read them) so their JSON stays free of irrelevant defaults. eta0/t0 are recorded for every RND
    # run (only load-bearing under sgd1t) so the O3 grouping key is always complete.
    if REGISTRY[cfg.algorithm].kind == "rnd":
        record["rnd_optimizer"] = cfg.rnd_optimizer
        record["rnd_bonus_readout"] = cfg.rnd_bonus_readout
        record["rnd_sgd_eta0"] = cfg.rnd_sgd_eta0
        record["rnd_sgd_t0"] = cfg.rnd_sgd_t0
        # run-3.2.3 / convergence-run-1 knobs: recorded for every RND run so the grouping key
        # (weight + bias scheme, reward normalization on/off + its filter discount) is always
        # complete in the analysis loader.
        record["rnd_weight_init"] = cfg.rnd_weight_init
        record["rnd_bias_init"] = cfg.rnd_bias_init
        record["rnd_reward_norm"] = cfg.rnd_reward_norm
        record["rnd_reward_norm_gamma"] = cfg.rnd_reward_norm_gamma
        # train-run-5 original-RND knobs: recorded for every RND run so the analysis can group the
        # original-small arm (lr, activation, deeper predictor, keep-mask, env-steps warmup) distinctly.
        record["rnd_lr"] = cfg.rnd_lr
        record["rnd_activation"] = cfg.rnd_activation
        record["rnd_predictor_extra_layers"] = cfg.rnd_predictor_extra_layers
        record["rnd_update_proportion"] = cfg.rnd_update_proportion
        record["rnd_obs_warmup_mode"] = cfg.rnd_obs_warmup_mode
        record["rnd_obs_warmup_steps"] = cfg.rnd_obs_warmup_steps
        # run-8.1.2 algorithm-2 knobs: recorded for every RND run so the analysis can distinguish
        # algorithm 1 (plain l2) from 2.1 (ratio bonus), 2.2 (normalized loss), 2.3 (LayerNorm).
        record["rnd_readout_norm_init"] = cfg.rnd_readout_norm_init
        record["rnd_readout_norm_eps"] = cfg.rnd_readout_norm_eps
        record["rnd_predictor_loss"] = cfg.rnd_predictor_loss
        record["rnd_layer_norm"] = cfg.rnd_layer_norm
    # visit-count (gt_*) oracles: record the count->bonus decay exponent so 1/sqrt(n) (-0.5) and 1/n (-1)
    # runs are distinguishable in the JSON. Omitted for non-visit-count algorithms.
    if REGISTRY[cfg.algorithm].kind == "visit_count":
        record["visit_count_decay"] = cfg.visit_count_decay
    if REGISTRY[cfg.algorithm].kind == "elliptical":
        record["elliptical_regularization"] = cfg.elliptical_regularization
        record["elliptical_update_timing"] = cfg.elliptical_update_timing
        record["elliptical_feature_input"] = cfg.elliptical_feature_input
        record["elliptical_feature_normalization"] = cfg.elliptical_feature_normalization
        # the clip is a swept axis in run 3.1.2; json emits float('inf') as Infinity (Python round-trips it)
        record["elliptical_bonus_clip"] = cfg.elliptical_bonus_clip
    # intrinsic-model summary stats (elliptical only today): mean feature norms for the effective ridge
    # ratio, covariance eigenvalue range, clip-hit fraction. None for models without diagnostics.
    if intrinsic_diagnostics is not None:
        record["intrinsic_diagnostics"] = intrinsic_diagnostics
    # atomic replace: write the tmp file, then rename over the run's path, so a concurrent reader (the
    # 20-min progress counter) never parses a half-written checkpoint.
    path = os.path.join(out_dir, _run_name(cfg) + ".json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(record, f)
    os.replace(tmp_path, path)
```

## 2. `LocalLogCheckpointCallback` — decides *when* to flush

`/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/callbacks/local_log_checkpoint.py:1-25` (whole file, verbatim):

```python
"""Checkpoint the per-run local JSON at the eval cadence, so a run killed mid-training (job walltime,
node failure, per-run timeout) leaves its partial curves instead of nothing. The flush callable is
provided by the driver (train.py) and atomically replaces the run's OWN file; this callback only decides
WHEN to fire. Run 3.1.1 lost 488 part-trained runs to walltime kills under the old write-once-at-end
convention — this callback is the fix (run-id-and-logging.md)."""
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


class LocalLogCheckpointCallback(BaseCallback):
    """Call `flush()` every `eval_freq` steps. Append LAST in the callback list so the flush sees the
    same step's freshly appended history rows from the callbacks that ran before it."""

    def __init__(self, eval_freq: int, flush: Callable[[], None], verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.flush = flush

    def _on_step(self) -> bool:
        # fire on the eval cadence only (eval_freq<=0 disables, mirroring the other callbacks)
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.flush()
        return True
```

## 3. `TrainEpisodeStatsCallback` — fills `train_history` and `train_episode_history`

`/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/callbacks/train_episode_stats.py:1-131` (whole file, verbatim):

```python
"""Train episode stats callback: log mean extrinsic/intrinsic/total reward and length."""
import collections

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rnd_exploration.common.debug import print_or_wandb_log


class TrainEpisodeStatsCallback(BaseCallback):
    """
    Logs average episode reward (extrinsic, intrinsic, total) and episode length
    from the training env at the same frequency as eval.
    Uses only the past n_eval_episodes training episodes for each log.
    Assumes Monitor wrapper is present. Tracks extrinsic/intrinsic from info dict.
    """

    def __init__(self, train_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, beta: float = 0.0, log_to_wandb: bool = None, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        # log_to_wandb: send train stats to wandb only in full mode (see WandbEvalLoggingCallback)
        self.log_to_wandb = use_wandb if log_to_wandb is None else log_to_wandb
        self.history = []            # per-eval-cadence mean over the past n_eval_episodes (the live summary)
        # NEW (run-4 logging convention, run-id-and-logging.md): one entry per COMPLETED training episode, so
        # the full first-to-last training-episode trajectory is saved (episode-level, not step-level). The
        # per-eval training curve = mean over the past n_eval_episodes of these, computed in the analysis.
        self.episode_history = []
        self.beta = beta
        self._monitor = None
        self._ep_extrinsic = 0.0
        self._ep_intrinsic = 0.0
        self._episode_extrinsics = []
        self._episode_intrinsics = []
        self._episode_successes = []

    def _get_monitor(self):
        """Locate Monitor wrapper (assumed to exist)."""
        if self._monitor is not None:
            return self._monitor
        env = self.train_env.envs[0]
        while hasattr(env, "env"):
            if isinstance(env, Monitor):
                self._monitor = env
                return self._monitor
            env = env.env
        raise RuntimeError("Monitor wrapper not found in training env")

    def _on_step(self) -> bool:
        # Accumulate extrinsic and intrinsic from info on every step
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])
        if len(infos) > 0:
            info = infos[0]
            ext = float(info.get("extrinsic_reward", 0.0))
            intr_raw = float(info.get("intrinsic_reward", 0.0))
            intr = self.beta * intr_raw
            self._ep_extrinsic += ext
            self._ep_intrinsic += intr
            if dones[0]:
                # record this completed episode (episode-level log): the Monitor holds its total reward +
                # length; we hold the accumulated extrinsic/intrinsic. before: episode just ended; after: one
                # row appended with the step it ended at -> the full episode trajectory is reconstructable.
                monitor = self._get_monitor()
                # success = the episode TERMINATED (reached the goal under continuing_task=False); a
                # time-limit end arrives as done=True + info["TimeLimit.truncated"]=True (SB3 VecEnv)
                success = not bool(info.get("TimeLimit.truncated", False))
                ep_length = int(monitor.get_episode_lengths()[-1])
                self.episode_history.append({
                    "step": int(self.num_timesteps),
                    "train/extrinsic_reward": float(self._ep_extrinsic),
                    "train/intrinsic_reward": float(self._ep_intrinsic),
                    "train/total_reward": float(monitor.get_episode_rewards()[-1]),
                    "train/episode_length": ep_length,
                    "train/success": success,
                    # steps to goal counts the terminating step; null for episodes that never reached it
                    "train/steps_to_goal": ep_length if success else None,
                })
                self._episode_extrinsics.append(self._ep_extrinsic)
                self._episode_intrinsics.append(self._ep_intrinsic)
                self._episode_successes.append(success)
                self._ep_extrinsic = 0.0
                self._ep_intrinsic = 0.0

        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        monitor = self._get_monitor()
        rewards = monitor.get_episode_rewards()
        lengths = monitor.get_episode_lengths()
        n_completed = len(rewards)
        if n_completed == 0:
            return True

        n_window = min(self.n_eval_episodes, n_completed)
        window_rewards = rewards[-n_window:]
        window_lengths = lengths[-n_window:]
        window_extrinsics = self._episode_extrinsics[-n_window:]
        window_intrinsics = self._episode_intrinsics[-n_window:]
        window_successes = self._episode_successes[-n_window:]

        mean_total = float(np.mean(window_rewards))
        mean_extrinsic = float(np.mean(window_extrinsics)) if window_extrinsics else 0.0
        mean_intrinsic = float(np.mean(window_intrinsics)) if window_intrinsics else 0.0
        mean_length = float(np.mean(window_lengths))
        success_rate = float(np.mean(window_successes)) if window_successes else 0.0

        summary = collections.OrderedDict([
            ("step", self.num_timesteps),
            ("train/mean_extrinsic_reward", mean_extrinsic),
            ("train/mean_intrinsic_reward", mean_intrinsic),
            ("train/mean_total_reward", mean_total),
            ("train/mean_episode_length", mean_length),
            ("train/success_rate", success_rate),
            ("train/n_episodes_averaged", n_window),
        ])
        self.history.append(dict(summary))
        # wandb mode still logs every snapshot; local mode prints the block only when verbose>0
        # (default silent — the per-run JSON carries the full history, the console adds nothing)
        if self.log_to_wandb or self.verbose > 0:
            print_or_wandb_log(
                self.log_to_wandb,
                summary,
                f"Train episode stats (step {self.num_timesteps})",
            )
        return True
```

## 4. `WandbEvalLoggingCallback` — fills `eval_history`

`/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/callbacks/wandb_eval_logging.py:43-143`, the `_on_step` body that builds the row (verbatim, from line 71 where the summary is built — the earlier lines are the wandb heatmap upload, skipped in local mode):

```python
        # The eval summary always carries the step; the eval/* reward keys are added only when the
        # standalone rollout runs. The visit-count coverage (below) is cheap and logged either way.
        summary = collections.OrderedDict([("step", self.num_timesteps)])
        # Standalone eval rollout (OFF by default): collect extrinsic, intrinsic, and total per
        # deterministic episode. run-4 convention: NO final-eval special case -- the final eval uses the
        # same n_eval_episodes as every other eval (the n_eval_episodes_final knob is retired).
        if self.eval_standalone:
            n_ep = self.n_eval_episodes
            ...
            # reward summary keys (only present when the rollout ran)
            summary["eval/n_eval_episodes"] = n_ep
            summary["eval/mean_extrinsic_reward"] = float(np.mean(episode_extrinsic))
            summary["eval/mean_intrinsic_reward"] = float(np.mean(episode_intrinsic))
            summary["eval/mean_total_reward"] = float(np.mean(episode_total))
            summary["eval/mean_ep_length"] = float(np.mean(episode_lengths))
        if self.visit_count_env is not None:
            visit_counts = self.visit_count_env.get_visit_counts()
            open_cells = (self.visit_count_env.maze_map == 0).sum()
            visited_cells = (visit_counts > 0).sum()
            total_visits = int(visit_counts.sum())
            summary["visit_counts/total_visits"] = total_visits
            summary["visit_counts/cells_visited"] = int(visited_cells)
            summary["visit_counts/open_cells"] = int(open_cells)
            summary["visit_counts/coverage_pct"] = 100.0 * visited_cells / max(1, open_cells)
        if self.visit_count_env_1m is not None:
            counts_1m = self.visit_count_env_1m.get_visit_counts()
            open_squares = (self.visit_count_env_1m.maze_map == 0).sum()
            visited_squares = (counts_1m > 0).sum()
            summary["visit_counts_1m/cells_visited"] = int(visited_squares)
            summary["visit_counts_1m/open_cells"] = int(open_squares)
            summary["visit_counts_1m/coverage_pct"] = 100.0 * visited_squares / max(1, open_squares)
        # capture every eval summary locally (both modes), then send to wandb only in full mode
        self.history.append(dict(summary))
```

`DistanceLoggingCallback` (`.../callbacks/distance_logging.py:48-64`) is a no-op when `enabled=False`, so `distance_history` stays `[]`. Run 8.1.2 passes `log_distance=False` (`slurm/build_queue.py:117`), so **every 8.1.2 record has `distance_history: []`**.

## 5. Callback wiring + the two flush sites

`/p/rlprojects/RND/07_reconstruction/train.py:610-645` builds the three metric callbacks in fixed order `[eval, train-episode-stats, distance]`; `train.py:739-772` appends the checkpoint callback last and does the final write:

```python
    callbacks = build_callbacks(
        cfg, train_vec, eval_vec, train_position, train_second,
        intrinsic_model, goal_cell, start_cell, _run_name(cfg), log_to_wandb,
    )

    # intrinsic-model diagnostics for the log record: only models exposing diagnostics() (the elliptical
    # family) have them; RND / VisitCount / no_exploration contribute None.
    def _diagnostics():
        """Return the intrinsic model's diagnostics dict, or None when the model has none."""
        if intrinsic_model is not None and hasattr(intrinsic_model, "diagnostics"):
            return intrinsic_model.diagnostics()
        return None

    # checkpoint the per-run JSON at every eval cadence (completed=False): a run killed at the job
    # walltime leaves its partial curves instead of nothing. Appended LAST so its flush sees the same
    # step's freshly appended history rows from the callbacks before it.
    if cfg.local_log_dir:
        def _flush_checkpoint():
            """Atomically rewrite this run's JSON with everything accumulated so far (completed=False)."""
            _write_local_log(cfg, time.time() - t_start, callbacks[0].history, callbacks[2].history,
                             callbacks[1].history, callbacks[1].episode_history,
                             intrinsic_diagnostics=_diagnostics(), completed=False)
        callbacks.append(LocalLogCheckpointCallback(eval_freq=cfg.eval_freq, flush=_flush_checkpoint))

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    # runtime = setup + training (wandb param-fetch wait is excluded — t_start is after params)
    runtime_seconds = time.time() - t_start
    if log_to_wandb:
        wandb.log({"runtime_seconds": runtime_seconds})
    if cfg.local_log_dir:
        # callbacks = [eval, train-episode-stats, distance, checkpoint]; the final write sets completed=True
        _write_local_log(cfg, runtime_seconds, callbacks[0].history, callbacks[2].history, callbacks[1].history,
                         callbacks[1].episode_history, intrinsic_diagnostics=_diagnostics(), completed=True)
```

Note the **positional argument order** at both call sites: `(cfg, runtime, eval_history, distance_history, train_history, train_episode_history, ...)` — `distance_history` is the third positional, before `train_history`. Easy to get wrong when copying.

---

## 6. The exact JSON record schema (measured, not inferred)

Read from `.../data/2026-08-01-02-03_run812/local/00000_of_24200.json` with the project env. Top-level keys **in file order**, with the values that record actually carried:

| key | type | example value |
|---|---|---|
| `completed` | bool | `true` |
| `run_id` | int | `0` |
| `run_total` | int | `24200` |
| `algorithm` | str | `"rnd_next_state"` |
| `beta` | float | `0.001` |
| `a_seed` | int | `0` |
| `z_logging_mode` | str | `"local"` |
| `env_setup` | str | `"AntMaze_UMaze-v5_start_bottom_left"` |
| `env_name` | str | `"AntMaze_UMaze-v5"` |
| `start_cell` | str | `"3,1"` |
| `goal_cell` | str | `"1,1"` |
| `continuing_task` | bool | `false` |
| `env_max_episode` | int | `-1` |
| `position_noise_range` | float | `0.0` |
| `reward_shift` | float | `-1.0` |
| `discount_factor` | float | `0.99` |
| `device` | str | `"cpu"` |
| `total_timesteps` | int | `1000000` |
| `eval_freq` | int | `50000` |
| `eval_standalone` | bool | `false` |
| `runtime_seconds` | float | `67297.37283110619` |
| `eval_history` | list of dict | 20 rows |
| `train_history` | list of dict | 20 rows |
| `train_episode_history` | list of dict | 1428 rows |
| `distance_history` | list of dict | 0 rows |
| `rnd_optimizer` | str | `"sgd"` |
| `rnd_bonus_readout` | str | `"l2"` |
| `rnd_sgd_eta0` | float | `0.01` |
| `rnd_sgd_t0` | float | `1000.0` |
| `rnd_weight_init` | str | `"orthogonal"` |
| `rnd_bias_init` | str | `"normal_0.5"` |
| `rnd_reward_norm` | bool | `false` |
| `rnd_reward_norm_gamma` | float | `0.99` |
| `rnd_lr` | float | `0.001` |
| `rnd_activation` | str | `"leaky_relu"` |
| `rnd_predictor_extra_layers` | int | `1` |
| `rnd_update_proportion` | float | `1.0` |
| `rnd_obs_warmup_mode` | str | `"env_steps"` |
| `rnd_obs_warmup_steps` | int | `6400` |
| `rnd_readout_norm_init` | bool | `false` |
| `rnd_readout_norm_eps` | float | `1e-08` |
| `rnd_predictor_loss` | str | `"mse"` |
| `rnd_layer_norm` | bool | `false` |

The `rnd_*` block appears only for `REGISTRY[algorithm].kind == "rnd"`; `visit_count_decay` only for visit-count oracles; the five `elliptical_*` keys only for elliptical; `intrinsic_diagnostics` only when the intrinsic model exposes `diagnostics()` (RND does not, so it is **absent** from every 8.1.2 record).

### `eval_history` — list of dict, one row per eval cadence

With `eval_standalone=False` (run 8.1.2), the `eval/*` reward keys are **absent**. Actual first and last rows:

```json
{"step": 50000, "visit_counts/total_visits": 50000, "visit_counts/cells_visited": 3, "visit_counts/open_cells": 7, "visit_counts/coverage_pct": 42.857142857142854, "visit_counts_1m/cells_visited": 37, "visit_counts_1m/open_cells": 112, "visit_counts_1m/coverage_pct": 33.035714285714285}
{"step": 1000000, "visit_counts/total_visits": 1000000, "visit_counts/cells_visited": 4, "visit_counts/open_cells": 7, "visit_counts/coverage_pct": 57.142857142857146, "visit_counts_1m/cells_visited": 60, "visit_counts_1m/open_cells": 112, "visit_counts_1m/coverage_pct": 53.57142857142857}
```

Key set when `eval_standalone=False`: `step`, `visit_counts/total_visits`, `visit_counts/cells_visited`, `visit_counts/open_cells`, `visit_counts/coverage_pct`, `visit_counts_1m/cells_visited`, `visit_counts_1m/open_cells`, `visit_counts_1m/coverage_pct`.
When `eval_standalone=True`, five more keys are inserted right after `step`: `eval/n_eval_episodes`, `eval/mean_extrinsic_reward`, `eval/mean_intrinsic_reward`, `eval/mean_total_reward`, `eval/mean_ep_length`.
The `visit_counts_1m/*` block exists only when a 1 m surface was passed (`visit_count_env_1m`), i.e. section-8 runs.

### `train_history` — list of dict, one row per eval cadence

```json
{"step": 50000, "train/mean_extrinsic_reward": -700.0, "train/mean_intrinsic_reward": 1.1076624315901542, "train/mean_total_reward": -700.0, "train/mean_episode_length": 700.0, "train/success_rate": 0.0, "train/n_episodes_averaged": 71}
{"step": 1000000, "train/mean_extrinsic_reward": -700.0, "train/mean_intrinsic_reward": 0.418006761880964, "train/mean_total_reward": -700.0, "train/mean_episode_length": 700.0, "train/success_rate": 0.0, "train/n_episodes_averaged": 100}
```

Exactly seven keys: `step`, `train/mean_extrinsic_reward`, `train/mean_intrinsic_reward`, `train/mean_total_reward`, `train/mean_episode_length`, `train/success_rate`, `train/n_episodes_averaged`. The window is the past `min(n_eval_episodes, n_completed)` training episodes; `train/mean_intrinsic_reward` is already multiplied by `beta`.

### `train_episode_history` — list of dict, one row per **completed training episode**

```json
{"step": 700, "train/extrinsic_reward": -700.0, "train/intrinsic_reward": 4.351003267049796, "train/total_reward": -700.0, "train/episode_length": 700, "train/success": false, "train/steps_to_goal": null}
{"step": 999600, "train/extrinsic_reward": -700.0, "train/intrinsic_reward": 0.49503831890225425, "train/total_reward": -700.0, "train/episode_length": 700, "train/success": false, "train/steps_to_goal": null}
```

Exactly seven keys: `step` (the global timestep the episode ENDED at), `train/extrinsic_reward`, `train/intrinsic_reward` (beta-scaled), `train/total_reward` (from the Monitor), `train/episode_length` (int), `train/success` (bool: terminated, not time-limit-truncated), `train/steps_to_goal` (int on success, `null` otherwise).

### `distance_history` — list of dict, `[]` unless `--log_distance True`

Rows are whatever `compute_intrinsic_vector_distance` returns (`distance_to_gt/*` metrics) plus a `"step"` key added at `distance_logging.py:63`.

---

## 7. Atomic write, flush cadence, `completed` flag

**Atomic write** (`train.py:601-607`): the record is `json.dump`ed to `<path>.tmp`, then `os.replace(tmp_path, path)` renames it over the target. `os.replace` is atomic within a filesystem, so a concurrent reader (the 20-minute monitor, the stage-1 controller) either sees the previous complete record or the new complete record, never a partial one. One file per run means no lock contention across the hundreds of concurrent workers.

**Cadence**: every `eval_freq` steps (`LocalLogCheckpointCallback._on_step`, `num_timesteps % eval_freq == 0`), plus once after `model.learn` returns. For run 8.1.2 that is `eval_freq=50000` on 1M-step task-S runs → 20 checkpoint flushes plus the final write; 10M-step task-R runs → 200 flushes plus the final write. The checkpoint callback is appended **last** so its flush sees the same step's rows already appended by the eval and train-stats callbacks.

**`completed` flag**:
- checkpoint flushes write `completed: false` (`train.py:756-760`);
- the post-`learn` write writes `completed: true` (`train.py:769-772`);
- a record with **no** `completed` field is treated as complete (old write-once convention).

Consumers implement exactly that, e.g. `20_mins_monitoring/env_metrics_tables.py` `load_completed`:

```python
def load_completed(sweep_id):
    """{config_key: [compact_record, ...]} over completed per-run JSONs that have a computable score.
    Each record is reduced to scalars immediately (compact_record) and the full record is dropped. A
    missing "completed" field counts as complete (the logging convention); a completed=false
    checkpoint of a killed attempt is skipped, as is a record with no scorable episode."""
    local = os.path.join(run_dir(), "data", sweep_id, "local")
    by_key = {}
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush this cycle; picked up next tick
        if not d.get("completed", True):
            continue
```

and `slurm/requeue_orphans.py:191-207` `archive_partial` moves a `completed: false` JSON into `data/<sweep_id>/killed_attempts_<date>/` when its marker is requeued, so a restarted attempt starts from an empty path and the partial is kept as evidence.

---

## 8. Run identity, ordering, filename pattern

**`run_id` / `run_total`** are plain `Config` fields (`train.py:135-136`): `run_id: int = -1` (0-based position in the sweep; -1 = standalone), `run_total: int = 0` (sweep size; 0 = standalone). The worker passes them from the queue marker (`worker.py:144-145`).

**Filename**, `train.py:363-381`:

```python
def _run_name(cfg: Config) -> str:
    """The run's JSON filename stem. A sweep run (run_total > 0) is named by its id, so each run owns one
    short, sweep-sortable file; a standalone run (run_total == 0) keeps the descriptive pipe-delimited name."""
    # sweep run: id-based name. before: run_id=42, run_total=600 -> after: "042_of_600" (id zero-padded to
    # run_total's width so names sort in sweep order). The descriptive fields live inside the JSON, not the name.
    if cfg.run_total > 0:
        return f"{cfg.run_id:0{len(str(cfg.run_total))}d}_of_{cfg.run_total}"
    # standalone run (no sweep id): descriptive pipe-delimited name with every load-bearing knob spelled out
    parts = [
        f"algorithm={cfg.algorithm}",
        cfg.env_name.replace("/", "-"),
        f"seed={cfg.a_seed}",
        f"goal={cfg.goal_position}",
        f"beta={cfg.beta}",
        f"discount_factor={cfg.discount_factor}",
        f"env_max_episode={cfg.env_max_episode}",
        f"apply_termination_wrapper={cfg.apply_termination_wrapper}",
    ]
    return "|".join(str(p) for p in parts)
```

**Full path** = `os.path.join(cfg.local_log_dir, cfg.z_logging_mode, _run_name(cfg) + ".json")`. The worker sets `--local_log_dir=<run>/data/<sweep_id>` and `--z_logging_mode=local`, so the `local/` component of `data/<sweep_id>/local/` **is the logging mode**, not a hardcoded folder name. For 8.1.2: `data/2026-08-01-02-03_run812/local/00000_of_24200.json` (id zero-padded to 5 digits, the width of `24200`).

**Seed-outermost ordering** — `slurm/build_queue.py:193-220` in the run folder:

```python
    def write_marker(pool, run_id, cfg_spec, seed, steps, task):
        # one marker JSON per (config, seed); `pool` routes it to the claiming worker class and
        # requeue_orphans re-pends a killed run into the same pool it came from.
        cfg = {
            "sweep_id": args.sweep_id, "run_id": run_id, "run_total": RUN_TOTAL,
            "task": task, "pool": pool,
            "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
            "arm": cfg_spec["arm"], "beta": cfg_spec["beta"], "a_seed": seed,
            "config_key": config_key(cfg_spec),
            "params": cfg_spec["params"],
            "fixed": dict(FIXED_COMMON, total_timesteps=steps),
        }
        name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{label(cfg_spec)}_seed{seed}.json"
        with open(os.path.join(sweep_queue, pool, name), "w") as fh:
            json.dump(cfg, fh)

    # task S: SEED outermost, config inner — seed s owns ids [240*s .. 240*s+239] in CONFIGS order
    run_id = 0
    for seed in SEEDS:
        for cfg_spec in CONFIGS:
            write_marker("pending_1m", run_id, cfg_spec, seed, STEPS_S, "S")
            run_id += 1
    # task R: ids [N_S, RUN_TOTAL), env outer, seed inner (no racing, order immaterial)
    for cfg_spec in BASELINE_CONFIGS:
        for seed in SEEDS:
            write_marker("pending_10m", run_id, cfg_spec, seed, STEPS_R, "R")
            run_id += 1
    assert run_id == RUN_TOTAL
```

So for 8.1.2: `RUN_TOTAL = 24200 = 24000 (task S: 240 configs x 100 seeds, seed outermost) + 200 (task R: 2 baselines x 100 seeds)`. The 240 task-S configs are ordered env setup (2) → arm (4: alg1, alg2.1, alg2.2, alg2.3) → learning rate (2) → beta (15 ascending) — `build_queue.py:83-96`. The **queue marker filename** carries a descriptive tail (`00000_of_24200_AntMaze_UMaze-v5_alg1_lr0.001_b0.001_seed0.json`) while the **output JSON filename is id-only** (`00000_of_24200.json`); `requeue_orphans.py:121-125` `json_path_for` maps one to the other with `re.match(r"(\d+_of_\d+)_", marker_name)`.

Ordering is pinned by `slurm/test_run_queue_convention.py` in the same folder.

---

## 9. Checkpoint / resume of training state — **none exists**

There is **no checkpoint or resume of training state anywhere in `07_reconstruction`**. I grepped the whole tree for `torch.save`, `load_state_dict`, `set_parameters(`, `model.save(`, `SAC.load`, `save_replay_buffer`, `load_replay_buffer`. The only hits:

- `07_reconstruction/tests/methods/rnd/goldens/_generate.py:83,85` — `torch.save(golden, OUT)`, a unit-test golden-fixture generator, unrelated to training.
- `07_reconstruction/src/rnd_exploration/methods/rnd.py:178` — `self.predictor.body.load_state_dict(self.target.body.state_dict())`, copying target weights into the predictor **at construction time** (an init scheme, not a resume).

`model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)` at `train.py:763` is the only `learn` call; there is no `reset_num_timesteps=False`, no model save, no replay-buffer save, no `--resume` flag anywhere in `Config` (`train.py:57-145`).

The run's own documents say this explicitly:

- `slurm/worker.py:18` — `"There are no checkpoints in this run: a run that cannot finish inside the job must not start."`
- `slurm/requeue_orphans.py:14-15` — `"a killed 10M baseline run restarts from scratch, this run has no checkpoints"`.
- `experiment_background.md:73-74` — `"The deferred second stage (survivors to 10M, both-must-fail trigger vs a task-R-defined 10M bar, checkpoint/resume) is documented in the plan and NOT part of this run."`

Elsewhere in the wider `/p/rlprojects/RND` repo there are model-saving code paths, but none of them is a resume, and none belongs to a `train_runs/` run:
- `06rl_integration/train_rl.py:1341-1346` uses stable-baselines3 `CheckpointCallback(save_freq=config.eval_freq, save_path=.../checkpoints, name_prefix='rl_model')` and `train_rl.py:1417` `model.save(final_model_path)`. There is no load-and-continue path; `06rl_integration/evaluate.py:130` `SAC.load(model_path, env=env)` loads a finished model **for evaluation only**.
- `rlkit/`, `08_cleanrl_ppo_rnd/cleanrl/`, `RND Linear Comparison/` are vendored third-party reference repositories, not this project's code.

**Bottom line: if the new training script needs resume, it has to be built from scratch — there is no in-repo precedent to copy, only the resumable *logging* described above.** The `completed` flag plus the `killed_attempts_<date>/` archive is the entire existing story for a killed run: the partial curves survive, the training state does not, and the run restarts from step 0.

---

## 10. Worker: claim, run, mark, and requeue after a kill

`.../slurm/worker.py` (whole file is 196 lines; the load-bearing parts verbatim).

**Paths and pools**, `worker.py:31-47`:

```python
RUN = os.environ["RUN_DIR"]
PROJ = os.environ.get("PROJ_DIR", "/p/rlprojects/RND/07_reconstruction")
SWEEP_ID = os.environ["SWEEP_ID"]
QUEUE = os.path.join(RUN, "queue", SWEEP_ID)
POOLS = os.environ.get("WORKER_POOLS", "pending_1m").split()
RUNNING = os.path.join(QUEUE, "running")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
DATA = os.path.join(RUN, "data", SWEEP_ID)  # train.py appends /local -> data/<sweep_id>/local/<id>.json

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'
```

**The atomic claim** — `os.rename` from a pending pool into `running/` is the lock; whoever wins the rename owns the run (`worker.py:102-128`):

```python
def claim():
    """Atomically claim one pending config from the first claimable non-empty pool in POOLS order,
    approximately in run-id order. Returns (name, running_path) or (None, None) when nothing is
    claimable. `all_empty` distinguishes 'nothing left anywhere' from 'blocked by the guard'."""
    any_blocked = False
    for pool in POOLS:
        pending = os.path.join(QUEUE, pool)
        try:
            names = sorted(os.listdir(pending))
        except FileNotFoundError:
            continue
        if not names:
            continue
        if not pool_claimable(pool):
            any_blocked = True
            continue
        # Sort pending names (zero-padded ids -> lexical order == id order) and pick randomly among
        # only the FIRST 32: workers stay inside the earliest-id window so early seeds FINISH first,
        # while the random pick keeps the rename-collision protection (run-8.1 convention).
        while names:
            name = random.choice(names[:32])
            try:
                os.rename(os.path.join(pending, name), os.path.join(RUNNING, name))
                return name, os.path.join(RUNNING, name), False
            except OSError:
                names.remove(name)  # another worker won this file; retry within the refreshed window
    return None, None, any_blocked
```

**Command construction** — this is where `--local_log_dir`, `--run_id`, `--run_total`, `--z_logging_mode` are handed to `train.py` (`worker.py:131-156`):

```python
def build_cmd(cfg):
    """Build the train.py argv for one claimed config; JSON lands in data/<sweep_id>/local/<id>.json.
    The per-config `params` and the global `fixed` args (which carry this task's total_timesteps)
    pass through verbatim; the per-node device override is appended last."""
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--algorithm={cfg["algorithm"]}',
        f'--beta={cfg["beta"]}',
        f'--a_seed={cfg["a_seed"]}',
        f'--env_setup={cfg["env_setup"]}',
        '--z_logging_mode=local',
        '--use_wandb=False',
        f'--local_log_dir={DATA}',
        f'--run_id={cfg["run_id"]}',
        f'--run_total={cfg["run_total"]}',
    ]
    # per-config arm knobs, then the global fixed args (incl. this task's total_timesteps)
    for k, v in cfg.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    # per-node device override: WORKER_DEVICE=cuda (GPU jobs) / cpu (CPU jobs) appended LAST so
    # argparse's last-occurrence-wins beats the queue's device-agnostic --device=cpu from `fixed`.
    if os.environ.get("WORKER_DEVICE"):
        args.append(f"--device={os.environ['WORKER_DEVICE']}")
    return args
```

**The claim-run-mark loop** — `running/` → `done/` on rc 0, `running/` → `failed/` otherwise (`worker.py:159-191`):

```python
def main():
    """Claim-run-mark loop over this worker's pools until nothing is left to claim."""
    # small startup jitter so workers don't all import torch / build the env at the same instant
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    log(f"pools={POOLS} device={os.environ.get('WORKER_DEVICE', '(queue default)')} "
        f"job_end={'unknown' if JOB_END is None else time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(JOB_END))}")
    n_done = n_fail = 0
    while True:
        name, path, blocked = claim()
        if name is None:
            if blocked:
                # 10M items remain but this job can no longer finish one: exit rather than idle-spin
                log(f"only walltime-blocked items remain; exiting after {n_done} done / {n_fail} failed")
            else:
                log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: task={cfg.get('task', '?')} {cfg['env_setup']} {cfg['arm']} "
            f"beta={cfg['beta']} seed={cfg['a_seed']} steps={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        # no per-run timeout: run train.py to natural completion (or until the job's Slurm walltime
        # kills the whole step). A slow-but-progressing run is never guillotined near the finish.
        rc = subprocess.call(build_cmd(cfg), cwd=PROJ)
        dest = DONE if rc == 0 else FAILED
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        n_done += (rc == 0)
        n_fail += (rc != 0)
        log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)} "
            f"(running totals: {n_done} done, {n_fail} failed)")
```

The `claimed <marker>` log line is load-bearing: it is the only record of which Slurm job owns which marker, and the requeue step parses it back out.

**The walltime guard** (8.1.2-specific, blocks claiming a 10M run a job cannot finish since there is no resume), `worker.py:57-99` — `job_end_epoch()` reads `SLURM_JOB_END_TIME` or falls back to `scontrol show job`, and `pool_claimable("pending_10m")` returns True only when `JOB_END - time.time() >= REQUIRED_10M_SECONDS` (90 h on cuda, 170 h on cpu, overridable via `WORKER_REQUIRED_10M_HOURS`; unknown job end ⇒ never claimable).

**Requeue after a kill** — the worker itself does nothing; the owner-side `slurm/requeue_orphans.py`, called every monitor tick, does it in three phases plus a blanket re-pend:

1. `phase1_detect` (`requeue_orphans.py:230-262`): for each marker in `running/`, find the latest `claimed <marker>` line across `logs/*.log` and `for_collaborator/logs/*.log` (`last_claim_jobid`, lines 78-97), skip if that claim line is older than the marker's last requeue (stale-info guard against double-running), then `job_is_terminal(jobid)` — absent from `squeue` **and** a terminal `sacct` state (`COMPLETED, FAILED, CANCELLED, TIMEOUT, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, DEADLINE, REVOKED, BOOT_FAIL`). Any command failure returns False (fail-safe). With no claim line at all, the fallback is `checkpoint_is_stale`: the run's JSON **exists** and has not been modified for 6 h. A marker with no claim line **and** no JSON is never declared stale. Orphans move `running/` → `failed/` and get a `killed` line in the append-only ledger `slurm/killed_orphans_<sweep_id>.txt`.
2. `phase2_requeue` (lines 265-269) + `requeue_one` (lines 210-227): archive the partial `completed:false` JSON to `data/<sweep_id>/killed_attempts_<date>/`, then `os.rename` the marker from `failed/` back to **its origin pool**, read from the marker JSON's own `"pool"` field — or to `pruned/` if the config already has a stage-1 verdict. Every action appends to the ledger; the ledger is read as **counts** (`Counter`), not sets, so a marker whose requeued attempt is itself killed can be requeued again.
3. `phase2b_repend_all_failed` (lines 299-310): blanket re-pend of everything left in `failed/` — justified only because no config in this run fails deterministically.
4. `phase3_problems` (lines 272-296): collaborator problem reports in `for_collaborator/problems/open/*.md` with `MARKER: <name>` lines get those markers requeued, then the report moves to `problems/resolved/` with a footer.

`requeue_one` verbatim (`requeue_orphans.py:210-227`):

```python
def requeue_one(name, p, dry, why=""):
    """Return one failed/ marker to its origin pool (or pruned/ for a decided config); ledger it."""
    src = os.path.join(p["failed"], name)
    if not os.path.exists(src):
        return False
    archive_partial(name, p, dry)
    key, pool = marker_fields(src)
    decided = key is not None and key in decided_config_keys(p)
    dest_state = "pruned" if decided else pool
    log(f"requeue: {name} failed/ -> {dest_state}/" + (f" ({why})" if why else "")
        + (" (config already decided)" if decided else ""))
    if not dry:
        try:
            os.rename(src, os.path.join(p[dest_state], name))
        except OSError:
            return False
        ledger_append(p, "pruned-instead" if decided else "requeued", name)
    return True
```

Because there is no training-state resume, a requeued marker means the run **starts over from step 0**; the archived partial JSON is kept only as evidence and is excluded from every aggregate by the `completed` check.

**Slurm wiring** for reference (`slurm/worker_28x1.slurm`): sets `OMP_NUM_THREADS=1`, `WORKER_DEVICE=cpu`, `WORKER_POOLS="pending_1m"`, `PROJ_DIR`, `RUN_DIR`, `SWEEP_ID` (required via `--export`), and launches `srun --wait=0 /p/rlprojects/RND/.venvs/exploration/bin/python "$RUN_DIR/slurm/worker.py"`.

---

## 11. Minimum set to copy for a new training script

1. `_write_local_log` — `/p/rlprojects/RND/07_reconstruction/train.py:505-607` (the whole function, including the `.tmp` + `os.replace` tail and the `out_dir = local_log_dir / z_logging_mode` join).
2. `_run_name` — `train.py:363-381`.
3. `LocalLogCheckpointCallback` — `/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/callbacks/local_log_checkpoint.py` (whole file, 25 lines).
4. `TrainEpisodeStatsCallback` — `.../callbacks/train_episode_stats.py` (whole file, 131 lines) for `train_history` + `train_episode_history`.
5. The eval-row builder — `.../callbacks/wandb_eval_logging.py:71-134` for `eval_history`.
6. The wiring at `train.py:739-772`: callbacks in order `[eval, train-stats, distance]`, checkpoint callback appended **last**, `_flush_checkpoint` closing over `t_start` writing `completed=False`, final write `completed=True` — watch the positional order `(eval, distance, train, train_episode)`.
7. Config fields `local_log_dir`, `run_id`, `run_total`, `z_logging_mode` — `train.py:133-136`.
8. Reader-side contract: `d.get("completed", True)` — skip `False`, treat missing as complete (`20_mins_monitoring/env_metrics_tables.py:113-131`, `20_mins_monitoring/status_table.py:103-116`).