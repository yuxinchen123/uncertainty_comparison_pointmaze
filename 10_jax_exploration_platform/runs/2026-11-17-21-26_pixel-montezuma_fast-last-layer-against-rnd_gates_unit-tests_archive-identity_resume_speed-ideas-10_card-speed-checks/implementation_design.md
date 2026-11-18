# Implementation design: fast last layer, event logging and sweep tooling for pixel Montezuma

Design of 2026-11-17 for the plan `/u/sl5nw/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md`.
P = `/p/rlprojects/RND/10_jax_exploration_platform`. Line numbers are those of `main` at commit 19d50152.

## Facts the design rests on

1. The pixel trainer's code on `main` equals the code that produced the archived baseline (the diff of
   `src/exploration_platform/atari_pixels`, `train_final.py` and `run_unit.py` against the archive's commit is
   empty), so a comparison with the archive starts clean.
2. `P/src/exploration_platform/bonuses/rnd/last_layer.py` imports only jax, works over a leading "policy" axis, and
   its arithmetic already broadcasts a per-policy step size. It is imported, not copied.
3. Adding fields to `PixelPPOConfig` changes `asdict(config)`, so archived run folders can no longer be resumed.
   Accepted: the baseline is complete; identity checks use fresh folders.
4. `train_final.py` has two obstacles for a fleet on several servers: `game_source()` (lines 54-79) puts absolute
   paths into the run identity, and the commit is read with `git rev-parse` (line 149), which inside a frozen code
   copy returns the enclosing repository's head.
5. The CleanRL reward filter runs along the environment axis (`upstream_reward_filter_axis=True`,
   `training.py:66-100`). The fast layer's decay inside a rollout changes that divisor; it is logged, not changed.

## The interface both work packages agree on

- Learner JSON keys (fields of `PixelPPOConfig`): `fast_last_layer_step` (float, default 0.0),
  `fast_last_layer_sharing_group` (int, default 0), `record_game_events` (bool, default false),
  `record_bonus_profile` (bool, default false). Per-copy values use the existing `per_copy_scalar_settings`
  mechanism; `fast_last_layer_step` is added to `SCALAR_SETTING_NAMES`. Per-copy step sizes are all zero or all
  positive in one process.
- Environment JSON key: `report_game_events` (bool, default false). The trainer raises when the learner flag is on
  and the environment flag is off.
- `train_final.py` gets `--copy-seeds 13,13,13,13` (one seed per copy; replaces `--seed-start` when given).
- Files beside the checkpoint in `data/<unit>/run/`: `metrics.jsonl` (new fields below), `game_first_times.json`,
  `game_windows.jsonl`, and the existing `run.json`, `events.jsonl`, `progress.json`, `checkpoint.pkl`.
- A unit is complete when `progress.json` says completed.

## A. Fast last layer in the pixel trainer

Config (`atari_pixels/config.py`): the four fields above. Validation in `__post_init__`: step in [0, 1]; the group
is 0 when the step is 0; when the step is positive the group is a positive divisor of `n_envs`; a positive step is
rejected together with `hoist_rollout_bonus` or `paired_rnd_rollout_trunks`.

Networks (`atari_pixels/networks.py:359-366`): add `rnd_hidden(parameters, normalized_frames, configuration)` (trunk
plus `dense[:-1]` with ReLU); `rnd_forward` returns `dense(parameters["dense"][-1], rnd_hidden(...))`, the same
operations in the same order as now.

Training (`atari_pixels/training.py`):

1. Imports (10-16): `rnd_hidden`; `fast_layer_scores`, `unit_rows_with_bias` from `..bonuses.rnd.last_layer`.
2. `Rollout` (19-37): add `bonus_hidden`, `bonus_targets`, `predictor_bonus`, `game_events`, all defaulting to
   `None` (a `None` adds no leaf to the compiled program).
3. New functions after `intrinsic_reward` (54-63):
   - `rollout_bonus_features(parameters, target, frames, statistics, configuration)`: the predictor's last hidden
     features and the target's output, float32 `[C,N,512]`, under the statistics of the start of the update;
   - `fast_last_layer_bonus(hidden, targets, output_layer, step_sizes, group)`: takes `[T,C,N,512]`, returns
     float32 `[T,C,N]`. `rows, length = unit_rows_with_bias(hidden)` (float64 `[T,C,N,513]`);
     `layer = concat([weight [C,512,512], bias[:,None,:]], axis=1)` in float64 (pixel dense weights are stored
     `[C, input, output]`, `networks.py:45`, the maze layout); groups: reshape every `[T,C,N,k]` to
     `[T, C*G, g, k]` with `G = N/g` (environments `j*g .. (j+1)*g-1` form group j), `jnp.repeat(layer, G, axis=0)`
     gives one temporary layer per group, and the step size becomes `[C*G,1,1]`; call
     `fast_layer_scores(..., bonus_of_error=half squared error)`, reshape back, cast to float32.
4. `PixelPPORND.__init__` (229-232): resolve the per-copy step sizes and set `self.fast_last_layer`.
5. `collect.step` (339-340): with the fast layer on, call `rollout_bonus_features` in place of `intrinsic_reward`,
   store both arrays in the `Rollout`, and set the bonus to zeros.
6. After the scan (beside 380-385): compute the bonus with `state.parameters["predictor"]["dense"][-1]`, then
   `rollout._replace(intrinsic_rewards=bonus, bonus_hidden=None, bonus_targets=None)`.
7. `normalize_intrinsic_rewards`, GAE and the update are untouched.

`cache_target_features` does not help here: it caches update-time targets of the current frames under the updated
statistics (433-437); the bonus needs targets of the next frames under the statistics of the start of the update.

Diagnostics under `record_bonus_profile`, added in `iteration` (534-547): mean bonus of the first and last 32
rollout steps; the real predictor's own mean bonus; the divisor `sqrt(reward_statistics.variance)`; row alignment
(squared length of the mean unit row).

Default path: every addition sits behind a Python `if` evaluated at trace time; with defaults the trace is the same
sequence of primitives as today.

## B. Event logging

Accumulation is per agent step: compare `state.game` with `game` in `_step_single` (`environment.py:606-656`), after
the 4-frame scan and before any reset. The scan carry is untouched. Known limit: a pickup followed by a room change
inside the same agent step is not seen.

New module `atari_pixels/game_events.py`: `GAME_EVENT_NAMES` and a pure `game_events(before, after, start_room)`.

| event | rule |
|---|---|
| key, gem, amulet, sword, torch pickup | `items_active` 1 to 0 by `before.items_type`, only when the room is unchanged |
| door opened | `inventory[0]` decreased |
| enemy killed | `inventory[1]` decreased |
| left start room | `before.room_id == 4` and `after.room_id != 4` |
| left start room with both doors locked | the above and `all(after.global_doors_active[4] == 1)`; expected never |
| life lost by fall, enemy, laser, other | `after.lives < before.lives`, cause from `after.death_type` (1 fall, 2 enemy, 3 laser) |

Per-lane values read only when a game ends: `game_has_key` (`items_active[0] == 0` in room 4, else
`global_items_active[4,0] == 0`) and `game_left_start_room` (`room_id != 4` or any `global_doors_active[4] == 0`).

Environment: a constructor flag `report_game_events=False` adds `info["game_events"]`, `info["room_after"]` and the
two per-game flags inside `_step_single`; they are reshaped with the rest at line 713.

Learner (`record_game_events`): `collect.step` reduces over environments (counts `[C,E]`, rooms entered `[C,33]`,
finished games with each flag `[C]`); `iteration` adds `event_counts [C,E]`, `event_first_step [C,E]`,
`room_first_step [C,33]` (rollout step index, -1 for none), `games_with_key`, `games_left_start_room`. The existing
`rooms_visited` stays as it is.

Host bookkeeping without any change to the checkpoint: first-time step =
`(update - 1) * 16384 + (rollout_step + 1) * 128`. New standard-library module
`scripts/atari_pixel_campaign/game_records.py` holds one fold over metric lines and produces
`game_first_times.json` (first key, door, exit and its destination, each room, each item type, first kill, score
milestones 400, 1,000, 2,500) and `game_windows.jsonl` (one line per 61 updates = 999,424 steps: games, score sum,
key and exit shares, lives lost by cause). On resume, after `keep_metrics_through`, `train_final.py` rebuilds both
by folding the kept lines, then folds each new line. The fold accepts archive lines without event fields.

## C. Fleet changes in `train_final.py`

Compare the game by content (`installed_version`, `asset_record_sha256`, a new hash of the game source files), with
paths kept as information only; read `SOURCE_COMMIT` when present; raise on a non-finite loss.

## D. Run management (`scripts/pixel_sweep_campaign/`, new)

- `build_units.py`, `worker.py`, `submit.py`, `monitor.py`, `remote_fleet.py`, `report.py`, `README.md`.
- The maze fleet files under `scripts/maze_campaign/` are in live use by a running study: they are imported, never
  edited. Where the pixel fleet needs something different (the claim filter by execution profile, the worker job
  text, the progress source, the files handed over and brought back), it is implemented in the new folder.
- Units are named in words (`grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-13`).
  Every run is built as a single; `build_units.py --pack-pending` turns four never-started singles of one seed, one
  group and one weight (the four step sizes) into one four-copy unit, under the queue lock, only for H100 capacity
  that exists. RND's weights at one seed pack four per unit. Units are ordered so that claims are interleaved by
  seed (all configurations at seed 13 first, then seed 14, ...).
- Execution profiles, fixed per unit when it first starts: "one copy" (the A4500 learner settings, reset budget 8,
  every card under 60 GB) and "four copies" (the H100 learner settings, reset budget 16). A worker claims only
  units of its card's profile; each start records the device in `events.jsonl`.
- Requests: UVA both accounts, cards free now first, 4 processors and 16 GiB per card, `--requeue`; Gautschi
  `-p ai -q normal`, one H100 and 14 cores, 24 h, at most 4 jobs, packs only; Gilbreth `-q standby`, 4 h; ACES
  `-p gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=39G`; DeltaAI last, after an ARM build.
- `report.py` reads the sidecar files only and writes `early_signals.md`: fleet status and speed per card type (the
  five throughput columns of rule `rl-throughput-tables.md`); per configuration the seeds done, the mean score over
  the run so far, the capped first exit, the rooms reached; medians of the first-time steps; per-million-step
  shares and lives lost by cause; score milestones; standing checks (exits with both doors locked, non-finite
  values).

## Gates (the full table is in the plan, section E)

1. Unit tests (`tests/atari_pixels/test_fast_last_layer.py`, `test_game_events.py`, config validation): step 0
   equals the plain bonus; one state repeated at step 0.5 gives b, b/4, b/16; a group equal to all environments
   equals a direct `fast_layer_scores` call; changing one group leaves the others bit-equal; per-copy step sizes
   equal separate calls; duplicate rows stay finite; events on edited `game.reset` states; a scripted walk off a
   ledge counts one fall.
2. Standard-library tests: the fold, the identity, the unit builder, the worker with a stand-in trainer, the claim
   filter.
3. Default path against the archive, one copy (archived seed 5, 200 updates, A4500 settings), logging off and on.
4. Default path against the archive, four copies (archived seeds 1 to 4, 100 updates, H100 settings), on serval05.
5. Resume with step 0.1, group 16, events on: 40 updates against 20 + 20, `compare_resumed.py`.
6. Speed and memory per card type. 7. One real unit through the whole path.

## Environment for every job

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png
export JAXTARI_SKIP_SPRITE_UPDATE=1
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
```

Card settings of the archive: RTX A4500 / RTX 5080 learner
`{"convolution_batching":"unrolled","cache_target_features":true,"predictor_selected_rows":1280}`, environment
`reset_lane_budget` 8, one copy; H100 learner
`{"convolution_algorithm":"space_to_depth","cache_target_features":true,"predictor_selected_rows":1280,"iteration_compiler_options":[["xla_gpu_enable_while_loop_double_buffering","true"]]}`,
`reset_lane_budget` 16, four copies. Both environments also set `separable_rectangle_drawing`,
`unroll_fixed_render_loops`, `masked_conditional_draws`, `two_way_masked_draws`, `render_final_two_together` and
`max_episode_steps: 27000` (copy the exact JSON from an archived `run.json` under
`/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_*/data/<unit>/run/`).
