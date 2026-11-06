## Purpose

Train the selected fast implementation of CleanRL's PPO with Random Network Distillation on pixel
Montezuma's Revenge (JAXtari recreation) for two billion training steps per independent copy, after
correcting the game. Before the long run, re-check the selected configuration on the corrected game:
speed at the selected packing, a learning check against the original reference, and a
stop-and-resume comparison.

Why the game was corrected (plan section 198): the alternative sprite pack installed for JAXtari
(the only pack available without declaring ownership of the Atari ROMs) ships room images twice, as
`.png` (2026-05-02) and `.npy` (2026-10-01). The game reads only the `.npy` files, and for five rooms
(the start room, the level-1 middle room, the pit room, level-2 rooms 0 and 6) the `.npy` collision
map and background are a copy of the corridor room: solid everywhere below row 47. In the start room
the player could then reach the conveyor only by climbing, was carried to x = 54 and stayed there
forever, and no fall was long enough to lose a life. Every environment of every earlier learning check
ended there. The newer upstream JAXtari (`dev` at ae1d7a0a, 2026-10-30) has the same problem and is
about five times slower per game step. The corrected pack takes those rooms from the same pack's
`.png` images.

## Key hyperparameters

| Parameter | Value |
|---|---|
| algorithm | CleanRL `ppo_rnd_envpool.py` equations and hyperparameters (one setting for all copies) |
| environments per copy | 128 |
| rollout steps per update | 128 |
| updates per copy | 122,070 (1,999,994,880 training steps; 15,624,960 per environment) |
| observation warm-up | 50 rollouts (CleanRL) |
| episode limit | 27,000 agent steps (108,000 frames, as EnvPool) |
| sticky actions | 0.25; up to 30 reset NOOPs; life loss ends the PPO episode |
| precision | float32 networks at highest matrix precision, float64 running statistics, deterministic GPU operations |
| H100 NVL (serval02 card 1) | 4 copies packed, seeds 1 to 4; `convolution_algorithm=space_to_depth`, cached target features, predictor on 1,280 selected rows, while-loop double buffering |
| RTX A4500 (jaguar03, 8 cards) | 1 copy per card, seeds 5 to 12; `convolution_batching=unrolled`, cached target features, predictor on 1,280 selected rows |
| environment speed settings (both) | cached room backgrounds, last two states rendered (in one call), separable rectangle drawing, unrolled fixed render loops, masked one- and two-way draws |
| game | JAXtari 04ab9aec (installed package) with the corrected sprite pack (`XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png`) |
| checkpoints | complete state every 20 minutes, at stop signals and at the end (`train_final.py`) |

## Code and config changes

- New sprite pack built by `scripts/atari_pixel_campaign/build_montezuma_room_geometry_assets.py`:
  every Montezuma `backgrounds/*.npy` with a same-size opaque `.png` is replaced by that `.png`'s pixels
  (30 files; `pitroom_level_3_diff_3.npy` kept, its `.png` is a 1550 x 670 map); its `asset_source.json`
  lists each file's checksums.
- `environment.py`: refuses a pack whose start room is solid below the top floor
  (`require_room_geometry`); episode limit default 27,000 agent steps (was 108,000); exposes `room_count`.
- `training.py`: each update records which rooms each copy's games were in (`rooms_visited`).
- `train_final.py`: the run identity includes the game source and the asset pack's record checksum.
- Tests: `tests/test_pixel_montezuma_room_geometry.py` (new); the small test environment gains the
  `render_final_two_together` attribute.

## Git state

Recorded in `command.txt` and the commit that adds this folder (branch `merge-atari-cleanrl`).
