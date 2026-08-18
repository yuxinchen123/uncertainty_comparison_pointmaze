# Montezuma's Revenge on the platform — reference semantics and deviations

Specification name: `montezuma_oc4500_sticky@1`.

Montezuma's Revenge is the hard-exploration Atari game the original random-network-distillation
paper (Burda et al., 2018) was built to solve. The platform's implementation is **JAXAtari's
pure-JAX reimplementation of the game** — repository `k4ntz/JAXAtari`, installed at commit
`035fc46bf149220a2f462c864873707ced334a4c` (package `jaxatari` 0.1.0, MIT license), the most
recent version at install time (2026-08-18), where the game carries the project's top quality
rating — wrapped in JAXAtari's own `AtariWrapper` and adapted to the platform contract by
`jax_montezuma.py`. The game-behaviour reference JAXAtari itself validates against is
`ALE/MontezumaRevenge-v5`.

Game assets: the environment's collision maps and sprites are JAXAtari's **alternative
(replacement) asset package** — the package the project provides for users who do not declare
ownership of the original Atari 2600 ROM — installed to `~/.local/share/jaxatari/sprites/` by
`python -m jaxatari.install_sprites` (ownership declined). No original ROM assets are used or
distributed.

## What follows the ALE-v5 / RND conventions

| item | value | source |
|---|---|---|
| action set | the game's full 18 ALE actions | `JaxMontezumaRevenge.ACTION_SET` |
| sticky actions | with probability 0.25 the previous action repeats — v5's `repeat_action_probability`, the Machado et al. protocol | `AtariWrapper(sticky_actions=0.25)` |
| no-op starts | none (`noop_max=0`): sticky actions are the stochasticity protocol that replaced them | `AtariWrapper` |
| frame skip | 4: one env step holds the action for 4 game frames | the adapter's step scan |
| frame stack | 4 most recent (post-skip) frames in the observation | the adapter |
| episode end | lives exhausted terminates (the game's own done); no episodic-life splitting | `AtariWrapper(episodic_life=False)` |
| episode cap | 4,500 env steps = 18,000 frames, the RND paper's cap; truncation only | `MontezumaConfig.max_episode_steps` |
| reward | the score delta, clipped to its sign — the RND convention | `MontezumaConfig.clip_reward` |
| lives | 5, as in the game | JAXAtari |

## Deviations, and why

1. **Object-centric observation, not pixels.** The original RND setup feeds 84x84 grayscale
   pixel stacks to convolutional networks. The platform's agent and bonus are the MLPs every
   other family uses, so this environment observes JAXAtari's object-centric observation —
   positions, sizes, activity, type and orientation of the player and every object group —
   flattened to 192 features per frame and stacked over 4 frames (768 features). A pixel/CNN
   variant is recorded as future work in the family ledger, not silently substituted.
2. **The room index, inventory and lives are appended to the observation** (6 features; 774
   in all). The reference's pixel observation distinguishes rooms visually; the object list
   alone does not name the room, so without this every room would alias onto every other and
   a novelty bonus could not tell new rooms from old. Same information the pixels carry,
   packaged explicitly.
3. **Deterministic resets up to the sticky-action stream.** No no-op starts; each environment
   holds its own PRNG key (seeded from base seed, copy seed index and environment index, per
   the platform's pairing convention), consumed by sticky-action draws and re-split across
   episodes, so auto-reset never replays the same stickiness sequence.
4. **Float32 under a trace-time x64-off boundary**, exactly as the MJX AntMaze: the platform
   enables x64 globally for its running statistics, and JAXAtari's integer game logic assumes
   the jax default, so every call into the game is traced under `jax.enable_x64(False)`.
5. **Coverage counts rooms.** The platform's visited-cell machinery indexes the 24 playable
   rooms (via the game's own `get_room_idx`), so "coverage" for this family is the classic
   Montezuma rooms-visited metric.

## Correctness gates

Run by `tests/envs/test_montezuma_adapter.py` and `tests/envs/test_montezuma_compose.py`:

1. Contract: shapes/dtypes of the six step outputs; observation is 774 float32 features; the
   appended room index equals the game state's room.
2. Determinism given keys: the same actions under the same base seed give bitwise-equal
   trajectories; different environment indices draw different sticky streams.
3. Semantics: reward is the clipped score delta (first reachable score in room 4 pays +1);
   truncation at the cap; auto-reset restores a fresh episode with 5 lives and advances the
   key; copies are bit-isolated.
4. The x64 boundary holds under the platform's global setting.
5. End to end: the fused iteration composes and trains with `rnd_next_state` and `none`;
   the PointMaze-only visit-count family is refused.
