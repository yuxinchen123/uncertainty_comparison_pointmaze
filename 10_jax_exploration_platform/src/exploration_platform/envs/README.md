# envs

One folder per environment; each holds the physics and the batched JAX stepper for it.

| folder | family | physics | obs / act | spec |
|---|---|---|---|---|
| `pointmaze/` | PointMaze (umaze / open / medium / large) | hand-derived exact contacts, probe-verified against Gymnasium-Robotics | 4 / 2 | `09_parallelization` physics_spec.md |
| `antmaze/` | AntMaze (umaze / medium / large) | MuJoCo MJX over the vendored Gymnasium ant.xml plus the maze walls | 29 / 8 | `antmaze/spec.md` |
| `atari_montezuma/` | Montezuma's Revenge | JAXAtari's pure-JAX game under its AtariWrapper (v5 sticky actions), object-centric observation | 774 / 18 discrete | `atari_montezuma/spec.md` |

Every stepper exposes the same contract the trainer composes against: `obs_dim` / `act_dim`,
`reset()`, `step(state, act) -> (state', obs, reward, terminated, truncated, final_obs)`,
`respawn(state)`, `cell_index(obs_flat)` and the `open_cells` / `n_cells` coverage mask. A
discrete-action family additionally sets `action_kind = "discrete"` and `n_actions`; the
composer then builds the categorical actor instead of the Gaussian one.
