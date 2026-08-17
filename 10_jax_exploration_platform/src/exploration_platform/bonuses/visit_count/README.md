# visit_count

Oracle visit counts over discretised position and velocity: the bonus of a state is
`min(1, n**decay)`, where n is how often that discrete state has been reached by that copy.
Nothing is learned — the count table IS the knowledge — so a state no copy has reached scores the
maximum, 1.0.

| file | what is in it |
|---|---|
| `config.py` | `VisitCountConfig` — the decay and the velocity discretisation |
| `spec.md` | **`bonus_spec@1`**: the counted state, the table's index arithmetic, the bonus, and the `post_rollout_update@1` schedule, with the reason for each |
| `reference.py` | the same semantics as a plain python loop — the definition, written so it is obviously the definition |
| `implementation.py` | the fused form: one scatter and one gather per iteration |

Two presets: `gt_position_velocity_sqrt` (decay -0.5, so `1/sqrt(n)`) and
`gt_position_velocity_linear` (decay -1, so `1/n`). `07_reconstruction`'s name
`gt_position_velocity` resolves to the sqrt preset, which is that project's own default.

The table is `int32` and `rows*cols*100` wide per copy — 10,800 entries for the large maze, 43 KB
per copy, about 0.35 GB at 8,448 copies.

This family cannot score one rollout step at a time: its bonus is read from a table that has to
have counted the whole rollout first. Composing it with `hoist_rollout=False` is therefore refused
when the program is built, rather than quietly producing different numbers.
