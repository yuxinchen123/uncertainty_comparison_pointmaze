# agents/ppo

Proximal policy optimisation with two value heads, batched over C independent copies. Every
parameter carries a leading copy axis, every forward is a batched matmul, every reduction keeps
the copy axis, the scalar loss is the SUM over copies, the gradient clip is per copy, and Adam is
elementwise — so one program is C independent trainers that never touch each other.

| file | what is in it |
|---|---|
| `config.py` | `PPOConfig` — every knob of the agent and of the loop that drives it |
| `networks.py` | the actor and the two-headed critic: their keyed per-copy initialisation and their forward passes |
| `losses.py` | `ppo_loss_per_copy` — the policy, value and entropy terms, returned as a [C] vector so a bonus's [C] vector can be added to it before the single sum |
| `update.py` | the per-copy gradient-norm clip, the hand-rolled Adam, and the two update styles, each built as its own function |

The agent knows nothing about any bonus. What ties them together is
`../../training/compose.py`, which picks one of each and only then calls `jax.jit`.

`SWEEP.md` beside this file describes sweeping a knob across copy groups.

This folder was the single 701-line module `jax_ppo_rnd.py` copied from
`09_parallelization/ppo/jax_ppo/` (see `../../../../ORIGIN.md`). The split is checked by the
golden-parity gate in `tests/golden_09/`, which requires the composed program to reproduce the
frozen baseline bit for bit.
