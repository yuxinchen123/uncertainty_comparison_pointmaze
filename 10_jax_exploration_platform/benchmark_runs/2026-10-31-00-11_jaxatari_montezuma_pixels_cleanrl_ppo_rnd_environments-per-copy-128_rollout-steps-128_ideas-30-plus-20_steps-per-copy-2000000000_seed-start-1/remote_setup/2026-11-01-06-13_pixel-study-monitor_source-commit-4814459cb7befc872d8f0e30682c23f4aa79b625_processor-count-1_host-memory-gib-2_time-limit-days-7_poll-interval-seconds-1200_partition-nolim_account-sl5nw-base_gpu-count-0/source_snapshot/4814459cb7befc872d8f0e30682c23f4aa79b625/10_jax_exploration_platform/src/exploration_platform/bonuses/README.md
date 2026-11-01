# bonuses

One folder per intrinsic-reward family, and one set of hooks they all present.

`protocol.py` defines those hooks and documents when each is called. A family is a frozen record
of pure functions; the composer picks one record in Python and calls its functions directly, so
the compiled program contains that family's arithmetic and nothing else — no switch over
families, no flag inside the program, and no cost at all for a family that was not chosen.

| file / folder | what it is |
|---|---|
| `protocol.py` | what a family has to provide |
| `registry.py` | the preset names a run's configuration carries, and the factory each selects |
| `none.py` | no bonus at all: the intrinsic reward is zero everywhere |
| `rnd/` | random network distillation on the next observation |
| `visit_count/` | oracle counts over discretised position and velocity |

## Selecting a family

```python
Runner(PPOConfig(n_copies=128), bonus="rnd_next_state")
```

The name is resolved on the host and the family is bound to the run — its copy count, its base
seed, its environment — before anything is traced. `07_reconstruction`'s names for the same
families (`no_exploration`, `rnd_state`) resolve to the same presets.

That the selection is real, and not a branch the compiler still carries, is checked directly:
`tests/bonuses/test_none_has_no_bonus_arithmetic.py` builds random network distillation with two
network widths that appear nowhere else in the program, finds hundreds of array shapes carrying
them when the bonus is selected, and requires exactly zero of them — and fewer matrix
multiplications overall — when it is not.

A family that scores a state from a table rather than a network provides no trainable parameters
and no loss; the composer's arithmetic then simply has nothing to add.
