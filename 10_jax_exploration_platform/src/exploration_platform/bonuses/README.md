# bonuses

One folder per intrinsic-reward family, and one set of hooks they all present.

`protocol.py` defines those hooks and documents when each is called. A family is a frozen record
of pure functions; the composer picks one record in Python and calls its functions directly, so
the compiled program contains that family's arithmetic and nothing else — no switch over
families, no flag inside the program, and no cost at all for a family that was not chosen.

| folder / file | family |
|---|---|
| `protocol.py` | what a family has to provide |
| `rnd/` | random network distillation on the next observation |
| `visit_count/` | oracle counts over discretised position and velocity |

A family that scores a state from a table rather than a network provides no trainable parameters
and no loss; the composer's arithmetic then simply has nothing to add.
