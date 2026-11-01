## Purpose

Test native allocation-failure capture and a fresh small array fixture. No pixel learner is executed and this produces zero environment transitions, learner updates, packing points, cases or ideas.

## Key hyperparameters

One RTX A4500 on jaguar03, eight processors, 64 GiB host memory and twenty minutes. The first fresh child requests one unsigned-byte device buffer just larger than measured current-context capacity. The second fresh child runs seven groups of ten synchronized small-array calls.

## Code and config changes

The native-capture helper remains unchanged. Diagnostic records have a separate kind that formal capacity and packing readers must reject. Allocation identity is written separately after the owned job starts; the frozen plan is never rewritten to insert a guessed identifier.

## Git state

Numerical source provenance: d28e6735e79788f618acb0ebfde8609db1f776af. Helper source: 474907ef8623aa50f2979954ef2466e14173c05f.
