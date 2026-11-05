# training

Where the environment, the agent and the bonus meet, and the only place that calls `jax.jit`.

| file | what is in it |
|---|---|
| `state.py` | `TrainState` — the whole of a run's mutable state — and `ParamLayout`, which knows where each named tensor sits in the one-array parameter form |
| `sweep.py` | splitting the copies into groups and turning a group's setting into a per-copy device vector |
| `train_step.py` | one training iteration and one warm-up pass, built for exactly one (environment, agent, bonus) combination |
| `compose.py` | picks the three pieces, fixes the parameter layout, and compiles the two programs |
| `runner.py` | the driver: build a state, warm it up, step the compiled iteration in a loop, record sparsely |

The compiled iteration takes a state and a learning-rate scalar and returns a new state and the
iteration's metrics. Its key comes from the state (`fold_in(rng, step)`), so the driver keeps no
counter beside it and a state fully describes where a run is.
