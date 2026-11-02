## Purpose

Prepare a request for later intermediate timing for the unconfirmed common combination of room-background caching and rendering only the final two game states on jaguar03. No experiment has been submitted or executed by this preparation.

## Key hyperparameters

1 independent policy copies, 128 pixel environments per copy, 128 rollout steps, four epochs, four minibatches, fifty normalization warmup rollouts, seed start one, and the original two-billion-transition annealing horizon. Networks and Adam use float32; statistics use float64; matrix precision is highest and deterministic operations remain required. Requested later timing retains seven alternating paired rounds with ten full updates per arm. No formal timing plan exists until actual correctness and independent review exist.

## Code and config changes

Only environment.cache_room_backgrounds and environment.render_final_two_only change from false to true. Every other full configuration field remains unchanged. The new startup wrapper will be reviewed separately; no numerical source is changed.

## Git state

Numerical source is d28e6735e79788f618acb0ebfde8609db1f776af. The existing clean published helper source is e5afbf6b51a3d1666b46e71c0631a45d47080cd8. Integration and original checkout edits are excluded from this preparation.
