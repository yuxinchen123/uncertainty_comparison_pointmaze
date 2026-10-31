## Purpose

Diagnose the held two-policy gathered-minibatch update memory failure in one fresh A4500 process. This is separate diagnostic evidence and grants no additional case attempt or optimization idea.

## Key hyperparameters

Each of two policies retains 128 pixel environments, 128 rollout steps, four epochs, four minibatches, fifty normalization warmup rollouts, and seeds one and two. Seven alternating timing pairs contain one complete update per arm. Neural arithmetic remains float32 with highest precision, statistics float64, and XLA operations deterministic.

## Code and config changes

Use the installed CUDA asynchronous allocator with the original 0.85 memory fraction and disabled preallocation. No large preparation array is used. The original frozen numerical functions and complete checks receive separate memory observations outside their work. Missing allocator counters are recorded as unavailable.

## Git state

Numerical source d28e673 and committed helper 5b055815757e8219747bedf87423ae791af054fe. Exact source, case, failure, parser and command hashes are in launch_plan.json. The installed parser and official JAX allocation documentation support the selected allocator option.
