## Purpose

Prepare one literal held submission for each independently reviewed intermediate common-confirmation packet. This utility submits only the selected copy count, once, and never releases a job.

## Key hyperparameters

Each packet requests one A4500, eight processors and 64 GiB. One policy copy requests two hours; two policy copies request three hours. Every measurement retains seven alternating paired rounds of ten full updates per arm, 128 environments per copy, 128 rollout steps, four epochs, four minibatches, fifty warmup rollouts and the full two billion transition annealing schedule.

## Code and config changes

Reuse the previously reviewed held-submission mechanism: write an exclusive intent, retain raw return bytes before parsing, synchronize both identifier ledgers and the exact submission row, then save private ownership snapshots and a separate held binding. Any partial attempt refuses further submission. Actual job and physical GPU identities remain unresolved until Slurm returns them. The common candidate changes only background caching and final-two rendering.

## Git state

The clean published helper source is 9fcc43a478b93d731df780805fd7c3c514cdf67f. The numerical source remains d28e6735e79788f618acb0ebfde8609db1f776af. Both exact source closures, actual prior correctness results and reviews, target qualification and reviewed reuse scope are retained in the packet records. The utility must itself be reviewed and published before any held submission.
