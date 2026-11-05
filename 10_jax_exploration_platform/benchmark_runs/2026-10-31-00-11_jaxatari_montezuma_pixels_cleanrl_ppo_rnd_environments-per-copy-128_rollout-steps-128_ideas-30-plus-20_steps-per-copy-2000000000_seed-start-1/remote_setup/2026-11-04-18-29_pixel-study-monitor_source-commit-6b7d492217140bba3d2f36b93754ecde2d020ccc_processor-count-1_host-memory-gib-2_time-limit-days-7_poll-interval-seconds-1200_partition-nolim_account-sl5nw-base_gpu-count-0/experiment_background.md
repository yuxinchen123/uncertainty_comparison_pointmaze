## Purpose

Prepare a replacement for the standing persistent CPU monitor from the reviewed source, including the corrected reporting of remaining validation work. The old monitor stays running.

## Key hyperparameters

One requested CPU, two GiB host memory, zero GPUs, seven days, no requeue, and one complete report every 1,200 seconds on the nolim partition under sl5nw_base with normal quality of service.

## Code and config changes

Before: the standing monitor uses source 79a01e851ff92ddba7d0e12eaf8164cab2028878. After: this packet uses source 6b7d492217140bba3d2f36b93754ecde2d020ccc. The reviewed allocation guard functions and resource shape are unchanged. The candidate is checked on the existing CPU allocation. A future actual replacement must pass independent runtime and first-full-report review before the old monitor is stopped, after its auxiliary steps finish. This infrastructure work grants zero experiment credit.

## Git state

Commit: `6b7d492217140bba3d2f36b93754ecde2d020ccc`. The preparation source is a clean detached checkout.
