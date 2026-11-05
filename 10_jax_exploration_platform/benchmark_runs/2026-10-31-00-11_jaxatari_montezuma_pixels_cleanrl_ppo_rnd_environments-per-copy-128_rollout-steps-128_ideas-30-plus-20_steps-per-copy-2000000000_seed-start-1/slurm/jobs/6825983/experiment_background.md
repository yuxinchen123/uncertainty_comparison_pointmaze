## Purpose

Prepare a replacement for the standing persistent CPU monitor from the reviewed source, including the corrected reporting of remaining validation work. The old monitor stays running.

## Key hyperparameters

One requested CPU, two GiB host memory, zero GPUs, seven days, no requeue, and one complete report every 1,200 seconds on the nolim partition under sl5nw_base with normal quality of service.

## Code and config changes

Before: the standing monitor uses source b431573a272ccec354cfbccb2b7528b4b846e7b9. After: this packet uses source 548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e. The reviewed allocation guard functions and resource shape are unchanged. The candidate is checked on the existing CPU allocation. A future actual replacement must pass independent runtime and first-full-report review before the old monitor is stopped, after its auxiliary steps finish. This infrastructure work grants zero experiment credit.

## Git state

Commit: `548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e`. The preparation source is a clean detached checkout.
