## Purpose

Verify complete checkpoint restart, original single-update fallback and interrupted checkpoint publication on one allocated RTX A4500. This is a bounded component validation with zero study or final-training credit.

## Key hyperparameters

The two frozen fixtures use seed 11, one policy copy, one pixel environment, two rollout steps, seven updates, one training epoch, one minibatch and zero observation warmup iterations. The grouped fixture resumes with both three-then-one and one-then-three update orders. The scheduled-report fixture reports every four updates and recovers interrupted publication at update three. Networks use float32 and statistics use float64, with highest matrix precision and deterministic operations. The allocation requests one typed A4500 on jaguar03, eight processors, 64 GiB host memory and a 40-minute limit.

## Code and config changes

The exact two fixtures are loaded from committed helper source 8a96999408d4e214b212074d932f0eeabcbb5eea. All ten pixel numerical files are checked byte for byte against D28. The source archive includes only committed src and scripts files. The packet prepares an owned held submission; it submits, releases and admits no worker.

## Git state

Helper commit: `8a96999408d4e214b212074d932f0eeabcbb5eea`. Committed source bytes match the working source paths. Other campaign records continue to change during monitoring. Source and packet hashes are recorded separately.
