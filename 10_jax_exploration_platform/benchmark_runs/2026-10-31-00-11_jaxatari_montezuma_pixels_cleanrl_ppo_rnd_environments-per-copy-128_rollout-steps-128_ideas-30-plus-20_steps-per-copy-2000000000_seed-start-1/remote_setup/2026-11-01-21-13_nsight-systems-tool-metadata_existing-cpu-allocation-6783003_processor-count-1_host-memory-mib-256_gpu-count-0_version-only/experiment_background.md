## Purpose

Discover the actual Nsight Systems executable path, version and file hash for later independently reviewed A4500 profiling packets. This is a metadata observation in the already owned CPU monitor allocation.

## Key hyperparameters

One CPU, 256 MiB, zero GPUs and at most five minutes. Execute only a discovered executable with --version; never start a capture or profile. Use bounded ordinary-file and module metadata reads and incremental hashing.

## Code and config changes

No experiment, numerical source, active monitor, environment or configuration is changed. Standard-library-only discovery calls the published tool_record function with a bounded captured version command. Missing or ambiguous tools remain explicit limits. No scientific, case or idea credit is granted.

## Git state

Published metadata helper source is 9fcc43a478b93d731df780805fd7c3c514cdf67f. The separate reviewed profile helper-closure edit stays fixed while this metadata observation runs. Actual source and executable hashes are saved with the raw command results.
