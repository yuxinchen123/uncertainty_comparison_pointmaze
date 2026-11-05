## Purpose

Collect the four independently accepted common performance contexts into the existing prerequisite format for profile-source qualification. This metadata step grants no qualification, worker admission or scientific credit.

## Key hyperparameters

The inputs cover H100 and RTX A4500, each with one and two independent training copies. Each accepted context used seven paired rounds of ten complete updates per arm and the original configuration. This step requests one CPU, 512 MiB and zero GPUs for at most 300 seconds. It performs no training or environment transitions.

## Code and config changes

Use the independently reviewed metadata producer and the published exact prerequisite reader from commit d1469fd063a44167ec56760729eb0beea7ecf7d3. Reuse the accepted initial snapshot and existing reviews. The output requires a separate independent review.

## Git state

The preparation base is 87fbd790e41e2abfb7d54eeaa77a0a2d4a440d73. These new packet files and current campaign records are uncommitted at preparation; publish the exact reviewed files before execution and record the actual publication separately. The runtime helper checkout is clean and detached. Unrelated original checkout edits are preserved.
