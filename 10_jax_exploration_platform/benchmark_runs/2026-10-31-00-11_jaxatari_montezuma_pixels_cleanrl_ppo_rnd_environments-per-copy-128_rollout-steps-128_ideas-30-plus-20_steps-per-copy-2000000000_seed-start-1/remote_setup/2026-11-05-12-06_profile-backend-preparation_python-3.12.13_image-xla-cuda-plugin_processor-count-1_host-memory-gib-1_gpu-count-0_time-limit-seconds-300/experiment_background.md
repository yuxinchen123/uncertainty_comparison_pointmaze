## Purpose

Create a fresh metadata record for the actual selected Python and the package that owns its CUDA backend image. The later qualification must compare its actual loaded bytes with this record. This preparation grants no GPU qualification or admission.

## Key hyperparameters

Use the selected Python 3.12.13, one requested CPU, one GiB and zero GPUs. The maximum step time is 300 seconds. A shared file lock permits only one large-image reader. There are no training updates or environment transitions.

## Code and config changes

Run the exact published separate profile-qualification backend preparer from helper commit d1469fd063a44167ec56760729eb0beea7ecf7d3. Read actual executable, package metadata, package record and image bytes. Do not reuse the old preparation or import numerical packages.

## Git state

The runtime helper checkout is clean and detached at d1469fd063a44167ec56760729eb0beea7ecf7d3. The integration base is 424ad85946a42eace23336baee5ea83ff57261f5. These new standards are uncommitted during preparation and will be published before execution. Unrelated original checkout work is preserved.
