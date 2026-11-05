## Purpose

Record the first observed parent, command and CPU setting of a child launched with the selected Python interpreter on serval02. Keep the first observation separate from the later child acknowledgment. The earlier failed timing attempt did not retain its first observation, so this test cannot reconstruct that historical value.

## Key hyperparameters

Use eight allowed logical CPUs and a 300-second maximum duration. The child uses only the Python standard library. It performs zero environment transitions, learner updates and GPU operations. It never opens the standing GPU lock or changes the campaign metadata files. These CPU settings are not exclusive allocations.

## Code and config changes

A separate diagnostic saves each initial identity comparison and the later acknowledged identity. Preserve the existing parent, exact command, CPU setting, owner and process birth requirements. A later matching observation does not make an initial mismatch successful. Source and interpreter hashes will be frozen and published before the one actual invocation.

## Git state

Preparation starts from published integration commit 3314cf64. The final exact diagnostic source commit and hashes are pending independent review. The integration checkout has separate historical progress files and untracked evidence; none is overwritten by this preparation.
