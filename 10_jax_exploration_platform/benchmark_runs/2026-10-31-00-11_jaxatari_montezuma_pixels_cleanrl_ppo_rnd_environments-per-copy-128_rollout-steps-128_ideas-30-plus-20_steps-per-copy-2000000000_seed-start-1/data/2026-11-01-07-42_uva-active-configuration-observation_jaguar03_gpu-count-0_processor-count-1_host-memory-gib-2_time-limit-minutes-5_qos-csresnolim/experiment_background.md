## Purpose

Observe the actual owned CPU allocation and Slurm configuration paths. No GPU library, device or learner is used. Missing or untrusted configuration remains evidence, and this observation grants no physical assignment, qualification or training admission.

## Key hyperparameters

One requested processor, two GiB host memory, five minutes and zero GPUs on jaguar03. Record actual allocated processors and affinity, including simultaneous multithreading.

## Code and config changes

Read only /etc/slurm and the conf-cache directory under the actual SlurmdSpoolDir reported by Slurm. Filter slurm.conf to the explicit non-secret setting list; retain ordinary gres.conf and cgroup.conf bytes and path ownership metadata.

## Git state

Observer source: 3df6419cb39ac20766af902c49f3b65a9e219d8a.
