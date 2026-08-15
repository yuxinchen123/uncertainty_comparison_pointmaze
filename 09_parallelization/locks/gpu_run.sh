#!/bin/bash
# Run a command on serval05 under the exclusive H100 lock.
# Usage (from any machine that can ssh serval05):  bash gpu_run.sh "<command>"
# The lock is a serval05-local flock; a second caller BLOCKS until the first releases it,
# so two agents/sessions never profile or train on the GPU at the same time.
# -w 21600: give up after 6 h of waiting rather than queue forever silently.
set -euo pipefail
CMD="$1"
exec ssh -o BatchMode=yes serval05 "mkdir -p /localtmp/sl5nw/locks && flock -w 21600 /localtmp/sl5nw/locks/h100.lock bash -lc $(printf '%q' "$CMD")"
