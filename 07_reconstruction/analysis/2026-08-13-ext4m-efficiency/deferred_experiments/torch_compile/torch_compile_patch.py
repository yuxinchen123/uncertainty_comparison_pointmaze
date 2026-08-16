#!/usr/bin/env python
"""torch.compile the SAC gradient step — DEFERRED, not applied.

Why it is not in the applied set:
- **numerics**: inductor is free to fuse and reassociate float operations, so the trained networks
  drift from every result already recorded. That alone disqualifies it this round.
- **the gain is probably small here**: the 2026-06-25 campaign estimated ~1% for this workload. The
  nets are 256x256 MLPs on batches of 256 — little to fuse — and CPU inductor pays a compilation
  cost per PROCESS. The sweep runs 900 short-lived worker processes, so that cost is paid 900 times
  and eats into whatever steady-state win exists.

Both of those are worth measuring rather than assuming, which is what this file is for: it makes the
experiment one line, for a future sweep that is allowed a fresh baseline.

    PYTHONPATH=<this dir> EFF_PATCHES=... python patched_entry_copy.py <train4m args>
    # with, in that copy:  from torch_compile_patch import patch_torch_compile; patch_torch_compile()

Measure it the same way everything else here was measured — `analysis/code/run_conditions.py`,
exclusive node, one job at a time — and expect the record check to report DIFFERS. The question for
this candidate is not "is it identical" (it will not be) but "is the speedup worth a new baseline".
"""
import torch

from stable_baselines3.sac.sac import SAC


def patch_torch_compile(mode="default"):
    """Compile SAC.train once per process, on first call.

    before: SAC.train(gradient_steps, batch_size) runs eager torch every env step
    after:  the same method, wrapped by torch.compile, compiled on its first invocation

    `dynamic=False` keeps a single shape specialization (batch size never changes here), which is
    what makes the one-off compile cost bounded."""
    original_train = SAC.train
    compiled = {}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if "fn" not in compiled:
            compiled["fn"] = torch.compile(original_train, mode=mode, dynamic=False)
            print(f"[torch_compile] SAC.train compiled (mode={mode})", flush=True)
        compiled["fn"](self, gradient_steps, batch_size)

    SAC.train = train
