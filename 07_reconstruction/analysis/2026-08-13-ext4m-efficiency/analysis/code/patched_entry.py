#!/usr/bin/env python
"""Run the production trainer (train4m.py) with candidate optimizations monkey-patched in.

Why a patch entry point instead of editing the repo: the ext4m sweep is TRAINING RIGHT NOW out of
`/p/rlprojects/RND/07_reconstruction` (450 runs). Editing train.py / rnd.py under a live sweep would
make later runs differ from earlier ones, so every candidate is applied here, in this process only,
and the live files stay untouched. A candidate that survives measurement + the record-identity check
becomes a real patch at (re)submit time, not before.

Select candidates with EFF_PATCHES (comma separated); anything not listed keeps stock behavior:

  adam_foreach   torch.optim.Adam(..., foreach=True). On CPU torch defaults to the SINGLE-TENSOR
                 Adam loop (`_default_to_fused_or_foreach` only opts in on CUDA), so SAC's three
                 optimizers (actor, critic, entropy coef) and the RND predictor's Adam walk their
                 parameter lists in Python every gradient step. The foreach kernels do the same
                 element-wise math per element, so the update is expected bit-identical — which the
                 record-identity check verifies rather than assumes.
  sgd_foreach    the same for torch.optim.SGD (algorithm 2.3 trains its predictor with plain SGD).
  norm_cache     RND._normalize_obs rebuilds the mean/var tensors from the numpy RunningMeanStd on
                 every call (twice per gradient step, once per env step in the logging wrapper).
                 Cache them and rebuild only when the RMS actually moves (its .count changes), so
                 the tensors are value-identical to the ones the stock code builds.
  gc_freeze      gc.freeze() after construction: the long-lived env/model/graph objects stop being
                 rescanned by every generational collection. Pure bookkeeping, no math.
  interop1       torch.set_num_interop_threads(1). The worker already pins the intra-op pool to one
                 thread through OMP_NUM_THREADS; the inter-op pool is separately sized from the
                 machine's core count (46 on this node) and only costs dispatch bookkeeping here.
  mkldnn_off     torch.backends.mkldnn.enabled = False. The nets are 256x256 MLPs on batches of 256;
                 oneDNN's layout/dispatch machinery can cost more than it saves at that size. This
                 one is NOT expected to be record-identical (different kernels may sum in a
                 different order) — the record check decides.
  inference_mode RND._raw_bonus computes under torch.inference_mode() rather than torch.no_grad().
                 Same values, cheaper: inference-mode tensors carry no version counter or autograd
                 metadata. The bonus only ever feeds the reward (a constant to the critic loss), so
                 it never needs to enter a graph.

Usage (same argv as train4m.py):
  EFF_PATCHES=adam_foreach,norm_cache python patched_entry.py --ckpt_dir ... --algorithm ...
"""
import gc
import os
import sys

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))


def patch_optimizer_foreach(cls):
    """Force foreach=True on an optimizer class whose CPU default is the single-tensor loop.
    before: SAC builds Adam(params, lr=3e-4) -> torch picks foreach=False on CPU
    after:  the same call becomes Adam(params, lr=3e-4, foreach=True)."""
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("foreach", True)
        original(self, *args, **kwargs)

    cls.__init__ = __init__


def patch_norm_cache():
    """Memoize RND._normalize_obs's mean/var tensors between RunningMeanStd updates.

    before: every call runs torch.as_tensor(rms.mean) + torch.as_tensor(rms.var) (float64 numpy ->
            float32 torch) even though the statistics move only in RND.update().
    after:  the two tensors are rebuilt only when rms.count changes; identical values, one
            conversion per statistics update instead of one per call."""
    from rnd_exploration.methods.rnd import RND
    import torch

    def _normalize_obs(self, x):
        if not self.use_obs_norm or self.obs_rms is None:
            return x
        key = (float(self.obs_rms.count), x.dtype, x.device)
        cached = getattr(self, "_eff_norm_cache", None)
        if cached is None or cached[0] != key:
            mean = torch.as_tensor(self.obs_rms.mean, device=x.device, dtype=x.dtype)
            var = torch.as_tensor(self.obs_rms.var, device=x.device, dtype=x.dtype)
            # the stock code computes sqrt(var + 1e-8) per call; precompute the same value once
            inv = torch.sqrt(var + 1e-8)
            cached = (key, mean, inv)
            self._eff_norm_cache = cached
        _, mean, inv = cached
        x = (x - mean) / inv
        return x.clamp(-5.0, 5.0)

    RND._normalize_obs = _normalize_obs


def patch_inference_mode():
    """Compute the RND bonus under inference_mode instead of no_grad (same values, less bookkeeping).

    before: _raw_bonus runs its forwards inside `with torch.no_grad()` — the outputs are still
            autograd-aware tensors carrying version counters
    after:  the same forwards inside `with torch.inference_mode()` — no version counter, no graph
            metadata; the bonus only becomes a reward constant, so nothing downstream needs them"""
    from rnd_exploration.methods.rnd import RND
    import torch

    original = RND._raw_bonus

    def _raw_bonus(self, samples):
        with torch.inference_mode():
            out = original(self, samples)
        # hand back a normal tensor: inference tensors may not be used by autograd-tracked ops, and
        # the reward it feeds is combined with tensors that are
        return out.clone()

    RND._raw_bonus = _raw_bonus


def main():
    """Apply the selected patches, then hand control to train4m.main() with argv untouched."""
    patches = {p.strip() for p in os.environ.get("EFF_PATCHES", "").split(",") if p.strip()}
    import torch
    if "adam_foreach" in patches:
        patch_optimizer_foreach(torch.optim.Adam)
    if "sgd_foreach" in patches:
        patch_optimizer_foreach(torch.optim.SGD)
    if "norm_cache" in patches:
        patch_norm_cache()
    if "interop1" in patches:
        # train4m.main() itself sets this from 2026-08-13 05:07 (the owner applied the same idea in
        # the live trainer), and torch raises if it is set twice -- so this patch is now a no-op
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            print(f"[patched_entry] interop1 already applied upstream: {exc}", flush=True)
    if "mkldnn_off" in patches:
        torch.backends.mkldnn.enabled = False
    if "inference_mode" in patches:
        patch_inference_mode()
    print(f"[patched_entry] patches applied: {sorted(patches) or ['none (baseline)']}", flush=True)

    import train4m
    if "gc_freeze" in patches:
        # freeze after the trainer's construction, right before learn() burns 4M steps: wrap run4m
        original_run = train4m.run4m

        def run4m(cfg, ext):
            gc.collect()
            gc.freeze()
            return original_run(cfg, ext)

        train4m.run4m = run4m
    train4m.main()


if __name__ == "__main__":
    main()
