# Applied changes — the ext4m fleet's torch-level optimizations (2026-08-13)

Every change below passed the bit-exactness gate on ALL THREE sweep configurations (same seed →
identical record histories and identical model tensors, sha256-verified; see analysis.md stages
1–2), so previous results remain valid by construction: an optimized run produces byte-identical
training numbers to an unoptimized one. Combined measured effect: −3.2% wall on algorithm 2.3,
−3.3% on the run-5 original RND configuration (30k-step A/B, idle puma01, single thread) —
about 2 hours per 4M run.

1. **`--opt_polyak_foreach=True`** (worker argv → train.py Config switch, default False).
   The codebase's own `torch._foreach_` polyak target-network update replaces SB3's per-tensor
   zip loop. The whole measured gain (−3.1% alone). Applied to every run of the fleet.
2. **`--opt_torch_reward=True`** (worker argv → Config switch, default False).
   Reward combine at buffer-sample time stays in torch — the default path round-trips
   torch → numpy → torch on every gradient step. Alone within noise (−0.4%) but free, gated
   clean everywhere including the VisitCount (gt) path.
3. **`torch.set_num_interop_threads(1)`** (one line in train4m.py main, before any torch work).
   A single-thread worker has nothing for the inter-op pool to schedule. Alone within noise;
   no numeric surface at all.

Where they live:
- `slurm/ext4m_worker.py` `build_cmd` appends the two Config switches to every train4m argv;
- `train4m.py` sets the interop thread count at startup;
- `train.py` is UNTOUCHED (the 1M sweep's trainer stays byte-identical to what produced runs
  5/6), and the switches reach it only as standard argv.

Deployment: committed before the fleet swap; the swap itself waited for the running fleet's
first 0.5M-step checkpoints so every in-flight run resumed (bit-exactly) under the optimized
trainer instead of restarting.
