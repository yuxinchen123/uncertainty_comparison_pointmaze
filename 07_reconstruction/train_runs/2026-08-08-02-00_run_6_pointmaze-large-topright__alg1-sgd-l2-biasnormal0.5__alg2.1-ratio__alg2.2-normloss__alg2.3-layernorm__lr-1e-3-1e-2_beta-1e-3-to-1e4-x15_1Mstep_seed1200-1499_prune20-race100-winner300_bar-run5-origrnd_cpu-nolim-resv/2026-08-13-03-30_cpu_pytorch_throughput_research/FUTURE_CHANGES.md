# Deferred improvements — meaningful but NOT applied (they would change the numbers)

The user's rule for this research: applied changes must keep previous results valid. Everything
below fails that bar (it changes logged values, the training math, or bit-level numerics) while
still being worth real throughput. Each entry names the idea, the expected gain, why it breaks
validity, and what a future run would need. Concrete artifacts live in `future_improvements/`.

1. **(WITHDRAWN by the profile.)** The imagined evaluation cost does not exist in production:
   the sweep configs run `eval_standalone=False`, so scores come from training episodes and the
   eval callback measured 0.0% of wall (analysis.md stage 4). Kept here as a corrected belief,
   not a plan.
2. **`torch.compile` on the SAC policy/critic and RND nets.** On CPU with small MLPs the
   compiled kernels can win 10-30%, but compiled kernels are not bit-identical to eager
   (fusion reorders floating-point reductions), and warmup adds minutes per run. Gate-fails by
   construction; worth testing wholesale on the NEXT experiment family.
3. **`torch.set_flush_denormal(True)`.** RND predictor errors shrink toward denormal range late
   in training; denormal arithmetic is slow on x86. Flushing to zero can win where runs stall —
   but it literally changes tiny values, so it is not bit-exact.
4. **Per-step intrinsic logging removal.** `ComputeIntrinsicRewardWrapper` runs an RND forward
   on EVERY env step purely to log per-episode intrinsic reward sums; the training bonus is
   recomputed at sample time anyway. Dropping it removes an entire forward per step — but
   `train/intrinsic_reward` in train_episode_history comes from it, and that field is part of
   the record format. A future record-format revision could sample it.
5. **Optimizer `foreach=True` for SAC's Adam / the RND optimizer on CPU.** Multi-tensor Adam
   batches the parameter-group loop; numerics are equivalent-but-not-guaranteed-bit-identical,
   so it belongs with a gate in front of it on the next family (if it gates clean there, it can
   be promoted).
6. **Vectorized environments (several mazes per worker).** SB3's DummyVecEnv with n_envs > 1
   amortizes python overhead, but changes the data-collection schedule (train_freq semantics),
   i.e., the training math. Structurally the biggest python-side win; needs its own validation
   run.
