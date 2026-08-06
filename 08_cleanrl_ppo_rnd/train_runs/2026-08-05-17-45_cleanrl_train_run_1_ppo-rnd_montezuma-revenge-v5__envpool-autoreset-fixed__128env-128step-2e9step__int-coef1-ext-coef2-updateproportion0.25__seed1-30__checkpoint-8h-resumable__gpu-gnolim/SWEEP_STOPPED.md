# Stopped and superseded, 2026-08-05

This run trained a single configuration — CleanRL's PPO + Random Network Distillation on
MontezumaRevenge, 30 seeds — and was stopped by the owner about two hours in, at roughly 0.5% of its
step budget, to be replaced by a five-arm ablation of the same implementation.

Its outputs were deleted on the owner's instruction: the per-run records, the checkpoints, the queue,
the canary tree and the job logs. What is kept is the design: `experiment_background.md`,
`configs.jsonl`, the resource estimate, and the submission scripts.

Nothing was learned from it that the replacement does not also cover; it ran for two hours and no run
reached 0.7% of its steps. What it did establish, and what the replacement inherits, is in
`../../shuze_experiment/2026-08-05_profilling/`: the per-node throughput measurements, the packing
and processor-count curves, and the two defects the canary exposed.

The replacement run folder is the five-arm ablation created the same day.
