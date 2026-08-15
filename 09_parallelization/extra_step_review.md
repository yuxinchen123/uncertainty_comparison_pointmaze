# Extra step — technique diff against the joint-replenishment efficiency campaign

Written 2026-08-15 for Phase 4 of `efficiency_improvement_user_prompt.md`. Source read in full:
`/p/rlprojects/RLforOR/inventory_management/joint_replenishment/efficiency/` — `PLAN.md`, `LOOP.md`,
`autoresearch_distilled.md`, `harness/bench_protocol.md`, all seven `module*/…/progress_and_changes.md`,
`module1_env/shared_notes.md`, `module2_arch/shared_notes.md`, `module1_env/literature_notes.md`,
`module3_unified/design.md`, all `ideas.md`, `e2e_campaign/{campaign_log,report,memo_fused_triton}.md`,
`e2e_campaign/v2_report/report.md`, `final_delivery/{report,gap_checklist,campaign_log}.md`,
`unified_report/report.md`, `reviews/2026-08-12_{measurement,plan_vs_prompt}_review.md`, and
`decided_model_infra/2026-08-14-v1-captured-e2e-recommended-precision/README.md`.

Compared against: `STRUCTURE.md`, `PROGRESS.md`, the seven `progress_and_changes.md` files,
`ppo/research/perf_tricks_checklist.md`, `ppo/torch_ppo/torch_ppo_rnd.py`,
`pointmaze/torch_env/torch_pointmaze.py`, `ppo/torch_ppo/tests/test_capture_gpu.py`, and the
scripts in `benchmarks/`.

Reference campaign headline, for calibration: environment 50x (PyTorch) / 57.6x (Triton) / 60.8x (JAX)
over their own version 0; training pipeline 1.82x; end to end 1.633 -> 0.1790 seconds per update, 9.1x.

---

## 1. Techniques used in the reference campaign that are missing here

### 1. Pack the two identically-shaped towers into one batched matmul

**Reference evidence.** `module3_unified/progress_and_changes.md` experiment 2: "per-layer weight
stacking turns each pair of latency-bound tower matmuls into one batched matmul: **+47.8% overall**,
worst cell +21.8%". `module6_e2e/progress_and_changes.md` experiment 4 repeated it inside the captured
update — weight packs built once per update plus a single (2, 3d, d) matmul for all six projections —
for **+13.6%** geometric mean. `e2e_campaign/v2_report/report.md` measured the same change on the
uncaptured stack at **+21.7%**, and notes it "also sheds 2400 launches of host time".
`module2_arch/jax/progress_and_changes.md` experiment 2 measured the same idea under XLA at **-11.5%**
and states the rule: no optimization ports across frameworks unmeasured.

**Module.** 2 (trainer) and 3 (end to end), PyTorch only. Do not port to the JAX trainer.

**Concrete experiment.** In `torch_ppo_rnd.py` the actor trunk and the critic trunk have identical
shapes (4 -> 64 -> 64), take the *same* input, and use the same activation — `_mlp2` is called twice on
`obs` in `_one_step_pure` and twice on `mb["obs"]` in `_losses`. The RND target trunk and predictor
trunk likewise share the input and the first layer shape (4 -> 256). Store the parameters packed from
the start — actor and critic trunk weights as one `[2, C, in, out]` tensor viewed as `[2C, in, out]`
for `baddbmm`, same for the RND first layer — so no per-call `torch.stack` exists at all, and the
optimizer sees fewer, larger tensors. Gate with the existing bitwise capture test (the reference's
tier A passed at 1e-6 for this change), then a paired speed run at C = 8 / 128 / 1024.

### 2. Put the whole rollout horizon inside one CUDA kernel

**Reference evidence.** `module1_env/cuda/progress_and_changes.md` experiment 1: moving the time loop
inside the kernel — one program owns its environment rows for the whole horizon, state kept in
registers, no grid barrier because environments never interact — was **20.57x**, every cell between
11.8x and 23.5x. Their version 0 was a per-step fused kernel, exactly what `pointmaze/cuda_env` is
today. Experiment 2 then captured the whole rollout as a CUDA graph for a further **1.87x**.
`module1_env/shared_notes.md` states the geometry rule that comes with it: "a rollout-level kernel
needs a DIFFERENT launch geometry than a per-step kernel … size the grid to the device (a few programs
per multiprocessor) before sizing the tile", and experiment 7 measured **+5.8%** from raising the
program count per multiprocessor by 4x, while experiment 6 measured **-18%** from the opposite change.

**Module.** 1 (environment), CUDA implementation.

**Concrete experiment.** Add a second entry point to `pointmaze_kernel.cu` that takes a horizon `T`
and an action buffer `[T, C, N, 2]` and runs the period loop inside the kernel, keeping `pos`, `vel`,
`goal`, `step_count`, `reset_count` in registers across all `T` steps. Keep the existing per-step
kernel for the correctness path. Then sweep the rows-per-program constant so the grid is a few programs
per multiprocessor. Report the per-step kernel and the whole-rollout kernel as separate curves — the
gap between them is the launch cost, measured rather than argued (the reference's C++ track made the
same split deliberately).

### 3. Hoist the RND target features out of the minibatch loop

**Reference evidence.** `module6_e2e/progress_and_changes.md` experiment 1, the campaign's second
largest single win: "the loss phase … recomputed, with a whole 4-layer transformer forward over H*B
rows, the exact logit the rollout's compiled step had already produced. The step now returns its logit"
— **+17.61%**, tier A 6.5e-07, and about 4 GB freed. The pattern is stated generally in
`module6_e2e/ideas.md`: "the loss-phase forward is a pure recomputation of what the compiled rollout
step already computes."

**Module.** 2 (trainer) and 3, PyTorch and JAX.

**Concrete experiment.** `_losses` calls `rnd_features(mb["rnd_input"])`, which runs the frozen target
network. The target network is frozen and `_U["rnd_input"]` is fixed for the whole update, so in style B
the same target features are recomputed **16 times per iteration** (4 epochs x 4 minibatches) on
identical rows. The target trunk is the widest network in the trainer (4 -> 256 -> 128). Compute
`target(_U["rnd_input"])` once in `_post_body`, store it as `_U["rnd_target_feat"]` of shape
`[C, B, 128]`, and gather it per minibatch alongside the other keys. This is exactly bit-identical
(the target has no gradient), so the existing bitwise capture test is the whole gate. The honest
caveat the reference's own rule demands: a gather of `[C, 128, 128]` floats replaces a forward, so
measure rather than assume — and keep it under the deletion clause if it comes out non-inferior.

### 4. Remove the redundant bootstrap critic forward

**Reference evidence.** Same experiment as item 3, and it is the exact shape the reference's own idea
list predicted before measuring — `module2_arch/pytorch/ideas.md` item 3, "delete the per-step critic
forwards … `values` are only used for GAE and are computed with the same weights the loss phase
re-forwards anyway. Removes 100 of 300 small forwards. Exact (bit-identical values)."

**Module.** 2 and 3, both frameworks.

**Concrete experiment.** `_post_body` runs `self.critic_values(nobs_buf…)` over all `T*N` stored next
observations, on top of the `T` per-step critic forwards the rollout already ran. For every row that
did not reset, `nobs_buf[t]` equals the entry observation of step `t+1`, so `vext_next[t]` equals
`vext_buf[t+1]` exactly. Under the default configuration `continuing_task=True`, so nothing ever
terminates, every environment truncates at exactly step 400, and resets are synchronized: about
**1 row in 400 (0.25%)** actually needs a fresh value. Rebuild `vext_next`/`vint_next` by shifting the
rollout's own value buffers, evaluating the critic on `final_obs` only at `t = T-1` and at reset steps.
Two smaller deletions fall out of the same reading and belong in the same experiment under the
simplicity criterion: `boot_mask` is identically zero under `continuing_task=True`, so the
`(1 - boot_mask[t])` multiply and the whole `b["term"]` buffer are dead work.

### 5. Compile the post-rollout phase

**Reference evidence.** `module1_env/pytorch/progress_and_changes.md` experiment 6, the environment
track's largest single keep: handing 25 steps at a time to one compiled function — the Python loop
unrolls at trace time — was **3.84x**, uniform across cells (3.69x to 4.20x).
`module1_env/jax/progress_and_changes.md` experiments 2 and 6 measured the same lever as scan unroll
(**1.89x** then **1.11x**), and experiment 4 measured **1.14x** from deleting one device-wide reduction
per step. The counter-evidence is equally clear and must be respected:
`module3_unified/progress_and_changes.md` experiment 1 measured the same chunking idea at **+0.12%,
not significant**, once the body was already kernel-bound, and states why — "three or four milliseconds
of transformer kernels per step dwarf the Python-loop microseconds the 25-step chunk removes (the
opposite regime from the bare environment, whose kernels are microseconds)."

**Module.** 2 and 3, PyTorch.

**Concrete experiment.** `_post_body` is **not compiled at all** — only `_one_step_pure` and `_losses`
are. It contains a 128-iteration Python loop for the intrinsic filter (`f = gamma*f + rint[t]`) and a
128-iteration Python loop for the two-stream GAE with two buffer writes each, plus the permutes,
reshapes and copies. Inside the captured iteration graph the launches are free, but the kernel *device
time* stands (the reference's fused-kernel memo makes exactly this point: "capture makes launching them
free, but their device time stands"). Compile `_post_body` with `fullgraph=True, dynamic=False` so the
two scans fuse. Only after that, and only if the diagnostic in item 13 shows the rollout is not already
kernel-bound, try a K-step compiled chunk of the rollout body (K = 8, 16, 32) — our networks are tiny
multilayer perceptrons, so we may be in the environment regime where it pays rather than the
transformer regime where it did not.

### 6. `max-autotune-no-cudagraphs` on every compiled region

**Reference evidence.** `module6_e2e/progress_and_changes.md` experiment 5: one keyword argument,
**+5.0%** geometric mean under capture (+6.01/+4.83/+5.13/+3.98% per cell), tier A 7.0e-06.
`e2e_campaign/v2_report/report.md` measured **+7.2%** on the uncaptured stack. The warning is explicit
and matters here: "the -no-cudagraphs mode is mandatory: plain max-autotune's cudagraph trees collide
with the manual whole-update capture."

**Module.** 2 and 3, PyTorch.

**Concrete experiment.** Change the two `torch.compile` calls in `PPORND.__init__` to
`mode="max-autotune-no-cudagraphs"`. Pin `TORCHINDUCTOR_CACHE_DIR` to serval05 local disk so the
autotune cost is paid once. Report cold-compile time separately from steady state. This changes kernel
selection and therefore float association, so it needs the behavioural gate of item 15, not only the
bitwise capture test.

### 7. Reduced precision in the trunk matmuls only, gated behaviourally

**Reference evidence.** The campaign's clearest cross-cutting law, measured twice on the same code.
`module2_arch/pytorch/progress_and_changes.md` experiment 7 measured bf16 autocast at **-14.7%** on the
launch-bound eager stack. `module6_e2e/progress_and_changes.md` precision arm measured the same change
at **+41.3% to +75.1%** (about +55% geometric mean) once the stack was captured and device-bound, and
the recommended layer split (bf16 in the trunk matmuls, fp32 LayerNorm statistics, fp32 heads, fp32
simulator/returns/losses/optimizer) at **+52.6%**. `unified_report/report.md` states it as a law:
"reduced precision pays where the stack is device-bound and the kernels engage the tensor cores; it
costs where launches dominate." In JAX the default matmul-precision knob was a null (**+0.07%**,
occupancy-bound matmuls) but explicit bf16 matmul inputs with `preferred_element_type=float32` gave
**+6.7% to +19.6%**. Every implementation's shared note repeats the same hard limit: state stays fp32,
because bf16's 8-bit mantissa cannot represent the accumulating state.

**Module.** 2 and 3, both frameworks.

**Concrete experiment.** Only after item 13 shows the stack is device-bound. Autocast only the trunk
matmuls of actor, critic, RND target and RND predictor; keep the environment recursion, the observation
whitening, the `BatchedRMS` statistics (already float64), the intrinsic filter, GAE, the losses, the
gradient clip and Adam in fp32. Note the reference's own end state before copying it wholesale: after a
throughput benchmark they flipped the shipped **default back to fp32 with TF32 matmuls** and left bf16
as a config field (`decided_model_infra/…/README.md` section 8 item 4), so build it as a switch, not a
replacement.

### 8. Delete the select from the serial dependency chain in the contact tournament

**Reference evidence.** `module1_env/cuda/progress_and_changes.md` experiment 8, **1.3834x** — "the
single largest arithmetic win came from removing ONE select, not from removing work … Freezing made the
cumulative sum depend on the comparison, so each search step ran compare, select, then the next compare
— a serial chain about thirty steps long inside every element … the win is latency, not arithmetic."
The general form is in `module1_env/shared_notes.md`: "Look for accumulators whose update is gated by a
comparison; the gate usually costs more as a dependency than as an operation." The same file also
records that the two things that *sounded* like the bottleneck were not: cutting random-number
generation by three quarters bought **0.6%**, halving memory traffic bought **3.6%**.

**Module.** 1 (environment), all three implementations.

**Concrete experiment.** `torch_pointmaze.dynamics_step` runs a two-smallest tournament over 8 candidate
contacts in which each candidate's update of every one of the ten carried values is a `torch.where`
gated on the previous candidate's comparison — a serial chain of roughly 40 dependent selects per
element. Two restructurings to measure: (a) compute all 8 candidate distances first, then select the
two smallest by an arithmetic rank (count how many candidates are strictly closer, which is the same
counting trick that bought 1.38x in the reference) rather than by a sequential tournament; (b) since
`d1`/`d2` are only used for ordering, carry the ordering decision and gather the payload once at the
end instead of moving five payload values at every candidate. The fixture checker is the gate; it must
run against the compiled path, not only the eager one (item 16).

### 9. Measure cache-versus-recompute per framework instead of choosing once

**Reference evidence.** The most explicitly transferable methodological finding.
`module1_env/pytorch/progress_and_changes.md` experiment 2 cached the constant per-item channels and won
**1.50x**; `module1_env/jax/progress_and_changes.md` experiment 5 **deleted** that same cache and won
**1.091x**, and the shared note spells out why: "in eager PyTorch the recompute is separate kernels,
under XLA it folds into a kernel that was going to run anyway. Do not port this decision across
implementations; measure it in each."

**Module.** 1, PyTorch and JAX separately.

**Concrete experiment.** Both our environments currently sit on the recompute side: `dynamics_step`
rebuilds each neighbour rectangle arithmetically from the cell index (`xl`, `yb`) at every candidate at
every step, deliberately to avoid a gather. That is the choice XLA rewarded in the reference and the
choice eager PyTorch penalised. Build the alternative — a precomputed `[n_cells, 8, 4]` table of
candidate rectangles gathered by cell index — and measure it in the torch environment and in the JAX
environment independently. Expect opposite verdicts; record both.

### 10. Scan unroll on the JAX rollout

**Reference evidence.** `module1_env/jax/progress_and_changes.md` experiments 2 and 6: `unroll=4` gave
**1.889x** and `4 -> 8` a further **1.112x** on the environment scan.
`module2_arch/jax/progress_and_changes.md` experiment 4 measured the same knob on a trainer scan whose
body is a transformer at **+4.67%** — "real but bounded by the body's compute density" — and the V2
campaign found `unroll=4` flipped from below the keep floor to **+2.3%** once the body changed
precision, so the knob has to be re-swept whenever the body changes.

**Module.** 1 and 2, JAX.

**Concrete experiment.** None of the six `lax.scan` calls in `jax_ppo_rnd.py` passes `unroll`. Sweep
`unroll` in {1, 2, 4, 8} on the rollout scan (line 345), the intrinsic filter scan (359), the GAE scan
(377) and the style-B minibatch scan (316), one at a time. Our body is a tiny multilayer perceptron, so
the environment-track numbers are the better prior than the transformer-track ones. Watch compile time
and peak memory, which the reference reported rising with the knob (377 to 662 MiB at unroll 4 -> 8).

### 11. Measure the environment's share of a training iteration by substitution

**Reference evidence.** This measurement changed the reference campaign's whole plan.
`module2_arch/cpp/progress_and_changes.md` timed it directly: "environment calls, one update … 0.005 s;
the whole update 60.4 s; the environment's share **0.008%**", and concluded "whoever optimizes this
pipeline should spend every hour on the transformer and none on the environment."
`module4_freestyle/progress_and_changes.md` built the general instrument — `env_share_breakdown(k)`,
which times the rollout twice against the same build, once with the real step and once with a
`_model_only_step` that runs identical network forwards on tokens handed in, so the difference is
attributable — and used it to retire the whole cross-stack environment pairing question:
"the environment is no longer where a training update spends its time, so which environment a GPU
pipeline pairs with has stopped being a throughput decision."

**Module.** 3 (end to end), and it decides how much of Module 1 is worth further effort.

**Concrete experiment.** Add a `_model_only_step` to `torch_ppo_rnd.py` that runs the identical policy,
value and RND forwards but returns a handed-in next state instead of calling `env.step_core`, and time
`_rollout_body` with each. Do the same for the reset-noise draw on its own. This number does **not**
port from the reference: their model was a four-layer transformer, ours is a 4-64-64 multilayer
perceptron, so our environment share could plausibly be tens of percent rather than a fraction of one.
Either answer reorders the remaining work — a large share justifies the CUDA environment pairing, a
small share caps it — and it costs one afternoon.

### 12. Decide the hand-fused megakernel by a written criterion, before building it

**Reference evidence.** `e2e_campaign/memo_fused_triton.md` is a complete, reusable procedure. The
criterion was written *before* the measurement, with a falsifiable prediction: define `h` as the
device-idle share of one replayed iteration, `g` as the matmul share of kernel time, `e` as the
elementwise/copy/reduction share, and `rho` as the median ratio of the best Triton template time to the
best cuBLAS time over the iteration's matmul shapes. Go only if `h > 15%`, or `e > 40%` with those
kernels under half of measured copy bandwidth, or `rho < 0.90` on at least half the shapes; the ceiling
is `1 / (rho*g + e/2)`. Measured: `h = 2.79%`, `g = 61.1%`, `e = 38.6%`, `rho ~ 1`, about 26,500 kernels
per update — **NO-GO at a 1.243x ceiling**, against a multi-week build.

**Module.** 3, and it settles our own listed candidate "a hand-written fused Triton/CUDA per-iteration
megakernel".

**Concrete experiment.** Run the same probe on our captured one-graph iteration at C = 128: 30 timed
replays plus 3 profiled replays, sum kernel device time against replay wall time, taxonomise kernels,
and read `rho` out of a max-autotune log. Our verdict may well differ from theirs — our kernels are far
smaller and far more numerous relative to the arithmetic, so `h` could exceed 15% here where it was
2.79% there. Write the criterion down first, then measure.

### 13. Record enqueue time beside wall time — the launch-bound diagnostic

**Reference evidence.** `harness/bench_protocol.md` Amendment 6 added one derived number to every row:
the host time from region start to just before the closing synchronization. It immediately paid for
itself — `module6_e2e/progress_and_changes.md` experiment 1: "Amendment 6's first reading:
enqueue_seconds == wall at every measured cell — the eager update never lets the host get ahead of the
device … This raises the capture experiment's prior substantially." Experiment 3 closed the loop:
"eager enqueue == wall (launch-bound); captured enqueue = 0.0003 s against a 0.312 s wall
(device-bound). The +5-20% estimate was conservative precisely because the launch-bound share was
bigger than the breakdown's kernel-time buckets could show."

**Module.** All three (it is a harness change, but it is the instrument every speed decision depends on).

**Concrete experiment.** In `benchmarks/bench_train.py` and `bench_env_step.py`, record
`time.perf_counter()` immediately after the last launch of the timed region and again after the closing
`torch.cuda.synchronize()`, and store both in the row. Enqueue near wall means launch-bound and fusion
or capture will pay; enqueue far below wall means device-bound and only fewer or better kernels will.
This single column tells us whether items 5, 6 and 7 are worth running at all.

### 14. Draw the per-iteration randomness inside the captured graph

**Reference evidence.** `module6_e2e/progress_and_changes.md` D0: "RNG-replay probe: PASS on all three
checks — graph replay draws from the default generator's host-visible state at replay time, and
re-pinning the state makes a replay reproduce the first eager draw." `module4_freestyle` item 2 states
the mechanism: "Order sampling stays inside, on the DEFAULT generator, which torch registers at capture
so each replay advances its philox offset. A private generator would not be registered." The cost of
leaving randomness outside is measured: experiment 3 notes "the mg replay enqueues ~0.1 s (the
per-update demand-block redraw's host poisson launch dominates its outside-the-graph work)", roughly a
quarter of that cell's wall, and `v2_combo/ideas.md` E2 queues side-stream double-buffering as the fix.

**Module.** 2 and 3, PyTorch.

**Concrete experiment.** `iteration_captured` does two things outside the replay every iteration:
`self._Z.normal_()` and `self._perm.copy_(torch.rand(...).argsort(dim=-1))`. Both use the default
generator, which torch registers at capture, so both can move inside `_iteration_body` and be redrawn
per replay. Measure the outside-the-graph time first (item 13 gives it directly) — at our sizes it is
probably well under one percent, which is why this is listed here and not in the ranked list, but it
also removes the last host work from the iteration and simplifies the capture contract.

### 15. Two-tier equivalence gating, and the metastable-basin lesson behind it

**Reference evidence.** `harness/bench_protocol.md` Amendment 2 is the most expensive lesson in the
campaign. A change that was "adversarially reviewed, lockstep-verified INERT (per-update parameter
deltas <= 6e-8)" landed **+24.8%** off the reference's final objective, because the method possessed a
rare metastable plateau selected in the early transient and "Adam converts single-ULP gradient
differences into a fresh basin draw immediately — single-trajectory comparison at these settings can
only be statistical." The fix: tier A lockstep (identical rollouts by RNG pinning, per-parameter deltas
over 5 updates, tolerance 1e-5) for exact refactors, and tier B (3 seeds x 200 updates, candidate median
against the reference *distribution* median, 6% band) for anything that changes numerics. Amendment 3
adds the companion trap: process-global flags "leak across the paired design: the last-built side's
build() set the flags for BOTH sides, so the first TF32 session measured TF32-vs-TF32".

**Module.** 2 and 3.

**Concrete experiment.** We have the tier-A equivalent already (`tests/test_capture_gpu.py` is bitwise
for the rollout and 1e-5 for the update — good), but no tier B at all, and **TF32 is already shipped**
(`e2e/torch_e2e/progress_and_changes.md` experiment 2, +9%) with no learning-equivalence check behind
it. Build the tier-B gate now: 3 seeds x a fixed short run at C = 8, comparing the median final
extrinsic reward and median coverage against a cached fp32 reference distribution, and run it
retroactively on TF32 before running it on max-autotune (item 6) and bf16 (item 7). Also add
`assert_global_state()` to `PPORND` before we ever put two trainers in one benchmark process:
`torch.set_float32_matmul_precision("high")` is set in `__init__` and is process-global, so a paired
comparison built in one process would measure TF32 against TF32, exactly as it did in the reference.

### 16. Correctness gates must execute the timed code path, checked adversarially

**Reference evidence.** `module1_env/shared_notes.md`, the JAX track's gate-strengthening note: "the
harness's correctness gates drive `replay_given` / `grad_rollout` / `observation_np`, which share the
step math with the rollouts but never execute the TIMED path. Any implementation that restructures its
rollout needs an adapter `self_check` comparing the timed path against the same recursion written step
by step … and it must be checked adversarially (inject a 1-2% error and confirm it is reported)." The
PyTorch track added the same check when it compiled its step, and the JAX track credits it with making
the custom backward safe to keep.

**Module.** 1, all three implementations.

**Concrete experiment.** `pointmaze/common/code/check_against_fixtures.py` drives the eager `step`. Add
a `--path` switch so the same fixtures run through the compiled step, through the captured rollout, and
through the CUDA kernel, and add the adversarial check — perturb one constant by 1% and confirm the
checker fails. This is also the only thing standing between the CUDA environment's "fixtures ALL PASS"
row and a claim about the kernel that will actually be timed.

### 17. A memory-for-speed keep rule, and fresh-process confirmation of every out-of-memory verdict

**Reference evidence.** `harness/bench_protocol.md` Amendment 5 pre-registered a rule for candidates
that make a previously impossible cell fit: keep if the paired ratio over already-fitting cells is at
least 0.97 and the newly-fit cell's throughput beats the best fitting cell — "the original keep rule had
no way to bank a memory win". Amendment 5 also requires fresh-process confirmation, and
`module2_arch/jax/progress_and_changes.md` S0 explains why: re-measuring one cell per process turned
**7 of the previous table's "does not fit" verdicts into artifacts** — "the OOM verdicts were taken
after earlier cells in the same process had grown the pool (it never shrinks)".
`unified_report/report.md` lists it among the five findings that generalize.

**Module.** 3, and it touches a headline deliverable number.

**Concrete experiment.** `e2e/torch_e2e/progress_and_changes.md` reports "32,768 fits / 65,536 OOM
during iteration-graph build" from an in-process sweep over eleven copy counts. Re-measure the top of
that ladder one cell per fresh process, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set, and
refine between 32,768 and 65,536 — the maximum copy count is one of the numbers the report is asked for.
Note in passing that the reference's Amendment-5 rule would not reward a newly-fit cell here: our total
throughput already saturates at about 5.6e6 environment steps per second from C = 8192, so cells above
that buy coverage, not speed.

### 18. Benchmark a frozen copy from git, never the working tree

**Reference evidence.** `LOOP.md` "Queued-session hygiene (added after a near-miss on day 1)": "A
session queued on the serval05 lock imports its --candidate/--reference FILES when it eventually starts,
not when it was queued. Never point a queued session at a working-tree file that a later edit could
change; point it at a frozen copy extracted from git." `module1_env/cpp/progress_and_changes.md` records
that it nearly happened to them on the baseline run, and that freezing each version into its own
directory "is what made a queued session safe to leave unattended" — with the follow-on warning that a
frozen directory also freezes the tooling inside it.

**Module.** All three — this is operational, and it applies directly because we run parallel fork agents
behind one H100 lock.

**Concrete experiment.** Extend `locks/gpu_run.sh` so a queued job runs against
`git show <commit>:<path>` written into a per-run frozen directory at the same directory depth (relative
imports resolve from the file's location), and record the commit in the result JSON. Adopt the
reference's commit discipline with it: one experiment is one committed edit to one file, committed
*before* the run, with the progress note committed separately after the keep-or-revert decision, so a
revert never touches the notes.

---

## 2. Reference-campaign techniques we already cover

| Technique | Reference measurement | Where we have it |
|---|---|---|
| Compiled fused per-step function | m1-pytorch exp3, 1.57x | `_one_step_pure` under<br>`torch.compile(fullgraph=True,`<br>`dynamic=False)` |
| Whole-update CUDA-graph capture | m6 exp3, +35.7% | `_build_update_graph`,<br>`_build_iteration_graph` |
| Warmup on a side stream with<br>snapshot and restore | m6 exp3 | `_build_iteration_graph` snapshots<br>state, parameters, optimizer |
| Capturable Adam with a<br>device-tensor learning rate | m4 items 3 and 4 | `capturable=True`, `self._lr_t` |
| TF32 matmuls | m2 exp6, +7.6% | `tf32` flag, e2e exp2, +9% |
| Batched matmuls over the copy axis | m3 exp2 family | `baddbmm` throughout, parameters as `[C, ...]` |
| Fewer, larger optimizer tensors | perf checklist row | 19 stacked tensors<br>regardless of C |
| Synchronization purge in the<br>hot loop | m6 exp2<br>(deletion clause) | no `.item()`/`float()`<br>outside the log cadence |
| Gradient clip without a host read | m6 exp2 | `_clip_per_copy_and_step` is pure tensor work |
| Randomness pre-drawn into<br>a static buffer | m1-pytorch exp5, 1.05x | `self._Z`, `self._perm` |
| Counter-based keyed randomness | cuda v0 design | `_hash_uniform`, frozen across<br>all three implementations |
| Branch-free reset by masked select | shared guardrail | `torch.where(d2, s_pos, pos)` |
| Loop body unrolled at trace<br>time, not a device loop | jax exp1 note | 8-candidate contact loop unrolled in Python |
| Style resolved at build time,<br>no branch in the hot path | m6 capture design | two separate update functions |
| Whole iteration in one jitted<br>function with donation | jax m2 v0 + exp5 | `jax_ppo_rnd._iterate`, `donate_argnums=(0,)` |
| One profiling job at a time<br>behind a file lock | `harness/serval05_run.sh` | `locks/gpu_run.sh` |
| Per-subtask experiment ledger,<br>keep or revert per row | LOOP.md | seven `progress_and_changes.md` files |
| Fused Adam | m2 exp8, +0.27% null there | enabled; needed for capturable anyway |

---

## 3. Do these next, in order

1. **Run the reference's D0 diagnostic block before changing any code** (items 13, 11, 12, and the null
   session of section 4). Zero speed effect by itself; it is what turns the rest of this list from
   guesses into decisions. In the reference this block predicted capture's +35.7% in advance, killed a
   multi-week megakernel build with a 1.243x ceiling, and retired the entire cross-stack environment
   pairing question with one 0.008% measurement. Half a day.
2. **Pack the actor and critic trunks, and the RND target and predictor first layers, into batched
   matmuls** (item 1). Reference: **+47.8%** eager, **+13.6%** captured, **+21.7%** uncaptured;
   **-11.5%** in JAX, so PyTorch only. Ours is the captured case, so expect the +13.6% class, with more
   upside than theirs because our per-copy matmuls are far smaller and therefore more launch- and
   occupancy-limited than their 2112-row ones.
3. **Whole-rollout persistent kernel in the CUDA environment** (item 2). Reference: **20.57x** over a
   per-step fused kernel, then **1.87x** more from capturing the rollout. This is the largest single
   number in the whole reference campaign and our CUDA environment is still at their version 0. It is
   also a required deliverable — the Module 1 throughput and knee curves per implementation.
4. **Compile `_post_body`, then test a K-step compiled rollout chunk** (item 5). Reference: **3.84x**
   for chunking in the environment track and **1.14x** for deleting one per-step reduction, against a
   measured **+0.12% null** once the body was kernel-bound. The filter and GAE scans are 256 Python
   iterations of tiny eager kernels inside the captured graph, so the compile is near-certain value; the
   chunk is conditional on the item-13 reading.
5. **The two recomputation hoists — RND target features out of the minibatch loop, and the redundant
   bootstrap critic forward** (items 3 and 4). Reference: **+17.61%** for the single equivalent hoist,
   tier A at 6.5e-07. Ours deletes 15 of 16 target-network forwards per iteration in style B, and about
   99.75% of a `T*N`-row critic forward. Both are exactly equivalence-preserving, so the existing
   bitwise test is the whole gate.
6. **`max-autotune-no-cudagraphs`, then bf16 trunk matmuls behind a config switch** (items 6 and 7).
   Reference: **+5.0%** for autotune under capture; bf16 **-14.7%** eager against **+52.6 to +55%**
   captured — run it only after item 13 confirms we are device-bound, keep the environment recursion,
   running statistics, GAE, losses and Adam in fp32, and gate both with the tier-B check of item 15
   before either is called a keep.

---

## 4. What our benchmark protocol is missing

Ranked by how much a wrong measurement would cost us.

1. **No paired comparison and no decision statistic.** Every verdict in our ledgers compares absolute
   numbers from different sessions ("226 -> 45.6 ms"). `reviews/2026-08-12_measurement_review.md` calls
   this the first fatal flaw: clocks cannot be pinned without root, so "hour-scale drift plausibly
   exceeds the 2-5% effects being chased. Every real decision must be an interleaved paired A/B in one
   session (ABBA order, which cancels linear drift to first order), scoring the *ratio*, never two
   absolute numbers from different sessions." Adopt the reference's rule verbatim: at least 5 ABBA pairs
   in one process, decision statistic `mean(log ratio)` with its standard error, keep only if
   `mean - 3*SE > 0` **and** the gain is at least 1% **and** no cell regresses more than 10%; a gain
   between the threshold and twice it needs a fresh-session confirmation; a change that deletes code
   keeps on non-inferiority.
2. **No noise floor.** The reference's first harness action of every subtask is a null run where the
   candidate is a byte-identical copy of the reference. Their training-bench null read per-cell standard
   error 1.2-1.8%, from which they concluded "keeps need roughly +3% to clear mean-3SE in practice";
   their C++ track's null read **+/-7%**, which retroactively explained two undecidable experiments. We
   have no such number, so we cannot presently tell a real +5% from noise. Run the null first, store it,
   re-run it on any toolchain change.
3. **Wrong number of synchronizations inside the timed region.** The protocol is "exactly two device
   synchronizations, at the region boundaries; nothing inside". `benchmarks/bench_train.py` synchronizes
   **once per iteration** inside the timed loop in one-graph mode and **three times per iteration** in
   the split-phase mode. At 40 ms per iteration the bias is small, but the phase split is exactly where
   it is largest and the reference records the same measurement (their diff-mode absolutes carry a
   "~0.3% sync share" footnote for the same reason). Time the whole block with two synchronizations and
   get the phase split from CUDA events or a separate profiled run.
4. **No enqueue column.** Amendment 6, covered as item 13 above. It is one derived number per row and it
   is the single instrument that decides whether a fusion or capture experiment is worth running.
5. **No behavioural equivalence gate.** Covered as item 15. TF32 is already in the production
   configuration with only a speed number behind it. The reference's Amendment 2 exists because a
   provably inert change landed +24.8% off through a metastable basin, and the whole point is that a
   single-trajectory comparison cannot detect that class of problem.
6. **The `one_graph` path does not force the capture preconditions on the optimizer.** In
   `torch_ppo_rnd.py`, `capturable=True` Adam and the device-tensor learning rate are gated on
   `cfg.capture_update`, not on `cfg.one_graph`, and `train()`'s annealing branch is gated the same way —
   so a `one_graph=True, capture_update=False` build writes the annealed learning rate into
   `opt.param_groups`, which a captured graph cannot see. `module4_freestyle/progress_and_changes.md`
   hit precisely this and its lockstep test caught it: "Its first run failed at 5.17e-03, the 2*lr
   signature the harness documents as a real bug: the learning-rate schedule was being written into the
   static tensor but never reaching the optimizer … A smoke test that only checked 'does it run and do
   parameters move' passed that build happily." Our `test_capture_gpu.py` calls `update_*` directly with
   a constant learning rate and never exercises `train()` or `one_graph` at all. Add a lockstep test
   that runs `train()` with `anneal_lr=True` for 5 iterations captured against 5 eager, comparing
   per-parameter deltas, and make `one_graph` imply `capture_update`.
7. **Ladder too coarse to locate the knee the deliverable asks for.** Our environment sweeps are decades
   (1e3, 1e4, 1e5, 1e6). The protocol asks for powers of two from a low start with at least four clearly
   launch-bound and four clearly saturated points, a half-power-of-two refinement around the knee, the
   sweep direction alternated between repeats so the batch axis is not confounded with thermal drift,
   and the knee reported **two ways** — the hyperbola fit `t(B) = sqrt(t0^2 + (B/S)^2)` giving
   `B* = S*t0`, and the first batch whose per-step time exceeds 1.25x the floor — with any disagreement
   stated rather than hidden. The user's Module 1 deliverable is literally this curve and this number.
8. **Out-of-memory ceilings not confirmed in fresh processes.** Covered as item 17; it affects a headline
   number (maximum copy count) that is currently taken from an in-process sweep.
9. **No session-preparation guard beyond the lock.** The reference refuses to start if a foreign process
   holds GPU memory, runs a fixed 10-second matmul clock warm before the first timed region, and samples
   clock, temperature, power and throttle reasons at about 2 Hz through every timed region, invalidating
   any cell measured under an active throttle or whose mean clock deviates more than 2% from the session
   reference. We have the lock, which covers our own jobs only.
10. **No steady-state assertions in the warmup.** The protocol requires at least three untimed
    iterations, the last one within 5% of the following timed median, `memory_reserved` unchanged over
    the last repeats, and zero allocation retries — a nonzero retry count invalidates the cell. It also
    requires re-measuring instead of reporting whenever the inter-repeat spread exceeds 3%.
11. **No "nothing compiled or synced during timing" gate.** Assert the dynamo compile-counter delta is
    zero across the timed region and run one correctness pass under
    `torch.cuda.set_sync_debug_mode("warn")`. The reference also documents the specific trap for any
    lazily-compiling implementation: a calibration probe that pays the compile sizes the cell at one to
    three steps and then measures launch overhead instead of throughput — "eleven of fifty-four ladder
    cells were lost to this before it was found", and 27 of 84 compiled-torch cells had to be re-run.
    Our `bench_env_step.py` computes its step count from a formula and warms up first, so we are clear
    today, but the CUDA environment and any new shape will re-open it. The row signature to watch for is
    a small step count per region together with a high enqueue fraction.
12. **Compile caches not pinned or cleared identically for both sides**, and cold-compile time not
    reported separately from steady state.
13. **Guardrails against measuring the wrong thing.** The reference's table of throughput-gaming failure
    modes maps onto ours directly: assert the step count executed equals the step count requested;
    consume and checksum the outputs inside the timed region so nothing is optimized away; assert two
    graph replays draw *different* action noise (a frozen generator under capture looks enormously fast
    and passes a mean test); check implied bandwidth against an achievable ceiling, since a "win" that
    implies more than peak means work was skipped.
14. **Thin provenance.** Our result JSONs carry git hash, GPU name and torch version. The reference's
    rows additionally carry driver version, protocol version, warmup count, repeat count, peak and
    reserved memory, allocation retries, clock and thermal samples, the cell-table hash, and whether the
    timed path is the native or the integrated one — with the standing rule that rows from different
    protocol versions are never compared.
15. **No results ledger beside the prose.** The reference pairs every `progress_and_changes.md` with a
    five-column `results.tsv` (`commit metric memory_gb status description`, status keep/discard/crash)
    and a per-commit sidecar JSON holding per-cell numbers, all repeat times and their spread, gate
    outcomes and provenance. Our ledgers are prose only, so no verdict can be recomputed.
16. **No stopping rule tied to a measured roofline.** The reference stops a subtask when three
    consecutive distinct-kind ideas are discarded, or when a measured roofline is reached — for the
    environment, sustained bytes per second against a *measured* device-copy bandwidth at the same
    tensor sizes; for the trainer, a stated fraction of achievable matmul throughput. They state the
    number rather than asserting the conclusion: "at n=128 B=4096 the kernel moves about 10 bytes per
    element-step and sustains roughly 1.1e11 element-steps/s, which is about 1.1 TB/s against a device
    that delivers around 3 TB/s — so this environment step is NOT bandwidth-bound even now."
17. **Report generated by hand rather than from the measurement files.** Every reference report is
    emitted by a `make_report.py` that reads the JSONL rows, so the prose cannot drift from the data;
    `final_delivery/report_data_inventory.md` tracks which number comes from which file. Our unified
    report should be generated the same way, and the markdown-editing rule then applies to it. The
    deliverable training runs at 8/16/32/64/128 copies additionally need the `experiment-background`
    skill's `experiment_background.md` in each run folder, a checkpoint manifest that refuses a resume
    when a behaviour-defining field changed (the reference added exactly this after a review finding),
    and the standing 20-minute monitor.

---

## 5. Two reference conclusions worth carrying as priors

1. **A technique's value is a property of the regime, not of the technique.** The reference measured the
   same bf16 change at -14.7% and +52.6%, and the same tower-fusion change at +47.8% in eager PyTorch
   and -11.5% under XLA. Every item in section 1 that is marked for one framework is marked that way on
   measured evidence, and every item ported to the other framework must be re-measured, not inherited.
2. **Once the loop is captured, launches are free but kernel device time is not.** The reference's
   captured update was 97.2% kernel-busy across roughly 26,500 kernels per update, which is why its
   remaining levers were batch size and precision rather than further fusion. Our stack is already
   captured, so the same question — how much device idle is left — is the first thing to measure and the
   thing that decides most of section 3.
