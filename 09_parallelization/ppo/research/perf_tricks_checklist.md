# Performance-trick checklist for batched multi-copy PPO+RND on one H100

Every row is one experiment for the optimization loops: change one thing, measure, keep or revert,
record in the subtask's `progress_and_changes.md`.

Reading the table:

- **Effect** is the expected wall-clock change **for this project** — a batched tiny-MLP PPO+RND
  (per-copy hidden widths in the 64–256 range) driving a closed-form PointMaze stepper, all on one
  H100, `n_copies` x `n_envs` on the GPU. It is not the effect the same trick has in LLM training.
  Where a number was measured in one of the reference repos, the number and the repo are named.
- Rows in the **Skip** group do not transfer. They are listed with the reason so the loops do not
  spend a run on them.
- The starting assumption, to be confirmed by the profiler before anything else: at these sizes the
  training step is **kernel-launch bound**, not compute bound. A PPO iteration written the obvious
  way is several hundred tiny kernels; at roughly 3–8 microseconds of launch overhead each, that is
  milliseconds of pure launch time per iteration with the GPU mostly idle. Almost every large win
  below is a way to issue fewer, bigger kernels — not a way to do less math.

## Checklist

| Group | Technique | What it does | Where it applies here | Effect | How to try |
|---|---|---|---|---|---|
| Precision | TF32 matmul mode | Runs fp32 matmuls on<br>tensor cores with a<br>shorter mantissa | Every linear layer in<br>policy, value, RND<br>predictor and target | Measured 2.2x in<br>LLMs-from-scratch<br>(big fp32 matmuls).<br>Here small — we are<br>launch bound — but<br>free | `torch.set_float32_matmul_precision("high")`<br>plus `torch.backends.cuda.matmul.allow_tf32 = True`<br>at startup |
| Precision | bf16 autocast | Casts activations and<br>matmul inputs to bf16 | Whole update step;<br>the copies axis makes<br>the batched matmuls<br>big enough to matter<br>at 64+ copies | Measured 1.6x in<br>LLMs-from-scratch.<br>Here likely small or<br>negative at low copy<br>counts (cast kernels<br>add launches) | Wrap the update in<br>`torch.amp.autocast("cuda", torch.bfloat16)`,<br>A/B at 8 and 128 copies |
| Precision | bf16 weights, fp32<br>master copy | Skips autocast<br>dispatch by casting<br>the weight inside the<br>layer instead | Alternative to the row<br>above if autocast shows<br>up in the profile | Small; removes<br>autocast bookkeeping<br>from the hot path | Copy nanochat's `Linear`<br>subclass: `F.linear(x, w.to(x.dtype))` |
| Precision | fp16 with GradScaler | Half precision with<br>loss scaling | Not worth it — the<br>scaler's inf check is a<br>host sync per step | Negative | Do not use. bf16 has<br>the same exponent range<br>and needs no scaler |
| Precision | Pad dimensions to<br>tensor-core multiples | Full tensor-core tiles<br>instead of ragged tails | Hidden widths, obs and<br>action dims, `n_envs`,<br>minibatch size | Measured 1.14x in<br>LLMs-from-scratch<br>(vocab 50257 to 50304) | Keep every shape a<br>multiple of 8, ideally 64.<br>Powers of two everywhere |
| Precision | int32 index tensors | Halves the bytes moved<br>by gathers and scatters | Minibatch indices, maze<br>cell indices, wall<br>lookup tables | Small | Build index tensors as<br>`dtype=torch.int32` |
| Compile | `torch.compile` on the<br>update function | Fuses many small ops<br>into few Triton kernels | The PPO loss, the RND<br>loss, GAE, the env step | Measured 1.22x in<br>LLMs-from-scratch on<br>a real model. Here<br>expect more (2x+):<br>most of our ops are<br>tiny and fusable | `torch.compile(fn, dynamic=False, fullgraph=True)` |
| Compile | `mode="reduce-overhead"` | Adds CUDA graph<br>capture on top of the<br>fused kernels | The whole iteration<br>once it is sync-free<br>and allocation-free | Likely the single<br>largest lever while<br>launch bound | Same call with<br>`mode="reduce-overhead"`,<br>after the sync and<br>allocation rows are done |
| Compile | `mode="max-autotune"` | Triton autotuning plus<br>CUDA graphs | Same target; costs<br>minutes of compile per<br>shape | Extra few percent<br>over reduce-overhead,<br>sometimes more for<br>odd shapes | Try once shapes are<br>frozen; keep the<br>Inductor cache so<br>reruns are cheap |
| Compile | `max-autotune-no-cudagraphs` | Autotuning without<br>graph capture | Fallback if graph<br>capture fights our RNG<br>or buffer reuse | Between default and<br>max-autotune | Use as the control when<br>debugging a graph<br>capture failure |
| Compile | `fullgraph=True` | Turns a silent graph<br>break into an error | Every compiled region | No direct speed;<br>protects the speed we<br>already bought | Set it and fix what it<br>reports |
| Compile | `dynamic=False` | Pins static shapes,<br>avoids dynamic-shape<br>kernels and recompiles | Every compiled region;<br>our shapes never change<br>after startup | Moderate; also kills<br>recompilation stalls | Set it; nanochat and<br>autoresearch both do |
| Compile | Two separate compiled<br>update functions | Full-batch update and<br>the 4-epoch 4-minibatch<br>update as distinct<br>functions | Exactly the user's<br>requirement: no `if`<br>inside the hot path | Prevents a branch from<br>splitting the graph | Write `update_fullbatch`<br>and `update_minibatch`,<br>pick once at startup |
| Compile | No graph breaks in the<br>hot region | Keeps one graph per<br>region | No `.item()`, no python<br>float from a tensor, no<br>`nonzero`, no boolean<br>mask indexing, no print | Large indirectly — one<br>break can undo the<br>whole compile win | Run with<br>`TORCH_LOGS=graph_breaks,recompiles`<br>and read the log |
| Compile | 0-D CPU tensors for<br>scheduled scalars | Changing a learning<br>rate does not trigger a<br>recompile | Learning rate, clip<br>coefficient, entropy<br>coefficient, RND beta | Removes a recompile<br>stall per schedule<br>change | nanochat's pattern:<br>`t.fill_(value)` on a<br>preallocated 0-D tensor,<br>pass the tensor in |
| Compile | Inductor on-disk cache | Reuses compiled kernels<br>across process restarts | Every benchmark rerun<br>in the optimization loop | Cuts startup, not<br>steady state; worth it<br>for loop throughput | Set `TORCHINDUCTOR_CACHE_DIR`<br>to a local disk path on<br>serval05 |
| Compile | Inductor coordinate<br>descent tuning | Extra autotuning of<br>Triton block sizes | Our small odd-shaped<br>batched matmuls | A few percent,<br>sometimes more | `torch._inductor.config.coordinate_descent_tuning = True` |
| Launch | Manual CUDA graph<br>capture | Replays a fixed kernel<br>sequence with one<br>launch | The full iteration:<br>rollout, GAE, update | Large; the fallback<br>when reduce-overhead<br>refuses our loop | `torch.cuda.graph(g)` with<br>static input and output<br>tensors, warmup on a<br>side stream first |
| Launch | `make_graphed_callables` | Graphs the forward and<br>backward of a module | The policy and RND nets<br>if graphing the whole<br>loop is too invasive | Moderate; a partial<br>version of the row<br>above | `torch.cuda.make_graphed_callables(module, sample_args)` |
| Launch | Graph-safe randomness | Keeps sampling inside<br>the captured region | Gaussian action noise,<br>minibatch permutation | Required for capture,<br>not a speedup by itself | Use a CUDA<br>`torch.Generator`; never<br>numpy or python `random` |
| Launch | Capture preconditions | No host sync, no new<br>allocation, no CPU<br>control flow inside the<br>captured region | Forces the sync and<br>buffer rows to be done<br>first | Gate, not a speedup | Verify with<br>`torch.cuda.set_sync_debug_mode("error")`<br>during a dry run |
| Launch | Persistent kernel env<br>(one kernel, many<br>steps) | A resident grid loops<br>over timesteps inside<br>the kernel | Module 1's fused CUDA<br>env: T rollout steps in<br>one launch instead of T | Large for the env<br>half; this is the main<br>reason the CUDA env<br>variant can beat the<br>torch one | Write the stepper loop<br>inside the kernel over<br>the horizon, one thread<br>per (copy, env) |
| Launch | One fused kernel per<br>env step | Position update, wall<br>clamp, reward,<br>termination and reset in<br>one kernel | Torch env: get it via<br>compile. CUDA env: hand<br>written | Large — the naive torch<br>env is 15+ kernels per<br>step | Compile the whole `step`<br>with `fullgraph=True`<br>and count kernels before<br>and after |
| Launch | GAE as a compiled scan | Replaces a python loop<br>of T tiny kernels | Advantage computation<br>after every rollout | Moderate; T is the<br>rollout length so this<br>is T times the fixed<br>overhead | Compile the reverse loop,<br>or write it as an<br>associative scan over<br>the horizon |
| Launch | Kernels per iteration<br>as the target metric | Makes the launch-bound<br>regime measurable | Every change while the<br>GPU shows idle gaps | The number to drive<br>down; stop when the<br>gaps close | Count kernels in the<br>profiler trace before<br>and after each change |
| Launch | Multiple CUDA streams | Overlaps independent<br>work | Low priority: the copies<br>axis already gives us<br>parallelism inside one<br>kernel | Small; adds complexity<br>and hides bugs | Only after batching and<br>graphs are exhausted |
| Syncs | No `.item()`, `.cpu()`,<br>`float()` in the loop | Each one blocks the CPU<br>until the GPU drains,<br>emptying the launch<br>queue | The cleanrl baseline has<br>several per step; the<br>20-minute log cadence<br>means we need none | Large. This is the<br>first thing to fix | Keep scalars as GPU<br>tensors, copy once per<br>log flush |
| Syncs | RND normalizer on GPU | Removes a host round<br>trip per step | cleanrl's `RunningMeanStd`<br>is numpy — it pulls<br>observations and rewards<br>to the host every step | Large in the baseline | Reimplement mean, var<br>and count as in-place<br>CUDA tensors |
| Syncs | Reward forward filter<br>and advantage<br>normalization on GPU | Same reason | The intrinsic-reward<br>discounted filter and<br>the per-batch advantage<br>standardization | Large in the baseline | Pure tensor ops, no<br>numpy anywhere in the<br>iteration |
| Syncs | GPU minibatch<br>permutation | Removes the numpy<br>shuffle and the index<br>upload | The 4-epoch 4-minibatch<br>update style | Moderate | `torch.randperm(n, device="cuda")`<br>or argsort of random keys |
| Syncs | Sync debug mode | Reports every hidden<br>host sync | Use during a dry run of<br>the whole iteration | Diagnostic; finds the<br>syncs we did not know<br>about | `torch.cuda.set_sync_debug_mode("warn")`,<br>then `"error"` |
| Syncs | Branchless episode<br>reset | Avoids data-dependent<br>shapes and the sync<br>they cause | Env termination,<br>truncation at the step<br>cap, goal reached | Large — a<br>`done.nonzero()` reset<br>both syncs and breaks<br>the graph | `torch.where(done, reset_state, state)`<br>on the full batch |
| Syncs | Schedule without python<br>floats | Keeps the learning-rate<br>schedule off the host | Learning-rate anneal<br>over the run | Small alone, but it is<br>one of the last syncs<br>blocking graph capture | Precompute the schedule<br>as a GPU tensor and index<br>by step, or use the 0-D<br>tensor pattern |
| Syncs | NaN and blow-up checks<br>at log cadence only | A per-step check costs<br>one sync per step | autoresearch's fast-fail<br>reads the loss every<br>step; we log every 20<br>minutes | Moderate | Accumulate a GPU flag,<br>read it when logging |
| Memory | Preallocated rollout<br>buffers | No allocation, no<br>`cat`, no `stack` in the<br>loop | Observations, actions,<br>log-probs, rewards,<br>values, dones for the<br>whole horizon | Large; also a<br>precondition for graph<br>capture | Allocate<br>(T, copies, envs, ...)<br>once, write by index |
| Memory | Preallocated env state<br>and in-place ops | Same, on the env side | Position, velocity,<br>step counters, visit<br>counts | Large in the env<br>throughput benchmark | Use `out=` and in-place<br>variants; never build a<br>new tensor per step |
| Memory | Expandable allocator<br>segments | Reduces caching-allocator<br>fragmentation and<br>`cudaMalloc` calls | Startup and any place<br>the buffer sizes vary | Small at steady state,<br>helps at high copy<br>counts | `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"`<br>before importing torch |
| Memory | Zero-allocation steady<br>state | Proves the loop is<br>allocation free | Assert during the<br>benchmark warmup | Gate for CUDA graphs | `torch.cuda.memory_allocated()`<br>identical across two<br>consecutive steps |
| Memory | Fixed layouts, no<br>`contiguous` in the loop | Avoids hidden copy<br>kernels | Batched matmul operand<br>layouts, the (T, C, N)<br>buffers | Small to moderate | Decide the layout once;<br>look for `copy` kernels<br>in the trace |
| Memory | Disable the cyclic<br>garbage collector after<br>warmup | Removes periodic<br>multi-hundred-millisecond<br>stalls | Long training runs, and<br>any timed benchmark | nanochat reports about<br>500 ms stalls; matters<br>for tail latency and<br>for clean measurements | `gc.collect()`,<br>`gc.freeze()`,<br>`gc.disable()` after the<br>first step; collect every<br>few thousand steps |
| Memory | Pinned host buffer for<br>the sparse log flush | Non-blocking copy of the<br>metrics that do leave<br>the GPU | The 20-minute log write<br>only | Small by design; the<br>point is that this is<br>the only host traffic | `torch.empty(..., pin_memory=True)`<br>plus `copy_(..., non_blocking=True)` |
| Memory | Reset peak memory stats<br>before each measurement | Makes the reported peak<br>meaningful | Every benchmark JSON | Measurement hygiene | `torch.cuda.reset_peak_memory_stats()`<br>after warmup |
| Optimizer | Fused AdamW | One kernel for the<br>whole parameter update | Every optimizer step | Measured 3% in<br>LLMs-from-scratch on a<br>large model; larger here<br>because our parameters<br>are many small tensors | `torch.optim.AdamW(..., fused=True)` |
| Optimizer | `foreach` grouping | One kernel per operation<br>across the parameter<br>list instead of per<br>parameter | Fallback when fused is<br>unavailable | Smaller than fused | `foreach=True` (already<br>the default on CUDA) |
| Optimizer | `capturable=True` | Keeps the step counter<br>on the GPU | Required for CUDA graph<br>capture of the optimizer<br>step; also removes a<br>host sync | Gate for the graph row;<br>small on its own | `torch.optim.AdamW(..., fused=True, capturable=True)` |
| Optimizer | Hand-written compiled<br>step | Full control, no<br>per-parameter python<br>loop | Our parameters are a<br>handful of stacked<br>tensors, so one compiled<br>function covers all | Moderate over fused<br>AdamW; this is what<br>nanochat and<br>autoresearch ship | Copy `adamw_step_fused`<br>from autoresearch's<br>`train.py` |
| Optimizer | Few large parameter<br>tensors | The optimizer cost<br>scales with tensor<br>count, not element<br>count | Direct consequence of<br>stacking per-copy<br>weights into (copies,<br>in, out) | Large at 128 copies:<br>tens of tensors instead<br>of hundreds | Falls out of the<br>batching row below |
| Optimizer | `set_to_none=True` on<br>zero_grad | Frees grads instead of<br>writing zeros | Every step | Small (already the<br>default in current<br>torch) | `model.zero_grad(set_to_none=True)` |
| Optimizer | Drop or fuse gradient<br>clipping | The global norm is an<br>extra reduction plus a<br>scan over parameters | PPO conventionally clips<br>at 0.5; measure whether<br>it is needed at our<br>scale | nanochat deleted<br>clipping outright and<br>gained a little MFU.<br>Changes learning, so<br>check reward too | If kept, do it with<br>`torch._foreach_norm`<br>inside the compiled step |
| Batching | Per-copy weights as<br>(copies, in, out) | Turns N independent<br>networks into one<br>batched matmul | The entire point of the<br>`n_copies` knob: policy,<br>value, RND predictor and<br>target | Very large — the<br>difference between one<br>kernel and `n_copies`<br>kernels per layer | `torch.baddbmm` or<br>`einsum` over the copy<br>axis; initialize each<br>copy with its own seed |
| Batching | `vmap` over stacked<br>module state | Same result written as a<br>transform of a single<br>model | Useful as the<br>correctness reference<br>for the batched matmul<br>version | Usually slower than the<br>explicit batched matmul;<br>keep it as the oracle | `torch.func.stack_module_state`<br>plus `functional_call`<br>plus `vmap` |
| Batching | One generator, shaped<br>draws | One RNG call for all<br>copies and envs | Action sampling, RND<br>target init, env resets | Large versus a<br>per-copy generator loop | Draw the whole<br>(copies, envs, act_dim)<br>tensor in one call |
| Batching | Hand-written Gaussian<br>log-prob and entropy | Avoids building a<br>distribution object and<br>its argument validation<br>every call | Continuous PPO — this is<br>called in the rollout and<br>again in every minibatch | Moderate; also removes<br>python objects from the<br>compiled region | Write the closed forms;<br>if using<br>`torch.distributions`,<br>pass `validate_args=False` |
| Batching | Copy count and env count<br>as compile-time<br>constants | No dynamic shapes, no<br>recompiles mid-run | Fixed for a given run;<br>the sweep varies them<br>across runs | Enables `dynamic=False`<br>and graph capture | Read them once at<br>startup, never from a<br>tensor |
| Batching | Largest batch that fits | Better tensor-core<br>utilization and fewer<br>launches per sample | The `n_envs` and<br>`n_copies` sweep that is<br>already a deliverable | Measured 1.11x in<br>LLMs-from-scratch just<br>from raising the batch | The throughput curves in<br>the deliverable double as<br>this experiment |
| Measure | CUDA event timing with<br>warmup | Correct timing of<br>asynchronous work | Every benchmark in<br>`benchmarks/` | Measurement<br>correctness; wall-clock<br>timing without a<br>synchronize is wrong | `torch.cuda.Event(enable_timing=True)`,<br>5 warmup iterations,<br>then many repeats,<br>report mean and spread |
| Measure | Torch profiler with CUDA<br>activity | Shows the kernel<br>timeline, kernel count<br>and the idle gaps | The first action of<br>every optimization loop,<br>before changing anything | Tells you which of the<br>rows above is worth<br>doing | `torch.profiler.profile`<br>with a wait/warmup/active<br>schedule, export a trace<br>and read it |
| Measure | `nsys` timeline | System-level view of<br>launch gaps and CPU-side<br>stalls | Confirming the<br>launch-bound diagnosis<br>and finding host syncs | The clearest evidence of<br>whether we are launch<br>bound | `nsys profile -t cuda,nvtx python bench.py`,<br>look for GPU idle between<br>kernels |
| Measure | `ncu` on one kernel | Occupancy, achieved<br>bandwidth, stall reasons | Only after the gaps are<br>closed and one kernel<br>dominates | Guides the last 20%,<br>mostly for the fused<br>CUDA env kernel | `ncu --set full --kernel-name <name>` |
| Measure | NVTX ranges | Names the phases inside<br>the trace | env step, GAE, forward,<br>backward, optimizer —<br>exactly the breakdown<br>table the deliverable<br>needs | Makes the required<br>per-part timing table<br>fall out of the profile | `torch.cuda.nvtx.range_push`<br>and `range_pop` around<br>each phase |
| Measure | Roofline check | Achieved FLOP per second<br>and bytes per second<br>against H100 peak | Tells us when to stop<br>optimizing a phase | Prevents chasing a<br>phase already at the<br>hardware limit | Compute both from the<br>measured time and the<br>known op counts |
| Measure | Allocation history | Shows allocation churn<br>inside the loop | When the zero-allocation<br>assertion fails | Diagnostic | `torch.cuda.memory._record_memory_history()` |
| Measure | Fixed-budget experiment<br>ledger | One row per attempted<br>change with before and<br>after | autoresearch's method,<br>which this project<br>already adopts | Keeps the loop honest<br>and reversible | `progress_and_changes.md`<br>per subtask, one commit<br>per experiment |
| Measure | Thread environment<br>variables | Stops CPU-side thread<br>oversubscription on a<br>128-core box | Every benchmark process | Small but removes<br>run-to-run noise | `OMP_NUM_THREADS=1`,<br>and `PYTHONNOUSERSITE=1`<br>so a user-site torch<br>cannot shadow the env |
| Skip | FlashAttention, FA3,<br>SDPA | Fused attention kernels | No attention anywhere in<br>our networks | Does not transfer | — |
| Skip | KV cache, sliding<br>window, grouped-query<br>attention, rotary<br>embeddings | Transformer inference<br>machinery | No sequence model, no<br>autoregressive decode | Does not transfer | — |
| Skip | fp8 training | 8-bit tensor cores | Needs dimensions that<br>are multiples of 16 and<br>at least 128 wide to pay<br>off; our layers are<br>narrower. nanochat only<br>reached about 5% at a<br>much larger model | Does not transfer | — |
| Skip | Mixture of experts,<br>grouped matmul | Sparse expert routing | No experts; nanochat<br>measured it as a net<br>loss even for LLMs at<br>this scale | Does not transfer | — |
| Skip | Muon and Polar Express<br>orthogonalization | Optimizer for large<br>weight matrices | Our per-copy matrices are<br>tiny; the iteration cost<br>would exceed the AdamW<br>step it replaces, and it<br>changes learning, not<br>speed | Does not transfer as a<br>speed trick | — |
| Skip | DDP, ZeRO sharding,<br>NCCL overlap | Multi-GPU training | One H100, one process | Does not transfer | — |
| Skip | Dataloader tricks:<br>worker processes,<br>pinned staging buffers,<br>prefetch-during-backward,<br>document packing | Hiding host-to-device<br>data movement behind<br>compute | There is no host dataset.<br>Every observation is<br>produced on the GPU by<br>our own env, so there is<br>nothing to prefetch and<br>no transfer to hide | Does not transfer.<br>The one surviving idea<br>is the general one:<br>preallocate buffers and<br>write into them, which<br>is its own row above | — |
| Skip | Gradient accumulation | Simulating a larger<br>batch than fits in<br>memory | Our whole rollout fits;<br>splitting it would only<br>add launches. The<br>minibatch update style is<br>an algorithm choice, not<br>a memory workaround | Does not transfer | — |
| Skip | Meta-device init and<br>`to_empty` | Avoids materializing a<br>large model twice at<br>startup | Our parameter count is<br>trivial; startup is<br>dominated by compile time | Does not transfer | — |
| Skip | Vocabulary padding | Padding an embedding<br>table to a multiple of 64 | No vocabulary. The<br>underlying lesson —<br>pad dimensions to<br>tensor-core multiples —<br>is its own row above | Does not transfer<br>literally | — |
| Skip | `cudnn.benchmark`,<br>channels-last memory<br>format | Convolution kernel<br>selection and layout | No convolutions | Does not transfer | — |

## Top 10 most likely wins for this project, in order

1. **Delete every host-device sync from the iteration.** The cleanrl PPO+RND baseline we are
   adapting has several per step: `action.cpu().numpy()`, a numpy running mean and standard
   deviation for both observations and intrinsic rewards, a numpy minibatch shuffle, and `.item()`
   calls for logging. Each one drains the launch queue. Nothing else on this list works until these
   are gone, and on its own it is likely the largest single change.
2. **Preallocate every buffer and make the steady state allocation-free.** Rollout storage as one
   fixed tensor written by index, env state in place, no `cat` and no `stack` in the loop. Second
   because it is also the precondition for graph capture.
3. **Stack the copies into batched weights of shape (copies, in, out).** One batched matmul per
   layer instead of `n_copies` separate ones. This is what makes 128 independent seeds cost close
   to what 8 cost, and it also shrinks the optimizer from hundreds of tiny tensors to a handful.
4. **`torch.compile(dynamic=False, fullgraph=True)` on the env step and on each update variant.**
   Fuses the dozens of small elementwise ops per phase into a few kernels. Keep the two update
   styles as two separate compiled functions so no branch sits in the graph.
5. **CUDA graphs.** Try `mode="reduce-overhead"` first; fall back to manual capture with static
   buffers if it refuses. Once the loop is sync-free and allocation-free this collapses the whole
   iteration to one launch, which is exactly what a launch-bound workload needs.
6. **Fuse across time, not just within a step.** One kernel for the whole rollout horizon in the
   CUDA env, and the GAE reverse loop as a compiled scan rather than T tiny kernels. This is where
   the fused-CUDA env variant can beat the torch one.
7. **Fused, capturable AdamW, or the hand-written compiled step from autoresearch.** Cheap to try,
   removes a host sync (the step counter), and is required for graph capture anyway.
8. **Freeze the shapes.** Copy count, env count, horizon and minibatch size as compile-time
   constants; GPU permutation for minibatches; no data-dependent indexing in resets — use masked
   `where`. Recompilation and dynamic shapes silently undo items 4 and 5.
9. **TF32 on, plus dimension padding to multiples of 8 or 64.** Two one-line changes with measured
   precedent in the reference material and no downside at our precision requirements.
10. **Sweep to the largest batch that fits, then test bf16.** The `n_envs` and `n_copies` curves are
    already a deliverable, so this experiment is free; run the bf16 test at the top of that range,
    where the batched matmuls are finally large enough for it to have a chance.

## Loop discipline

- First run of every subtask is the baseline, unchanged, with the measurement protocol fixed.
- One row of the table per experiment, one commit per experiment, keep or revert on the measured
  number alone.
- Profile before choosing the next row. The ordering above is a prediction, not a plan; the trace
  decides.
- Record the before and after numbers, the verdict, and the reason in the subtask's
  `progress_and_changes.md`, including the reverts — a measured negative result is the main product
  of a run.

## Sources

- `reference_repo/nanoGPT`: `train.py`, `model.py`, `bench.py`, `README.md`.
- `reference_repo/nanochat`: `nanochat/optim.py`, `nanochat/gpt.py`, `nanochat/common.py`,
  `nanochat/dataloader.py`, `nanochat/engine.py`, `scripts/base_train.py`, `scripts/chat_rl.py`,
  `scripts/infer_bench.py`, `dev/LOG.md`, `README.md`.
- `reference_repo/autoresearch`: `train.py`, `program.md`, `README.md`.
- `reference_repo/cleanrl`: `cleanrl/ppo_rnd_envpool.py` (the baseline whose syncs item 1 removes).
- `/p/rlprojects/RLforOR/LLMs-from-scratch`: `ch05/10_llm-training-speed/` (README and
  `01_opt_single_gpu.py`), `ch03/02_bonus_efficient-multihead-attention/`, `ch04/03_kv-cache/`,
  `ch05/11_qwen3/`, `appendix-D/`.
