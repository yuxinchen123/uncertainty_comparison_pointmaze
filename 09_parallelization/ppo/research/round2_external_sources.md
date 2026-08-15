# Round two — techniques from outside this repository

Search done 2026-08-15 against the round-one record (`PROGRESS.md`, the seven
`progress_and_changes.md` ledgers, `extra_step_review.md`, `report/2026-08-15-.../report.md`).
Nothing already measured in those files is proposed again; where a technique is already listed
there as planned-but-unmeasured, the status column says so.

## The regime these recommendations are ranked against

The ranking below follows from three numbers already in our own measurements.

1. One captured rollout step costs about 157 microseconds at C=128 (20.11 ms of rollout graph
   replay divided by 128 steps).
2. The environment is a small part of that: swapping the compiled torch environment for the fused
   CUDA kernel changed the iteration by 1.0 ms (36.6 to 35.6 ms), i.e. about 8 microseconds per
   step. So roughly 149 of the 157 microseconds are the policy, value, RND and buffer work.
3. That work is about 9 batched matmuls plus sampling, RND whitening and 10 buffer writes — call
   it 30 to 40 kernels per step. 149 microseconds over 35 kernels is about 4 microseconds each,
   which is close to the floor cost of a kernel that does almost no arithmetic. At C=128, N=4 a
   trunk matmul is 128 independent 4x64x64 products — a few megaflops, microseconds of fixed cost.

So the loop is not launch-bound (capture already removed launches) and not bandwidth-bound; it is
**kernel-count bound**: the iteration time is roughly the number of kernels times a fixed
per-kernel cost. Every technique below is judged by whether it reduces kernel count, lets kernels
run at the same time, or makes each kernel carry more work. A kernel census of one replayed
iteration (extra-step items 12 and 13) should confirm the 30-to-40 estimate before anything
expensive is built on it.

## A. LLM serving engines: vLLM, SGLang, megakernel work

| Technique | What it is | Here already? | Expected effect here, and why | Cost | Source |
|---|---|---|---|---|---|
| Piecewise CUDA-graph<br>capture | Split the graph at ops<br>that cannot be captured<br>(attention), capture the<br>rest | Not needed | None. vLLM splits because<br>attention resists capture.<br>We have no such op —<br>the whole iteration is<br>already one graph | — | [vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) |
| Uncapturable call<br>registered as a custom<br>operator with a fake<br>implementation | Lets Dynamo keep one<br>graph across an opaque<br>kernel call | NO — `cuda_env` is a<br>plain pybind call, so<br>the CUDA backend runs<br>as compiled subgraphs<br>around it | Lets `fullgraph=True` cover<br>the whole step with the<br>kernel inside, so inductor<br>can fuse the neighbouring<br>elementwise work instead of<br>emitting it separately.<br>A few kernels per step | Small<br>(half a day) | [custom ops](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html), [vLLM torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/) |
| One shared graph memory<br>pool across all captured<br>graphs | `graph_pool_handle()`<br>passed to every capture | NO — rollout, update<br>and iteration graphs<br>each take their own<br>pool | No steady-state speed.<br>Cuts capture-time memory,<br>and our copy ceiling<br>(32,768; 65,536 fails<br>during graph build) is set<br>by exactly that memory | Small | [graph_pool_handle](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph_pool_handle.html) |
| One graph per padded<br>batch shape plus a<br>runtime dispatcher | `cudagraph_capture_sizes`<br>and a batch descriptor | Not applicable | None. Copies, horizon and<br>envs are fixed for a run,<br>so there is one shape | — | [vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) |
| Persistent megakernel<br>with an on-GPU<br>instruction schedule | One kernel resident on<br>every multiprocessor,<br>walking a pre-scheduled<br>instruction list; the<br>schedule is reused for<br>hundreds of passes | NO | The largest lever we have.<br>It removes the fixed<br>per-kernel cost that<br>section "regime" says is<br>most of the 157 us step.<br>Their setting (batch-1<br>decode, many small<br>dependent kernels) is the<br>same shape as ours | Large<br>(weeks) | [Hazy Research megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [whole-GPU follow-up](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) |
| Compiler that turns a<br>tensor program into one<br>persistent kernel | MPK: lowers to<br>multiprocessor-level task<br>graphs, decentralized<br>in-kernel scheduling.<br>1.0-1.7x over vLLM and<br>SGLang, up to 10x over<br>torch with CUDA graphs | NO | Not a drop-in: it targets<br>LLM tensor programs, and<br>our loop has an<br>environment recursion and<br>an optimizer in it. Value<br>is as the design reference<br>for the item above | Large | [MPK paper](https://arxiv.org/abs/2512.22219), [mirage repo](https://github.com/mirage-project/mirage) |
| Compile cache pinned to<br>local disk, keyed by<br>config | Reuse inductor output<br>across processes | Listed in our checklist,<br>never actually set | No steady-state effect.<br>Cuts turnaround of the<br>experiment loop, which is<br>worth having before an<br>autotune sweep | Minutes | [inductor config](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) |

## B. Massively parallel RL infrastructure

| Technique | What it is | Here already? | Expected effect here, and why | Cost | Source |
|---|---|---|---|---|---|
| Batch simulator as a<br>GPU-resident megakernel | Madrona keeps the whole<br>simulation graph in one<br>persistent kernel, one<br>program per world | Our environment is one<br>kernel per step; the<br>rollout-level kernel is<br>planned<br>(extra-step item 2)<br>but never measured | Environment-only, it is a<br>large win on the<br>environment curve and a<br>required deliverable, but<br>only about 3% end to end<br>(measured: 35.6 vs 36.6 ms).<br>Its real value is as the<br>first half of an<br>environment-plus-policy<br>kernel | Medium | [Madrona](https://github.com/shacklettbp/madrona) |
| One block per<br>environment, agents as<br>threads, sampling written<br>in CUDA | WarpDrive: in-place state,<br>no host traffic, its own<br>action sampler about 3x<br>faster than the framework's | Our CUDA kernel is one<br>thread per environment,<br>no block structure;<br>sampling is torch | The block-per-copy layout<br>is the right shape for a<br>fused rollout kernel:<br>copies never interact, so<br>one block can own a copy's<br>weights with no grid-wide<br>barrier. In-kernel sampling<br>removes 2-3 kernels a step | Medium | [WarpDrive](https://arxiv.org/abs/2108.13976), [repo](https://github.com/salesforce/warp-drive) |
| Replace the framework<br>training loop with hand-<br>written CUDA | PufferLib reports 3-5<br>million steps a second for<br>small models on one GPU;<br>version 4 replaced PyTorch<br>with about 5,000 lines of<br>CUDA C | NO. We are at 2.68e6<br>environment steps a<br>second at C=128,<br>style A | Evidence, not a technique:<br>at our model size a hand-<br>written loop beats a<br>framework loop, which is<br>the same conclusion as the<br>megakernel row | Large | [PufferLib](https://github.com/PufferAI/PufferLib), [paper](https://openreview.net/forum?id=qRyteMTgn0) |
| Whole training loop in<br>one compiled function,<br>vectorized over seeds | PureJaxRL, gymnax, Brax,<br>Stoix: jit plus vmap plus<br>scan over the whole loop | DONE — the jax trainer<br>is exactly this shape;<br>the torch trainer is<br>the CUDA-graph<br>equivalent | None left | — | [PureJaxRL](https://github.com/luchris429/purejaxrl), [Stoix](https://github.com/EdanToledo/Stoix) |
| GPU-resident<br>environment plus policy,<br>no host traffic | Isaac Lab, MuJoCo<br>Playground | DONE | None left | — | [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) |
| Asynchronous actor and<br>learner | cleanba runs actors on<br>the GPU and overlaps them<br>with the learner; needs an<br>off-policy correction | NO | The update is 15.5 ms of a<br>50 ms split iteration.<br>Overlapping it with the<br>next rollout could give<br>about 1.3x, and our tiny<br>kernels leave the device<br>idle enough for real<br>overlap. But the data<br>becomes one iteration<br>stale, so it is a labeled<br>algorithm variant and needs<br>a learning check, not only<br>a speed number | Medium | [Cleanba](https://arxiv.org/abs/2310.00036) |
| CPU-side vectorized<br>environments with a C++<br>thread pool | EnvPool | Not applicable | None. Our environment runs<br>on the GPU; there is no<br>host step to hide | — | [EnvPool](https://github.com/sail-sg/envpool) |

## C. CUDA and PyTorch mechanics at our size

| Technique | What it is | Here already? | Expected effect here, and why | Cost | Source |
|---|---|---|---|---|---|
| Capture work from several<br>streams into one graph | Fork with an event onto a<br>side stream, join before<br>ending capture; the graph<br>keeps the fork and join<br>as dependencies, so the<br>branches can run at the<br>same time | NO — side streams are<br>used only for capture<br>warmup | Actor trunk, critic trunk,<br>RND target and RND<br>predictor all read the same<br>observation and do not<br>depend on each other. Each<br>kernel is far too small to<br>fill the device, so running<br>the branches concurrently<br>should shorten the rollout<br>step materially. Caveat:<br>a known allocator problem<br>with multi-stream capture,<br>so gate on the bitwise test<br>and on memory | Small to<br>medium | [CUDA graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [torch issue 114320](https://github.com/pytorch/pytorch/issues/114320) |
| Graph launch is now<br>constant time | Straight-line graph launch<br>went from 2 us plus 200 ns<br>a node (CUDA 11.8) to<br>about 2.5 us plus 1 ns a<br>node (12.6) | Finding, not a change | Tells us to stop here:<br>2.5 us against a 36.6 ms<br>iteration. Any further<br>graph merging, including<br>the ideas below, cannot<br>pay | 0 | [NVIDIA launch blog](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-graphs-and-other-performance-enhancements) |
| Conditional graph nodes<br>(if, while, switch) | Run many iterations inside<br>one graph launch without<br>returning to the host | NO | About 2.5 us saved per<br>iteration, i.e. 0.007%.<br>Not worth it, and not<br>exposed through torch | High | [conditional nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes) |
| Device-side graph launch | A kernel launches a graph;<br>needs an instantiate flag<br>and restricts node types | NO | None. We have no data-<br>dependent host decision to<br>remove | High | [CUDA graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) |
| Graph upload before first<br>launch | Pays the mapping cost<br>outside the timed region | NO | Measurement hygiene for the<br>first replay only, not<br>throughput. Not exposed in<br>torch | Small,<br>needs a<br>C++ shim | [CUDA graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) |
| Programmatic dependent<br>launch | A dependent kernel starts<br>its prologue before its<br>predecessor finishes;<br>compute capability 9.0 | NO | With about 35 dependent<br>tiny kernels a step there<br>is a real tail to recover,<br>but it needs hand-written<br>kernels and graph node<br>dependency types torch does<br>not expose — and a<br>megakernel removes the<br>chain outright | High | [PDL docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) |
| Thread-block clusters and<br>distributed shared memory | Hopper: blocks in a cluster<br>read each other's shared<br>memory | NO | Only meaningful inside a<br>hand-written megakernel, to<br>share one copy's weights<br>across the blocks working<br>on it | High, only<br>in the<br>megakernel<br>path | [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) |
| Cooperative-groups grid<br>synchronization | A persistent kernel with a<br>device-wide barrier between<br>phases | NO | Design note: we do NOT need<br>it. Copies and environments<br>never interact, so a block-<br>per-copy persistent kernel<br>can run the whole horizon<br>with no grid barrier and no<br>cooperative launch | 0 | [CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) |
| Persistent and stream-K<br>matmul kernels | Fixed grid, tiles pulled<br>from a work queue; splits<br>the reduction to avoid a<br>ragged last wave | NO | Stream-K fixes wave<br>quantization on large<br>matmuls. Ours are 128<br>independent 4x64x64<br>products — the problem is<br>fixed cost, not<br>quantization. The useful<br>half (weights resident,<br>loop over work) is the<br>megakernel row | — | [Triton persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html), [Colfax stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) |
| Triton matmul templates<br>with fused epilogue | `max_autotune_gemm_backends`<br>set to Triton, plus<br>coordinate-descent tuning;<br>bias and activation fold<br>into the matmul kernel | Autotune is planned<br>(extra-step item 6) but<br>unmeasured, and today<br>every matmul goes to<br>cuBLAS through ATen,<br>which cannot fuse an<br>epilogue | Directly attacks kernel<br>count: at about 9 matmuls a<br>step, folding the bias add<br>and tanh or relu into each<br>one removes on the order of<br>a third of the step's<br>kernels. This is the<br>cheapest large-effect item<br>on the list | Small<br>(one keyword<br>plus autotune<br>time) | [inductor config](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), [MoE Triton grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) |
| Regional compilation | Compile one repeated block<br>and reuse its code for the<br>others | NO | Cold-start compile time<br>only, and our repeated<br>region (the per-step<br>function) is already<br>compiled once. Little to<br>gain | Small | [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) |
| Automatic cudagraph trees<br>via reduce-overhead | Inductor captures and<br>manages graphs itself | MEASURED AND REJECTED<br>(torch_ppo row 2:<br>capture silently<br>skipped) | Stays rejected; manual<br>capture is faster and<br>verifiable | — | [torch CUDA graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) |
| Expandable allocator<br>segments | Reduces allocator<br>fragmentation | In the checklist, not<br>confirmed set in any<br>run script | No steady-state speed;<br>may raise the copy ceiling,<br>which is a headline number | One<br>environment<br>variable | [torch docs](https://docs.pytorch.org/docs/stable/notes/cuda.html) |

## D. Many independent small models on one GPU

| Technique | What it is | Here already? | Expected effect here, and why | Cost | Source |
|---|---|---|---|---|---|
| Strided-batched matmul<br>over the copy axis | One kernel for all copies'<br>weights of shape<br>[copies, in, out] | DONE — `baddbmm`<br>throughout | None left | — | [strided batched](https://developer.nvidia.com/blog/pro-tip-cublas-strided-batched-matrix-multiply) |
| cuBLAS grouped GEMM<br>(12.5) | One kernel for many<br>matmuls of DIFFERENT<br>shapes | NO | None. Our shapes are<br>identical across copies, so<br>strided-batched is already<br>the right call; the<br>measured 1.2x was against<br>looping batched calls in a<br>mixture-of-experts layer | — | [cuBLAS grouped GEMM](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) |
| CUTLASS grouped GEMM | Same idea, hand-tuned per<br>precision and architecture | NO | Same verdict, plus a<br>per-shape tuning burden | — | [CUTLASS](https://github.com/NVIDIA/cutlass) |
| `grouped_mm` in torch | Mixture-of-experts API,<br>jagged group sizes,<br>compute capability 9.0+ | NO | None. Aimed at variable<br>group sizes with a shared<br>weight layout; no gain for<br>uniform per-copy weights | — | [grouped_mm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html) |
| Segmented gather matmul<br>for many adapters | Punica: one fused kernel<br>serves requests whose<br>weights differ per row | NO | Not reusable directly<br>(their inner dimension is a<br>rank of 8 to 64 over token<br>rows). Useful as the<br>reference for how a kernel<br>that owns many small weight<br>sets is organized | — | [Punica](https://arxiv.org/abs/2310.18547) |
| Fully fused multilayer<br>perceptron | tiny-cuda-nn: one kernel<br>for the whole network,<br>weights loaded once into<br>registers, activations in<br>shared memory; built for<br>hidden widths 16 to 128 | NO | The closest published<br>design to what we need. Our<br>actor and critic trunks<br>(4 to 64 to 64) fit the<br>intended shape exactly. The<br>RND networks (4 to 256 to<br>128, and the predictor's<br>extra layer) are 130 to 260<br>kilobytes of weights per<br>copy, so they must stream<br>from cache rather than sit<br>in shared memory — size<br>this before building | Medium to<br>large | [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), [fused kernel source](https://github.com/NVlabs/tiny-cuda-nn/blob/master/src/fully_fused_mlp.cu) |
| Concatenate the weights<br>of towers that share an<br>input | One matmul instead of two | DONE — torch_ppo row 8,<br>plus 1.9% style B and<br>2.9% style A | None left | — | — |
| Block-diagonal dense<br>matmul instead of batched | Put all copies in one big<br>matmul with a block-<br>diagonal weight | NO, and should stay no | Multiplies arithmetic by<br>the copy count for no<br>reduction in kernel count | — | — |
| More environments per<br>copy | Raise the rows each matmul<br>processes so the same fixed<br>kernel cost carries more<br>work | Measured ONLY as the<br>labeled T=32, N=16<br>variant: 3.0x in jax<br>style A | The largest measured single<br>number we have, and it is<br>the direct consequence of<br>the kernel-count regime. It<br>shortens the advantage<br>horizon, so it is an<br>algorithm change and must<br>stay a labeled variant with<br>a learning check | None to<br>implement | ledger `jax_ppo` row 3 |

## E. JAX and XLA specifics

| Technique | What it is | Here already? | Expected effect here, and why | Cost | Source |
|---|---|---|---|---|---|
| XLA command buffers | XLA's own CUDA-graph<br>mechanism; on by default,<br>covering fusions, cuBLAS,<br>custom calls and a few<br>others | Implicitly on, never<br>checked | Verify that our jitted<br>iteration is actually<br>recorded as a command<br>buffer and that the<br>minimum-region-size<br>threshold is not excluding<br>our small regions. If it<br>is, the jax numbers are<br>launch-bound for no reason | One flag<br>plus a<br>profile | [XLA flags](https://openxla.org/xla/flags_guidance), [JAX GPU tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) |
| Triton matmul emitter for<br>every matmul | `xla_gpu_triton_gemm_any` | NO | The jax twin of the<br>inductor Triton template<br>row: the Triton emitter can<br>fuse the surrounding<br>elementwise work into the<br>matmul, cutting kernel<br>count. Cheap to test | One<br>environment<br>variable | [JAX GPU tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) |
| Whole-iteration jit and<br>buffer donation | One program per<br>iteration, state donated | DONE | None left | — | ledger `jax_e2e` rows 0-1 |
| Scan unroll | Trade compile time and<br>memory for fewer loop<br>boundaries | Planned (extra-step<br>item 10), unmeasured | Still worth running as<br>planned | Small | [XLA flags](https://openxla.org/xla/flags_guidance) |

## Worth trying now, in order

1. **Kernel census plus device-idle share of one replayed iteration.** Sum kernel device time
   against replay wall time and count kernels by kind at C=8 and C=128. Everything above is ranked
   on an estimate of 30 to 40 kernels a step; this makes it a measurement. Half a day, no code
   change. Already on the extra-step list as items 12 and 13.
2. **Triton matmul templates with fused epilogues in the torch trainer.** Set the inductor matmul
   backend to Triton with `max-autotune-no-cudagraphs` and coordinate-descent tuning, and pin the
   compile cache to local disk. cuBLAS cannot fold a bias add and a tanh into the matmul; a Triton
   template can. Directly reduces kernel count, which is the binding constraint. One keyword plus
   autotune time.
3. **Multi-stream capture of the four independent network branches.** Actor trunk, critic trunk,
   RND target and RND predictor all read the same observation and do not depend on each other;
   recording them on forked streams lets the captured graph run them at the same time on a device
   our kernels cannot fill. Gate on the existing bitwise capture test and watch for the known
   allocator problem with multi-stream capture.
4. **Register the fused CUDA environment kernel as a custom operator with a fake implementation.**
   Today it forces the step into subgraphs around an opaque call. With a proper registration the
   whole step compiles as one graph with the kernel inside, so the surrounding elementwise work
   fuses instead of becoming separate kernels.
5. **Share one graph memory pool across all captured graphs, and set expandable allocator
   segments.** No steady-state speed, but the maximum copy count is a headline deliverable and it
   is currently limited by memory during graph build.
6. **In jax: confirm command buffers are actually being formed, then test the Triton matmul
   emitter and the planned scan unroll.** Two environment variables and a profile; the jax trainer
   is the faster of the two, so a further gain there moves the reported best number.
7. **Fused per-step rollout kernel covering environment, policy, value and RND for all copies.**
   One block per copy, weights loaded once, no grid barrier needed because copies never interact.
   This is the item that attacks the 149 microseconds of non-environment work directly, and it is
   the same design as tiny-cuda-nn's fused network, WarpDrive's block-per-environment layout and
   Madrona's resident simulator. Build it only after step 1, and only if the written go-or-no-go
   criterion from extra-step item 12 clears.
8. **Two labeled algorithm variants, measured for learning as well as speed:** more environments
   per copy with a shorter horizon (measured at 3.0x in jax), and overlapping the update with the
   next rollout in the style of cleanba (about 1.3x available, at the cost of one-iteration-stale
   data and an off-policy correction).

## Explicitly not worth doing, with the reason

1. Conditional graph nodes, device-side graph launch and graph upload: graph launch is now about
   2.5 microseconds against a 36.6 millisecond iteration, so there is nothing left to win there.
2. Grouped GEMM in any of its forms (cuBLAS, CUTLASS, `grouped_mm`): it exists for matmuls with
   different shapes per group; ours are identical across copies, so the strided-batched call we
   already use is the right one.
3. Stream-K matmul scheduling: it fixes a ragged last wave on large matmuls, and our problem is
   fixed cost on tiny ones.
4. EnvPool-style host-side vectorization: there is no host environment step to hide.
5. Piecewise capture: we have no operation that resists capture.
