# How far from the H100's limit is this project, and what stage of optimisation is it at

Everything below is measured on the same machine the project used: one NVIDIA H100 NVL in
serval05, under the project's exclusive-use lock. Two new measurement programs were written for
this document and their outputs are stored beside it:

- `code/ceiling_probe.py` → `data/ceiling_probe.json` — what this card can actually do
  (matrix multiplication rate, memory bandwidth, cost of one kernel).
- `code/matmul_floor.py` → `data/matmul_floor.json` — the time every matrix multiply in one
  training iteration takes when timed on its own, which is the fastest the iteration could
  possibly go with the algorithm written the way it is.

Repository numbers are taken from `report/2026-08-15-gpu-parallel-rl-environment-training-endtoend/report.md`
(the round-2 measurements), `report/2026-08-15-pointmaze-gpu-parallelization/report.md`, the
per-module `progress_and_changes.md` ledgers, and `benchmarks/results/*.json`.

The repository's `*.json` rule already keeps `data/` out of version control, as the analysis-folder
convention requires. `code/` is left tracked on purpose: the probe programs are the instrument every
number below is read from, and a reader cannot check the arithmetic without them.

Two companion documents sit in this same folder and go deeper on one component each:
`environment_roofline.md` (the environment step, including the compiled instruction count this
document uses in section 7) and `trainer_roofline.md` (the training iteration). This document is the
one that places both on a scale and compares them with published systems.

## 1. The three ceilings, and why a program only ever hits one of them

A graphics processor can be thought of as three separate supplies that a program consumes, and a
program runs at the speed of whichever supply it exhausts first.

1. **Arithmetic.** How many multiply-and-add operations per second the chip can perform. Reported
   in FLOP/s, floating-point operations per second.
2. **Memory bandwidth.** How many bytes per second can move between the chip and its memory.
   Reported in bytes/s.
3. **Instruction issue and kernel dispatch.** A GPU executes *kernels*: one program launched once
   and run by many threads. Every kernel costs a fixed amount of time to start regardless of how
   much work it does, and inside a kernel each thread can only issue one instruction per clock
   cycle. A program made of many tiny kernels, or of long chains of dependent scalar arithmetic,
   runs out of this supply long before it runs out of the other two.

**The roofline model** (Williams, Waterman and Patterson, 2009) is the standard way to decide which
of the first two binds. Define the *arithmetic intensity* of a computation as the number of
floating-point operations it performs per byte it moves to and from memory. Plot achievable
performance against arithmetic intensity: the result is a slanted line (bandwidth times intensity)
that stops at a horizontal line (peak arithmetic rate). The corner where the two meet is the
*ridge point*. A computation to the left of the ridge point can never be arithmetic-limited — it
is memory-limited no matter how good the code is. One to the right can never be memory-limited.

For the card in this project the ridge point is

    60.3 TFLOP/s (single precision, ordinary arithmetic) / 3.94 TB/s = 15.3 FLOP per byte
    417.5 TFLOP/s (reduced-precision matrix units)      / 3.94 TB/s = 106 FLOP per byte

So: a computation performing fewer than about 15 arithmetic operations per byte it touches cannot
be made arithmetic-limited on this card, and one that wants to use the matrix units needs more
than 100 operations per byte before arithmetic is even in principle the constraint.

The third supply — instruction issue and kernel dispatch — is not on the roofline picture at all,
and it is the one that binds this project. That is the single most important fact in this
document; sections 6 and 7 measure it.

## 2. The card: specification numbers, each with a source and a check

| quantity | value | source |
|---|---|---|
| product | H100 NVL, one GPU | device query |
| memory | 94 GB HBM3 (95,830 MiB reported) | product brief, device |
| peak memory bandwidth | 3,938 GB/s | product brief, Table 2 |
| maximum clock | 1,785 MHz | product brief; device |
| streaming multiprocessors | 132 | device query |
| L2 cache | 60 MiB | device query |
| board power limit | 400 W | product brief |
| FP32 (ordinary arithmetic) | 60 TFLOP/s | H100 NVL datasheet |
| TF32 matrix units, dense | 417.5 TFLOP/s | datasheet, 835 halved |
| BF16 matrix units, dense | 835 TFLOP/s | datasheet, 1,671 halved |

Terms used above:

- **Streaming multiprocessor (SM)**: one of the 132 independent processor blocks on the chip. Each
  has 128 lanes that can do one single-precision multiply-and-add per clock.
- **TF32**: a reduced-precision number format that the matrix units accept in place of full single
  precision. Inputs keep single-precision range but carry 10 bits of mantissa instead of 23. This
  project turns it on (`+9%` on the trainer, recorded in the torch ledger row 6).
- **"With sparsity"**: NVIDIA's datasheet tensor-core rows are quoted for a mode that skips half
  the multiplications in a specially structured matrix. No ordinary program gets that, so the
  honest peak is half the quoted number. This is why 835 becomes 417.5.

Two checks that these numbers are consistent:

- FP32 peak from the device's own reported geometry: 132 SMs x 128 lanes x 2 FLOP per
  multiply-and-add x 1.785e9 clocks/s = **60.3e12 FLOP/s**, which matches the datasheet's 60.
- The dense TF32 figure implies a clock of 417.5e12 / (132 x 2048 FLOP per SM per clock) =
  **1.54 GHz**, below the 1.785 GHz maximum. That is expected: this is a 400 W card and it cannot
  hold its top clock while the matrix units are saturated. It also predicts that a measured
  matrix-multiply rate will land below 417.5, which is exactly what section 3 finds.

## 3. What this card actually delivers (measured, not quoted)

Datasheet peaks are not reachable. The practice of measuring your own ceiling and comparing
against that instead is argued for in Stas Bekman's `ml-engineering` handbook, which calls the
measured number the *maximum achievable matmul FLOPS* and points out that a metric expressed
against the theoretical peak "provides no indication of actual headroom".

From `data/ceiling_probe.json`:

| measurement | result | fraction of quoted peak |
|---|---|---|
| TF32 matmul, 4096 cube | 265.3 TFLOP/s | 63.5% of 417.5 |
| TF32 matmul, 8192 cube | 261.2 TFLOP/s | 62.6% |
| TF32 matmul, 16384 cube | 222.3 TFLOP/s | 53.2% |
| FP32 matmul, 4096 cube | 45.5 TFLOP/s | 75.4% of 60.3 |
| memory copy, 4 GiB | 3.568 TB/s | 90.6% of 3.938 |
| read-read-write, 1 GiB | 3.601 TB/s | 91.4% |
| one operation issued from Python | 4.2 microseconds | — |
| one kernel inside a replayed recording | 1.28 microseconds | — |

Arithmetic for the first row: a 4096-cube matrix multiply performs 2 x 4096^3 = 1.374e11
operations; it took 0.518 ms; 1.374e11 / 5.18e-4 = 2.653e14 = **265.3 TFLOP/s**.

Arithmetic for the copy row: 4 GiB read plus 4 GiB written is 8.59e9 bytes, in 2.407 ms;
8.59e9 / 2.407e-3 = 3.57e12 = **3.568 TB/s**.

Three things follow, and they set up everything after.

1. **Memory is the healthy supply.** A simple copy reaches 91% of the quoted bandwidth. When
   something is bandwidth-limited on this card you will see it near 3.5 TB/s, not near 1 TB/s.
2. **Arithmetic is not.** The best matmul reaches 63% of the dense quoted peak, and drops as the
   problem grows and the card heats. Any statement of the form "we reached X% of peak" should be
   read against 265, not 417.
3. **Kernels are expensive.** Even a kernel that adds one to a number costs 1.28 microseconds
   inside a pre-recorded replay, and 4.2 microseconds when issued one at a time from Python. A
   training iteration that launches 5,000 kernels has spent 6.4 ms before doing any arithmetic.

## 4. What one training iteration of this project actually computes

The networks, from `ppo/research/ppo_rnd_algorithm_spec.md` section 3. Observation width 4, action
width 2. One *copy* is one complete independent experiment; the shipped campaign runs C = 128 of
them at once, each holding N = 4 environments and collecting T = 128 steps, so B = T x N = 512
rows of experience per copy per iteration.

| network | layers | multiply-adds per row |
|---|---|---|
| actor | 4→64, 64→64, 64→2 | 4,480 |
| critic | 4→64, 64→64, two 64→1 heads | 4,480 |
| RND target (frozen) | 4→256, 256→128 | 33,792 |
| RND predictor (trained) | 4→256, 256→128, 128→128 | 50,176 |

One multiply-add is 2 FLOP. A backward pass costs about twice a forward pass, because it computes
a gradient with respect to the layer's input and a gradient with respect to its weights, each one
matrix multiply of the same size. So forward-and-backward is about 3 times forward.

Per copy, one iteration in the shipped configuration (style B, four passes over the data in
sixteen small updates). The wide passes are read off `_post_body_impl` in
`ppo/torch_ppo/torch_ppo_rnd.py`: the critic runs once on the observations and the next
observations concatenated, so 1024 rows; the frozen RND target runs twice, once for the curiosity
bonus under the statistics as they stood at the start of the iteration, and once more after those
statistics are advanced, cached for the update phase to reuse.

    rollout, actor only:   128 steps x 4 rows x 8,960 FLOP       =    4.59 MFLOP
    wide pass, critic:          1024 rows x 8,960 FLOP           =    9.18 MFLOP
    wide pass, RND target:       512 rows x 67,584 FLOP          =   34.60 MFLOP
    wide pass, RND predictor:    512 rows x 100,352 FLOP         =   51.38 MFLOP
    wide pass, RND target again: 512 rows x 67,584 FLOP          =   34.60 MFLOP
    update: 16 x 128 rows x (actor+critic+predictor) x 3         =  726.66 MFLOP
    ----------------------------------------------------------------------------
    total per copy                                               =  861.0  MFLOP

At 128 copies: 861.0e6 x 128 = **110.2 GFLOP per iteration**. Two programs written independently
for this folder — `code/matmul_floor.py` here and `code/count_arithmetic.py` for the companion
trainer analysis — arrive at 110,209,531,904 and 110.2 GFLOP respectively, so the hand arithmetic
above is confirmed twice.

The one-update-per-batch configuration (style A) does the update once over 512 rows instead of
sixteen times over 128, giving 316.0 MFLOP per copy and **40.4 GFLOP** per iteration.

## 5. Model FLOPs Utilisation: the standard yardstick, and our number

**Model FLOPs Utilisation (MFU)** was introduced in the PaLM paper (Chowdhery et al., 2022) as
"the ratio of the observed throughput relative to the theoretical maximum throughput of a system
operating at peak FLOPs". You count only the arithmetic the model is *defined* to need, divide by
the time taken, and divide again by the chip's peak rate.

**Hardware FLOPs Utilisation (HFU)** counts every operation the chip actually performed, including
work an implementation repeats — most commonly activations recomputed during the backward pass to
save memory. The distinction was made explicit in NVIDIA's Megatron activation-recomputation paper
(Korthikanti et al., 2022). HFU is always at least MFU; the gap is implementation overhead that
does not help the model.

This project has its own small example of the gap, and it is worth following because of how the
numbers come out. The curiosity mechanism holds a frozen network whose outputs cannot change within
an iteration. Until row 7 of the trainer ledger, that network was evaluated inside every one of the
sixteen update steps: 16 x 128 rows x 67,584 FLOP = 138.4 MFLOP per copy, where the algorithm
requires only 512 rows once, 34.6 MFLOP. The implementation was therefore performing 12.6% more
arithmetic than the model defines — an HFU above MFU by that margin — and computing the network
once per iteration removed all of it. The measured speedup was **1.6%** (6.7% in the
one-update-per-batch configuration), not 12.6%. That is the whole lesson of this document in one
line: on a workload that is not limited by arithmetic, removing arithmetic buys almost nothing.
After that change MFU and HFU are the same number here.

What counts as a good MFU, in the community that uses the term:

| system | MFU | hardware |
|---|---|---|
| GPT-3 as originally trained | 21.3% | reported in PaLM paper |
| Gopher | 32.5% | reported in PaLM paper |
| Megatron with recomputation | 42.1% | 2,240 A100 |
| Megatron, recomputation avoided | 54.2% | 2,240 A100 |
| PaLM 540B | 57.8% | TPU v4 |
| llm.c reproducing GPT-2 | about 60% | 8x A100 |

So a large language model training run is judged good at roughly 40-60% and excellent above 55%.

Ours, at C = 128, style B, 20.4 ms per iteration (the shipped round-2 configuration):

    110.21 GFLOP / 0.0204 s = 5.40 TFLOP/s
    5.40 / 417.5 = 1.29%   against the quoted dense TF32 peak
    5.40 /   265 = 2.04%   against what this card actually delivers

Style A, 8.1 ms: 40.45 / 0.0081 = 4.99 TFLOP/s, so 1.20% and 1.88%.
The unoptimised starting point, 777 ms: 0.14 TFLOP/s, so 0.03% and 0.05%.

**Two percent looks like a catastrophe and is not one.** The next section explains why, and
replaces the yardstick with one that means something here.

## 6. The right yardstick here: the ceiling for the algorithm as written

MFU asks "what fraction of the chip's arithmetic did you use". That question is only fair when the
work is shaped so that the chip's arithmetic *could* be used — that is, large matrices. A matrix
multiply reaches peak when all three of its dimensions are in the thousands. This project's
matrices are 4 wide, 64 wide, at most 256 wide, and in the rollout only 4 rows tall. No
implementation can make such a multiply fast; the hardware has no way to keep 132 processor blocks
of 128 lanes busy with a 4-by-64 matrix.

So the honest question is not "what fraction of the chip did you use" but **"how close are you to
the fastest this exact set of operations can possibly run on this chip"**. That is measurable:
time every matrix multiply of one iteration on its own, with nothing else in the way, and add up
the times. Nothing can be faster than that sum, because those multiplies still have to happen, one
after another, in the order the algorithm requires.

From `data/matmul_floor.json` (two independent runs agreed to 0.5%):

| part of the iteration | measured time | matmuls alone | matmuls' share |
|---|---|---|---|
| rollout, 128 sequential steps | 4.97 ms | 2.74 ms | 55.1% |
| wide passes after the rollout | 1.30 ms | 0.34 ms | 26.4% |
| update, 16 small steps | 13.89 ms | 3.47 ms | 25.0% |
| whole iteration | 20.17 ms | 6.55 ms | 32.5% |

(The measured column is the round-2 phase split at C = 128 from the end-to-end report section 4.2;
the shipped one-recording configuration measures 20.2 to 20.4 ms for the same work.)

Read this table twice.

**First reading — how far we are from the limit.** 6.55 ms is the floor for this algorithm on this
card. We run at 20.4 ms. So the implementation is at **6.55 / 20.4 = 32% of the ceiling for the
algorithm as written**, with at most a 3.1x further gain available and only if every non-matrix
operation on the device were made free, which is impossible. Style A is at 3.63 / 8.1 = **45%**.
For comparison, the same measure applied to the starting point (777 ms) gives 0.8%, and to the
end of round 1 (36.6 ms) gives 18%.

**Second reading — why MFU is the wrong number.** Those floor matmuls, run in isolation with
nothing in the way, themselves achieve:

    rollout matmuls: 0.587 GFLOP / 2.739 ms  =  0.21 TFLOP/s  = 0.08% of 265
    wide passes:    16.609 GFLOP / 0.343 ms  = 48.4  TFLOP/s  = 18.2% of 265
    update matmuls: 93.013 GFLOP / 3.472 ms  = 26.8  TFLOP/s  = 10.1% of 265
    all together:  110.210 GFLOP / 6.553 ms  = 16.8  TFLOP/s  =  6.3% of 265

**The best conceivable MFU for this algorithm on this card is about 6%.** We are at 2%. Judging
this workload against the 50% that a language model reaches is judging a bicycle by a lorry's
payload: the arithmetic is not there to be done. The 4-row rollout multiplies run at 0.08% of the
card and no code change fixes that; only changing the algorithm's shape does (section 9).

## 7. The environment: which supply does it exhaust?

The hand-written CUDA environment reaches 1.92e10 environment steps per second at 3 million
environments (cuda ledger row R2-3). The companion analysis in this folder,
`environment_roofline.md`, derives from the source that one environment step moves **82 bytes** and
performs about **250 floating-point operations**, and confirms the operation count by compiling the
kernel for this chip and counting its intermediate assembly. Those two numbers decide everything
below. (The cuda ledger's own rougher traffic model says 94 bytes; the conclusions do not turn on
which is used.)

**Is it memory-limited?** At 1 million environments the step takes 51.8 microseconds:

    1e6 envs x 82 bytes = 82 MB moved, in 51.8 microseconds
    82e6 / 51.8e-6 = 1.58 TB/s = 44% of the 3.57 TB/s this card actually delivers

44% is not the signature of a bandwidth-limited kernel. And the project proved it directly:
experiment E2 in the cuda ledger packed the state so it loads in one wide transaction and deleted a
redundant output, and the throughput did not move at all (59.8 versus 59.6 microseconds on the
build of the day). A change that removes bytes and buys nothing is proof that bytes were not the
constraint.

**What the roofline says, and why it is wrong here.** The arithmetic intensity is

    250 FLOP / 82 bytes = 3.05 operations per byte

against a ridge point of 15.3 for ordinary single-precision arithmetic. That is far to the *left*
of the corner, so the classical roofline calls this kernel memory-bound and puts its ceiling at
3.938e12 / 82 = 4.8e10 environment steps per second. The kernel reaches 1.92e10, which is 40% of
that — and moving fewer bytes bought nothing. The roofline's verdict is wrong here, and why it is
wrong is the most instructive thing in this section.

**The third supply decides.** The instruction count shows that **only about a quarter of the step's
instructions are floating-point arithmetic**. The rest are the comparisons that rank the eight
candidate walls, the selects that implement every branch without branching, and address arithmetic.
Those consume issue slots exactly as arithmetic does, and the roofline does not count them.
Counting all of them — 921 intermediate-assembly instructions, plus the expansion of correctly
rounded divides and square roots into the final machine code — gives about 1,050 machine
instructions per environment step, against an issue capacity of

    132 SMs x 128 lanes x 1.785e9 clocks/s = 3.02e13 instructions per second
    3.02e13 / 1,050 = 2.87e10 environment steps per second

So the ceiling that actually binds is 2.87e10, not 4.8e10, and the kernel at 1.92e10 is running at
**67% of it**. That is a finished-looking number, and it is set by a resource that appears on
neither axis of the roofline plot.

Two honest qualifications. The 1,050 is a counted 921 plus an estimate of the divide and
square-root expansion, with a stated range of 950 to 1,250, because no disassembler able to read a
Hopper binary is installed on the machine (the system `cuobjdump` is from CUDA 11.5 and refuses
it). And an instruction count is a property of the code as written: the cuda ledger records a coded
but unmeasured candidate — recomputing the full contact law for only the two winning wall
candidates instead of all eight — whose entire purpose is to lower it.

**Where the small-batch floor comes from.** Below about 30,000 environments the step time is flat
at 5.6 to 6.2 microseconds no matter how many environments there are. That is not simulation time;
it is one kernel launch plus one Python-to-C++ call. Compare with the measured 4.2 microseconds
for a single operation issued from Python in section 3: the environment's small-batch floor *is*
the dispatch cost, and nothing about the physics.

## 8. Amdahl's law applied to this profile: what is actually left

Amdahl's 1967 argument: if a part of a program takes fraction f of the total time, then making
that part infinitely fast speeds up the whole program by at most 1/(1-f). It is the arithmetic that
tells you when to stop working on something.

Applying it to the round-2 profile at C = 128:

| part | share of iteration | best possible speedup from perfecting it |
|---|---|---|
| update phase | 68.9% | 3.2x |
| rollout phase | 24.6% | 1.33x |
| post-rollout passes | 6.4% | 1.07x |
| non-matmul work, all phases | 67.5% | 3.1x |
| the environment itself | at most about 1.5% | at most 1.015x |

The last row is the one this project established by experiment rather than by arithmetic, and it is
a bound rather than a share. The hand-written CUDA environment is 24 times faster than the compiled
PyTorch one when measured alone. Substituting it into the training loop moved the iteration from
20.4 ms to 20.7 ms — that is, slightly the wrong way, within the ordinary spread. An environment
made 24 times faster that changes nothing can only have been a very small share to begin with. This
is Amdahl's law arriving as a measurement, and it is the single most useful result in the repository
for deciding where *not* to work next.

The same arithmetic says where to work: the update phase is 69% of the iteration and only a
quarter of it is matrix multiplication. The remaining roughly 10 ms is activations, gradient
clipping, the optimiser, and gathering minibatches — none of which has been attributed kernel by
kernel. That attribution is the next measurement, not the next optimisation.

## 9. Changing the algorithm's shape, which is the only way past the ceiling

Section 6's ceiling is "for the algorithm as written". Rewriting what is computed moves the
ceiling itself. The project has already done this three times, and measured a fourth it declined
to adopt:

1. **Train many independent copies at once.** One copy uses a negligible slice of the card. At the
   unoptimised starting point, 128 copies cost the same 777 ms per iteration as 8 copies did, so
   the copy axis alone multiplied throughput by 16 at no cost; the implementation work then
   multiplied it by a further 38 (777 ms to 20.4 ms). End to end, the training rate went from
   5.27e3 environment steps per second (8 copies, unoptimised) to 8.19e6 (128 copies, final,
   one update per batch) — a factor of 1,550, and none of it visible in an MFU figure computed for
   a single copy.
2. **Move work out of the sequential loop.** The critic, the action log-probability and the
   curiosity bonus were computed inside the 128-step loop; each is a function of data the loop
   already stores, so all 128 steps can be done in one wide pass afterwards. Worth 42% at 128
   copies, the largest single change of round two. It converts 128 small matrix multiplies into one
   large one, so it lowers the floor of section 6 while computing exactly the same numbers — the
   best kind of shape change, because nothing is traded for it.
3. **Fuse a hyperparameter sweep into one run.** Sixteen learning rates x 128 copies costs 1.0%
   more than training the same 2,048 copies at one rate, and 2.23x less than running the sixteen
   groups one after another.
4. **Declined: shorten the horizon.** Collecting 512 rows per copy as 32 steps of 16 environments
   instead of 128 steps of 4 gives a 3.0x faster iteration in the JAX build (style A) because it
   cuts the sequential chain fourfold and makes every matmul four times taller. It is recorded as a
   labelled variant and not adopted, because it truncates the advantage-estimation horizon and so
   changes the algorithm rather than its implementation.

Item 4 is the honest illustration of what stage (e) means: the remaining large factor is available,
it is understood, and taking it costs something other than engineering time.

## 10. How well-known codebases justify their performance numbers

Four conventions are in use, and they answer different questions.

**Fraction of peak arithmetic (MFU).** nanoGPT computes it in the training loop itself
(`model.py`, `estimate_mfu`), following the PaLM paper's appendix B, and prints it every step next
to the loss; the peak it divides by is written into the source as a constant (`flops_promised =
312e12`, one A100 in bfloat16). nanochat carries a table of peak rates per device
(`nanochat/common.py`) — it lists this exact card, "h100 nvl", at 835e12 — and logs `train/mfu` to
its dashboard. llm.c's GPT-2 reproduction is justified as "up to ~60% model flops utilisation".
The convention is: publish the fraction, publish the peak you divided by.

**Fraction of peak bandwidth (MBU).** nanochat's inference benchmark introduces the memory-side
analogue explicitly: "MBU (model bandwidth utilization) is the decode counterpart of training MFU:
achieved bytes/sec over the peak bandwidth of the GPU. It measures how far the implementation is
from the physical ceiling." It reports MFU for the compute-bound phase of inference and MBU for
the bandwidth-bound phase — the same two-supply reasoning as the roofline, applied per phase.

**Percentage of peak, per kernel, from the profiler.** NVIDIA's Nsight Compute opens every kernel
report with a "GPU Speed Of Light Throughput" section that states, in its own words, "the achieved
percentage of utilization with respect to the theoretical maximum" for compute and for memory, and
a breakdown naming the highest contributor. Practitioners read the pair: high memory and low
compute means bandwidth-bound; low both means latency- or occupancy-bound, which is this project's
regime.

**Wall-clock to a fixed quality on fixed hardware.** The nanoGPT speedrun leaderboard ranks
attempts by time to reach a fixed validation loss on a fixed 8xH100 node, with the deliverable
being a commit plus a training log, and rules forbidding changes to the data pipeline. This
convention deliberately refuses to separate algorithmic from systems improvements, on the grounds
that the wall clock is what is actually paid. Serving systems report in the same spirit: the vLLM
and SGLang papers publish throughput-versus-latency curves at a fixed model and fixed hardware
against named baselines, not fractions of peak.

**The stopping rule.** The most direct written guidance is Bekman's `ml-engineering`: measure your
device's achievable matrix-multiply rate and compare against that, because a percentage of the
theoretical peak "provides no indication of actual headroom". This project's own
`extra_step_review.md` carries the same rule from the previous efficiency campaign (item 16): stop
a subtask "when three consecutive distinct-kind ideas are discarded, or when a measured roofline is
reached", and it insists the roofline be stated as a number rather than asserted. Item 12 of the
same review records the practice of writing a *go/no-go criterion before* building a large
optimisation: the reference campaign wrote down thresholds on device-idle share, elementwise share,
and the ratio of a hand-written kernel to the vendor library, measured them, computed a ceiling of
1.243x, and cancelled a multi-week build.

## 11. This project beside published parallel reinforcement-learning systems

Throughput comparisons across reinforcement-learning systems are only meaningful with the
environment's cost stated, so it is in the table. "Environment only" means no learning is
happening; "end to end" means the number is measured while training.

| system | what is measured | steps/second | hardware | environment |
|---|---|---|---|---|
| **ours, CUDA kernel** | environment only | 1.9e10 | 1x H100 NVL | 2-D point, exact contacts |
| **ours, JAX fused** | environment only | 9.9e9 | 1x H100 NVL | same |
| **ours, PyTorch compiled** | environment only | 4.2e9 | 1x H100 NVL | same |
| **ours, JAX trainer** | end to end | 1.9e7 | 1x H100 NVL | same, 2-4k copies |
| **ours, PyTorch, 128 copies** | end to end | 8.2e6 | 1x H100 NVL | same |
| Madrona, Overcooked | environment only | 4.0e7 | 1x RTX 4090 | grid kitchen |
| Madrona, Hanabi | environment only | 2.0e7 | 1x RTX 4090 | card game |
| PureJaxRL, CartPole | environment only | 2.0e7 | 1x A100 | 4-state cart |
| WarpDrive, Tag | environment only | 9.8e6 | 1x GPU | 2-D chase, 5 agents |
| EnvPool, MuJoCo | environment only | 3.0e6 | DGX-A100 host, 128 cores | 3-D articulated |
| Madrona, hide and seek | environment only | 1.9e6 | 1x RTX 4090 | 3-D rigid body |
| EnvPool, Atari | environment only | 1.0e6 | DGX-A100 host, 128 cores | emulator |
| PufferLib 2.0 | end to end | 3e5 to 1.2e6 | 1x RTX 4090 | its C environments |
| Isaac Gym | end to end | 5.4e5 | 1x A100 | 3-D articulated robots |
| Isaac Lab, humanoid | end to end | about 1.5e5 | 1x RTX 4090 | 3-D articulated robot |

Provenance note: every row above comes from the system's own paper or project page, except the
Isaac Lab row, which comes from secondary reporting — the Isaac Lab paper's abstract does not state
a throughput figure. EnvPool's numbers are for environments running on the host's CPU cores, not on
the GPU, which is why the host core count is given instead of a GPU.

**Reading this fairly.** Our environment is a two-dimensional point mass with an exact two-contact
solve against up to eight candidate wall boxes — considerably cheaper per step than a
three-dimensional articulated body (Isaac Gym, Isaac Lab, Madrona's hide and seek, EnvPool's
MuJoCo) and considerably more expensive than a grid world or a cart on a rail (Overcooked,
CartPole). Our networks are also tiny, which raises our step rate and would not transfer to a
system training a larger policy. The number in the table that carries the least distortion is the
end-to-end row against PufferLib and Isaac Gym: 8.2e6 against 1.2e6 and 5.4e5, on a larger GPU,
with a cheaper environment and smaller networks. The claim that survives all the caveats is
narrow and true: **for a small-network, cheap-environment workload, this system's end-to-end
training rate is in the same band as the fastest published systems of that kind, and its
environment-only rate is at the top of the published range.**

## 12. A grading scale a non-specialist can apply

The scale answers "what stage is this at", not "is it fast". Each stage is defined by the
*evidence that exists*, not by a speed. Read down the middle column and stop at the last row whose
evidence you actually have.

| stage | the artefact that proves it | next win |
|---|---|---|
| **(a) unoptimised** | a runtime, nothing else | 10x to 100x |
| **(b) obvious waste removed** | a ledger of changes, each with a before and after | 3x to 30x |
| **(c) resource named, profile flat** | a profile plus a named binding resource | 1.2x to 2x |
| **(d) at the hardware limit** | a measured ceiling, and the fraction of it reached | under 1.3x |
| **(e) algorithm reformulated** | a change to what is computed, priced both ways | resets the ceiling |

In full:

1. **(a) Unoptimised.** Someone timed it. There is no profile, no stated ceiling, and comparisons
   are single runs against single runs.
2. **(b) Obvious waste removed.** A written ledger records every change tried, kept or rejected,
   with a measurement for each. The structural things are done: the work runs on the device rather
   than the host, in batches rather than loops, compiled rather than interpreted.
3. **(c) Critical resource named, profile flat.** "Flat" does not mean the phases are equal; it
   means no single *removable* item is a large share of the time — what is left is the algorithm's
   own work, in proportion to how much of it there is. One resource is *named* as the binding one,
   with a measurement that rules the others out — most convincingly by a change that should have
   helped if a different resource were binding, and did not. Individual changes are now worth 1%
   to 15%, which cannot be told from run-to-run
   drift without running the two variants alternately in separate processes and reporting the
   spread between repeats of the same thing.
4. **(d) At the hardware limit for the algorithm as written.** There exists a ceiling *measured on
   this machine for these exact operation shapes*, and the achieved fraction of it is stated as a
   number. Remaining candidates have been costed and are worth less than they take.
5. **(e) The algorithm reformulated.** What is computed changes — batch shape, sequence length,
   number of passes over the data, arithmetic precision — and the effect on the result is stated
   alongside the effect on the clock, so a reader can see what was traded.

Two rules for using the scale.

- **Stages are not a ranking of quality.** A system at (c) with a stated ceiling is in better shape
  than one claiming (d) with no measured ceiling. The evidence is the grade.
- **(d) requires a ceiling, not a feeling.** "We tried everything we could think of" is stage (c)
  with the profile missing. The distinguishing artefact is a number of the form "we are at X% of
  Y, and Y was measured on this machine".

### Where this project sits

| component | stage |
|---|---|
| CUDA environment | (d), on an instruction count with one estimated step |
| PyTorch environment | (c) |
| JAX environment | (c) |
| Trainer, PyTorch | (c), moving to (d) |
| Trainer, JAX | (c) |
| End-to-end loop | (c), with (e) already executed twice |

**CUDA environment — (d), on an instruction count with one estimated step.** The binding resource is
named and the alternatives are ruled out by measurement: the kernel runs at 44% of the bandwidth
this card actually delivers, and packing the state so it loads in one wide transaction, plus
deleting a redundant output, changed the time by 0.3% — so bytes are not the constraint. The
small-batch floor of 5.6 microseconds was shown to be one dispatch, not physics. The ceiling is
stated as a number and the fraction reached is stated too: 2.87e10 environment steps per second
from the instruction count, and the kernel runs at 67% of it. What keeps this short of an
unqualified (d) is that one input to the ceiling — how far correctly rounded divides and square
roots expand in the final machine code — is estimated rather than disassembled.

**PyTorch environment — (c).** Profiled and restructured so the compiler can fuse it: the contact
selection was rewritten so that no operation depends on data, and the maze geometry became one byte
per square, together worth 9x at one million environments. It sits 4.6x behind the hand-written
kernel there, and that gap was measured rather than assumed.

**JAX environment — (c).** Same physics through a second compiler; unrolling the step loop was
swept and measured (1.77x at small batches, 1.11x at one million). Two further refactors were
declined with a reason drawn from measurement rather than taste: the environment is a small share
of a training step, so the end-to-end value of either is bounded.

**Trainer, PyTorch — (c), moving to (d).** 110.2 GFLOP per iteration in 20.4 ms is 5.40 TFLOP/s,
and the iteration is at 32% of the 6.55 ms matmul-only floor measured for these exact shapes. The
profile is flat in the sense of stage (c): the phases are rollout 25%, wide passes 6% and update
69%, but the update's 69% is the algorithm doing four passes over its data, not a removable item,
and the thing that used to dominate — the environment, once 84% of raw kernel time — is now 1.5%.
Round-2 changes were worth 5% to 45% and needed alternating paired runs in separate processes to be
believed. It is not (d) because the 67% of the iteration that is not matrix multiplication has not
been attributed kernel by kernel.

**Trainer, JAX — (c).** The same algorithm, independently written, at 13.4 ms per iteration at 128
copies in the one-update-per-batch configuration. The equivalent of the PyTorch build's
hand-written recording turned out to be on by default in this framework and worth 15%, measured by
switching it off. Two candidate compiler flags were measured and rejected as inside the noise.

**End-to-end loop — (c), with (e) already executed twice.** The whole iteration is recorded as one
replay, verified to compute bit-identical results, and nothing returns to the host during an
iteration. The two shape changes that reset the ceiling — batching independent copies, and fusing a
learning-rate sweep into one run — are done and measured. A third, the shorter rollout horizon
worth 3.0x, is measured and deliberately declined because it changes the algorithm.

### What would move each component a stage

- **CUDA environment, to remove the last qualification on (d).** Disassemble the kernel on an
  sm_90-capable toolchain, or take one Nsight Compute "speed of light" reading, so the 1,050
  machine instructions become a count rather than a count plus an estimate. Then measure the
  deferred winner-only contact law, which is an attempt to lower that count and therefore to raise
  the ceiling itself — a stage (e) move on a component already at (d).
- **Trainer to (d).** One Nsight Compute pass over the replayed update graph, sorting kernel time
  into matrix multiplication, elementwise, optimiser, and gather, and checking each elementwise
  kernel against the 3.57 TB/s copy ceiling. That converts the 10 ms of unattributed update time
  into a list with a ceiling on each line, which is exactly the artefact stage (d) requires.
- **Anything to (e).** The available moves are already written down and priced: the shorter horizon
  (3.0x, changes the algorithm), and running the update in bfloat16 rather than TF32 (the card's
  matrix units are twice as fast in that format, and the shapes are small enough that the gain
  would be far below 2x — worth measuring before believing).

### The one-paragraph answer

This project is at stage (c) throughout, except the hand-written environment kernel, which is at
(d) — running at 67% of a ceiling computed from a counted instruction total — and it has already
executed the stage (e) moves that matter most for research throughput. The training loop uses about
2% of the card's arithmetic, which sounds alarming and is not: the best conceivable figure for this
algorithm on this card is about 6%, because the matrices are 4 to 256 numbers wide and no
implementation can make such matrices fill a chip built for matrices thousands wide. Against the
ceiling that can actually be reached, the implementation is at 32% (and 45% in the
one-update-per-batch configuration), having started at 0.8%. What remains is not a factor of fifty
hiding in the hardware; it is at most a factor of three, most of it in the update phase, and
claiming it requires either attributing that phase kernel by kernel or changing what the algorithm
computes.

## Sources

Performance-model and metric definitions:

- Williams, Waterman, Patterson, "Roofline: an insightful visual performance model for multicore
  architectures", *Communications of the ACM* 52(4), 2009 —
  https://dl.acm.org/doi/10.1145/1498765.1498785
- Amdahl, "Validity of the single processor approach to achieving large scale computing
  capabilities", AFIPS 1967 —
  https://www3.cs.stonybrook.edu/~rezaul/Spring-2012/CSE613/reading/Amdahl-1967.pdf
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", 2022 (introduces Model FLOPs
  Utilisation; appendix B gives the counting rule) — https://arxiv.org/abs/2204.02311
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", MLSys 2023
  (Model versus Hardware FLOPs Utilisation) — https://arxiv.org/abs/2205.05198
- NVIDIA Nsight Compute Profiling Guide, "GPU Speed Of Light Throughput" section —
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Bekman, `ml-engineering`, training performance chapter (MFU/HFU) and the maximum-achievable
  matmul benchmark —
  https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md and
  https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks

Hardware specifications:

- NVIDIA H100 NVL GPU Product Brief, PB-11773-001_v01, March 2024 (clocks, memory, bandwidth,
  power) —
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf
- NVIDIA H100 NVL datasheet (FP32 60 TFLOPS, TF32 Tensor Core 835 TFLOPS, BF16 1,671 TFLOPS) —
  https://www.pny.com/file%20library/company/support/product%20brochures/nvidia%20data%20center%20gpus/english/h100-nvl-datasheet.pdf
- NVIDIA H100 Tensor Core GPU datasheet (the SXM and PCIe columns, and the sparsity footnote) —
  https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf
- The device itself: `torch.cuda.get_device_properties` and `nvidia-smi` on serval05, recorded in
  `data/ceiling_probe.json`

How open codebases report performance:

- karpathy/nanoGPT, `model.py` `estimate_mfu` and the training loop's per-step MFU print —
  https://github.com/karpathy/nanoGPT
- karpathy/nanochat, per-device peak table in `nanochat/common.py`, `train/mfu` logging, and the
  MFU/MBU inference benchmark in `scripts/infer_bench.py` — https://github.com/karpathy/nanochat
- karpathy/llm.c, the discussion reproducing GPT-2 (124M) in 90 minutes on one 8x A100 node —
  https://github.com/karpathy/llm.c/discussions/481
- KellerJordan/modded-nanogpt, the nanoGPT speedrun leaderboard and its rules —
  https://github.com/KellerJordan/modded-nanogpt
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
  (vLLM), SOSP 2023 — https://arxiv.org/abs/2309.06180
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", 2023 —
  https://arxiv.org/abs/2312.07104

Parallel reinforcement-learning systems in the comparison table:

- Makoviychuk et al., "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot
  Learning", 2021 — https://arxiv.org/abs/2108.10470
- Mittal et al., "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot
  Learning", 2025 — https://arxiv.org/abs/2511.04831
- Weng et al., "EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine",
  NeurIPS 2022 Datasets and Benchmarks — https://arxiv.org/abs/2206.10558
- Freeman et al., "Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation",
  NeurIPS 2021 Datasets and Benchmarks — https://arxiv.org/abs/2106.13281
- Shacklett et al., "An Extensible, Data-Oriented Architecture for High-Performance, Many-World
  Simulation", SIGGRAPH 2023, and the project page carrying the per-environment throughput figures
  — https://madrona-engine.github.io/
- Suarez, "PufferLib 2.0: Reinforcement Learning at 1M steps/s", *Reinforcement Learning Journal*
  2025 — https://rlj.cs.umass.edu/2025/papers/Paper151.html
- Lu et al., PureJaxRL, and the accompanying write-up "Achieving 4000x Speedups and Meta-Evolving
  Discoveries" — https://github.com/luchris429/purejaxrl and https://chrislu.page/blog/meta-disco/
- Lan et al., "WarpDrive: Extremely Fast End-to-End Deep Multi-Agent Reinforcement Learning on a
  GPU", JMLR 2022 — https://arxiv.org/abs/2108.13976

Repository measurements used above:

- `report/2026-08-15-gpu-parallel-rl-environment-training-endtoend/report.md` — round-2 phase
  split, copy scaling, pairing measurements, training campaign
- `report/2026-08-15-pointmaze-gpu-parallelization/report.md` — environment throughput ladders,
  optimisation ledgers, sweep scaling
- `pointmaze/{torch_env,cuda_env,jax_env}/progress_and_changes.md`, `ppo/{torch_ppo,jax_ppo}/progress_and_changes.md`
- `pointmaze/common/physics_spec.md`, `ppo/research/ppo_rnd_algorithm_spec.md`
- `extra_step_review.md`, items 12 and 16 (the written go/no-go criterion and the stopping rule)
- `benchmarks/results/*.json`
