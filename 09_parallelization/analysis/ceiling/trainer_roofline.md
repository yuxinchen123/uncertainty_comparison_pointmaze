# How far the trainer is from the maximum of the H100, and what that distance means

Question answered here: one training iteration of the batched PPO+RND trainer at 128 copies takes
20.4 milliseconds in PyTorch and 13.4 in JAX. How much of the graphics processor's capability is
that, and by the standards a production training team would apply, is this implementation finished,
half finished, or barely started?

Everything below is arithmetic on two things: the operation counts of the code in
`ppo/torch_ppo/torch_ppo_rnd.py` and `ppo/jax_ppo/jax_ppo_rnd.py`, and the times already measured
in `benchmarks/results/`. The counting script is `code/count_arithmetic.py`; running it reproduces
every number in this note.

## The machine, and the specification numbers used

The device is one NVIDIA H100 NVL in serval05. Its identity was read from the card itself
(`nvidia-smi` on serval05, 2026-08-15): product name `NVIDIA H100 NVL`, 95,830 MiB of memory,
maximum multiprocessor clock 1,785 MHz, maximum memory clock 2,619 MHz, compute capability 9.0,
driver 580.159.04.

| quantity | value | source |
|---|---|---|
| memory size | 94 GB HBM3 | NVIDIA H100 NVL datasheet |
| memory bandwidth | 3.9 TB/s | NVIDIA H100 NVL datasheet |
| FP32 arithmetic rate | 60 TFLOP/s | NVIDIA H100 NVL datasheet |
| TF32 matrix rate, quoted | 835 TFLOP/s | datasheet, marked "with sparsity" |
| TF32 matrix rate, dense | 417.5 TFLOP/s | half the quoted figure |
| streaming multiprocessors | 132 | NVIDIA Hopper architecture description |
| FP32 lanes per multiprocessor | 128 (16,896 per GPU) | NVIDIA Hopper architecture description |
| L2 cache | 50 MB | NVIDIA Hopper architecture description |
| clock, maximum | 1,785 MHz | read from the card |

Terms used above, each defined once:

- **FLOP** — one floating-point operation, one multiply or one add. **TFLOP/s** is a million
  million of them per second.
- **TF32** — a number format NVIDIA's matrix units accept in place of ordinary 32-bit floating
  point. It keeps the same exponent range but shortens the fraction from 23 bits to 10, which is
  what lets the matrix units run about seven times faster than the ordinary arithmetic lanes. The
  trainer switches this on (`tf32=True` in every production configuration), so 417.5 TFLOP/s is the
  correct ceiling to measure against.
- **"With sparsity"** — a doubled figure that only applies when half the weights in every group of
  four are zero and the hardware is told to skip them. No part of this trainer is sparse, so the
  honest peak is half the marketed number. Quoting 835 here would understate our utilisation by a
  factor of two in the flattering direction, which is the usual way these figures are abused.
- **Streaming multiprocessor** — one of the 132 independent processing blocks the chip is divided
  into. Work that does not produce at least 132 independent pieces cannot occupy the whole chip.

Two internal consistency checks that the datasheet numbers are the right ones for this card:

- 16,896 lanes at 2 FLOP each at 1.785 GHz is $16{,}896 \times 2 \times 1.785\times10^{9} =
  60.3\times10^{12}$, which reproduces the datasheet's 60 TFLOP/s from the clock read off the card.
- The 6,016-bit HBM3 bus at twice the 2,619 MHz memory clock is
  $\dfrac{6016}{8} \times 2 \times 2.619\times10^{9} = 3.94\times10^{12}$ bytes per second, which
  reproduces the datasheet's 3.9 TB/s.

## 1. The arithmetic in one training iteration

### The counting rule

A matrix multiplication of an $m$ by $k$ matrix by a $k$ by $n$ matrix produces $m \times n$
results, each the sum of $k$ products, so it costs $2mkn$ operations — one multiply and one add per
term. A backward pass through a layer costs about twice its forward pass, because it computes two
matrices instead of one: the gradient with respect to the layer's input and the gradient with
respect to its weights. Element-wise work (tanh, the rectifier, adding a bias, clipping) is
proportional to the number of values rather than to their products, and comes to well under one
percent of the totals below; it is excluded, which is the standard convention.

Every layer is run for all 128 copies at once as a batched matrix multiplication, so a per-copy
count multiplied by 128 is the whole machine's work.

### The network shapes

From `ppo/research/ppo_rnd_algorithm_spec.md` sections 3.1 to 3.3: observation width 4, action
width 2, policy and value trunks 64 wide, RND networks 256 wide with 128 outputs.

| network | layers | trained |
|---|---|---|
| actor | 4→64, 64→64, 64→2 | yes |
| critic | 4→64, 64→64, two 64→1 heads | yes |
| RND target | 4→256, 256→128 | no, frozen |
| RND predictor | 4→256, 256→128, 128→128 | yes |

### Stage one — the rollout, 128 sequential steps

After the round-two change that moved the critic, the log-probability and the RND bonus out of the
loop (`torch_ppo/progress_and_changes.md` row 11), the only network left inside the sequential loop
is the actor, run on the 4 environments of each copy:

| layer | shape | arithmetic |
|---|---|---|
| first | 4×4 by 4×64 | 2 × 4 × 4 × 64 = 2,048 |
| second | 4×64 by 64×64 | 2 × 4 × 64 × 64 = 32,768 |
| mean head | 4×64 by 64×2 | 2 × 4 × 64 × 2 = 1,024 |
| one step, one copy | | **35,840** |

Over the horizon and the copies, 35,840 by 128 steps by 128 copies:
$35{,}840 \times 128 \times 128 = 587{,}202{,}560$, i.e. **0.587 GFLOP**.

### Stage two — the post-rollout pass over the stored rows

Each copy stores $128 \times 4 = 512$ rows. Three wide passes run here. Reading the code rather
than the description matters in two places:

- The critic runs on **1,024** rows, not 512: the values of the stored observations and the
  bootstrap values of the stored next observations are concatenated into one call
  (`_post_body_impl`, the line `self.critic_values(torch.cat([flat_obs, flat_nobs], dim=1))`).
- The frozen RND target runs **twice** on 512 rows: once for the intrinsic bonus, using the
  observation statistics as they stood at the start of the iteration, and once more after those
  statistics are updated, to cache the features the update stage will use. The two whitened inputs
  differ, so the second pass is not redundant.

| pass | shapes | arithmetic per copy |
|---|---|---|
| critic, 1024 rows | 4→64, 64→64, two 64→1 | 9,175,040 |
| RND bonus, 512 rows | target and predictor | 85,983,232 |
| cached target, 512 rows | 4→256, 256→128 | 34,603,008 |
| total per copy | | **129,761,280** |

The RND bonus line in full: the target's and the predictor's first layers are packed into one
$512\times4$ by $4\times512$ multiplication, $2 \times 512 \times 4 \times 512 = 2{,}097{,}152$;
the target's second layer $2 \times 512 \times 256 \times 128 = 33{,}554{,}432$; the predictor's
second layer the same again; the predictor's third layer
$2 \times 512 \times 128 \times 128 = 16{,}777{,}216$.

Over 128 copies: $129{,}761{,}280 \times 128 = 16{,}609{,}443{,}840$, i.e. **16.6 GFLOP**.

### Stage three — the update, sixteen minibatch steps

Style B is four passes over the data, each split into four portions: sixteen forward-and-backward
steps on 128 rows per copy. The frozen target is not recomputed here — its features were cached in
stage two — so the forward pass covers the actor, the critic and the predictor only.

| layer | shape | arithmetic |
|---|---|---|
| packed actor and critic first layer | 128×4 by 4×128 | 131,072 |
| actor second layer | 128×64 by 64×64 | 1,048,576 |
| critic second layer | 128×64 by 64×64 | 1,048,576 |
| actor mean head | 128×64 by 64×2 | 32,768 |
| packed critic heads | 128×64 by 64×2 | 32,768 |
| predictor first layer | 128×4 by 4×256 | 262,144 |
| predictor second layer | 128×256 by 256×128 | 8,388,608 |
| predictor third layer | 128×128 by 128×128 | 4,194,304 |
| forward, one copy, one step | | **15,138,816** |

Backward is twice that, so forward and backward together are
$3 \times 15{,}138{,}816 = 45{,}416{,}448$ per copy per minibatch step. Over sixteen steps and 128
copies: $45{,}416{,}448 \times 16 \times 128 = 93{,}012{,}885{,}504$, i.e. **93.0 GFLOP**.

### The iteration total

| stage | arithmetic | share of arithmetic | measured time | share of time |
|---|---|---|---|---|
| rollout | 0.587 GFLOP | 0.53% | 4.97 ms | 24.7% |
| post-rollout | 16.6 GFLOP | 15.07% | 1.30 ms | 6.5% |
| update | 93.0 GFLOP | 84.40% | 13.89 ms | 68.9% |
| **one iteration** | **110.2 GFLOP** | 100% | **20.17 ms** | 100% |

Times are from `benchmarks/results/2026-08-15-17-03-56_profile_phases_C128.json`; the three stages
recorded as one sequence measure 20.31 ms, and the end-to-end benchmark reports 20.42 ms.

The two "share" columns disagree, and that disagreement is the central fact of this note: the
rollout is half a percent of the arithmetic and a quarter of the time.

## 2. The utilisation figure, and what it means

### Definition

Define $F$ as the arithmetic in one iteration, $t$ as the time that iteration takes, and $P$ as the
machine's peak matrix rate. The utilisation is

$U = \dfrac{F}{t \times P}$

It is the fraction of the arithmetic the machine could have done in that time that this program
actually asked it to do. In production training teams the same quantity is called *model FLOPs
utilisation*: the arithmetic the model definition requires, divided by the arithmetic the hardware
could have retired. A related figure, *hardware FLOPs utilisation*, additionally counts arithmetic
that was executed but thrown away (recomputed activations, padding); for this trainer the two are
close, so only the first is reported.

### Our numbers

With $F = 110.2$ GFLOP and $P = 417.5$ TFLOP/s (dense TF32):

- PyTorch, $t = 20.42$ ms: $\dfrac{110.2\times10^{9}}{0.02042} = 5.40$ TFLOP/s, and
  $\dfrac{5.40}{417.5} = \mathbf{1.29\%}$.
- JAX, $t = 13.42$ ms: $\dfrac{110.2\times10^{9}}{0.01342} = 8.22$ TFLOP/s, and
  $\dfrac{8.22}{417.5} = \mathbf{1.97\%}$.

Against the ordinary 32-bit arithmetic rate of 60 TFLOP/s instead — the rate that applies to every
part of the work that does not run on the matrix units — the same runs are 9.0% and 13.7%.

Per stage, the same division applied to the measured phase times:

| stage | achieved rate | percent of dense TF32 |
|---|---|---|
| rollout | 0.118 TFLOP/s | 0.03% |
| post-rollout | 12.7 TFLOP/s | 3.05% |
| update | 6.70 TFLOP/s | 1.60% |

The post-rollout stage, which runs the widest matrices, is the most efficient part of the program
by a factor of a hundred over the rollout. That is the pattern the rest of this note explains.

### Why a large language model training run reports 40 to 60 percent

Three differences, all of them about size rather than about code quality:

1. **The matrices are enormous.** A transformer layer multiplies a matrix of perhaps 500,000 rows
   (batch times sequence length) by a weight matrix of 4,096 by 4,096. That single multiplication
   is $2 \times 5\times10^{5} \times 4096 \times 4096 = 1.7\times10^{13}$ operations, about
   40 milliseconds at peak. Our largest single multiplication in a whole iteration is the
   predictor's second layer across all copies: $8{,}388{,}608 \times 128 = 1.07$ GFLOP, which is
   **2.6 microseconds** at peak. The launch cost of a single operation on this card is one to three
   microseconds. Their largest multiplication runs about twenty thousand times longer than a
   launch; ours runs about as long as one.
2. **The blocks are full.** The matrix units work on fixed blocks of values. The TF32
   multiply-accumulate instruction shapes are 16 rows by 8 columns by 4 deep and 16 by 8 by 8. A
   transformer's 500,000 rows and 4,096 columns fill those blocks exactly. Our rollout
   multiplication has 4 rows, so three quarters of every 16-row block is empty and three quarters
   of the arithmetic the hardware performs is on padding. Before any other consideration, that
   alone throws away a factor of four in the rollout.
3. **There is almost nothing between the multiplications.** A transformer step is a few hundred
   large operations. A PPO iteration written the obvious way is several thousand tiny ones, and
   between them sit gathers, running statistics, gradient clipping and an optimiser step that touch
   every parameter individually.

So 40 to 60 percent for a large language model and 1 to 2 percent here are not two scores on one
scale. They are measurements of two workloads whose per-operation size differs by four orders of
magnitude, and the utilisation figure is mostly reporting that size difference.

## 3. Why the number is low, and why that is expected rather than a failure

### The matrices are tiny

The largest multiplication inside a rollout step, taken over all copies, is 128 separate
$4\times64$ by $64\times64$ products. That is $2 \times 4 \times 64 \times 64 \times 128 =
4.19$ MFLOP, which at 417.5 TFLOP/s takes **10 nanoseconds**.

The occupancy side of the same fact: 128 independent products give the scheduler at most 128 blocks
of work for 132 multiprocessors — one each, none spare, and each with 16,384 multiply-accumulates
to do. A multiprocessor's share of the peak rate is $417.5/132 = 3.2$ TFLOP/s, so its 32,768
operations take about 10 nanoseconds, or roughly 18 clock cycles at 1,785 MHz. A single operation
cannot be launched, run and retired in 18 cycles; the fixed cost of *having* an operation is one to
three microseconds, a hundred to three hundred times the work inside it.

### The work is dominated by the number of operations, not their size

The project's own reference figures (`pointmaze/common/research/vectorized_env_design.md`) are a
launch cost of 3 to 7 microseconds for an ordinary operation and 1 to 2 microseconds for the same
operation replayed from a recorded sequence. The trainer records the whole iteration as one
sequence, so it pays the lower figure.

That model can be checked against a measurement taken before the round-two change. A static count
of the rollout step at that time gave 47 operations per step
(`ppo/research/round2_module23_ideas.md`), and the rollout was measured at 20.11 ms over 128 steps:

$\dfrac{20{,}112}{128} = 157.1$ microseconds per step, and
$\dfrac{157.1}{47} = 3.34$ microseconds per operation.

The same count applied to the current step body (`_one_step_pure`) gives 16 to 21 operations —
copy the observation (1), three matrix multiplications with their biases and activations (5), the
standard deviation and the action sample (2), the fused environment step (8, recorded in
`pointmaze/torch_env/progress_and_changes.md`), and five buffer writes now folded into the
operations that produce the values. The rollout is now measured at 4.97 ms:

$\dfrac{4{,}974}{128} = 38.9$ microseconds per step, and $\dfrac{38.9}{18} = 2.16$ microseconds
per operation.

Two independent counts, two independent measurements, the same per-operation cost of about 2 to
3.3 microseconds. The rollout is therefore *entirely* accounted for by the number of separate
operations in it. Its arithmetic would take 1.4 microseconds at peak; it takes 4,974.

### There are 128 steps that cannot be merged

Step $t+1$ needs the observation the environment returns from step $t$, which needs the action the
policy produced from step $t$'s observation. Nothing widens that chain: adding copies makes each
step wider, not shorter. The horizon is a hyperparameter of the algorithm (`num_steps = 128`), so
the only ways to shorten the chain are to change the algorithm or to remove the per-step cost
rather than the steps.

### What the launch-bound regime implies on its own

Take $k$ operations per rollout step and $\ell$ microseconds of fixed cost each. The rollout alone
costs $128 \times k \times \ell$ microseconds ("us" below is microseconds):

| operations per step | at 2 us | at 3.3 us | at 5 us |
|---|---|---|---|
| 35 | 8.96 ms | 14.78 ms | 22.40 ms |
| 47 | 12.03 ms | 19.85 ms | 30.08 ms |

So a build with 35 to 47 operations per rollout step is committed to somewhere between 9 and 30
milliseconds of rollout before a single useful multiplication happens. The pre-round-two build sat
at 47 operations and 3.34 microseconds, which is 19.85 ms — and its measured rollout was 20.11 ms.
That is the whole of today's 20.4 ms iteration spent on one stage, and it is why the round-two work
targeted operation count and nothing else.

The current build is at 18 operations and 2.16 microseconds:
$128 \times 18 \times 2.16 = 4{,}977$ microseconds = 4.98 ms, against 4.97 measured.

## 4. The ceiling that actually applies

Three ceilings can be computed for this workload. The binding one is the third.

**Ceiling A — arithmetic.** 110.2 GFLOP at 417.5 TFLOP/s is **0.264 ms**. At the ordinary 32-bit
rate of 60 TFLOP/s it is 1.84 ms. The measured 20.42 ms is 77 times the first. This ceiling is not
reachable and not informative, which is exactly why the 1.29 percent figure should not be read as a
grade.

**Ceiling B — memory traffic.** This is the ceiling that a roofline analysis actually gives for a
workload whose operations are small. Counting the bytes moved by the update stage per minibatch
step, with 7,668,480 trainable parameters across the 128 copies (30.7 MB):

| traffic | bytes |
|---|---|
| minibatch gathers, read and write | 18.7 MB |
| activations, forward and backward | 201.3 MB |
| Adam: read parameters, gradients, two moments; write three | 214.7 MB |
| gradient clip: read, scale, zero | 92.0 MB |
| one minibatch step | 526.8 MB |

Sixteen steps is 8.43 GB; adding the rollout (0.31 GB, dominated by re-reading the actor's weights
128 times) and the post-rollout stage (0.44 GB) gives **9.17 GB per iteration**. At 3.9 TB/s that
is **2.33 ms**.

The same figures give the workload's arithmetic intensity — operations performed per byte moved —
for the update stage: $\dfrac{93.0\times10^{9}}{8.43\times10^{9}} = 11.0$ FLOP per byte. This card
balances at $\dfrac{417.5\times10^{12}}{3.939\times10^{12}} = 106$ FLOP per byte. Ten times below
the balance point means the roofline ceiling for this workload is not 417.5 TFLOP/s but
$3.939 \times 11.0 = 43.5$ TFLOP/s, which is 10.4 percent of the peak. **No implementation of this
algorithm at this size can exceed about ten percent of the marketed matrix rate**, however it is
written. That single sentence reframes the 1.29 percent: it is not 1.29 percent of what is
achievable, it is 1.29 out of a possible 10.4, i.e. about an eighth of the reachable ceiling.

**Ceiling C — operation count and the sequential chain.** Define $K$ as the number of separate
operations in the iteration's dependent chain and $\ell$ as the fixed cost of one, measured above
at 2.16 to 3.34 microseconds. The floor is $t \ge K \times \ell$. Counting from the code:

| stage | operations |
|---|---|
| rollout, 128 steps × 18 | 2,304 |
| post-rollout (compiled scans and wide passes) | about 350 |
| update, 16 steps × about 215 | about 3,440 |
| **iteration** | **about 6,100** |

At 2.16 microseconds that is **13.2 ms**; at 3.34 it is 20.4 ms. The update's 215 operations per
step are dominated by the loops over parameter tensors: with 21 trainable tensors, the squared-norm
reduction, the gradient scaling and the gradient zeroing are $21 \times 3 = 63$ operations before
any arithmetic happens.

**Where the two implementations sit against the ceilings:**

| ceiling | value | PyTorch 20.42 ms | JAX 13.42 ms |
|---|---|---|---|
| A, arithmetic at dense TF32 | 0.26 ms | 77x above | 51x above |
| B, memory traffic at 3.9 TB/s | 2.33 ms | 8.8x above | 5.8x above |
| C, operation count × fixed cost | 13.2 to 20.4 ms | at it | at it |

The reading: the implementation is **at** the ceiling that binds it. The PyTorch build's iteration
time is, to within the uncertainty in the per-operation cost, exactly the number of operations it
issues multiplied by the cost of issuing one. The JAX build is faster for exactly one reason — its
compiler fuses more aggressively, so it issues fewer operations — not because it computes anything
differently (the two agree numerically to 8.6e-7).

**How a production team would label this stage.** The standard vocabulary sorts a workload into
launch-bound, latency-bound, memory-bandwidth-bound, or compute-bound, and the label determines
which optimisations can possibly help. This trainer is **launch-bound and latency-bound**, one
regime short of memory-bandwidth-bound and two short of compute-bound. Against that label the
implementation is not early-stage work: the operations have already been fused (compilation), the
launches already removed (the whole iteration is one recorded sequence), the sequential chain
already emptied of everything that did not have to be in it (the round-two hoist), and the
arithmetic already moved onto the matrix units (TF32). What remains is the cost of a 128-step
dependency chain and of an optimiser that touches each parameter tensor separately, and the honest statement of position is
"one structural change short of the practical floor", not "1.29 percent of peak".

## 5. Remaining headroom, and what it would cost

Each row assumes a fixed cost of 2.2 microseconds per operation, the figure derived in section 3.

| change | mechanism | operations removed | estimated gain |
|---|---|---|---|
| One flat buffer for gradient clip and Adam | 21 tensors × 3 loops → about 3 operations, ×16 steps | about 960 | 2.1 ms, 10% |
| Gather all sixteen minibatches once | 9 gathers × 16 → 9 | 135 | 0.30 ms, 1.5% |
| Persistent rollout kernel | 128 steps stay resident on the chip; no launches | 2,304 | up to 4.7 ms, 23% |
| More copies per run | same operations, more work inside each | none | 2.1x to 2.4x utilisation |
| Shorter horizon (T=32, N=16) | quarter the sequential steps, four times wider | 1,728 | measured 3.0x in JAX |
| bf16 instead of TF32 | halves the arithmetic, which is 1.3% of the time | none | at most 0.6% |

Details and caveats, in the same order:

1. **Flat parameter buffer.** `_clip_per_copy_and_step` loops over the 21 trainable tensors three
   times per minibatch step: once to accumulate each tensor's squared gradient norm, once to scale
   the gradients, once to zero them. That is $21 \times 3 \times 16 = 1{,}008$ operations per
   iteration, $1{,}008 \times 2.2 = 2.22$ ms. Holding every parameter in one contiguous buffer with
   views for the individual tensors reduces it to about three operations per step, saving about
   2.1 ms of 20.4, i.e. 10 percent. This was ranked and estimated independently in
   `round2_module23_ideas.md` at 1.5 to 3 ms; it is the largest inexpensive improvement left. Risk: a view that
   detaches from the parameter and silently stops receiving updates, which the existing
   zero-learning-rate capture test would catch.
2. **Gather once.** The nine batch fields are gathered separately for each of the sixteen
   minibatches, 144 operations where 9 would do, saving $135 \times 2.2 = 0.30$ ms. Bitwise
   identical result; a day of work.
3. **Persistent rollout kernel.** One program that stays resident on the chip and walks all 128
   steps internally, with the environment state and the actor's weights held in registers and
   shared memory. It removes the 2,304 launches, which are the whole of the rollout's 4.97 ms; what
   is left is the arithmetic (1.4 microseconds) plus the dependent latency of about 128 steps of
   register-resident work, plausibly 0.1 to 0.5 ms. Gain up to about 4.7 ms, 23 percent of the
   iteration. Cost: this is a rewrite of the policy and the environment into one hand-written
   kernel, weeks rather than days, and it forfeits the ability to change the network shape without
   rewriting the kernel. It is the only remaining change that alters the regime rather than
   trimming inside it.
4. **More copies.** The arithmetic per copy is fixed, so utilisation is a direct measure of how
   well a given copy count fills the machine:

   | copies | ms per iteration | achieved rate | percent of dense TF32 |
   |---|---|---|---|
   | 8 | 13.82 | 0.50 TFLOP/s | 0.12% |
   | 128 | 20.42 | 5.40 TFLOP/s | 1.29% |
   | 512 | 43.76 | 10.07 TFLOP/s | 2.41% |
   | 1,024 | 76.89 | 11.47 TFLOP/s | 2.75% |
   | 16,384 | 1,079 | 13.07 TFLOP/s | 3.13% |

   Nothing is optimised here; the same operations simply carry more work each. It costs no
   engineering at all and it is worth more than every code change in this table combined — but it
   gives throughput across experiments, not a faster single experiment, so it is the right lever
   only when the science needs many runs. The curve flattens near 3.1 percent because the
   per-operation cost stops dominating and the shapes' padding waste and the memory traffic take
   over; that 3.1 percent, against ceiling B's 10.4, is where this algorithm's shape finally binds.
5. **Shorter horizon.** Four times fewer sequential steps with four times more environments per
   step is the same 512 rows per copy, and it moves the rollout from 2,304 operations to 576. It
   was measured at 3.0x in the JAX build (style A). It is excluded from the default because it
   shortens the advantage-estimation horizon from 128 steps to 32 — a different algorithm, not a
   faster implementation of the same one, and it is labelled as such in the report.
6. **bf16.** The entire arithmetic of an iteration is 264 microseconds of the 20,423 measured, so
   halving it saves at most 0.6 percent, and the cast operations it adds are launches in the very
   regime that is binding. Correctly rejected in round two.

Taken together, items 1 and 2 are about 2.4 ms of the 20.4, reachable in a few days, with the existing
bitwise capture tests as the check. Item 3 is another 4.7, at weeks of work and a large loss of flexibility.
The floor for this algorithm at this size, on this card, with everything above done, is ceiling B:
about 2.3 ms, at which point the program would be moving bytes as fast as the memory can supply
them and doing 43 TFLOP/s of arithmetic — ten percent of the marketed peak, and one hundred percent
of what the workload's shape permits.

## Sources

- [NVIDIA H100 NVL datasheet (PDF, NVIDIA document hosted by PNY)](https://www.pny.com/file%20library/company/support/product%20brochures/nvidia%20data%20center%20gpus/english/h100-nvl-datasheet.pdf) — FP64 30, FP64 Tensor Core 60, FP32 60, TF32 Tensor Core 835, BFLOAT16 and FP16 Tensor Core 1,671, FP8 3,341 teraFLOPS (the Tensor Core rows marked with an asterisk for "with sparsity"); 94 GB memory; 3.9 TB/s bandwidth.
- [NVIDIA H100 product page, technical specifications table](https://www.nvidia.com/en-us/data-center/h100/) — the same figures for the H100 NVL column.
- [NVIDIA Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — "132 SMs per GPU", "128 FP32 CUDA Cores per SM, 16896 FP32 CUDA Cores per GPU", "4 fourth-generation Tensor Cores per SM, 528 per GPU", "50 MB L2 cache".
- [PTX instruction set, warp-level matrix multiply-accumulate](https://docs.nvidia.com/cuda/parallel-thread-execution/) and [a worked description of the TF32 shapes](https://accelerated-computing.academy/fall25/labs/lab6/) — the TF32 matrix instruction shapes are 16 by 8 by 4 and 16 by 8 by 8, which is where the 16-row tile in section 2 comes from.
- The card itself: `nvidia-smi` on serval05, 2026-08-15 — `NVIDIA H100 NVL`, 95,830 MiB, maximum multiprocessor clock 1,785 MHz, memory clock 2,619 MHz, compute capability 9.0, driver 580.159.04.
- Measurements, all in `09_parallelization/benchmarks/results/`:
  `2026-08-15-17-09-21_trainbench_torch_epoch_minibatch_final_styleB.json` (20.423 ms at 128
  copies), `2026-08-15-17-17-40_trainbench_jax_ppo_lfl_sync.json` (13.415 ms),
  `2026-08-15-17-03-56_profile_phases_C128.json` (the three stage times),
  `2026-08-15-02-51-10_profile_breakdown_C128.json` (the pre-round-two 20.11 ms rollout).
- Operation counts and per-operation costs: `ppo/research/round2_module23_ideas.md` (the 47-operation
  rollout step), `pointmaze/torch_env/progress_and_changes.md` (8 operations per environment step),
  `pointmaze/common/research/vectorized_env_design.md` (launch cost 3 to 7 microseconds normally,
  1 to 2 from a recorded sequence).
- Network shapes: `ppo/research/ppo_rnd_algorithm_spec.md` sections 3.1 to 3.3, 11 and 12.
- The arithmetic in this note: `code/count_arithmetic.py`, which prints every figure quoted above.
