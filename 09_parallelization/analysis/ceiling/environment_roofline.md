# How close the PointMaze environment step is to the limit of one H100 NVL

One environment step moves 82 bytes and performs about 250 floating-point operations, which on
this machine sets a highest possible rate of **48,000 million environment steps per second**. The
hand-written CUDA kernel reaches 40% of that, JAX 21%, compiled PyTorch 8.7%. The CUDA kernel's
60% shortfall is not waste: the step needs 12.8 machine instructions for every byte it moves,
while the machine can only afford 7.7 instructions per byte at full memory speed, so its real
ceiling is 28,700 million steps per second and it is running at two-thirds of that. The two
framework implementations fall short for a different and simpler reason — they move about 400 and
1,013 bytes per step instead of 82.

The sections below derive each of those numbers from the source code and the hardware
specification, with the arithmetic shown.

## 0. Terms, defined once

- **Byte moved**: a byte read from or written to the graphics processor's own memory (called
  HBM3 here, or "device memory"). Moving bytes is the slowest thing a small computation does.
- **Memory bandwidth**: how many bytes per second the memory can deliver. A hard limit.
- **FLOP**: one floating-point arithmetic operation — one add, one multiply, one divide, one
  square root. "FLOP/s" is operations per second.
- **Instruction**: one machine command. A FLOP is an instruction, but so is a comparison, a
  select between two values, an address calculation, and an integer add. Most of this kernel's
  instructions are *not* FLOPs, and that turns out to be the whole story.
- **Arithmetic intensity** $I$: operations divided by bytes moved, for the same piece of work.
  Define $I = F / B$ where $F$ is operations per environment step and $B$ is bytes per
  environment step. Units: operations per byte.
- **Machine balance**: the same ratio for the hardware — peak operations per second divided by
  peak bytes per second. If a computation's intensity is below the machine's balance, memory is
  the limit; if above, arithmetic is the limit. This comparison is called a **roofline**.
- **SM (streaming multiprocessor)**: one of the 132 independent processors on this chip. Each has
  4 instruction schedulers, and each scheduler starts at most one instruction per clock cycle.
- **Warp**: a group of 32 threads that execute the same instruction at the same time. This kernel
  runs one thread per environment, so one warp advances 32 environments in lockstep.
- **Occupancy**: how many warps are resident on an SM at once, out of the hardware maximum of 64.
  Resident warps are what the scheduler switches between while one of them is waiting.
- **L2 cache**: 60 MiB of fast memory in front of HBM3. Data that fits in it is not read from
  HBM3 at all, which matters below for the JAX measurement.
- **TF32**: a reduced-precision floating-point format used by the chip's *matrix* units (the
  hardware that multiplies small matrices for neural-network layers). This environment kernel
  uses no matrix units at all; TF32 appears here only to compute the machine balance point that
  the question asks for.

## 1. The machine

Read from the device itself on serval05 (`nvidia-smi` and `cudaGetDeviceProperties` through
PyTorch, 2026-08-15):

| property | value |
|---|---|
| name | NVIDIA H100 NVL, compute capability 9.0 |
| memory | 95,830 MiB HBM3 |
| SM count | 132 |
| L2 cache | 62,914,560 bytes (60 MiB) |
| max SM clock | 1,785 MHz |
| memory clock | 2,619 MHz |
| max threads per SM | 2,048 (= 64 warps) |
| registers per SM | 65,536 |

Read from NVIDIA's *H100 NVL GPU Product Brief*, PB-11773-001_v01, March 2024
([nvidia.com](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf)),
Tables 1 and 2:

| property | value |
|---|---|
| memory type / size | HBM3, 94 GB |
| memory clock | 2,619 MHz |
| memory bus width | 6,016 bits |
| **peak memory bandwidth** | **3,938 GB/s** |
| GPU clocks | base 1,080 MHz, boost 1,785 MHz |
| board power | 400 W maximum (default) |

Read from the *NVIDIA H100 Tensor Core GPU* datasheet, H100 NVL column
([PNY-hosted copy](https://www.pny.com/file%20library/company/support/product%20brochures/nvidia%20data%20center%20gpus/english/h100-nvl-datasheet.pdf)).
The datasheet marks the matrix-unit rows "with sparsity", which is a rate reachable only when
half the matrix entries are zero, so the ordinary (dense) rate is half the printed one:

| property | printed | dense |
|---|---|---|
| FP64 | 30 teraFLOPS | 30 |
| FP32 (non-matrix) | 60 teraFLOPS | 60 |
| TF32 matrix units, with sparsity | 835 teraFLOPS | 417.5 |
| FP16 matrix units, with sparsity | 1,671 teraFLOPS | 835.5 |
| memory bandwidth | 3.9 TB/s | — |

Two consistency checks, so these numbers can be trusted together:

1. Bandwidth from the memory clock and bus width: 6,016 bits = 752 bytes per transfer; HBM3
   transfers twice per clock; 2,619 × 10^6 × 2 × 752 = 3.939 × 10^12 bytes per second. The
   product brief prints 3,938 GB/s. Agreement to 0.03%.
2. FP32 rate from the chip layout: 132 SMs × 128 FP32 lanes × 2 operations per fused
   multiply-add × 1.785 × 10^9 clocks = 60.3 × 10^12 FLOP/s. The datasheet prints 60 teraFLOPS.
   Agreement to 0.5%. This also confirms that the datasheet's FP32 figure is quoted at the boost
   clock the device reports.

One derived quantity that is used throughout and is not printed on any datasheet — the rate at
which the machine can start instructions of any kind:

- 132 SMs × 4 schedulers × 1 instruction per clock × 1.785 × 10^9 = 9.43 × 10^11 warp-instructions
  per second.
- Each warp-instruction acts on 32 threads, so 9.43 × 10^11 × 32 = **3.02 × 10^13 thread-level
  instructions per second**.

## 2. What one environment step must move

Source: `pointmaze/cuda_env/pointmaze_kernel.cu` (`step_kernel`) and its Python wrapper
`cuda_pointmaze.py`, which allocates the buffers. State is float32 and packed as one
`[B, 4]` array holding position and velocity. One thread handles one environment.

| buffer | direction | bytes |
|---|---|---|
| state (x, y, vx, vy) | read | 16 |
| action (2 values) | read | 8 |
| goal position (2 values) | read | 8 |
| step counter (int32) | read | 4 |
| reset counter (int32) | read | 4 |
| state (x, y, vx, vy) | write | 16 |
| pre-reset observation | write | 16 |
| reward | write | 4 |
| terminated flag | write | 1 |
| truncated flag | write | 1 |
| step counter | write | 4 |
| **total, every step** | | **82** |
| goal + reset counter, on reset only | write | 12 × 1/400 = 0.03 |

Three points about the count:

- The **wall bitmask** is read every step (`nb_mask[cell]`, 4 bytes) but is *not* counted as
  traffic: the whole table is 9 × 12 × 4 = 432 bytes and lives permanently in cache. It still
  costs an instruction and a cache lookup, which matters in section 6.
- The **reset branch** writes the goal and the reset counter, but an episode is 400 steps long,
  so those 12 bytes are paid once per 400 steps: 0.03 bytes per step. Negligible.
- Every write is a full 16-byte or 4-byte value written by consecutive threads, so whole cache
  lines are written and the memory system never has to read a line before writing it.

**82 bytes per environment step**, of which 16 (the pre-reset observation) are written every step
but genuinely needed only on the 1-in-400 steps that reset — the trainer needs the observation
from before the reset for its value bootstrapping. Trimming that would give 66 bytes. Section 6
shows why trimming it does nothing.

For comparison, the PyTorch implementation (`torch_env/torch_pointmaze.py`, `step_core`) keeps
position, velocity and goal in separate arrays, keeps the reset counter in int64, and returns
*both* an observation and a pre-reset observation: 44 bytes read and 74 written = **118 bytes of
state per environment step**, before counting anything its compiler writes between kernels.

## 3. What one environment step must compute

Source: `pointmaze/cuda_env/pointmaze_dynamics.h`, function `dynamics`, in the default
`PM_RANK_TOURNAMENT` form, plus the reward and episode logic in `step_kernel`. The physics is
the exact MuJoCo update documented in `pointmaze/common/physics_spec.md`: clip the action and
velocity, compute the unconstrained acceleration, test the eight neighbouring cells for wall
contact, keep the two nearest, evaluate MuJoCo's solref/solimp force law for those two, solve the
two-contact quadratic program by enumerating its four cases, then take one implicit-damping Euler
step.

Counted by hand from the source:

| part of the step | floating-point operations |
|---|---|
| action and velocity clips, unconstrained acceleration | 8 |
| cell index of the ball | 2 |
| eight contact candidates, 14 operations each | 112 |
| force law for the two winners, 40 operations each | 80 |
| two-contact solve (four enumerated cases) | 20 |
| implicit-damping Euler integrator | 20 |
| reward, goal test, episode flags | 6 |
| **total** | **248** |

The two multi-line entries, so they can be checked:

- **One contact candidate (14)**: rebuild the neighbour rectangle from the cell index (2 for its
  left edge, 3 for its bottom edge), clamp the ball centre into the rectangle (1 add each for the
  right and top edge), the two differences to the nearest point (2), the squared distance (2
  multiplies and 1 add), the square root (1), subtract the ball radius (1). Eight candidates:
  8 × 14 = 112.
- **The force law for one winner (40)**: recompute the rectangle and nearest point (11 as above,
  without the radius subtraction), the two divides that normalise the contact direction (2), the
  penetration relative to the margin (1), its absolute value and scaling (2), MuJoCo's impedance
  curve, computed both ways and selected (6), the impedance itself (2), the reference acceleration
  (11 — three inner products and two subtractions), the regularizer (3). Two winners: 2 × 40 = 80.

Checked against the compiler. PTX is the compiler's intermediate assembly: one line per
machine-level operation, just before the final translation into this chip's own instruction set.
The kernel was compiled for this exact chip
(`nvcc -arch=sm_90 -O3 --fmad=false -DPM_RANK_TOURNAMENT`, CUDA 13.0 on serval05; source copy in
`code/kernel_probe.cu`, which is `step_kernel` verbatim with the PyTorch headers removed) and its
intermediate assembly counted:

| instruction class | count in PTX |
|---|---|
| float add / subtract / multiply | 239 |
| float divide (IEEE-rounded) | 15 |
| float square root (IEEE-rounded) | 10 |
| float absolute value | 2 |
| **float arithmetic subtotal** | **266** |
| float comparisons | 122 |
| selects between two values | 177 |
| integer arithmetic, shifts, logic, integer tests | ~200 |
| loads and stores | 77 |
| conversions and moves | 60 |
| **all instructions** | **921** |

The 266 float operations include about 16 inside the reset branch, which runs once per 400 steps,
so the steady-state figure is **about 250 floating-point operations per environment step** — the
hand count of 248 and the compiler agree to within 1%.

The important line in that table is not the FLOP count. It is that **only about a quarter of the
instructions are floating-point arithmetic**. The rest are the comparisons that rank the eight
candidate walls, the selects that implement every branch without branching, and address
arithmetic.

Two further facts from the compilation, used in section 6:

- **Registers: 39 per thread, zero spills** (reported by `ptxas -v`), and a 32-byte stack frame
  holding the array of eight distances. Two of the reads from that array use an index that is
  only known at run time (`dists[k1]`, `dists[k2]`), so those two reads go through the cache
  rather than staying in registers.
- Machine instructions are more numerous than PTX instructions, because an IEEE-correct float
  divide and an IEEE-correct square root each expand into a sequence of roughly 8 to 12 machine
  instructions. With 25 such operations, the estimate is
  921 − 25 + 25 × 9 ≈ 1,120, less the reset branch that is not executed on a normal step,
  giving **about 1,050 machine instructions per environment step** (range 950 to 1,250; no
  disassembler is installed on serval05, so this one number is an estimate rather than a
  measurement, and section 6 shows how much the conclusion depends on it).

## 4. The roofline: which resource runs out first

Arithmetic intensity of the environment step, $I = F / B$ from section 0:

- I = 250 ÷ 82 = **3.05 operations per byte**.

Machine balance, computed two ways because the chip has two kinds of arithmetic hardware:

- Matrix units, TF32, dense: 417,500 GFLOP/s ÷ 3,938 GB/s = **106 operations per byte**.
- Ordinary FP32 arithmetic, which is all this kernel can use: 60,300 ÷ 3,938 = **15.3 operations
  per byte**.

3.05 is far below both. By the classical roofline the kernel is **limited by memory bandwidth**,
and the ceiling follows directly:

- 3,938 × 10^9 bytes/s ÷ 82 bytes per environment step = **4.80 × 10^10 environment steps per
  second** = 48,000 million per second.
- Equivalently 20.8 picoseconds per environment step; at one million environments, a batched step
  would take 82 MB ÷ 3,938 GB/s = 20.8 microseconds.

That is the number the question asks for. Section 5 measures against it, and section 6 explains
why it is not the ceiling that actually binds.

## 5. Measured against the ceiling

Measurements taken on the same H100 NVL with an exclusive lock held, float32:

| implementation | best rate | at | µs per<br>batched step | benchmark file |
|---|---|---|---|---|
| CUDA fused kernel | 19,300 M/s | 1e6 envs | 51.8 | `envbench_cuda_fused`<br>`_tourn_count_grid` |
| JAX, many steps<br>per call | 9,920 M/s | 3e5 envs | 30.2 | `envbench_jax`<br>`_scan_grid` |
| PyTorch, compiled | 4,166 M/s | 3e6 envs | 720.2 | `envbench_torch`<br>`_compile_grid` |
| PyTorch, not<br>compiled | 201 M/s | 1e6 envs | 4,975 | `envbench_torch`<br>`_eager` |

Files are in `benchmarks/results/`, each prefixed by its timestamp of 2026-08-15 and suffixed
`.json`; each records the git revision it was measured at.

As a fraction of the 48,000 million per second ceiling:

| implementation | rate | share of ceiling |
|---|---|---|
| CUDA fused kernel | 19,300 M/s | **40.2%** |
| JAX, many steps per call | 9,920 M/s | **20.7%** |
| PyTorch, compiled | 4,166 M/s | **8.7%** |
| PyTorch, not compiled | 201 M/s | 0.4% |

The CUDA kernel's effective bandwidth: 82 bytes × 10^6 environments ÷ 51.8 microseconds
= 1,582 GB/s, which is the same 40% of 3,938 GB/s.

### What the two framework implementations' gaps are made of: bytes

Both frameworks split one environment step into several separate programs, and every value that
one program computes and the next one needs is written to memory and read back. That is extra
traffic the fused kernel does not pay.

Working backwards from the measurements, on the assumption that these two are bandwidth-limited:

| implementation | envs | µs/step | implied bytes per env-step |
|---|---|---|---|
| PyTorch compiled | 1e6 | 257.2 | 3,938e9 × 257.2e-6 ÷ 1e6 = **1,013** |
| PyTorch compiled | 3e6 | 720.2 | 3,938e9 × 720.2e-6 ÷ 3e6 = **945** |
| JAX, many steps/call | 3e5 | 30.2 | 3,938e9 × 30.2e-6 ÷ 3e5 = **397** |
| JAX, many steps/call | 1e6 | 107.0 | 3,938e9 × 107.0e-6 ÷ 1e6 = **421** |

Two batch sizes three times apart imply nearly the same byte count for each framework. That is
what a bandwidth limit predicts and what a fixed per-call overhead would not: overhead would make
the implied byte count fall as the batch grows.

The PyTorch figure can also be built up from the code, which is the check that matters. The
torch ledger's profile records **8 separate programs per step** at one million environments
(`torch_env/progress_and_changes.md`, row 3). Across a boundary between two of them, the
contact pipeline must carry the two-slot tournament state (distance, force-law constant,
regularizer, and two direction components, for each of the two slots = 10 values) plus the six
per-environment quantities the next program still needs (position, clipped velocity, unconstrained
acceleration) = 16 floats = 64 bytes, written once and read once = 128 bytes per boundary.

- 7 boundaries × 128 bytes = 896 bytes, plus the 118 bytes of genuine state = **1,014 bytes per
  environment step**.
- Measured-implied: 1,013 at one million environments, 945 at three million.

So compiled PyTorch is close to the bandwidth ceiling *for the bytes it moves*; it simply moves
about 12 times more bytes than the task requires. JAX's implied 400 bytes corresponds to roughly
two to three such boundaries instead of seven — its compiler fuses more, so it moves less, so it
runs faster, in that order. Neither framework is slow because its arithmetic is slow.

One caveat on the JAX number: its best point is 300,000 environments, where the state is about
118 bytes × 300,000 = 35 MB and therefore fits inside the 60 MiB L2 cache, so a large part of its
traffic never reaches HBM3 at all. Comparing that point against an HBM3 bandwidth ceiling is
generous to it — and the shape of its own curve says the same thing, since its throughput peaks
exactly where the data stops fitting. At one million environments, where the
data no longer fits, the best JAX form measured drops to 9,343 M/s
(`envbench_jax_scan_unroll8`).

### What the CUDA kernel's gap is made of: not bytes

The fused kernel moves only 82 bytes, so the same explanation cannot apply to it. Section 6
takes it apart.

## 6. Why a kernel can be far from the bandwidth ceiling and still be near its own

### The roofline drawn in instructions instead of FLOPs

The classical roofline counts only floating-point operations. This kernel's instructions are only
about a quarter floating-point arithmetic (section 3). Redo the comparison in the currency the
hardware actually spends — instructions:

- The kernel needs about 1,050 instructions per 82 bytes = **12.8 instructions per byte**.
- The machine can issue 3.02 × 10^13 instructions per second against 3.938 × 10^12 bytes per
  second = **7.7 instructions per byte** at full bandwidth.

12.8 is greater than 7.7, so on this roofline the kernel is on the *instruction* side, not the
memory side. Its ceiling is set by how fast instructions can be started:

- 3.02 × 10^13 instructions per second ÷ 1,050 instructions per environment step
  = **2.87 × 10^10 environment steps per second** = 28,700 million per second.
- Measured 19,300 ÷ 28,700 = **67% of its own ceiling.**

The two ceilings multiply out to the measured share of the bandwidth ceiling, which is the whole
answer in one line:

- 48,000 M/s × (28,700 ÷ 48,000) × 0.67 = 48,000 × 0.60 × 0.67 = 19,300 M/s.
- The first factor, 0.60, is structural: this algorithm costs 12.8 instructions per byte where the
  machine can only afford 7.7. No amount of tuning changes it.
- The second factor, 0.67, is the part an engineer can still work on.

Sensitivity to the one estimated quantity, the machine instruction count:

| instructions per env-step | ceiling | measured share |
|---|---|---|
| 950 | 31,700 M/s | 61% |
| 1,050 (central) | 28,700 M/s | 67% |
| 1,250 | 24,100 M/s | 80% |

The integer instructions run on a pipeline half as wide as the floating-point one on this chip,
which pushes the effective figure toward the top of that range. The honest statement is: **the
kernel is running at somewhere between two-thirds and four-fifths of the rate at which this
machine can issue its instructions.**

### Occupancy: is the machine kept busy?

Occupancy is how many warps are resident on an SM to switch between. Computed from the compiler's
register report:

- 39 registers per thread, rounded up to the hardware's allocation granularity: 40 per thread,
  1,280 per warp.
- 65,536 registers per SM ÷ 1,280 = 51 warps by register budget.
- The kernel launches blocks of 256 threads = 8 warps, so 6 whole blocks fit: **48 warps of the
  64 maximum = 75% occupancy**, with no spilling.
- Resident environments across the chip: 132 SMs × 48 warps × 32 threads = **202,752**.

That last number explains the shape of the measured curve: at 100,000 environments only half the
machine has work (8.31 µs per batched step, 12,000 M/s); at one million there are 4.9
full passes over the machine (19,300 M/s); at three million there are 14.8 and the rate is the
same 19,100 M/s. **Throughput stops improving once there is enough work to fill the machine**,
which is evidence that a shortage of parallel work is not the limiter at one million and above.

### Bandwidth-bound against latency-bound, and which this is

- A kernel is **bandwidth-bound** when so many memory requests are in flight that the memory
  system is the queue everything waits in. Symptom: time is proportional to bytes moved. Fix:
  move fewer bytes.
- A kernel is **latency-bound** when each operation takes a fixed time to produce its result and
  there is not enough independent work to do meanwhile, so the processor waits. Symptom: adding
  more parallel work helps; removing bytes does not. Fix: more resident warps, or more independent
  work inside each thread.
- A kernel is **issue-bound** when the schedulers have something to start on nearly every clock
  and the limit is simply the number of instructions the work requires. Symptom: neither fewer
  bytes nor more warps help; only fewer instructions do.

The CUDA ledger contains the experiment that separates these
(`cuda_env/progress_and_changes.md`, row 4, "E2 packed state"). It packed the state into one
16-byte load and store per environment and removed the separate observation output — 16 bytes of
the 98 the kernel moved before, a **16.3% reduction in traffic**, and fewer, wider memory
instructions.

| batch | before | after | predicted if bandwidth-bound |
|---|---|---|---|
| 100,000 | 9.1 µs | 9.1 µs | 7.6 µs |
| 1,000,000 | 59.6 µs | 59.8 µs | 49.9 µs |
| 3,000,000 | 171.7 µs | 173.1 µs | 143.7 µs |

Nothing moved. The prediction failed by the entire size of the effect. Read together with the
opposite experiment in the same ledger — row R2-3 replaced the running two-slot tournament with an
independent ranking that recomputes the force law for only the two winners, removing arithmetic
and nothing else, and gained **14% at one million environments** — the conclusion is exact:

- Changing the bytes: no effect.
- Changing the instructions: 14%.

This kernel is limited by the instructions it has to issue, not by the memory it has to touch.
The memory system is running at 40% of its peak and has capacity to spare. There is also a
component of dependent-operation waiting: the arithmetic is one long chain (distances → ranking →
force law → two-contact solve → integrator), each step needing the previous one's result, and two
of the reads in the middle of that chain (`dists[k1]` and `dists[k2]`, whose indices are only
known at run time) go through the cache instead of staying in registers. With 48 resident warps
per SM the schedulers can cover most of that waiting by switching warps, which is why the kernel
lands at two-thirds to four-fifths of the issue rate rather than well below it.

## 7. What an expert would try next, and the realistic best case

Because the limiter is instruction count, every worthwhile idea removes instructions. Ordered by
expected value:

1. **Replace the IEEE-exact divides and square roots with the hardware's approximate forms.**
   There are 15 divides and 10 square roots per step, each expanding into roughly 9 machine
   instructions; the approximate hardware instructions cost 1 to 2. Saving: about 200 of 1,050
   instructions, so **around 19%**, taking 19,300 to roughly 23,000 M/s. Price: the results stop
   being bit-identical to the PyTorch implementation, which is the project's cross-implementation
   correctness gate, and the reported one-step agreement of 2.1e-7 against the reference would
   have to be re-established at a looser tolerance.
2. **Give each thread two or four environments.** The address arithmetic, the constant loads and
   the loop bookkeeping are then paid once for several environments, and the independent chains
   overlap, covering the dependent waiting. Typical gain for a kernel of this shape: **5 to 15%**.
   Risk: more registers per thread, which could drop occupancy below the current 48 warps.
3. **Loop only over the cells that actually contain a wall**, instead of always testing all eight
   neighbours. Over the 46 open cells of the Large map the average is 4.78 wall neighbours out of
   8 — an apparent 40% saving. Most of it does not survive on this hardware: a warp costs the
   *maximum* over its 32 threads, and since 44 of the 46 open cells have 6 or fewer wall
   neighbours, the probability that all 32 threads of one warp are in such a cell is
   (44/46)^32 = 0.9565^32 = 0.24. Three warps in four still run 7 candidates, so the real saving is about one
   candidate in eight of the first pass — **roughly 4% overall**. This is the clearest example in
   the whole exercise of an algorithmic saving that does not transfer to hardware that executes
   32 environments in lockstep.
4. **Skip the contact pipeline entirely when no wall is within contact range.** Correct, and it
   would remove most of the work on most steps — but again only if all 32 environments in a warp
   agree, and independent environments do not. Making them agree means re-sorting environments by
   cell every step, which costs more than it saves at these batch sizes.
5. **Keep the two run-time-indexed distance reads in registers** by selecting over the eight
   values instead of reading an array. Replaces two cache round-trips in the middle of the
   dependency chain with about 14 select instructions. Could go either way; it is a
   twenty-minute experiment with a clean A/B measurement, which is the ledger's own standard.

**Realistic best case.** Items 1 and 2 together plausibly reach 750 to 800 machine instructions
per environment step. At the same 67% issue efficiency that gives
3.02 × 10^13 ÷ 775 × 0.67 = **2.6 × 10^10 environment steps per second**, about **1.35 times**
today's kernel and 54% of the bandwidth ceiling. Optimistically, at 80% efficiency, 3.1 × 10^10,
or 65% of the bandwidth ceiling.

**The bandwidth ceiling itself is not reachable with this physics.** Reaching 48,000 million steps
per second requires 3.02 × 10^13 ÷ 4.80 × 10^10 = **628 machine instructions per environment step
with no stalls at all** — a 40% cut on top of never waiting. The exact MuJoCo contact model (eight
candidate boxes, a solref/solimp force law, a two-contact quadratic program) does not fit in that
budget. A cheaper wall model does, and was tried and rejected on accuracy: the ledger records
one-step errors up to 48 millimetres against the reference, against the 4.4e-16 of the exact
model. That speed is available only by giving up the accuracy, which this project rejected.

## 8. What stage this optimization is at

A conventional way to grade this kind of work is by which resource has been made the limiter, and
how close the implementation is to *that* resource's ceiling:

| stage | limiter | example here |
|---|---|---|
| 1 | one environment at a time on the CPU | the original MuJoCo environment |
| 2 | per-call overhead | PyTorch not compiled: 4,975 µs per step |
| 3 | dispatch overhead | PyTorch compiled below 3e5 envs: a flat 155-171 µs floor |
| 4 | memory traffic the<br>framework adds | PyTorch compiled and JAX at scale: near their<br>own bandwidth limit, but moving 1,013 and about<br>400 bytes per env-step against a required 82 |
| 5 | instruction issue | the fused CUDA kernel: 82 bytes moved,<br>67% of the issue rate |
| 6 | the algorithm itself | not attempted; would mean approximate physics |

The fused kernel is at stage 5. The usual judgment at that point is that the kernel is finished:
the limiter has been identified by experiment rather than assumed, the implementation is within
about a third of that limiter's ceiling, and the remaining ideas each trade a few percent against
either bit-exactness or extra complexity. Further speed is an algorithm change, not tuning.

There is one more piece of context that decides whether any of section 7 is worth doing. The
end-to-end measurement in the main report already substituted this kernel into the training loop:
at 128 copies it changed the iteration time from 20.4 ms to 20.7 ms, about one percent, because
after the trainer's own optimizations the environment is a small part of a training step. So the
correct judgment is that the environment kernel is done for the purpose it serves in this project,
and the 1.35x still theoretically available should be spent only if a future workload is
environment-simulation only — data generation or evaluation sweeps — where the environment is
again the whole cost.

## 9. Summary of the derivation

| quantity | value | how obtained |
|---|---|---|
| bytes moved per env-step | 82 | counted from `step_kernel` buffers |
| float operations per env-step | 250 | hand count 248; PTX count 266<br>less the reset branch |
| all machine instructions<br>per env-step | ~1,050 | 921 PTX instructions plus<br>divide/root expansion |
| arithmetic intensity | 3.05 op/byte | 250 ÷ 82 |
| machine balance, TF32 matrix | 106 op/byte | 417,500 ÷ 3,938 |
| machine balance, FP32 | 15.3 op/byte | 60,300 ÷ 3,938 |
| verdict of the classical roofline | memory-limited | 3.05 well below 15.3 |
| **bandwidth ceiling** | **48,000 M steps/s** | 3,938e9 ÷ 82 |
| instructions per byte, kernel | 12.8 | 1,050 ÷ 82 |
| instructions per byte, machine | 7.7 | 3.02e13 ÷ 3.938e12 |
| verdict of the instruction roofline | issue-limited | 12.8 above 7.7 |
| **issue ceiling** | **28,700 M steps/s** | 3.02e13 ÷ 1,050 |
| CUDA kernel measured | 19,300 M steps/s | benchmark JSON, 1e6 envs |
| — share of bandwidth ceiling | 40.2% | 19,300 ÷ 48,000 |
| — share of issue ceiling | 67% | 19,300 ÷ 28,700 |
| JAX measured | 9,920 M steps/s | 20.7% of the bandwidth ceiling |
| PyTorch compiled measured | 4,166 M steps/s | 8.7% of the bandwidth ceiling |

Sources for the hardware numbers: the device's own report on serval05
(`nvidia-smi`, `cudaGetDeviceProperties`, 2026-08-15);
[NVIDIA H100 NVL GPU Product Brief PB-11773-001_v01](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf);
[NVIDIA H100 Tensor Core GPU datasheet, H100 NVL column](https://www.pny.com/file%20library/company/support/product%20brochures/nvidia%20data%20center%20gpus/english/h100-nvl-datasheet.pdf).
