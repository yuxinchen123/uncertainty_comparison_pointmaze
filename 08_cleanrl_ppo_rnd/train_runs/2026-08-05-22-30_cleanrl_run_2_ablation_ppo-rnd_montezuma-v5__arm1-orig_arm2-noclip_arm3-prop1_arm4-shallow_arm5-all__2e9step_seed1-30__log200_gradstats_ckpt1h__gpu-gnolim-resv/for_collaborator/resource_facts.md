# Resource estimate — where each number came from

Every figure here was measured on this cluster on 2026-08-05, not guessed. The measurements live in
`08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling/`; the report built from them is
`2026-08-05_throughput_and_resource_profiling.md` in that folder.

## `gpu_mem_mb` = 4900

Peak device memory of one run, from `torch.cuda.max_memory_allocated()` sampled every update after
warm-up, in the run's real configuration:

| node | GPU | peak allocated (MB) |
|---|---|---|
| `jaguar03` | RTX A4500 | 4,772 |
| `lotus` | Quadro RTX 6000 | 4,770 |
| `cheetah02` | RTX 4000 Ada | 4,770 |
| `cheetah08` | RTX A4000 | 4,772 |
| `jaguar02` | A16 | 4,770 |
| `adriatic01` | Quadro RTX 4000 (8 GB) | 6,212 |
| `ai07` | GTX 1080 Ti (11 GB) | 6,212 |

The figure is set from the cluster of five cards at ~4,770 MB. **The two 6,212 MB readings are real
and are worth knowing about**: on those cards cuDNN's autotuner selects a convolution algorithm with a
larger workspace. Both cards ran the whole ladder without failing, so the larger workspace fits; but
if a future change packs runs onto an 8 GB card, size against 6,300 MB, not 4,900.

The bound is not tight anywhere: the smallest card in scope is 8 GB.

## `host_mem_mb` = 6000

`sacct MaxRSS` of the profiling jobs, which ran one training process at a time:

| job | node | MaxRSS |
|---|---|---|
| 6533835 | `jaguar03` | 4,727 MB |
| 6533839 | `lotus` | 4,367 MB |

6,000 MB carries about 27% headroom over the larger reading.

## `cpu_threads_busy` = 8, and `c_min = c_max = 8`

From the cores-versus-throughput sweep: one GPU held constant, only the cpu allocation varying.

| cpus | steps/s | steps/s per cpu |
|---|---|---|
| 4 | 2,714 | 679 |
| 8 | 3,876 | 485 |
| 16 | 4,182 | 261 |
| 32 | 3,906 | 122 |
| 48 | 4,120 | 86 |

The curve flattens between 8 and 16 cpus and does not rise after. Past that point the rollout stops
falling at about 17 ms per step and the update is a flat GPU cost that cpus do not touch — envpool
has stopped being the limit and the per-step policy forward, bonus forward and host transfers have
become it.

**8 is chosen over 16 deliberately.** It reaches 93% of the peak rate on half the cores, and the cpu
is the capped resource in both GPU partitions (400 in `gpu`, 80 in `gnolim`), so the run count this
allows matters more than the last 7% of single-run speed. `c_min = c_max = 8` pins it, because
letting the tool pick a smaller value would land on the steep part of the curve.

## `w_max` = 3

Measured directly by running N copies of the trainer on one GPU at once (`jaguar03`, RTX A4500,
20 GB, 8 cpus per copy):

| runs on the GPU | steps/s per run | aggregate steps/s | steps/s per cpu |
|---|---|---|---|
| 1 | 4,618 | 4,618 | 577 |
| 2 | 3,595 | 7,190 | 449 |
| 3 | 3,343 | 10,029 | 418 |
| 4 | 2,915 | 11,661 | 364 |
| 6 | 2,290 | 11,450 | 286 |

Aggregate throughput per GPU keeps rising to 4 and then stops; at 6 copies the card held 19,869 MB
of its 20,470 MB and one of the six died. So the hard ceiling on a 20 GB card is 5, and 3 is the cap
recorded here because it is the largest packing with a comfortable memory margin and the best
aggregate-per-cpu trade among the packed options.

**Packing helps only where the GPU is the scarce side.** Throughput per cpu falls monotonically with
packing, so:

- In `gpu` (40 GPU slots, 400 cpus) and `gnolim` (20 GPU slots, 80 cpus) the cpu runs out first, so
  unpacked `W = 1` is correct — 40 unpacked runs beat 50 packed ones.
- On the `jaguar03` reservation (8 GPU slots, 222 cpus) the GPU is the scarce side, so packing turns
  8 slots into 24 at 2.17x the aggregate rate.

## `compatible_gpu_types`: 6.0 to 9.0

Canary jobs ran the RND predictor's real convolution stack forward and backward on each architecture
with `torch 2.6.0+cu124`:

| node | GPU | compute capability | verdict |
|---|---|---|---|
| `ai05` | GTX 1080 | 6.1 | runs |
| `ai07` | GTX 1080 Ti | 6.1 | runs |
| `affogato13` | GTX 1080 Ti | 6.1 | runs |
| `titanx03` | TITAN X (Pascal) | 6.1 | runs |
| `jaguar03` | RTX A4500 | 8.6 | runs |
| `nekomata01` | RTX 5080 | 12.0 | **fails**: `sm_120 is not compatible with the current PyTorch installation` |

The floor of 6.0 covers the Tesla P100 nodes (lynx05-07), which are the only cc 6.0 cards present.
The ceiling of 9.0 excludes nekomata01 and nekomata02 — 4 GPUs, the only ones on the cluster this
build cannot reach.

Two notes that do not fit in the JSON:

1. **The shared node catalog is wrong about `titanx03`.** It records a Titan X, Maxwell (2015),
   compute capability 5.2. The card reports **TITAN X (Pascal), compute capability 6.1, 12 GB**.
   Worth correcting at the source, since the packing tools read that catalog.
2. **`torch.compile` is unavailable below compute capability 7.0** because Inductor emits Triton
   kernels. It is switched off for this run anyway, so this only matters for future work.
