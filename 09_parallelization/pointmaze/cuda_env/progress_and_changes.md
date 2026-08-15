# cuda_env — experiment ledger (autoresearch style)

Goal metric (correctness): fixture checker (`../common/code/check_against_fixtures.py --impl cuda`)
both dtypes; resets bit-identical to torch; A/B trajectory agreement vs TorchPointMaze.
Goal metric (speed): env steps/second on the serval05 H100 (JSONs in `../../benchmarks/results/`).
Baseline to beat (torch compile): 4.03e9 env-steps/s at 1M envs; 153-171 us/step floor at 1k-100k.

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: whole step in one kernel (pointmaze_kernel.cu), one thread per env; exact transliteration of the torch dynamics; --fmad=false for float32 bit-equality; frozen fmix32 reset RNG in-kernel; shared math extracted to pointmaze_dynamics.h (compiles as host C++ too) | fixture checker ALL PASS both dtypes with error numbers IDENTICAL to the torch implementation digit for digit (float64 one-step <= 4.4e-16; float32 <= 4.6e-7) — evidence of bit-equality. A/B and benchmark pending on the GPU lock. | KEEP — correctness baseline |
| 0a | (root-cause record, no code change) every kernel launch was a silent no-op: device printf never fired, outputs untouched, NO error from any check even with CUDA_LAUNCH_BLOCKING=1. Isolated with a minimal 5-line kernel (also no-op) -> environment, not source. Cause: the pip cu13 lib dir ships only `libcudart.so.13` (no unversioned `libcudart.so`), so the link step's `-lcudart` fell back to the SYSTEM CUDA 11.5 runtime, and an 11.5 runtime silently ignores a CUDA-13 fatbin registration — launches become no-ops with "no error". Fix: cuda_home/lib64 rebuilt as a real dir of symlinks plus `libcudart.so -> libcudart.so.13`; extension then linked correctly (ldd shows .so.13) and everything ran. | FIXED (toolchain) |

| 1 | v0 benchmark + full gate results | fixture ALL PASS both dtypes; torch-vs-cuda resets BIT-IDENTICAL (incl. generation-1 respawns); A/B vs TorchPointMaze 500 steps within 1e-6 (worst 2.1e-7, most steps bit-equal, rare 1-2 ULP diffs on contact steps), 320 auto-resets exercised. Throughput (block 256): 1k envs 6.4 us/step (1.57e8/s), 10k 6.6 us (1.53e9/s), 100k 9.1 us (1.094e10/s), 1M 62.7 us (1.595e10/s), 4M 243 us (1.644e10/s). vs torch compile at 1M: 4.0x; vs jax scan upper bound at 1M: 1.9x. Small-batch floor 6.4 us vs torch 153 us = 24x. Traffic model ~94 B/env-step -> ~1.5 TB/s at 1M (~45% of peak) | KEEP — v0 baseline |

| 2 | block-size sweep at 1M envs (128/256/512/1024) | 59.8 / 59.6 / 61.3 / 72.3 us — 128 and 256 tie, larger blocks worse | NO CHANGE (keep 256) |
| 3 | full benchmark grid (tag _grid, repeats 5): 100..3M envs | floor 5.6-6.2 us/step up to 30k envs (launch-bound; one kernel + one pybind call); 100k 9.1 us (1.104e10/s), 300k 19.7 us (1.520e10/s), 1M 59.6 us (1.678e10/s), 3M 171.7 us (1.747e10/s) | recorded (baseline curve) |
| 4 | E2 packed state: one [B,4] state buffer loaded/stored as float4/double4; obs output ELIMINATED (post-reset state IS the observation, returned as the state tensor); goal as vector loads | all gates re-PASS (fixture both dtypes, A/B worst 2.1e-7, resets bit-identical). Throughput FLAT vs v0 (1M: 59.8 vs 59.6 us; 3M: 173.1 vs 171.7; 100k: 9.1 = 9.1) — the kernel is NOT bandwidth-limited at these sizes; the eliminated 16 B/env write bought nothing measurable | KEEP (equal speed, one buffer fewer, simpler aliasing story) |
| 5 | E3 candidate (coded, CPU-exact, NOT GPU-measured): distance-only tournament tracking (dist, k), full contact law recomputed only for the 2 winners — cuts ~8x of the in-loop law arithmetic; targets the compute/latency limit exposed by E4 | applied and reverted without GPU measurement: the main session's final training campaign took the lock for hours and this fork's window closed. CPU-port check passed exactly. The diff recipe is in this row and in git history; re-run gates + bench (tag _winnerlaw) when the GPU frees — first candidate for the extra-step revisit | DEFERRED (source left at E2) |

## Toolchain notes (serval05)

- torch 2.13.0+cu130 uses the consolidated `site-packages/nvidia/cu13/` tree (include+lib, no compiler).
- The cu13-suffixed pip names are DEPRECATED stubs that fail to install; the real packages are the
  plain names: `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-cuda-crt`,
  `nvidia-nvvm`. All install into `nvidia/cu13/`.
- Driver is 580.159.04 (CUDA 13.0 era). nvcc 13.3 + runtime 13.3 compiled fine but also produced
  the silent no-op; downgraded the whole toolchain to 13.0.88/13.0.96 to match the driver —
  no change, so the version story is NOT (yet) the explanation.
- Header pitfalls hit on the way: missing `nv/target` (fixed by installing `nvidia-cuda-cccl`);
  "CUDA compiler and CUDA toolkit headers are incompatible" (fixed by matching runtime minor to
  nvcc minor).
- CUDA_HOME is the symlink tree /localtmp/sl5nw/cuda_home -> nvidia/cu13/{bin,include,lib,nvvm}.
- ninja lives in the venv: prepend /localtmp/sl5nw/venvs/rnd09_torch/bin to PATH for builds.
- NFS caveat: the .cu source lives on /p; after editing it, delete
  /localtmp/sl5nw/torch_ext/pointmaze_cuda_ext before rebuilding (mtime skew can defeat ninja).
