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

## Round 2

| # | change | result | verdict |
|---|---|---|---|
| R2-1 | BLOCKER FIX: all three kernels launched with no stream argument, so they went to the legacy default stream instead of the stream the caller (and any CUDA-graph capture) is using. Added `#include <c10/cuda/CUDAStream.h>` and `at::cuda::getCurrentCUDAStream()` as the fourth launch argument in `env_step`, `env_reset`, `env_dynamics`. | The bug was confirmed BEFORE the fix by a new test: PyTorch itself warned "The CUDA Graph is empty", the captured step EXECUTED during capture instead of being recorded, and replaying the graph did not move the state. After the fix the graph records the step and 5 replays match 5 eager steps bit-exactly. All gates re-pass with numbers identical to before (fixture both dtypes ALL PASS, A/B vs torch worst 2.1e-7, resets bit-identical) — the fix touches no arithmetic. Env-only throughput unchanged to slightly better (1M: 58.8-59.1 us vs 59.6 before; two back-to-back runs spread 0.3 us). | KEEP — and see the note below about the pairing numbers |
| R2-2 | New test `test_stream_capture_gpu.py`: captures a graph containing one env step, replays it, and requires the state to move and to match the eager path; plus the same steps on an explicit side stream. | Fails loudly on the pre-fix build (the check that capture must RECORD rather than execute is what catches it), passes after. | KEEP — this is the test that was missing |
| R2-3 | Rank-by-counting tournament (round-2 idea list, CUDA row 2), built behind `-DPM_RANK_TOURNAMENT` with the old running tournament kept under `PM_TOURNAMENT=select` so the two can be compared back to back. Eight distances computed independently, each ranked by counting how many candidates precede it (ties to the lower index, which is what the running form's strict-improvement rule does), then geometry and contact law recomputed for the two winners only. A winner that is not a wall box is given the initial payload the running form leaves in an untouched slot, so the two agree by construction rather than by a downstream cancellation argument. | EXACT: 200 steps with 1,280 auto-resets are BIT-IDENTICAL between the two builds (states, rewards, reset counts). All gates pass with the same error numbers. Paired ABBA timing (count, select, select, count), env-steps/s: 1k 1.74e8 vs 1.67e8 (+4%); 100k 1.20e10 vs 1.104e10 (+8.5%); 1M **1.93e10 vs 1.69e10 (+14%)**; 3M **1.92e10 vs 1.76e10 (+9%)**. Within-form spread 0.1-3.3 us against differences of 7-14 us. | KEEP as the default form |
| R2-4 | Captured-step benchmark mode (`--capture-chunk K`): K env steps recorded into one CUDA graph and replayed, which is the floor a captured trainer actually pays. | 1k envs 4.7 us/step vs 5.7 uncaptured (-18%); 30k 5.1 vs ~6.0; 1M 54.3 vs 51.9 (SLOWER by 4.6%). Capture pays where the per-call launch cost dominates and not where the kernel does. Caveat recorded: the captured mode replays ONE fixed action buffer (that is how capture works), while the uncaptured bench rotates eight, so the large-batch comparison carries a workload difference and should not be read as a pure capture cost. | KEEP the mode as a measurement tool; no default change |

### Note for the main session: the previously published cuda-env PAIRING numbers are INVALID

The rows "CUDA env + torch trainer, one-graph: 25.1 ms (C=8) / 35.6 ms (C=128) style B, 23.2 ms
style A" were measured with the pre-fix kernel. Under `torch.cuda.graph(...)` the env kernel was
not recorded into the graph at all (PyTorch reported an empty graph in the isolated test), so each
replay ran the policy, the post-processing and the update but never stepped the environment. Those
numbers must be discarded and the pairing re-measured on the fixed build. The env-only numbers in
the report are unaffected — they never used capture.


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
