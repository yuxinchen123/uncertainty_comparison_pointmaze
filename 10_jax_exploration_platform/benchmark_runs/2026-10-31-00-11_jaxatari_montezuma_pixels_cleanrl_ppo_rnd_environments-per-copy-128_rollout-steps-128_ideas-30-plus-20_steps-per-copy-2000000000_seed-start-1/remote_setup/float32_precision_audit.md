# Float32 network precision review

Reviewed on 2026-10-31. The frozen networks request `HIGHEST` for convolutions, matrix products and Einstein summation. The installed JAX sources matched commit `a1521744c6dc074443fe549f19f48d7197abf759`, which pins OpenXLA `131bf41acb4650e4391a640c3f1859c1c86ad74b`.

The current settings select full Float32 calculations on the H100 and RTX A4500:

- The cuBLAS selection uses Float32 computation for precision level 2 (`HIGHEST`). Its TF32 path requires precision level 1 or lower. See [GPU computation selection](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/stream_executor/gpu/gpu_blas_lt.cc#L214) and [CUDA computation types](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/stream_executor/cuda/cuda_blas_utils.cc#L66).
- The cuDNN convolution path disables TF32 for `HIGHEST` and excludes engines that reduce input precision. See [convolution precision selection](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/backends/gpu/autotuner/cudnn.cc#L305) and [engine filtering](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/stream_executor/cuda/cuda_dnn.cc#L2643).
- The ordinary Triton dot path uses IEEE input precision for `HIGHEST`. TF32x3 needs its explicit algorithm selection. See [Triton lowering](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/backends/gpu/codegen/triton/transforms/stablehlo_lower_to_triton.cc#L596).

Running statistics explicitly use Float64 in [state.py](../../../src/exploration_platform/atari_pixels/state.py) and [training.py](../../../src/exploration_platform/atari_pixels/training.py), with X64 enabled for the learner. The game adapter separately scopes native game operations with X64 disabled. The successful restart check emitted one integer-truncation warning; its log does not locate the exact game operation, and no failed numerical comparison resulted.

Saved H100 and A4500 kernel summaries contain Float32 and SIMT names without reduced-input name matches. Those summaries do not contain instruction disassembly. This review covers the current frozen implementation; later candidates retain the same precision requirements and need their own code and numerical checks.
