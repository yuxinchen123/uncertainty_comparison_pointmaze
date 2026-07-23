# Resource facts — point maze + ant maze train run 1 (probe + canary, 2026-07-23)

Sources: resource probe jobs 6519792 (cpu, panther01) / 6519793 (gpu, ai06, COMPLETED 28 min), and
the canary wave jobs 6519804 (cpu 14x1, lynx08) / 6519805 (nolim 12x1, heartpiece) / 6519806
(gpu W=4, ai06) — 36 short 50k-step runs of the real worker path across every algorithm class and
both env families; 0 failures.

## Speed (steps/s per run; 1M-step run time)

| family | device / node class | steps/s (min / median / max) | 1M-step run |
|---|---|---|---|
| PointMaze | cpu (lynx08 / heartpiece class) | 14.7 / 16.7 / 21.0 | ~17 h |
| AntMaze | cpu (same) | 13.5 / 17.4 / 19.9 | ~16 h |
| PointMaze | cuda (ai06, 4 packed/GPU) | 25.7 | ~11 h |
| AntMaze | cuda (ai06, 2-4 packed/GPU) | 30-34 | ~8-9 h |

AntMaze runs as fast as PointMaze per step (SAC updates dominate, not the ant physics). The GPU
gives ~2x the per-run speed of the old CPU node classes.

## Memory

- CPU worker host RSS: ~730 MB at 50k steps (probe sstat MaxRSS); grows with the replay buffer to
  an estimated ~1.0-1.2 GB at 1M steps (AntMaze buffer ~270 MB at full). The 2 GB per worker in
  the worker files holds.
- GPU worker: ~230 MiB GPU memory per run (457 MiB for 2 packed); host RSS ~1.9 GB per run at 50k
  (CUDA runtime overhead) -> ~2.3 GB at 1M. GPU utilization 85% during the obs-warmup phase,
  ~12% steady state at 2 packed; W=4 showed no per-run slowdown vs 2.

## Packing decision

- **W = 8 workers per GPU** (worker_gpu_1x8.slurm): GPU memory 8 x 230 MiB ~= 1.9 GiB (fits any
  gpu-partition card), host --mem = ceil(8 x 2.4 GB x 1.25) = 24 GB, 8 single-cpu tasks.
- CPU shapes unchanged: 30x1 / 28x1 (32-core class), 14x1 / 12x1 (16-core class), 2 GB/worker.

## Load estimate

Racing minimum ~6,160 runs x 1M steps at ~17 h (cpu) / ~9 h (gpu) per run; winner phase adds
~28 configs to 300 seeds. With ~300-500 workers the racing phase is roughly a week of wall time.
