"""
Reusable CPU profiling for the 07_reconstruction training scripts.

Purpose: measure (a) wall-clock throughput and (b) how many CPU cores a run
actually keeps busy, separating program startup from steady-state training.

Design goal stated by the user: "make sure the profiling code can be easily
turned off later during training." Everything here is opt-in:

  - Set the environment variable RND_PROFILE=1 to turn profiling ON.
  - Leave it unset (or =0) and every function below becomes a no-op with no
    measurable overhead, so the same code can stay in a training script.

Two ground-truth signals are recorded, both at the level of *this process only*
(the env runs in-process via DummyVecEnv, so a single process captures it all):

  effective_cores = (process_cpu_time_end - process_cpu_time_start)
                    / (wall_time_end - wall_time_start)

    This is the windowed version of `/usr/bin/time -v`'s "Percent of CPU"
    divided by 100. It measures how many cores the process *kept busy*
    (consumed) — NOT how many did useful work: it counts CPU-seconds the OS
    charged to the process and divides by elapsed seconds. 1.0 = one core fully
    busy; 8.0 = eight cores fully busy on average. Because OpenMP/OpenBLAS worker
    threads spin-wait (busy-loop) when idle, spin-waste counts here as "busy" —
    that is the point: compare effective_cores against the actual speedup to see
    how much of the consumption was wasted.

  cpu_percent samples (psutil) and per-thread CPU times give the distribution
    and the count of threads that actually accumulated CPU time.

WARNING about CPU% as a proxy for speed: OpenMP / MKL worker threads spin-wait
by default (OMP_WAIT_POLICY=active), so an idle thread pool can report ~100%
CPU per thread without doing useful work. High CPU% therefore does NOT imply a
wall-clock speedup. Always read effective_cores together with steps_per_sec.
"""
import json
import os
import threading
import time
from collections import deque
from typing import Optional

PROFILE_ENABLED = os.environ.get("RND_PROFILE", "0") == "1"

# Thread-pool environment variables that control each math backend. These must
# be set BEFORE numpy / torch are imported for the *_NUM_THREADS variants to be
# read by OpenBLAS / MKL at load time. apply_thread_limit() additionally calls
# torch.set_num_threads() which works at any time.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",        # OpenMP -> torch (MKL/oneDNN), some numpy paths
    "MKL_NUM_THREADS",        # Intel MKL -> torch CPU matmul / linalg
    "OPENBLAS_NUM_THREADS",   # OpenBLAS -> numpy matmul / linalg
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def export_thread_env(n: int) -> dict:
    """Return a dict of the thread-cap env vars set to n (for subprocess env)."""
    return {k: str(n) for k in _THREAD_ENV_VARS}


def apply_thread_limit(n: Optional[int]) -> None:
    """Cap intra-op CPU threads for torch (MKL/OpenMP). Env vars for OpenBLAS/MKL
    should already be exported by the launcher; this re-affirms inside torch."""
    if n is None:
        return
    for k in _THREAD_ENV_VARS:
        os.environ[k] = str(n)
    try:
        import torch
        torch.set_num_threads(int(n))
    except Exception:
        pass


class CpuSampler:
    """Background daemon thread that samples whole-process CPU usage.

    No-op unless RND_PROFILE=1. Samples, every `interval` seconds:
      - process cpu_percent (sum over threads; >100 means multiple cores)
      - number of OS threads in the process
      - process cumulative cpu_times (user+system) and wall time, so any time
        window can be turned into an effective-cores figure afterwards.
    """

    def __init__(self, interval: float = 0.1, enabled: Optional[bool] = None):
        self.enabled = PROFILE_ENABLED if enabled is None else enabled
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples = deque()  # (wall, cpu_percent, num_threads, cpu_time_total)
        self.peak_rss = 0  # bytes, peak resident set size of this process
        self._proc = None

    def start(self):
        if not self.enabled:
            return self
        import psutil
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime; first call always returns 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        proc = self._proc
        while not self._stop.is_set():
            try:
                ct = proc.cpu_times()
                self.samples.append(
                    (time.perf_counter(),
                     proc.cpu_percent(None),
                     proc.num_threads(),
                     ct.user + ct.system)
                )
                rss = proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def window_summary(self, wall_start: float, wall_end: float) -> dict:
        """Effective cores and CPU% distribution over [wall_start, wall_end]."""
        if not self.enabled or len(self.samples) < 2:
            return {}
        s = [x for x in self.samples if wall_start <= x[0] <= wall_end]
        if len(s) < 2:
            s = list(self.samples)
        eff_cores = (s[-1][3] - s[0][3]) / max(1e-9, (s[-1][0] - s[0][0]))
        pcts = [x[1] for x in s[1:]]  # skip first (priming artifacts)
        thr = [x[2] for x in s]
        pcts_sorted = sorted(pcts)
        n = len(pcts_sorted)
        return {
            "effective_cores_window": round(eff_cores, 3),
            "mean_cpu_percent": round(sum(pcts) / max(1, len(pcts)), 1),
            "median_cpu_percent": round(pcts_sorted[n // 2], 1) if n else None,
            "max_cpu_percent": round(max(pcts), 1) if pcts else None,
            "mean_num_threads": round(sum(thr) / len(thr), 1),
            "max_num_threads": max(thr),
            "peak_rss_mb": round(self.peak_rss / 1e6, 1),
            "n_samples": len(s),
        }


def per_thread_cpu_active(threshold_s: float = 0.05) -> dict:
    """Snapshot: how many OS threads have accumulated real CPU time so far.
    Strong evidence of how many cores actually did work (vs. spin-waiting)."""
    if not PROFILE_ENABLED:
        return {}
    try:
        import psutil
        proc = psutil.Process()
        ts = proc.threads()
        active = [t for t in ts if (t.user_time + t.system_time) >= threshold_s]
        return {
            "n_threads_total": len(ts),
            "n_threads_with_cpu_time": len(active),
            "thread_cpu_seconds": sorted(
                round(t.user_time + t.system_time, 2) for t in ts
            )[::-1][:20],
        }
    except Exception:
        return {}


try:
    from stable_baselines3.common.callbacks import BaseCallback

    class StepRateProfiler(BaseCallback):
        """SB3 callback that records (wall, step, process_cpu_time) every K steps
        and computes a steady-state throughput excluding the first `warmup_steps`.

        No-op unless RND_PROFILE=1. Overhead when on is one psutil cpu_times()
        call per K steps.
        """

        def __init__(self, every: int = 50, warmup_steps: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.enabled = PROFILE_ENABLED
            self.every = every
            self.warmup_steps = warmup_steps
            self.records = []  # (wall, num_timesteps, cpu_time_total)
            self._proc = None
            self._t0 = None

        def _on_training_start(self) -> None:
            if not self.enabled:
                return
            import psutil
            self._proc = psutil.Process()
            self._t0 = time.perf_counter()
            self._record()

        def _record(self):
            ct = self._proc.cpu_times()
            self.records.append(
                (time.perf_counter(), int(self.num_timesteps), ct.user + ct.system)
            )

        def _on_step(self) -> bool:
            if self.enabled and (self.num_timesteps % self.every == 0):
                self._record()
            return True

        def _on_training_end(self) -> None:
            if self.enabled:
                self._record()

        def summary(self) -> dict:
            if not self.enabled or len(self.records) < 3:
                return {}
            rec = self.records
            last_step = rec[-1][1]
            # steady-state window = records with step >= warmup_steps
            w = [r for r in rec if r[1] >= self.warmup_steps]
            if len(w) < 2:
                w = rec
            wall = w[-1][0] - w[0][0]
            steps = w[-1][1] - w[0][1]
            cpu = w[-1][2] - w[0][2]
            full_wall = rec[-1][0] - rec[0][0]
            return {
                "total_steps": last_step,
                "full_run_wall_s": round(full_wall, 3),
                "warmup_steps": self.warmup_steps,
                "steady_window_steps": steps,
                "steady_window_wall_s": round(wall, 3),
                "steady_steps_per_sec": round(steps / max(1e-9, wall), 1),
                "steady_effective_cores": round(cpu / max(1e-9, wall), 3),
                "steady_ms_per_step": round(1000.0 * wall / max(1, steps), 3),
            }
except Exception:  # stable_baselines3 not importable in a pure-microbench context
    StepRateProfiler = None
