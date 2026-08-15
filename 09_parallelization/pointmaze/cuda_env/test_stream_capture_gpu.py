"""GPU-only: the fused CUDA env must obey the caller's stream, and must work under graph capture.

The kernels are launched from C++. If they are issued to the legacy default stream instead of
the stream PyTorch is currently using, then inside a `torch.cuda.graph(...)` region they carry no
ordering against the surrounding PyTorch kernels, and may not be recorded into the graph at all.
A replay would then leave the environment state untouched while everything around it advances —
which looks fast and passes any test that only checks shapes or means.

Two checks:
  1. Capture a graph containing one environment step, replay it five times, and require the
     resulting state to be bit-equal to five eager steps from the same start with the same action.
     A graph that did not record the env kernel leaves the state at its starting value and fails.
  2. Run the env step on an explicit non-default stream and require the same answer, so the
     kernels demonstrably follow the current stream rather than a fixed one.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_stream_capture_gpu.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
from pm_common import EnvConfig  # noqa: E402
from cuda_pointmaze import CudaPointMaze  # noqa: E402

C, N, STEPS = 2, 8, 5


def fresh():
    """A reset environment plus the constant action used by every check."""
    env = CudaPointMaze(EnvConfig(), C, N, device="cuda", base_seed=7)
    env.reset()
    act = torch.full((C, N, 2), 0.5, device="cuda")
    return env, act


def eager_reference():
    """Five ordinary steps; returns the start state and the final state."""
    env, act = fresh()
    start = env.state.clone()
    for _ in range(STEPS):
        env.step(act)
    return start, env.state.clone()


def test_capture_records_the_env_step():
    """A captured env step must advance the state exactly as the eager step does."""
    start, ref = eager_reference()
    assert not torch.equal(start, ref), "the eager reference did not move; the check is vacuous"

    env, act = fresh()
    # warm up on a side stream (required before capture), then put the state back to the start
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            env.step(act)
    torch.cuda.current_stream().wait_stream(side)
    env.reset()
    captured_start = env.state.clone()
    assert torch.equal(captured_start, start), "reset did not return to the same start state"

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        env.step(act)
    # capture records without executing, so the state must still be at the start here
    assert torch.equal(env.state, start), "capture executed the step instead of recording it"

    for _ in range(STEPS):
        g.replay()
    torch.cuda.synchronize()

    moved = not torch.equal(env.state, start)
    same = torch.equal(env.state, ref)
    print(f"  replayed state moved: {moved}")
    print(f"  replayed state matches eager: {same} "
          f"(worst |d| {(env.state - ref).abs().max().item():.3e})")
    assert moved, "the replayed graph did not step the environment at all"
    assert same, "the replayed graph stepped the environment differently than the eager path"
    print("ok test_capture_records_the_env_step")


def test_step_follows_the_current_stream():
    """The same steps issued on an explicit side stream must give the same answer."""
    _, ref = eager_reference()
    env, act = fresh()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(STEPS):
            env.step(act)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(env.state, ref), "stepping on a side stream gave a different result"
    print("ok test_step_follows_the_current_stream")


if __name__ == "__main__":
    test_capture_records_the_env_step()
    test_step_follows_the_current_stream()
