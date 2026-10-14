"""Confirm the selected 128-copy execution on real episode-window accumulation."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def measure(bonus, length):
    """Time training-window accumulation at 128 copies, excluding compilation."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.blocks import iteration_block
    runner = Runner(PPOConfig(n_copies=128, update_style="full_batch", track_coverage=True,
                              parameter_settings=((1e-3, 1.),), copies_per_group=128), bonus)
    state = runner.prime(runner.init_state(0))
    programs = {}

    # Both modes compute and accumulate the same 200-iteration episode windows.
    def window(state):
        """Run one complete recording window, carrying per-iteration annealed rates."""
        reward = jnp.zeros((128,), jnp.float32)
        intrinsic = jnp.zeros((128,), jnp.float32)
        rates = np.asarray(1 - np.arange(200, dtype=np.float64) / 200000, np.float32)
        index = 0
        while index < 200:
            count = min(length, 200 - index)
            if count == 1:
                state, metrics = runner.iterate(state, jnp.asarray(rates[index]))
            else:
                if count not in programs:
                    programs[count] = iteration_block(runner, count)
                state, metrics = programs[count](state, jnp.asarray(rates[index:index + count]))
            reward += metrics["reward_ext_sum"]
            intrinsic += metrics["rint_mean"]
            index += count
        jax.block_until_ready((state, reward, intrinsic))
        return state

    state = window(state)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(8):
            state = window(state)
        times.append((time.perf_counter() - start) / 1600)
    seconds = sorted(times)[1]
    return {"bonus": bonus, "copies": 128, "iterations_per_dispatch": length,
            "seconds_per_iteration": seconds, "steps_per_second_per_copy": 512 / seconds,
            "timing_repeats": times,
            "peak_device_memory_mb": jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20}


def main():
    """Confirm both algorithm families using the same seed and recording shape."""
    output = Path(sys.argv[1])
    rows = []
    for bonus in ("rnd_next_state", "gt_position_velocity_sqrt"):
        for length in (1, 32):
            row = measure(bonus, length)
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
