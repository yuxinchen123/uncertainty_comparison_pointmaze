"""Per-layer cost of each convolution algorithm on this GPU (deterministic float32, highest precision).

For each CleanRL layer shape: update cost = forward + both gradients at 4096 rows; rollout cost =
forward at 128 rows. Writes one JSON line per (copies, layer, algorithm) to the output file.
Usage: per_layer_convolution_cost.py <output.jsonl> <batching>
"""
import json
import os
import statistics
import sys
import time

os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
from exploration_platform.atari_pixels.networks import convolution

OUTPUT, BATCHING = sys.argv[1], sys.argv[2]
LAYERS = {"policy first (4 to 32, 8x8 stride 4)": (4, 32, 8, 4, 84),
          "RND first (1 to 32, 8x8 stride 4)": (1, 32, 8, 4, 84),
          "second (32 to 64, 4x4 stride 2)": (32, 64, 4, 2, 20),
          "third (64 to 64, 3x3 stride 1)": (64, 64, 3, 1, 9)}
ALGORITHMS = ("compiler", "space_to_depth", "patch_weight_gradient", "patch_matrix")


def timed(function, *arguments):
    for _ in range(2):
        jax.block_until_ready(function(*arguments))
    durations = []
    for _ in range(7):
        started = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        durations.append(time.perf_counter() - started)
    return statistics.median(durations) * 1000


with open(OUTPUT, "a") as output:
    for copies in (1, 2):
        for name, (channels, outputs, kernel, stride, size) in LAYERS.items():
            key = jax.random.PRNGKey(0)
            layer = {"weight": jax.random.normal(key, (copies, kernel, kernel, channels, outputs), jnp.float32) * 0.05,
                     "bias": jnp.zeros((copies, outputs), jnp.float32)}
            for algorithm in ALGORITHMS:
                if algorithm == "space_to_depth" and stride == 1:
                    continue

                def loss(parameters, images):
                    return jnp.sum(convolution(parameters, images, stride, batching=BATCHING,
                                               algorithm=algorithm) ** 2)

                # The first layer reads data, so training needs no input gradient there.
                update = jax.jit(jax.grad(loss, argnums=(0,) if stride == 4 else (0, 1)))
                forward = jax.jit(lambda parameters, images: convolution(
                    parameters, images, stride, batching=BATCHING, algorithm=algorithm))
                big = jax.random.uniform(key, (copies, 4096, size, size, channels), jnp.float32)
                small = jax.random.uniform(key, (copies, 128, size, size, channels), jnp.float32)
                record = {"device": jax.devices()[0].device_kind, "batching": BATCHING, "copies": copies,
                          "layer": name, "algorithm": algorithm,
                          "update_milliseconds": timed(update, layer, big),
                          "rollout_forward_milliseconds": timed(forward, layer, small)}
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(record, flush=True)
