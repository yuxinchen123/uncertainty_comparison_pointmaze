"""Which per-executable command-graph options does this XLA accept, and do loops get faster?"""
import time
import jax
import jax.numpy as jnp

def loop_heavy(x):
    def outer(i, v):
        def inner(j, w):
            return jnp.sin(w) * 1.0001 + j
        return jax.lax.fori_loop(0, jnp.int32(3) + (v[0, 0] > 1e9), inner, v)  # dynamic count: a real while loop
    return jax.lax.fori_loop(0, 128, outer, x)

x = jnp.ones((128, 33600), jnp.float32)
variants = {
    "default": None,
    "add WHILE and CONDITIONAL": {"xla_gpu_enable_command_buffer": "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL"},
    "plus syntax": {"xla_gpu_enable_command_buffer": "+WHILE,+CONDITIONAL"},
    "disabled": {"xla_gpu_enable_command_buffer": ""},
}
for name, options in variants.items():
    try:
        f = jax.jit(loop_heavy, compiler_options=options) if options else jax.jit(loop_heavy)
        jax.block_until_ready(f(x)); jax.block_until_ready(f(x))
        t = time.perf_counter()
        for _ in range(5):
            jax.block_until_ready(f(x))
        print(f"{name}: ok, {(time.perf_counter() - t) / 5 * 1000:.1f} ms per call", flush=True)
    except Exception as error:
        print(f"{name}: ERROR {type(error).__name__}: {str(error)[:200]}", flush=True)
