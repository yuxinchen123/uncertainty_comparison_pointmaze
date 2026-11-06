"""Does this XLA accept these per-executable options? (compile and run a small loop with each)"""
import jax
import jax.numpy as jnp

def f(x):
    return jax.lax.fori_loop(0, 8, lambda i, v: jnp.tanh(v @ v.T @ v) * 0.5, x)

x = jnp.ones((256, 256), jnp.float32)
for options in ({"xla_gpu_enable_while_loop_double_buffering": "true"},
                {"xla_gpu_enable_triton_gemm": "false"},
                {"xla_gpu_enable_while_loop_unrolling": "WHILE_LOOP_UNROLLING_AUTO_UNROLL"}):
    try:
        jax.block_until_ready(jax.jit(f, compiler_options=options)(x)); print("accepted", options, flush=True)
    except Exception as error:
        print("REFUSED", options, type(error).__name__, str(error)[:160], flush=True)
