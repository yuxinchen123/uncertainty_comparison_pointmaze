"""Keyed per-copy weight initialisation, shared by the agent and by every bonus family.

Copy c's weights are the same whether the run has 8 copies or 8,448, and the same across sweep
groups when the sweep pairs seeds, because each weight tensor is drawn from a key folded from
(base seed, network id, layer index, copy seed index) rather than from one sequential stream.
"""
import jax
import jax.numpy as jnp

from . import F32

# Stable integer ids for the keyed init (the reproducible-randomness rule: one named quantity,
# one key). These five are FROZEN — every number 09_parallelization measured depends on them, and
# the golden-parity gate compares against those numbers. A new network takes the next free id and
# never reuses one of these.
NET_IDS = {"actor": 1, "critic": 2, "critic_heads": 3, "rnd_target": 4, "rnd_predictor": 5}


def orthogonal(key, rows, cols, gain):
    """Orthogonal [rows, cols] like torch nn.init.orthogonal_ (QR with sign fix), float32."""
    flat = jax.random.normal(key, (max(rows, cols), min(rows, cols)), F32)
    q, r = jnp.linalg.qr(flat)
    q = q * jnp.sign(jnp.diagonal(r))
    if rows < cols:
        q = q.T
    return (gain * q)[:rows, :cols].astype(F32)


def stacked_orthogonal(net: str, layers, n_copies: int, base_seed: int, copy_seed_index):
    """Weights and biases for one network, stacked over copies: {"W0": [C, in, out], "b0": ...}.

    layers gives (rows, cols, gain) per layer in the torch convention (rows = outputs), and the
    weight is stored transposed so a forward pass is `x @ W`.

    before: net "actor", layers [(64, 4, sqrt2), (64, 64, sqrt2), (2, 64, 0.01)], 4 copies
    after:  {"W0": [4, 4, 64], "b0": [4, 64], "W1": [4, 64, 64], "b1": [4, 64],
             "W2": [4, 64, 2],  "b2": [4, 2]}
    """
    out = {}
    for li, (rows, cols, gain) in enumerate(layers):
        ws = []
        for c in range(n_copies):
            k = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
                jax.random.PRNGKey(base_seed), NET_IDS[net]), li), copy_seed_index[c])
            # stored [in, out] for x @ W; orthogonal drawn on [out, in] then transposed
            ws.append(orthogonal(k, rows, cols, gain).T)
        out[f"W{li}"] = jnp.stack(ws)
        out[f"b{li}"] = jnp.zeros((n_copies, rows), F32)
    return out
