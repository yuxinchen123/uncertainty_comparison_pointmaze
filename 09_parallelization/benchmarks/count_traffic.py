"""Arithmetic and memory traffic of one PyTorch training iteration, as a function of copies.

Companion to `analysis/ceiling/code/count_arithmetic.py`, which counts arithmetic at 128
copies. At the copy counts this trainer is actually used at (1,024 to 4,096) the arithmetic is
no longer the interesting quantity: the tensors have grown past every cache, so what the
iteration costs is set by how many bytes it moves. This script counts both, per copy, and
prints the two ceilings they imply on the card.

Counting rules, stated once:

- a matrix multiplication of (m x k) by (k x n) costs 2*m*k*n arithmetic operations, and its
  backward pass two more multiplications of the same size, so 6*m*k*n in total;
- every tensor written once and read once counts twice; a tensor that a later stage reads
  again counts again;
- element-wise arithmetic is excluded from the arithmetic count (well under one percent) but
  its traffic is counted, because at these sizes the traffic is what matters.

Run anywhere (no GPU needed): python count_traffic.py --n-copies 1024 4096
"""
import argparse

F32 = 4                       # bytes in one single-precision number
BW = 3.939e12                 # measured-consistent H100 NVL bandwidth, bytes per second
TF32_DENSE = 417.5e12         # dense TF32 matrix rate (half the "with sparsity" datasheet figure)
FP32 = 60.3e12                # ordinary single-precision arithmetic rate

T, N, B, MB, NMB = 128, 4, 512, 128, 16   # horizon, envs per copy, rows, minibatch rows, steps

# parameter counts per copy, read off the trainer's `stack` calls
ACTOR_P = 4 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2 + 2          # includes logstd
CRITIC_P = 4 * 64 + 64 + 64 * 64 + 64 + 2 * (64 + 1)
PRED_P = 4 * 256 + 256 + 256 * 128 + 128 + 128 * 128 + 128
TARGET_P = 4 * 256 + 256 + 256 * 128 + 128                      # frozen, never trained
TRAINED_P = ACTOR_P + CRITIC_P + PRED_P


def gemm(m, k, n):
    """Arithmetic of one (m x k) by (k x n) matrix multiplication."""
    return 2 * m * k * n


def counts(style):
    """(arithmetic, bytes moved) per COPY for one iteration in the given update style."""
    # ---- rollout: T sequential steps of the actor on N rows, plus the environment ----
    # arithmetic: three actor layers per step
    roll_f = T * (gemm(N, 4, 64) + gemm(N, 64, 64) + gemm(N, 64, 2))
    # traffic: the actor's weights are re-read on every one of the T steps (this is the whole
    # story of the rollout at large copy counts), plus the six per-step buffer writes and the
    # pre-drawn noise
    # before: per step the kernel reads ACTOR_P weights and writes obs 4, act 2, nobs 4 and
    #         three scalars per environment; after: multiplied by T steps and N environments
    roll_b = T * F32 * (ACTOR_P + N * (4 + 2 + 4 + 3) + N * 2 + N * 6)

    # ---- post-rollout: wide passes over the stored rows ----
    post_f = (gemm(2 * B, 4, 64) + gemm(2 * B, 64, 64) + 2 * gemm(2 * B, 64, 1)   # critic
              + gemm(B, 4, 512) + 2 * gemm(B, 256, 128) + gemm(B, 128, 128)       # bonus
              + gemm(B, 4, 256) + gemm(B, 256, 128))                              # cached target
    # traffic: each hidden activation is written by one kernel and read by the next
    post_b = F32 * (2 * (2 * B * 64) * 2                    # critic two hidden layers
                    + 2 * (B * 512) + 2 * (B * 256) + 2 * (B * 128) * 2   # bonus networks
                    + 2 * (B * 256) + B * 128               # cached target
                    + 12 * T * N                            # filter, GAE and statistics scans
                    + 2 * (B * 4 + B * 2 + B * 4))          # flatten the stored rows
    # the batch the update stage reads: nine fields, of which the cached target features
    # (B x 128 per copy) are nine tenths of the bytes
    batch_floats = B * (4 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 128)
    post_b += F32 * batch_floats                            # written once, in place

    # ---- update ----
    rows, steps = (B, 1) if style == "full_batch" else (MB, NMB)
    fwd_f = (gemm(rows, 4, 128) + 2 * gemm(rows, 64, 64) + 2 * gemm(rows, 64, 2)
             + gemm(rows, 4, 256) + gemm(rows, 256, 128) + gemm(rows, 128, 128))
    upd_f = 3 * fwd_f * steps
    # activations written by the forward and read by the backward, which writes gradients of
    # the same shapes; weights read by the forward and again by the backward; weight gradients
    # written, accumulated into the gradient buffer, then read by the optimiser
    act_floats = rows * (128 + 64 + 64 + 2 + 2 + 256 + 128 + 128)
    per_step = F32 * (3 * act_floats                        # forward write, backward read+write
                      + 2 * TRAINED_P                       # weights read forward and backward
                      + 2 * TRAINED_P                       # gradient written, then copied in
                      + rows * (4 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 128))   # the batch fields read
    # the optimiser: one reduction pass over the gradient, then one fused pass that reads the
    # gradient, both moments and the parameters and writes both moments and the parameters.
    # The gradient is not zeroed — the next backward pass overwrites it.
    per_step += F32 * TRAINED_P * (1 + 7)
    upd_b = steps * per_step
    if style != "full_batch":
        # style B also permutes the whole batch into a second buffer once per epoch
        upd_b += 4 * 2 * F32 * batch_floats

    return {"rollout": (roll_f, roll_b), "post": (post_f, post_b), "update": (upd_f, upd_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, nargs="+", default=[1024, 2048, 4096])
    args = ap.parse_args()
    print(f"trainable parameters per copy: {TRAINED_P:,}  "
          f"(actor {ACTOR_P:,}, critic {CRITIC_P:,}, predictor {PRED_P:,}); "
          f"frozen target {TARGET_P:,}")
    for style in ("full_batch", "epoch_minibatch"):
        c = counts(style)
        print(f"\n=== {style} ===")
        for C in args.n_copies:
            tf = sum(v[0] for v in c.values()) * C
            tb = sum(v[1] for v in c.values()) * C
            print(f"  C={C:>5d}  arithmetic {tf/1e9:8.1f} GFLOP  traffic {tb/1e9:8.2f} GB  "
                  f"intensity {tf/tb:5.1f} FLOP/byte  "
                  f"floor(bandwidth) {tb/BW*1e3:7.2f} ms  "
                  f"floor(TF32) {tf/TF32_DENSE*1e3:6.3f} ms")
            for k, (f, b) in c.items():
                print(f"          {k:>8s}: {f*C/1e9:7.1f} GFLOP  {b*C/1e9:7.2f} GB  "
                      f"{b*C/BW*1e3:6.2f} ms at full bandwidth")


if __name__ == "__main__":
    main()
