"""Count the matrix-multiplication arithmetic of one PPO+RND training iteration at C=128.

Convention: a matrix multiplication of (m x k) by (k x n) costs 2*m*k*n floating-point
operations (one multiply and one add per accumulation). A backward pass costs about twice
the forward pass (one matrix multiplication for the input gradient, one for the weight
gradient). Element-wise work (tanh, relu, bias add, clipping) is counted separately and
is negligible against the matrix multiplications.
"""

C = 128          # independent training copies
N = 4            # environments per copy
T = 128          # rollout steps
B = T * N        # 512 stored rows per copy
MB = 128         # rows per minibatch per copy
NMB = 16         # minibatch steps per iteration (4 epochs x 4 minibatches)


def gemm(m, k, n):
    """Floating-point operations of one (m x k) by (k x n) matrix multiplication."""
    return 2 * m * k * n


# ---------------- rollout: 128 sequential steps, actor only (post round-2 hoist) ----------
# per copy, per step, M = N = 4 rows: 4->64, 64->64, 64->2
actor_step = gemm(N, 4, 64) + gemm(N, 64, 64) + gemm(N, 64, 2)
rollout_per_copy = actor_step * T
rollout = rollout_per_copy * C

# ---------------- post-rollout: wide passes over the stored rows ---------------------------
# critic runs on 1024 rows per copy (512 stored observations + 512 next observations,
# concatenated into one call): 4->64, 64->64, then two 64->1 heads
critic_post = gemm(2 * B, 4, 64) + gemm(2 * B, 64, 64) + 2 * gemm(2 * B, 64, 1)
# RND bonus on 512 next observations: target (4->256, 256->128) and predictor
# (4->256, 256->128, 128->128); their first layers run as one packed 4->512 multiplication
rnd_bonus = gemm(B, 4, 512) + gemm(B, 256, 128) + gemm(B, 256, 128) + gemm(B, 128, 128)
# the frozen target is evaluated a second time, on the re-whitened rows, and cached for the
# update stage (the round-2 "target hoist")
target_cache = gemm(B, 4, 256) + gemm(B, 256, 128)
post_per_copy = critic_post + rnd_bonus + target_cache
post = post_per_copy * C

# ---------------- update: 16 minibatch steps of forward and backward ----------------------
# forward per copy per minibatch, M = 128 rows
actor_critic_fwd = (gemm(MB, 4, 128)          # packed first layer: actor 64 + critic 64
                    + gemm(MB, 64, 64)        # actor second layer
                    + gemm(MB, 64, 64)        # critic second layer
                    + gemm(MB, 64, 2)         # actor mean head
                    + gemm(MB, 64, 2))        # packed critic heads (extrinsic + intrinsic)
predictor_fwd = gemm(MB, 4, 256) + gemm(MB, 256, 128) + gemm(MB, 128, 128)
fwd_per_copy_per_mb = actor_critic_fwd + predictor_fwd
bwd_per_copy_per_mb = 2 * fwd_per_copy_per_mb
update_per_copy = (fwd_per_copy_per_mb + bwd_per_copy_per_mb) * NMB
update = update_per_copy * C

total = rollout + post + update

# ---------------- measured times (seconds) -------------------------------------------------
t_torch = 0.020423381298314780      # trainbench_torch_epoch_minibatch_final_styleB, C=128
t_jax = 0.013415080145350658        # trainbench_jax_ppo_lfl_sync, epoch_minibatch, C=128
t_roll = 4973.728179931641e-6       # profile_phases_C128
t_post = 1304.8959970474243e-6
t_upd = 13887.264251708984e-6

# ---------------- hardware ----------------------------------------------------------------
TF32_DENSE = 417.5e12   # 835 TFLOPS datasheet figure is "with sparsity"; dense is half
FP32 = 60e12
BW = 3.939e12           # 3.9 TB/s
SM = 132
CLK = 1.785e9

print("=== arithmetic per iteration, C=128, T=128, N=4, style B ===")
print(f"actor, one rollout step, one copy      : {actor_step:>15,} FLOP")
print(f"rollout, whole iteration               : {rollout:>15,} FLOP")
print(f"post-rollout, whole iteration          : {post:>15,} FLOP")
print(f"  critic on 1024 rows, one copy        : {critic_post:>15,}")
print(f"  RND bonus on 512 rows, one copy      : {rnd_bonus:>15,}")
print(f"  cached target on 512 rows, one copy  : {target_cache:>15,}")
print(f"update, whole iteration                : {update:>15,} FLOP")
print(f"  forward, one copy, one minibatch     : {fwd_per_copy_per_mb:>15,}")
print(f"  forward+backward, one copy, one mb   : {fwd_per_copy_per_mb + bwd_per_copy_per_mb:>15,}")
print(f"TOTAL                                  : {total:>15,} FLOP  = {total/1e9:.1f} GFLOP")
print(f"  shares: rollout {100*rollout/total:.2f}%  post {100*post/total:.2f}%  update {100*update/total:.2f}%")

print("\n=== achieved rate and utilisation ===")
for name, t in [("torch 20.42 ms", t_torch), ("jax 13.42 ms", t_jax)]:
    r = total / t
    print(f"{name}: {r/1e12:6.2f} TFLOP/s "
          f"= {100*r/TF32_DENSE:5.2f}% of TF32 dense peak, {100*r/FP32:5.2f}% of FP32 peak")

print("\n=== per phase (torch, from profile_phases_C128) ===")
for name, f, t in [("rollout", rollout, t_roll), ("post", post, t_post), ("update", update, t_upd)]:
    r = f / t
    print(f"{name:8s}: {f/1e9:7.3f} GFLOP in {t*1e3:6.3f} ms = {r/1e12:6.3f} TFLOP/s "
          f"= {100*r/TF32_DENSE:5.2f}% of TF32 dense")

print("\n=== time each phase's arithmetic would take at peak ===")
for name, f in [("rollout", rollout), ("post", post), ("update", update), ("whole", total)]:
    print(f"{name:8s}: {1e6*f/TF32_DENSE:8.1f} us at TF32 dense, {1e6*f/FP32:8.1f} us at FP32")

print("\n=== single largest matrix multiplication in the iteration ===")
big = gemm(MB, 256, 128) * C
print(f"predictor second layer, all copies, one minibatch: {big/1e9:.3f} GFLOP "
      f"= {1e6*big/TF32_DENSE:.2f} us at TF32 dense peak")
roll_gemm = gemm(N, 64, 64) * C
print(f"largest rollout-step multiplication, all copies  : {roll_gemm/1e6:.3f} MFLOP "
      f"= {1e9*roll_gemm/TF32_DENSE:.1f} ns at TF32 dense peak")

print("\n=== machine fill ===")
wave = SM * 2048
print(f"one full wave of resident threads              : {wave:,}")
print(f"output elements of one rollout-step trunk GEMM : {C*N*64:,} "
      f"({100*C*N*64/wave:.1f}% of a wave at one element per thread)")
print(f"rows per copy in that GEMM                     : {N} (a TF32 tensor-core tile is 16 rows"
      f" -> {16/N:.0f}x of the tile is padding)")
print(f"input width of the first layer                 : 4 (tile depth 8 -> 2x padding)")

print("\n=== memory traffic of the update stage (float32) ===")
params = C * (4*64+64 + 64*64+64 + 64*2+2 + 2          # actor incl. logstd
              + 4*64+64 + 64*64+64 + 64*1+1 + 64*1+1   # critic
              + 4*256+256 + 256*128+128 + 128*128+128) # predictor
gather_floats = C * MB * (4 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 128)
act_floats = C * MB * (128 + 64 + 64 + 256 + 128 + 128)
gather_bytes = 2 * gather_floats * 4
act_bytes = 4 * act_floats * 4          # write in forward, read in forward and backward, gradient
adam_bytes = 7 * params * 4             # read p,g,m,v; write p,m,v
clip_bytes = 3 * params * 4             # read g, write g (scale), zero g
mb_bytes = gather_bytes + act_bytes + adam_bytes + clip_bytes
upd_bytes = mb_bytes * NMB
print(f"trainable parameters, all copies : {params:,} ({params*4/1e6:.1f} MB)")
print(f"per minibatch step: gathers {gather_bytes/1e6:.1f} MB, activations {act_bytes/1e6:.1f} MB,"
      f" Adam {adam_bytes/1e6:.1f} MB, clip {clip_bytes/1e6:.1f} MB -> {mb_bytes/1e6:.1f} MB")
print(f"whole update stage: {upd_bytes/1e9:.3f} GB -> {1e3*upd_bytes/BW:.2f} ms at 3.9 TB/s "
      f"(measured {t_upd*1e3:.2f} ms)")
I = update / upd_bytes
print(f"arithmetic intensity of the update stage: {I:.1f} FLOP per byte")
print(f"balance point of this card: {TF32_DENSE/BW:.0f} FLOP per byte "
      f"-> bandwidth ceiling {BW*I/1e12:.1f} TFLOP/s = {100*BW*I/TF32_DENSE:.1f}% of peak")

print("\n=== operation-count model ===")
anchor_ops, anchor_us = 47, 20112.224578857422/128
print(f"pre-round-2 rollout: {anchor_ops} operations per step, {anchor_us:.1f} us per step "
      f"-> {anchor_us/anchor_ops:.2f} us per operation")
now_us = t_roll*1e6/128
for k in (16, 18, 21):
    print(f"post-round-2 rollout: {now_us:.1f} us per step / {k} operations "
          f"-> {now_us/k:.2f} us per operation")
print("\ntask's stated range, 35-47 operations per step, 2-5 us each:")
for k in (35, 47):
    for l in (2e-6, 3.3e-6, 5e-6):
        print(f"  {k} ops x {l*1e6:.1f} us x 128 steps = {k*l*128*1e3:6.2f} ms of rollout alone")

print("\n=== copy-count scaling: utilisation against C ===")
rows = [(8, 0.013824692799244077), (16, 0.014901482209097594), (32, 0.01567829289706424),
        (64, 0.01721478020772338), (128, 0.02042338129831478), (256, 0.028049297898542137),
        (512, 0.043761171505320814), (1024, 0.07688662139698862)]
for c, t in rows:
    f = total * c / C
    r = f / t
    print(f"C={c:5d}: {t*1e3:7.2f} ms, {f/1e9:8.1f} GFLOP, {r/1e12:6.2f} TFLOP/s, "
          f"{100*r/TF32_DENSE:5.2f}% of TF32 dense, {100*r/FP32:5.2f}% of FP32")

print("\n=== whole-iteration memory traffic and the perfect-fusion floor ===")
roll_w = C * (4*64+64 + 64*64+64 + 64*2+2)      # actor parameters re-read each step
roll_bytes = T * roll_w * 4 + T * C * N * 13 * 4
post_bytes = 4 * C * (2*B*(64+64) + B*(512+128+128+128) + B*(256+128) + B*143)
all_bytes = roll_bytes + post_bytes + upd_bytes
print(f"rollout {roll_bytes/1e6:7.1f} MB, post {post_bytes/1e6:7.1f} MB, update {upd_bytes/1e6:7.1f} MB"
      f"  -> total {all_bytes/1e9:.2f} GB")
print(f"time to move it at 3.9 TB/s            : {1e3*all_bytes/BW:.2f} ms")
print(f"time to do the arithmetic at TF32 dense: {1e3*total/TF32_DENSE:.3f} ms")
print(f"perfect-fusion floor = max of the two  : {1e3*max(all_bytes/BW, total/TF32_DENSE):.2f} ms")
print(f"measured torch 20.42 ms is {t_torch/(all_bytes/BW):.1f}x that floor; jax 13.42 ms is "
      f"{t_jax/(all_bytes/BW):.1f}x")

print("\n=== headroom estimates ===")
ell = 2.2e-6
per_tensor = 21 * 3          # squared-norm reduction, gradient scaling, gradient zeroing
print(f"per-parameter-tensor operations in clip+Adam: {per_tensor} per minibatch step, "
      f"{per_tensor*NMB} per iteration -> {per_tensor*NMB*ell*1e3:.2f} ms at {ell*1e6:.1f} us each")
print(f"  flattening them into one buffer leaves ~3 -> saves {(per_tensor-3)*NMB*ell*1e3:.2f} ms "
      f"({100*(per_tensor-3)*NMB*ell/t_torch:.1f}%)")
gathers = 9 * NMB
print(f"minibatch gathers: {gathers} per iteration -> gathering once leaves 9, saves "
      f"{(gathers-9)*ell*1e3:.2f} ms ({100*(gathers-9)*ell/t_torch:.1f}%)")
print(f"rollout launches removed by a persistent kernel: {128*18} operations = "
      f"{128*18*ell*1e3:.2f} ms of the measured {t_roll*1e3:.2f} ms")
print(f"arithmetic is {1e6*total/TF32_DENSE:.0f} us of the {t_torch*1e6:.0f} us iteration "
      f"({100*(total/TF32_DENSE)/t_torch:.1f}%) -> halving it with bf16 saves at most "
      f"{100*0.5*(total/TF32_DENSE)/t_torch:.1f}%")
t_16384 = 1.079
f_16384 = total * 16384 / C
print(f"C=16384: {f_16384/1e12:.1f} TFLOP in {t_16384*1e3:.0f} ms = {f_16384/t_16384/1e12:.2f} TFLOP/s "
      f"= {100*f_16384/t_16384/TF32_DENSE:.2f}% of TF32 dense, {100*f_16384/t_16384/FP32:.1f}% of FP32")
