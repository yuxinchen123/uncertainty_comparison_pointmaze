// Shared exact-dynamics math for the fused CUDA env; compiles as CUDA device code
// (PM_DEV = __device__) or as plain host C++ (PM_DEV = inline) for the CPU port test.
#pragma once
#include <cstdint>
// physics constants (pm_common.py, probe-verified); doubles here, cast to scalar_t per use
#define K_H       0.01
#define K_M       4.1887902047863905
#define K_D       1.0
#define K_G       100.0
#define K_R       0.1
#define K_VCLIP   5.0
#define K_ACLIP   1.0
#define K_INVMHD  (1.0 / (K_M + K_H * K_D))
#define K_MARGIN  0.002
#define K_SD0     0.9
#define K_SDMAX   0.95
#define K_SWIDTH  0.001
#define K_SMID    0.5
#define K_BREF    (2.0 / (K_SDMAX * 0.02))
#define K_KOD     (1.0 / (K_SDMAX * K_SDMAX * 0.02 * 0.02))

#ifdef __CUDACC__
#define PM_DEV __device__ __forceinline__
__constant__ int DI8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
__constant__ int DJ8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
#else
#define PM_DEV inline
#define __restrict__
static const int DI8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int DJ8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
#include <cmath>
#include <cstdio>
using std::sqrt; using std::fabs;
#endif

// clamp matching torch.clamp for finite inputs
template <typename T>
PM_DEV T clampv(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// FROZEN reset RNG: two fmix32 rounds, uniform = (h >> 8) * 2^-24 (float32-exact)
PM_DEV float hash_uniform(uint32_t x) {
#pragma unroll
  for (int r = 0; r < 2; r++) {
    x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
  }
  return (float)(x >> 8) * (1.0f / 16777216.0f);
}

// one keyed uniform(-noise, +noise) draw (matches TorchPointMaze._reset_noise)
template <typename T>
PM_DEV T reset_noise(uint32_t id_key, uint32_t rc, uint32_t q, T noise) {
  uint32_t key = id_key + rc * 0x27D4EB2Fu + q * 0x165667B1u;
  T u = (T)hash_uniform(key);
  return (u * (T)2.0 - (T)1.0) * noise;
}

// exact dynamics: 8 neighbor-box contact candidates, two-smallest tournament, MuJoCo
// solref/solimp force law, exact 2-contact QP, implicit-damping Euler (torch twin line
// by line; expression grouping kept identical for float32 bit-equality)
template <typename T>
PM_DEV void dynamics(
    T px, T py, T vx_in, T vy_in, T ax, T ay,
    const int* __restrict__ nb_mask, int rows, int cols,
    T& px_o, T& py_o, T& vx_o, T& vy_o) {
  const T aX = clampv(ax, (T)(-K_ACLIP), (T)K_ACLIP);
  const T aY = clampv(ay, (T)(-K_ACLIP), (T)K_ACLIP);
  const T vx = clampv(vx_in, (T)(-K_VCLIP), (T)K_VCLIP);
  const T vy = clampv(vy_in, (T)(-K_VCLIP), (T)K_VCLIP);
  const T Fx0 = (T)K_G * aX;
  const T Fy0 = (T)K_G * aY;
  const T aux = (Fx0 - (T)K_D * vx) / (T)K_M;
  const T auy = (Fy0 - (T)K_D * vy) / (T)K_M;

  // cell of the current center
  long jc = (long)(px + (T)(cols / 2.0));
  long ic = (long)((T)(rows / 2.0) - py);
  jc = jc < 0 ? 0 : (jc > cols - 1 ? cols - 1 : jc);
  ic = ic < 0 ? 0 : (ic > rows - 1 ? rows - 1 : ic);
  const int mask = nb_mask[ic * cols + jc];
  const T fi = (T)ic;
  const T fj = (T)jc;

  // Pick the two nearest wall boxes, then evaluate the contact law for those two only.
  //
  // Two forms of the same selection are kept. PM_RANK_TOURNAMENT computes the eight distances
  // independently, ranks each by counting how many candidates precede it, and recomputes the
  // geometry and the law for the two winners; the older form carries the winners' payload
  // through a running two-slot tournament, which makes each candidate wait for the previous
  // candidate's comparison and keeps ten values live across the loop. Both orders are "sort by
  // distance, ties to the lower candidate index", so they select the same two candidates: the
  // running form takes a candidate only on a STRICT improvement, and the counting form breaks
  // an exact tie with (j < k). The two builds are compared against each other by benchmark.
  T d1, b1 = (T)-1e30, R1 = (T)0, s1x = (T)0, s1y = (T)0;
  T d2, b2 = (T)-1e30, R2 = (T)0, s2x = (T)0, s2y = (T)0;
#ifdef PM_RANK_TOURNAMENT
  // pass 1: the eight distances, independent of each other
  T dists[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const bool is_wall = (mask & (1 << k)) != 0;
    const T xl = (fj + (T)DJ8[k]) - (T)(cols / 2.0);
    const T yb = (T)(rows / 2.0) - (fi + (T)DI8[k] + (T)1.0);
    const T nx = clampv(px, xl, xl + (T)1.0);
    const T ny = clampv(py, yb, yb + (T)1.0);
    const T dx = px - nx;
    const T dy = py - ny;
    T dsq = dx * dx + dy * dy;
    dsq = dsq < (T)1e-24 ? (T)1e-24 : dsq;
    dists[k] = is_wall ? sqrt(dsq) - (T)K_R : (T)1e9;
  }
  // pass 2: rank by counting the candidates that precede each one (stable: ties to lower index)
  int k1 = 0, k2 = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      rank += (dists[j] < dists[k] || (dists[j] == dists[k] && j < k)) ? 1 : 0;
    }
    k1 = (rank == 0) ? k : k1;
    k2 = (rank == 1) ? k : k2;
  }
  // pass 3: geometry and contact law for the two winners only, expression for expression the
  // same as the running form computes for the same candidate
  d1 = dists[k1];
  d2 = dists[k2];
#pragma unroll
  for (int w = 0; w < 2; w++) {
    const int kw = (w == 0) ? k1 : k2;
    int di = 0, dj = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      di = (k == kw) ? DI8[k] : di;
      dj = (k == kw) ? DJ8[k] : dj;
    }
    const T xl = (fj + (T)dj) - (T)(cols / 2.0);
    const T yb = (T)(rows / 2.0) - (fi + (T)di + (T)1.0);
    const T nx = clampv(px, xl, xl + (T)1.0);
    const T ny = clampv(py, yb, yb + (T)1.0);
    const T dx = px - nx;
    const T dy = py - ny;
    T dsq = dx * dx + dy * dy;
    dsq = dsq < (T)1e-24 ? (T)1e-24 : dsq;
    const T dn = sqrt(dsq);
    const T dist = (w == 0) ? d1 : d2;
    const T sx = dx / dn;
    const T sy = dy / dn;
    const bool active = dist <= (T)K_MARGIN;
    const T r = dist - (T)K_MARGIN;
    T xs = fabs(r) / (T)K_SWIDTH;
    xs = clampv(xs, (T)0.0, (T)1.0);
    const T ys = xs < (T)K_SMID ? xs * xs / (T)K_SMID
                                : (T)1.0 - ((T)1.0 - xs) * ((T)1.0 - xs) / (T)K_SMID;
    const T imp = (T)K_SD0 + (T)(K_SDMAX - K_SD0) * ys;
    T b_qp = (T)(-K_BREF) * (vx * sx + vy * sy) - (imp * (T)K_KOD) * r - (aux * sx + auy * sy);
    b_qp = active ? b_qp : (T)-1e30;
    const T Rreg = ((T)1.0 - imp) / (imp * (T)K_M);
    // A winner that is not a wall box means the maze cell has fewer than two wall neighbours.
    // The running form never lets such a candidate enter a slot (its distance is the same 1e9
    // the slot starts at, and entry needs a strict improvement), so the slot keeps its initial
    // payload. Reproduce that exactly rather than relying on the forces cancelling later.
    const bool w_is_wall = (mask & (1 << kw)) != 0;
    const T bw = w_is_wall ? b_qp : (T)-1e30;
    const T Rw = w_is_wall ? Rreg : (T)0;
    const T sxw = w_is_wall ? sx : (T)0;
    const T syw = w_is_wall ? sy : (T)0;
    if (w == 0) { b1 = bw; R1 = Rw; s1x = sxw; s1y = syw; }
    else        { b2 = bw; R2 = Rw; s2x = sxw; s2y = syw; }
  }
#else
  d1 = (T)1e9;
  d2 = (T)1e9;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const bool is_wall = (mask & (1 << k)) != 0;
    const T xl = (fj + (T)DJ8[k]) - (T)(cols / 2.0);
    const T yb = (T)(rows / 2.0) - (fi + (T)DI8[k] + (T)1.0);
    const T nx = clampv(px, xl, xl + (T)1.0);
    const T ny = clampv(py, yb, yb + (T)1.0);
    const T dx = px - nx;
    const T dy = py - ny;
    T dsq = dx * dx + dy * dy;
    dsq = dsq < (T)1e-24 ? (T)1e-24 : dsq;
    const T dn = sqrt(dsq);
    const T dist = is_wall ? dn - (T)K_R : (T)1e9;
    const T sx = dx / dn;
    const T sy = dy / dn;
    const bool active = dist <= (T)K_MARGIN;
    const T r = dist - (T)K_MARGIN;
    T xs = fabs(r) / (T)K_SWIDTH;
    xs = clampv(xs, (T)0.0, (T)1.0);
    const T ys = xs < (T)K_SMID ? xs * xs / (T)K_SMID
                                : (T)1.0 - ((T)1.0 - xs) * ((T)1.0 - xs) / (T)K_SMID;
    const T imp = (T)K_SD0 + (T)(K_SDMAX - K_SD0) * ys;
    T b_qp = (T)(-K_BREF) * (vx * sx + vy * sy) - (imp * (T)K_KOD) * r - (aux * sx + auy * sy);
    b_qp = active ? b_qp : (T)-1e30;
    const T Rreg = ((T)1.0 - imp) / (imp * (T)K_M);
    const bool take1 = dist < d1;
    const bool take2 = (!take1) && (dist < d2);
    d2 = take1 ? d1 : (take2 ? dist : d2);
    b2 = take1 ? b1 : (take2 ? b_qp : b2);
    R2 = take1 ? R1 : (take2 ? Rreg : R2);
    s2x = take1 ? s1x : (take2 ? sx : s2x);
    s2y = take1 ? s1y : (take2 ? sy : s2y);
    d1 = take1 ? dist : d1;
    b1 = take1 ? b_qp : b1;
    R1 = take1 ? Rreg : R1;
    s1x = take1 ? sx : s1x;
    s1y = take1 ? sy : s1y;
  }
#endif

  // exact 2-contact QP by case enumeration
  const T A11 = (T)(1.0 / K_M);
  const T A12 = (s1x * s2x + s1y * s2y) / (T)K_M;
  const T dd1 = A11 + R1;
  const T dd2 = A11 + R2;
  T f1_single = b1 / dd1;
  f1_single = f1_single < (T)0.0 ? (T)0.0 : f1_single;
  T f2_single = b2 / dd2;
  f2_single = f2_single < (T)0.0 ? (T)0.0 : f2_single;
  const T det = dd1 * dd2 - A12 * A12;
  const T f1_joint = (dd2 * b1 - A12 * b2) / det;
  const T f2_joint = (dd1 * b2 - A12 * b1) / det;
  const bool use_joint = (f1_joint >= (T)0.0) && (f2_joint >= (T)0.0);
  const bool case1 = (b1 >= (T)0.0) && (A12 * f1_single >= b2);
  const T f1 = use_joint ? f1_joint : (case1 ? f1_single : (T)0.0);
  const T f2 = use_joint ? f2_joint : (case1 ? (T)0.0 : f2_single);

  const T Fx = Fx0 + f1 * s1x + f2 * s2x;
  const T Fy = Fy0 + f1 * s1y + f2 * s2y;
  vx_o = ((T)K_M * vx + (T)K_H * Fx) * (T)K_INVMHD;
  vy_o = ((T)K_M * vy + (T)K_H * Fy) * (T)K_INVMHD;
  px_o = px + (T)K_H * vx_o;
  py_o = py + (T)K_H * vy_o;
}

