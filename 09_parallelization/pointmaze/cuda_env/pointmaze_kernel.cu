// Fused CUDA PointMaze: the entire env step (exact MuJoCo-matched dynamics + reward +
// episode ends + auto-reset with the frozen fmix32 reset RNG) in ONE kernel, one thread
// per env. Direct transliteration of pointmaze/torch_env/torch_pointmaze.py; compiled
// with --fmad=false so float32 results match torch eager bit for bit.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAException.h>

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

__device__ __constant__ int DI8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
__device__ __constant__ int DJ8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// clamp matching torch.clamp for finite inputs
template <typename T>
__device__ __forceinline__ T clampv(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// FROZEN reset RNG: two fmix32 rounds, uniform = (h >> 8) * 2^-24 (float32-exact)
__device__ __forceinline__ float hash_uniform(uint32_t x) {
#pragma unroll
  for (int r = 0; r < 2; r++) {
    x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
  }
  return (float)(x >> 8) * (1.0f / 16777216.0f);
}

// one keyed uniform(-noise, +noise) draw (matches TorchPointMaze._reset_noise)
template <typename T>
__device__ __forceinline__ T reset_noise(uint32_t id_key, uint32_t rc, uint32_t q, T noise) {
  uint32_t key = id_key + rc * 0x27D4EB2Fu + q * 0x165667B1u;
  T u = (T)hash_uniform(key);
  return (u * (T)2.0 - (T)1.0) * noise;
}

// exact dynamics: 8 neighbor-box contact candidates, two-smallest tournament, MuJoCo
// solref/solimp force law, exact 2-contact QP, implicit-damping Euler (torch twin line
// by line; expression grouping kept identical for float32 bit-equality)
template <typename T>
__device__ __forceinline__ void dynamics(
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

  // two-smallest-by-distance tournament over the 8 candidates, payload (b, R, sx, sy)
  T d1 = (T)1e9, b1 = (T)-1e30, R1 = (T)0, s1x = (T)0, s1y = (T)0;
  T d2 = (T)1e9, b2 = (T)-1e30, R2 = (T)0, s2x = (T)0, s2y = (T)0;
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

// full env step: dynamics + reward + episode ends + auto-reset, one thread per env
template <typename T>
__global__ void step_kernel(
    T* __restrict__ pos, T* __restrict__ vel, T* __restrict__ goal,
    int* __restrict__ step_count, int* __restrict__ reset_count,
    const T* __restrict__ act,
    T* __restrict__ obs, T* __restrict__ reward,
    bool* __restrict__ terminated, bool* __restrict__ truncated,
    T* __restrict__ final_obs,
    const int* __restrict__ nb_mask,
    int B, int n_envs, int rows, int cols,
    T start_x, T start_y, T goal_cx, T goal_cy,
    T position_noise, T goal_radius_sq, T reward_shift,
    int max_episode_steps, int continuing, uint32_t base_seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;

  T px, py, vx, vy;
  dynamics(pos[2 * i], pos[2 * i + 1], vel[2 * i], vel[2 * i + 1],
           act[2 * i], act[2 * i + 1], nb_mask, rows, cols, px, py, vx, vy);
  int sc = step_count[i] + 1;

  // reward and episode ends on the post-step position
  const T ddx = px - goal[2 * i];
  const T ddy = py - goal[2 * i + 1];
  const bool at_goal = (ddx * ddx + ddy * ddy) <= goal_radius_sq;
  reward[i] = (at_goal ? (T)1.0 : (T)0.0) + reward_shift;
  const bool term = continuing ? false : at_goal;
  const bool trunc = (sc >= max_episode_steps) && !term;
  terminated[i] = term;
  truncated[i] = trunc;

  final_obs[4 * i] = px;
  final_obs[4 * i + 1] = py;
  final_obs[4 * i + 2] = vx;
  final_obs[4 * i + 3] = vy;

  // auto-reset (same keys and draws as the torch twin's branch-free where-select)
  if (term || trunc) {
    const int rc = reset_count[i] + 1;
    reset_count[i] = rc;
    const uint32_t c = (uint32_t)(i / n_envs);
    const uint32_t e = (uint32_t)(i % n_envs);
    const uint32_t id_key = base_seed * 0x9E3779B1u + c * 0x85EBCA77u + e * 0xC2B2AE3Du;
    px = start_x + reset_noise(id_key, (uint32_t)rc, 0u, position_noise);
    py = start_y + reset_noise(id_key, (uint32_t)rc, 1u, position_noise);
    vx = (T)0.0;
    vy = (T)0.0;
    goal[2 * i] = goal_cx + reset_noise(id_key, (uint32_t)rc, 2u, position_noise);
    goal[2 * i + 1] = goal_cy + reset_noise(id_key, (uint32_t)rc, 3u, position_noise);
    sc = 0;
  }
  step_count[i] = sc;
  pos[2 * i] = px;
  pos[2 * i + 1] = py;
  vel[2 * i] = vx;
  vel[2 * i + 1] = vy;
  obs[4 * i] = px;
  obs[4 * i + 1] = py;
  obs[4 * i + 2] = vx;
  obs[4 * i + 3] = vy;
}

// reset all envs at their CURRENT reset generation (matches TorchPointMaze.reset)
template <typename T>
__global__ void reset_kernel(
    T* __restrict__ pos, T* __restrict__ vel, T* __restrict__ goal,
    int* __restrict__ step_count, const int* __restrict__ reset_count,
    T* __restrict__ obs,
    int B, int n_envs, T start_x, T start_y, T goal_cx, T goal_cy,
    T position_noise, uint32_t base_seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;
  const uint32_t c = (uint32_t)(i / n_envs);
  const uint32_t e = (uint32_t)(i % n_envs);
  const uint32_t id_key = base_seed * 0x9E3779B1u + c * 0x85EBCA77u + e * 0xC2B2AE3Du;
  const uint32_t rc = (uint32_t)reset_count[i];
  const T px = start_x + reset_noise(id_key, rc, 0u, position_noise);
  const T py = start_y + reset_noise(id_key, rc, 1u, position_noise);
  pos[2 * i] = px;
  pos[2 * i + 1] = py;
  vel[2 * i] = (T)0.0;
  vel[2 * i + 1] = (T)0.0;
  goal[2 * i] = goal_cx + reset_noise(id_key, rc, 2u, position_noise);
  goal[2 * i + 1] = goal_cy + reset_noise(id_key, rc, 3u, position_noise);
  step_count[i] = 0;
  obs[4 * i] = px;
  obs[4 * i + 1] = py;
  obs[4 * i + 2] = 0;
  obs[4 * i + 3] = 0;
}

// dynamics only (the fixture checker's step_batch contract)
template <typename T>
__global__ void dynamics_kernel(
    const T* __restrict__ pos_in, const T* __restrict__ vel_in, const T* __restrict__ act,
    T* __restrict__ pos_out, T* __restrict__ vel_out,
    const int* __restrict__ nb_mask, int B, int rows, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) printf("DEVICE dynamics_kernel B=%d rows=%d cols=%d\n", B, rows, cols);
  if (i >= B) return;
  T px, py, vx, vy;
  dynamics(pos_in[2 * i], pos_in[2 * i + 1], vel_in[2 * i], vel_in[2 * i + 1],
           act[2 * i], act[2 * i + 1], nb_mask, rows, cols, px, py, vx, vy);
  pos_out[2 * i] = px;
  pos_out[2 * i + 1] = py;
  vel_out[2 * i] = vx;
  vel_out[2 * i + 1] = vy;
}

static inline int nblocks(long B, int t) { return (int)((B + t - 1) / t); }

void env_step(torch::Tensor pos, torch::Tensor vel, torch::Tensor goal,
              torch::Tensor step_count, torch::Tensor reset_count, torch::Tensor act,
              torch::Tensor obs, torch::Tensor reward, torch::Tensor terminated,
              torch::Tensor truncated, torch::Tensor final_obs, torch::Tensor nb_mask,
              long n_envs, long rows, long cols,
              double start_x, double start_y, double goal_cx, double goal_cy,
              double position_noise, double goal_radius_sq, double reward_shift,
              long max_episode_steps, bool continuing, long base_seed, long block) {
  const long B = pos.numel() / 2;
  AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "env_step", [&] {
    step_kernel<scalar_t><<<nblocks(B, (int)block), (int)block>>>(
        pos.data_ptr<scalar_t>(), vel.data_ptr<scalar_t>(), goal.data_ptr<scalar_t>(),
        step_count.data_ptr<int>(), reset_count.data_ptr<int>(), act.data_ptr<scalar_t>(),
        obs.data_ptr<scalar_t>(), reward.data_ptr<scalar_t>(),
        terminated.data_ptr<bool>(), truncated.data_ptr<bool>(),
        final_obs.data_ptr<scalar_t>(), nb_mask.data_ptr<int>(),
        (int)B, (int)n_envs, (int)rows, (int)cols,
        (scalar_t)start_x, (scalar_t)start_y, (scalar_t)goal_cx, (scalar_t)goal_cy,
        (scalar_t)position_noise, (scalar_t)goal_radius_sq, (scalar_t)reward_shift,
        (int)max_episode_steps, continuing ? 1 : 0, (uint32_t)base_seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void env_reset(torch::Tensor pos, torch::Tensor vel, torch::Tensor goal,
               torch::Tensor step_count, torch::Tensor reset_count, torch::Tensor obs,
               long n_envs, double start_x, double start_y, double goal_cx, double goal_cy,
               double position_noise, long base_seed, long block) {
  const long B = pos.numel() / 2;
  AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "env_reset", [&] {
    reset_kernel<scalar_t><<<nblocks(B, (int)block), (int)block>>>(
        pos.data_ptr<scalar_t>(), vel.data_ptr<scalar_t>(), goal.data_ptr<scalar_t>(),
        step_count.data_ptr<int>(), reset_count.data_ptr<int>(), obs.data_ptr<scalar_t>(),
        (int)B, (int)n_envs, (scalar_t)start_x, (scalar_t)start_y,
        (scalar_t)goal_cx, (scalar_t)goal_cy, (scalar_t)position_noise,
        (uint32_t)base_seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void env_dynamics(torch::Tensor pos_in, torch::Tensor vel_in, torch::Tensor act,
                  torch::Tensor pos_out, torch::Tensor vel_out, torch::Tensor nb_mask,
                  long rows, long cols, long block) {
  const long B = pos_in.numel() / 2;
  printf("HOST env_dynamics B=%ld rows=%ld cols=%ld block=%ld\n", B, rows, cols, block);
  AT_DISPATCH_FLOATING_TYPES(pos_in.scalar_type(), "env_dynamics", [&] {
    dynamics_kernel<scalar_t><<<nblocks(B, (int)block), (int)block>>>(
        pos_in.data_ptr<scalar_t>(), vel_in.data_ptr<scalar_t>(), act.data_ptr<scalar_t>(),
        pos_out.data_ptr<scalar_t>(), vel_out.data_ptr<scalar_t>(),
        nb_mask.data_ptr<int>(), (int)B, (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("env_step", &env_step, "fused PointMaze step");
  m.def("env_reset", &env_reset, "PointMaze reset");
  m.def("env_dynamics", &env_dynamics, "PointMaze dynamics only");
}
