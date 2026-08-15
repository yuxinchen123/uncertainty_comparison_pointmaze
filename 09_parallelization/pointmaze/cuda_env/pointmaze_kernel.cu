// Fused CUDA PointMaze: the entire env step (exact MuJoCo-matched dynamics + reward +
// episode ends + auto-reset with the frozen fmix32 reset RNG) in ONE kernel, one thread
// per env. Direct transliteration of pointmaze/torch_env/torch_pointmaze.py; compiled
// with --fmad=false so float32 results match torch eager bit for bit.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "pointmaze_dynamics.h"

// 16B/32B vectorized load-store types per scalar type
template <typename T> struct VecT;
template <> struct VecT<float>  { using v4 = float4;  using v2 = float2;  };
template <> struct VecT<double> { using v4 = double4; using v2 = double2; };

// full env step: dynamics + reward + episode ends + auto-reset, one thread per env.
// state is packed [B, 4] = (x, y, vx, vy), loaded and stored as one vector; the state
// buffer after auto-reset IS the observation, so there is no separate obs output.
template <typename T>
__global__ void step_kernel(
    T* __restrict__ state, T* __restrict__ goal,
    int* __restrict__ step_count, int* __restrict__ reset_count,
    const T* __restrict__ act,
    T* __restrict__ reward,
    bool* __restrict__ terminated, bool* __restrict__ truncated,
    T* __restrict__ final_obs,
    const int* __restrict__ nb_mask,
    int B, int n_envs, int rows, int cols,
    T start_x, T start_y, T goal_cx, T goal_cy,
    T position_noise, T goal_radius_sq, T reward_shift,
    int max_episode_steps, int continuing, uint32_t base_seed) {
  using V4 = typename VecT<T>::v4;
  using V2 = typename VecT<T>::v2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;

  const V4 s = reinterpret_cast<const V4*>(state)[i];
  const V2 a = reinterpret_cast<const V2*>(act)[i];
  const V2 g = reinterpret_cast<const V2*>(goal)[i];
  T px, py, vx, vy;
  dynamics(s.x, s.y, s.z, s.w, a.x, a.y, nb_mask, rows, cols, px, py, vx, vy);
  int sc = step_count[i] + 1;

  // reward and episode ends on the post-step position
  const T ddx = px - g.x;
  const T ddy = py - g.y;
  const bool at_goal = (ddx * ddx + ddy * ddy) <= goal_radius_sq;
  reward[i] = (at_goal ? (T)1.0 : (T)0.0) + reward_shift;
  const bool term = continuing ? false : at_goal;
  const bool trunc = (sc >= max_episode_steps) && !term;
  terminated[i] = term;
  truncated[i] = trunc;

  reinterpret_cast<V4*>(final_obs)[i] = V4{px, py, vx, vy};

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
    reinterpret_cast<V2*>(goal)[i] =
        V2{goal_cx + reset_noise(id_key, (uint32_t)rc, 2u, position_noise),
           goal_cy + reset_noise(id_key, (uint32_t)rc, 3u, position_noise)};
    sc = 0;
  }
  step_count[i] = sc;
  reinterpret_cast<V4*>(state)[i] = V4{px, py, vx, vy};
}

// reset all envs at their CURRENT reset generation (matches TorchPointMaze.reset)
template <typename T>
__global__ void reset_kernel(
    T* __restrict__ state, T* __restrict__ goal,
    int* __restrict__ step_count, const int* __restrict__ reset_count,
    int B, int n_envs, T start_x, T start_y, T goal_cx, T goal_cy,
    T position_noise, uint32_t base_seed) {
  using V4 = typename VecT<T>::v4;
  using V2 = typename VecT<T>::v2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B) return;
  const uint32_t c = (uint32_t)(i / n_envs);
  const uint32_t e = (uint32_t)(i % n_envs);
  const uint32_t id_key = base_seed * 0x9E3779B1u + c * 0x85EBCA77u + e * 0xC2B2AE3Du;
  const uint32_t rc = (uint32_t)reset_count[i];
  reinterpret_cast<V4*>(state)[i] =
      V4{start_x + reset_noise(id_key, rc, 0u, position_noise),
         start_y + reset_noise(id_key, rc, 1u, position_noise), (T)0.0, (T)0.0};
  reinterpret_cast<V2*>(goal)[i] =
      V2{goal_cx + reset_noise(id_key, rc, 2u, position_noise),
         goal_cy + reset_noise(id_key, rc, 3u, position_noise)};
  step_count[i] = 0;
}

// dynamics only (the fixture checker's step_batch contract)
template <typename T>
__global__ void dynamics_kernel(
    const T* __restrict__ pos_in, const T* __restrict__ vel_in, const T* __restrict__ act,
    T* __restrict__ pos_out, T* __restrict__ vel_out,
    const int* __restrict__ nb_mask, int B, int rows, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
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

void env_step(torch::Tensor state, torch::Tensor goal,
              torch::Tensor step_count, torch::Tensor reset_count, torch::Tensor act,
              torch::Tensor reward, torch::Tensor terminated,
              torch::Tensor truncated, torch::Tensor final_obs, torch::Tensor nb_mask,
              long n_envs, long rows, long cols,
              double start_x, double start_y, double goal_cx, double goal_cy,
              double position_noise, double goal_radius_sq, double reward_shift,
              long max_episode_steps, bool continuing, long base_seed, long block) {
  const long B = state.numel() / 4;
  AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "env_step", [&] {
    step_kernel<scalar_t><<<nblocks(B, (int)block), (int)block, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        state.data_ptr<scalar_t>(), goal.data_ptr<scalar_t>(),
        step_count.data_ptr<int>(), reset_count.data_ptr<int>(), act.data_ptr<scalar_t>(),
        reward.data_ptr<scalar_t>(),
        terminated.data_ptr<bool>(), truncated.data_ptr<bool>(),
        final_obs.data_ptr<scalar_t>(), nb_mask.data_ptr<int>(),
        (int)B, (int)n_envs, (int)rows, (int)cols,
        (scalar_t)start_x, (scalar_t)start_y, (scalar_t)goal_cx, (scalar_t)goal_cy,
        (scalar_t)position_noise, (scalar_t)goal_radius_sq, (scalar_t)reward_shift,
        (int)max_episode_steps, continuing ? 1 : 0, (uint32_t)base_seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void env_reset(torch::Tensor state, torch::Tensor goal,
               torch::Tensor step_count, torch::Tensor reset_count,
               long n_envs, double start_x, double start_y, double goal_cx, double goal_cy,
               double position_noise, long base_seed, long block) {
  const long B = state.numel() / 4;
  AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "env_reset", [&] {
    reset_kernel<scalar_t><<<nblocks(B, (int)block), (int)block, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        state.data_ptr<scalar_t>(), goal.data_ptr<scalar_t>(),
        step_count.data_ptr<int>(), reset_count.data_ptr<int>(),
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
  AT_DISPATCH_FLOATING_TYPES(pos_in.scalar_type(), "env_dynamics", [&] {
    dynamics_kernel<scalar_t><<<nblocks(B, (int)block), (int)block, 0,
        at::cuda::getCurrentCUDAStream()>>>(
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
