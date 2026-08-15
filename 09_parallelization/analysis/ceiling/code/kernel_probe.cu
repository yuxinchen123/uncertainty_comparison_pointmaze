// Instruction-count probe: the fused PointMaze step kernel, copied verbatim from
// pointmaze/cuda_env/pointmaze_kernel.cu with the torch headers and the host wrappers
// removed (the kernel body already uses raw pointers, so the device code is identical).
// Compiled to PTX and to a cubin only to count instructions and registers; never run.
#include <cstdint>
#include "../../../pointmaze/cuda_env/pointmaze_dynamics.h"

template <typename T> struct VecT;
template <> struct VecT<float>  { using v4 = float4;  using v2 = float2;  };
template <> struct VecT<double> { using v4 = double4; using v2 = double2; };

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

  const T ddx = px - g.x;
  const T ddy = py - g.y;
  const bool at_goal = (ddx * ddx + ddy * ddy) <= goal_radius_sq;
  reward[i] = (at_goal ? (T)1.0 : (T)0.0) + reward_shift;
  const bool term = continuing ? false : at_goal;
  const bool trunc = (sc >= max_episode_steps) && !term;
  terminated[i] = term;
  truncated[i] = trunc;

  reinterpret_cast<V4*>(final_obs)[i] = V4{px, py, vx, vy};

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

// force one float instantiation so the compiler emits the code
template __global__ void step_kernel<float>(
    float*, float*, int*, int*, const float*, float*, bool*, bool*, float*, const int*,
    int, int, int, int, float, float, float, float, float, float, float,
    int, int, uint32_t);
