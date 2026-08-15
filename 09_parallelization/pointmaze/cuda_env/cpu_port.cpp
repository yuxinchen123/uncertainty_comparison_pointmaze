// CPU port of the kernel dynamics: validates the transliteration without a GPU.
#include "pointmaze_dynamics.h"
extern "C" void dynamics_batch_f64(const double* pos, const double* vel, const double* act,
                                   double* pos_out, double* vel_out, const int* nb_mask,
                                   int B, int rows, int cols) {
  for (int i = 0; i < B; i++) {
    dynamics<double>(pos[2*i], pos[2*i+1], vel[2*i], vel[2*i+1], act[2*i], act[2*i+1],
                     nb_mask, rows, cols, pos_out[2*i], pos_out[2*i+1],
                     vel_out[2*i], vel_out[2*i+1]);
  }
}
