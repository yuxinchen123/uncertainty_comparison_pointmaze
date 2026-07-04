"""Confirm optuna's installed Matern 5/2 kernel equals the textbook closed form. Self-contained."""
import numpy as np
import torch
from optuna._gp.gp import Matern52Kernel

# Optuna's kernel takes the SQUARED scaled distance t = (r/L)^2 (dims summed after ARD scaling).
t = np.array([0.0, 0.01, 0.25, 1.0, 2.25, 4.0, 9.0], dtype=np.float64)

# Installed autograd kernel: exp(-sqrt5d)*(1 + sqrt5d + sqrt5d^2/3), sqrt5d = sqrt(5 t).
installed = Matern52Kernel.apply(torch.from_numpy(t)).detach().numpy()

# Textbook closed form k(r)/s2 = (1 + sqrt5 r/L + 5 r^2/3L^2) exp(-sqrt5 r/L) with s=sqrt5 r/L=sqrt(5t).
s = np.sqrt(5.0 * t)
closed = (1.0 + s + s * s / 3.0) * np.exp(-s)

print(f"{'t=(r/L)^2':>12}{'installed':>14}{'closed_form':>14}")
for ti, a, c in zip(t, installed, closed):
    print(f"{ti:12.4f}{a:14.10f}{c:14.10f}")
print("max abs diff = %.2e ; matches:" % np.max(np.abs(installed - closed)),
      bool(np.allclose(installed, closed, atol=1e-12)))
