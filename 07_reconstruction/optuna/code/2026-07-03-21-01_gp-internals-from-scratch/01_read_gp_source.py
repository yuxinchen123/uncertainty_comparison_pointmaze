"""Read the installed optuna 4.9.0 GPSampler source and verify the Matern 5/2 kernel numerically.

This script quotes exact source/docstring lines for: the kernel, input/objective normalization,
kernel-hyperparameter fitting, the single-objective acquisition function, and the acquisition
optimizer. It then checks that the installed Matern52Kernel forward matches the closed form
    k(r)/s2 = (1 + sqrt(5) r/L + 5 r^2/(3 L^2)) * exp(-sqrt(5) r/L).
"""

import inspect
import textwrap

import numpy as np
import torch

import optuna
from optuna._gp import acqf as acqf_mod
from optuna._gp import gp as gp_mod
from optuna._gp import optim_mixed as optim_mod
from optuna._gp import prior as prior_mod
from optuna._gp import search_space as ss_mod
from optuna.samplers._gp import sampler as sampler_mod


def banner(title):
    """Print a labeled separator so each quoted block is easy to find in the output."""
    # Print a fixed-width header line for the given section title.
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_source(obj, header):
    """Print the exact installed source of a function/class, labeled by header."""
    # Fetch the source text straight from the installed file and print it verbatim.
    banner(header)
    print(inspect.getsource(obj))


# Report versions so the reader knows exactly what was observed.
banner("VERSIONS")
print("optuna", optuna.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)

# Quote the GPSampler class docstring: names the kernel, the fitting objective, the acqf, and deps.
banner("GPSampler CLASS DOCSTRING (verbatim)")
print(inspect.getdoc(sampler_mod.GPSampler))

# The Matern 5/2 kernel is a torch.autograd.Function; quote it and its forward docstring.
show_source(gp_mod.Matern52Kernel, "KERNEL: optuna._gp.gp.Matern52Kernel (verbatim)")

# The kernel() method shows how squared distances are scaled by inverse squared lengthscales (ARD).
show_source(gp_mod.GPRegressor.kernel, "KERNEL APPLY: optuna._gp.gp.GPRegressor.kernel (verbatim)")

# Objective standardization (zero mean, unit std) before fitting.
show_source(sampler_mod._standardize_values,
            "OBJECTIVE NORMALIZATION: optuna.samplers._gp.sampler._standardize_values (verbatim)")

# Input normalization of one numeric parameter to the unit interval (log params take log first).
show_source(ss_mod._normalize_one_param,
            "INPUT NORMALIZATION: optuna._gp.search_space._normalize_one_param (verbatim)")

# Kernel-hyperparameter fitting: maximize marginal log-likelihood + log prior via l-bfgs-b.
show_source(gp_mod.GPRegressor._fit_kernel_params,
            "HYPERPARAMETER FITTING: optuna._gp.gp.GPRegressor._fit_kernel_params (verbatim)")
show_source(gp_mod.GPRegressor.marginal_log_likelihood,
            "MARGINAL LOG LIKELIHOOD: optuna._gp.gp.GPRegressor.marginal_log_likelihood (verbatim)")
show_source(prior_mod.default_log_prior,
            "PRIOR ON HYPERPARAMETERS: optuna._gp.prior.default_log_prior (verbatim)")
print("DEFAULT_MINIMUM_NOISE_VAR =", prior_mod.DEFAULT_MINIMUM_NOISE_VAR)

# Single-objective acquisition function: log expected improvement (LogEI).
show_source(acqf_mod.logei, "ACQUISITION (single-objective): optuna._gp.acqf.logei (verbatim)")
show_source(acqf_mod.standard_logei,
            "ACQUISITION helper: optuna._gp.acqf.standard_logei (verbatim)")
show_source(acqf_mod.LogEI, "ACQUISITION class: optuna._gp.acqf.LogEI (verbatim)")

# Acquisition optimizer: QMC preliminary samples, roulette pick, then local search (l-bfgs-b).
show_source(optim_mod.optimize_acqf_mixed,
            "ACQ OPTIMIZER: optuna._gp.optim_mixed.optimize_acqf_mixed (verbatim)")

# The posterior mean/variance formulas, quoted from the method docstring.
banner("POSTERIOR FORMULAS: optuna._gp.gp.GPRegressor.posterior docstring (verbatim)")
print(inspect.getdoc(gp_mod.GPRegressor.posterior))


def matern52_closed_form(scaled_sq_dist):
    """Closed-form Matern 5/2 correlation as a function of scaled squared distance t=(r/L)^2."""
    # Build sqrt(5)*(r/L) = sqrt(5 t), then evaluate (1 + s + s^2/3) exp(-s) with s=sqrt(5)(r/L).
    s = np.sqrt(5.0 * scaled_sq_dist)
    return (1.0 + s + (s * s) / 3.0) * np.exp(-s)


# Numerically confirm the installed kernel forward equals the closed form for several distances.
banner("NUMERICAL CHECK: installed Matern52Kernel.forward vs closed form")
t_vals = np.array([0.0, 0.01, 0.25, 1.0, 2.25, 4.0, 9.0], dtype=np.float64)
installed = gp_mod.Matern52Kernel.apply(torch.from_numpy(t_vals)).detach().numpy()
closed = matern52_closed_form(t_vals)
print(f"{'t=(r/L)^2':>12} {'installed':>14} {'closed_form':>14} {'abs_diff':>12}")
for t, a, b in zip(t_vals, installed, closed):
    # One row per test distance: installed forward, hand-coded closed form, absolute difference.
    print(f"{t:12.4f} {a:14.10f} {b:14.10f} {abs(a - b):12.2e}")
print("max abs diff:", float(np.max(np.abs(installed - closed))))
print("Matches closed form:", bool(np.allclose(installed, closed, atol=1e-12)))

# Show the default acquisition-optimizer control constants set in the sampler constructor.
banner("GPSampler default acquisition-optimizer settings (observed on an instance)")
gps = optuna.samplers.GPSampler(seed=0)
print("n_startup_trials        =", gps._n_startup_trials)
print("n_preliminary_samples   =", gps._n_preliminary_samples, "(QMC candidates per ask)")
print("n_local_search          =", gps._n_local_search, "(local-search restarts per ask)")
print("tol                     =", gps._tol)
print("minimum_noise (var)     =", gps._minimum_noise)
print("deterministic_objective =", gps._deterministic)
