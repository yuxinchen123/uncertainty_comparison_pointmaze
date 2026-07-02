"""Optional, bit-exact performance patches for Stable-Baselines3 internals.

Currently: a torch._foreach_ replacement for SB3's Polyak (soft) target update, which otherwise spends
pure-Python time in `zip_strict` iterating the critic parameters one-by-one every gradient step. The
foreach version is the SAME two element-wise ops with the same scalars in the same per-element order, so
it is bit-identical to SB3's loop. Installed only when the perf switch is on (train.py build_sac).
"""
import torch as th
import stable_baselines3.sac.sac as _sacmod


def polyak_update_foreach(params, target_params, tau: float) -> None:
    """Bit-exact soft target update via torch._foreach_ (drop-in for SB3 common.utils.polyak_update).

    target = (1-tau)*target + tau*param, for each (param, target) pair. before: SB3 loops zip_strict and
    does target.data.mul_(1-tau); th.add(target.data, param.data, alpha=tau, out=...) per parameter
    (14 Python resumptions/step for the 12-param critic + the empty batch_norm list). after: two fused C++
    foreach kernels over the whole parameter list.
    """
    with th.no_grad():
        # materialize the generators; foreach kernels need concrete tensor lists
        t = list(target_params)
        s = list(params)
        # SAC also calls polyak_update on the (empty for MlpPolicy) batch_norm_stats list; foreach raises
        # on an empty list, so guard it -- an empty update is a no-op anyway.
        if not t:
            return
        # same math as the SB3 loop: target *= (1-tau); target += tau*param  -- in the same element order
        th._foreach_mul_(t, 1.0 - tau)
        th._foreach_add_(t, s, alpha=tau)


def patch_polyak_foreach() -> None:
    """Rebind the name SAC actually calls. sac.sac does `from ...common.utils import polyak_update`, binding
    the reference at import time, so we must patch `stable_baselines3.sac.sac.polyak_update` (not common.utils)."""
    _sacmod.polyak_update = polyak_update_foreach
