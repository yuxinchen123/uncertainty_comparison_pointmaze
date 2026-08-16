#!/usr/bin/env python
"""Fused RND: one predictor+target forward per gradient step instead of two.

DEFERRED — not wired into patched_entry.py and not applied to any run. See README.md: the fusion
must pick one of the two observation-normalizations that the split code uses (bonus-time vs
update-time), so the predictor it trains differs from today's by a one-batch lag in the observation
statistics. That is a fresh-baseline change, not a free one.

Implements choice (a): both the bonus and the loss use the BONUS-time normalization (the statistics
as they stood before this batch), and the statistics are updated afterwards so the next step sees
them — the same sequence of statistics, applied one batch later.

Enable in a future experiment with:
    from fused_rnd import patch_fused_rnd; patch_fused_rnd()
"""
import torch

from rnd_exploration.methods.rnd import RND


def _fused_compute_and_update(self, samples):
    """Bonus + predictor step from ONE forward pass. Returns the bonus, shape (batch_size,).

    before: compute() forwards predictor+target (no grad) and update() forwards them again (grad),
            each on its own normalization of the same batch
    after:  one forward with grad; the bonus is its detached readout, the loss its differentiable
            one; the observation statistics advance after the step
    """
    x = self._get_feature_tensor(samples)
    xn = self._normalize_obs(x)                      # pre-update statistics, as compute() uses today

    src = self.predictor(xn)
    if self.n_predictors == 1:
        src = src.unsqueeze(1)
    with torch.no_grad():
        tgt = self.target(xn)
    dist = self._dist_ensemble(src, tgt)             # B_mse = 0.5*||e||^2 per predictor

    # ---- the bonus: the readout the live code computes in _raw_bonus, off the same tensors ----
    with torch.no_grad():
        readout = dist.detach()
        if self.bonus_readout == "l2":
            readout = (2.0 * readout).clamp(min=1e-8).sqrt()
            if self.readout_norm_init:
                init_src = self.init_predictor(xn)
                if self.n_predictors == 1:
                    init_src = init_src.unsqueeze(1)
                init_l2 = (2.0 * self._dist_ensemble(init_src, tgt)).clamp(min=1e-8).sqrt()
                readout = readout / (init_l2 + self.readout_norm_eps)
        elif self.bonus_readout == "mse_mean":
            readout = (2.0 / self.output_dim) * readout
        mean_dist = readout.mean(dim=1)
        std_dist = (torch.zeros_like(mean_dist) if self.n_predictors == 1
                    else readout.std(dim=1, unbiased=True))
        bonus = mean_dist + self.beta_std * std_dist
        if self.reward_norm:
            import numpy as np
            bonus = bonus / float(np.sqrt(self.reward_rms.var))

    # ---- the predictor step: the loss update() builds, from the forward already done ----
    self.opt.zero_grad()
    if self.predictor_loss == "mse_init_normalized":
        with torch.no_grad():
            init_src = self.init_predictor(xn)
            if self.n_predictors == 1:
                init_src = init_src.unsqueeze(1)
            init_sq = 2.0 * self._dist_ensemble(init_src, tgt)
        per_sample = (2.0 * dist) / (init_sq + self.predictor_loss_delta)
        loss = per_sample.mean() if self.update_proportion >= 1.0 else \
            self._masked_mean(per_sample.mean(dim=1))
    elif self.update_proportion >= 1.0:
        loss = dist.mean()
    else:
        loss = self._masked_mean(dist.mean(dim=1))
    loss.backward()
    if self.optimizer == "sgd1t":
        lr_t = self._sgd_1t_lr(self.sgd_eta0, self.sgd_t0, self._n_updates)
        for group in self.opt.param_groups:
            group["lr"] = lr_t
    self.opt.step()
    self._n_updates += 1

    # statistics advance after the step: the same sequence of statistics, one batch later
    if self.use_obs_norm and self.obs_rms is not None:
        self.obs_rms.update(x.detach().cpu().numpy())
    return bonus


def patch_fused_rnd():
    """Route compute() through the fused path and make update() a no-op for the same batch."""
    RND.compute = _fused_compute_and_update
    RND.update = lambda self, samples: None
