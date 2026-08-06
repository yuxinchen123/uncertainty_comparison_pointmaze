"""Gradient statistics for the RND predictor, collected without stalling the GPU.

The question these answer: **how often is the predictor's gradient clipped, and is the predictor the
reason?** The run clips at a global norm of 0.5 over the policy and the predictor *together*, so when
the clip fires it scales the predictor's gradient down whether or not the predictor caused the spike.

Why this needs care. The update loop runs 16 optimizer steps per policy update, and three forced
host-device synchronisations were already removed from it because each one drains the GPU pipeline.
Calling `.item()` on a gradient norm per optimizer step would put all three back and then some. So
every statistic is accumulated **into GPU tensors** and read to the host **once per logging
interval**.

The headline number is free. `torch.nn.utils.clip_grad_norm_` computes the total norm from the
gradients and only then scales them, and returns that pre-clip norm — the trainer already pays for
the computation and was throwing the result away.
"""

import torch


class GradientStatistics:
    """Accumulate per-optimizer-step gradient norms on the GPU; read them once per interval."""

    def __init__(self, predictor_parameters, clip_threshold, device):
        """Prepare the accumulators for a predictor and a clip threshold.

        `clip_threshold` of 0 or None means the run does not clip; the statistics are still
        collected, and the clip-fire fraction then reports how often the run *would* have clipped at
        0.5. That is what makes the no-clipping arm comparable to the others.
        """
        self.predictor_parameters = [p for p in predictor_parameters if p.requires_grad]
        self.clip_threshold = float(clip_threshold) if clip_threshold else 0.0
        # The threshold the fire-fraction is measured against. With clipping off there is no
        # threshold in the optimizer, so the reference value is CleanRL's 0.5 and the field then
        # means "how often it would have fired".
        self.reference_threshold = self.clip_threshold if self.clip_threshold > 0 else 0.5
        self.device = device
        # One flat buffer of six running sums, so a step adds to a tensor rather than to six.
        # order: steps, sum(combined), sum(combined^2), sum(predictor), sum(predictor^2), fires
        self._sums = torch.zeros(6, dtype=torch.float64, device=device)
        self._max_combined = torch.zeros((), dtype=torch.float64, device=device)
        self._max_predictor = torch.zeros((), dtype=torch.float64, device=device)
        self._min_scale = torch.ones((), dtype=torch.float64, device=device)
        self._sum_scale = torch.zeros((), dtype=torch.float64, device=device)
        self._nonfinite = torch.zeros((), dtype=torch.float64, device=device)

    def predictor_norm(self):
        """The predictor's own gradient norm, without clipping it.

        Computed the same way clip_grad_norm_ computes its total: the 2-norm of the vector of
        per-tensor 2-norms. The predictor holds eight gradient tensors, so this is eight small
        kernels plus one stack — negligible beside a 3,840-row forward and backward pass.
        """
        grads = [p.grad for p in self.predictor_parameters if p.grad is not None]
        if not grads:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]), 2
        ).double()

    def record(self, combined_norm):
        """Record one optimizer step. `combined_norm` is clip_grad_norm_'s return value.

        Everything here stays on the GPU. Nothing is read to the host.
        """
        combined = combined_norm.detach().double()
        predictor = self.predictor_norm()
        finite = torch.isfinite(combined)
        # A non-finite norm only happens under float16 autocast, where the gradient scaler skips the
        # step. Excluding it keeps one inf from destroying every mean in the interval.
        c = torch.where(finite, combined, torch.zeros_like(combined))
        p = torch.where(finite, predictor, torch.zeros_like(predictor))
        one = finite.double()

        fired = (c > self.reference_threshold).double() * one
        # The factor the gradients were actually multiplied by: 1.0 when the clip did not fire.
        scale = torch.clamp(self.reference_threshold / (c + 1e-6), max=1.0)
        scale = torch.where(finite, scale, torch.ones_like(scale))

        self._sums += torch.stack([one, c, c * c, p, p * p, fired])
        self._max_combined = torch.maximum(self._max_combined, c)
        self._max_predictor = torch.maximum(self._max_predictor, p)
        self._min_scale = torch.minimum(self._min_scale, scale)
        self._sum_scale += scale
        self._nonfinite += (1.0 - one)

    def read_and_reset(self):
        """Transfer the interval's statistics to the host once, then zero the accumulators."""
        # The single synchronisation point. One stack means one copy rather than ten.
        packed = torch.stack([*self._sums, self._max_combined, self._max_predictor,
                              self._min_scale, self._sum_scale, self._nonfinite]).cpu().tolist()
        steps, s_c, s_cc, s_p, s_pp, fires = packed[:6]
        max_c, max_p, min_scale, sum_scale, nonfinite = packed[6:]
        self._sums.zero_()
        self._max_combined.zero_()
        self._max_predictor.zero_()
        self._min_scale.fill_(1.0)
        self._sum_scale.zero_()
        self._nonfinite.zero_()

        if steps == 0:
            return {"grad/optimizer_steps_counted": 0}
        mean_c, mean_p = s_c / steps, s_p / steps
        # Variance from the running sums; clamped at zero because floating-point cancellation can
        # make the difference very slightly negative when every sample is identical.
        var_c = max(s_cc / steps - mean_c * mean_c, 0.0)
        var_p = max(s_pp / steps - mean_p * mean_p, 0.0)
        return {
            "grad/clip_threshold": self.clip_threshold,
            "grad/clipping_enabled": self.clip_threshold > 0,
            "grad/reference_threshold": self.reference_threshold,
            "grad/optimizer_steps_counted": int(steps),
            "grad/clip_fired_fraction": fires / steps,
            "grad/mean_combined_norm_before_clipping": mean_c,
            "grad/max_combined_norm_before_clipping": max_c,
            "grad/standard_deviation_combined_norm": var_c ** 0.5,
            "grad/mean_predictor_norm_before_clipping": mean_p,
            "grad/max_predictor_norm_before_clipping": max_p,
            "grad/standard_deviation_predictor_norm": var_p ** 0.5,
            # Squared shares are used because they partition: the predictor's and the policy's
            # squared norms sum to the combined squared norm, while plain norm ratios do not.
            "grad/mean_predictor_share_of_squared_norm": (s_pp / s_cc) if s_cc > 0 else None,
            "grad/mean_scale_applied_to_gradients": sum_scale / steps,
            "grad/min_scale_applied_to_gradients": min_scale,
            "grad/nonfinite_norm_steps": int(nonfinite),
        }


def total_norm_without_clipping(parameters):
    """The 2-norm of the gradients, computed exactly as clip_grad_norm_ does, but never clipping.

    The no-clipping arm still needs the number, and calling clip_grad_norm_ with a huge threshold
    would be a lie in the code even though it would not scale anything.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]), 2
    )
