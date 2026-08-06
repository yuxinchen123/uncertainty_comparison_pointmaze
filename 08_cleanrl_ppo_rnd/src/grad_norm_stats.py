"""Gradient-norm statistics for the PPO + RND update loop, with no host-device synchronisation.

The question this answers: **how often is the predictor network's gradient clipped, and is the
predictor the reason?** The run clips at global norm 0.5 over the combined agent + predictor
parameters, so the predictor is never clipped on its own — every gradient in the optimizer, the
predictor's included, is multiplied by the same factor `min(0.5 / (combined_norm + 1e-6), 1)`.
Therefore:

* *How often is the predictor's gradient clipped* is exactly *how often the combined norm exceeded
  0.5*. One number, and `clip_grad_norm_` already computes it — see `clip_and_record` below.
* *Whether the predictor is responsible* needs a second number, the predictor's own gradient norm,
  which is NOT recoverable from the combined norm: the combined squared norm is
  `agent_norm**2 + predictor_norm**2`, one equation in two unknowns.

The hard constraint is throughput. Three forced host-device synchronisations were already removed
from this loop because each drains the GPU pipeline sixteen times per iteration (4 minibatches x
4 update epochs). Nothing here may put one back, so:

* every per-step quantity stays a GPU tensor — no `.item()`, no `.cpu()`, no Python `if` on a tensor;
* the per-step work is two extra kernel launches: one vector norm over a 32-element vector, and one
  store of two floats into a preallocated ring buffer;
* all the arithmetic that turns those two columns into statistics (squares, the comparison against
  the clip threshold, the masked means) runs ONCE per logging interval, on the GPU, over the whole
  buffer at once;
* exactly one host transfer per logging interval, of a 13-element vector, inside the block that
  already synchronises for `losses/value_loss` and friends.
"""

import torch


class GradNormStats:
    """Clip the combined gradient and record its norm and the predictor's norm, without any host sync."""

    def __init__(self, agent_parameters, predictor_parameters, clip_threshold, capacity, device):
        """Set up the ring buffer and the accumulators for one run's whole training."""
        # The parameter order is fixed here and never changes, so the predictor's gradients are
        # always the tail of the list and `split` is a constant slice index into the norms vector.
        # before: agent 20 tensors (1,407,476 params), predictor 12 tensors (2,203,296 params)
        # after:  self.parameters is 32 tensors, self.split == 20, norms[20:] is the predictor's
        self.agent_parameters = list(agent_parameters)
        self.predictor_parameters = list(predictor_parameters)
        self.parameters = self.agent_parameters + self.predictor_parameters
        self.split = len(self.agent_parameters)
        self.clip_threshold = float(clip_threshold)
        self.checked = False

        # Column 0 is the combined pre-clip norm, column 1 is the predictor-only pre-clip norm.
        # Capacity is the number of optimizer steps in one logging interval, so in a normal run the
        # buffer is folded exactly once, at drain time.
        self.buffer = torch.zeros((capacity, 2), dtype=torch.float32, device=device)
        self.fill = 0

        # Ten running sums and counts, one running maximum pair, one running minimum. All on device,
        # and all float64: the standard deviation comes from a sum minus a sum of squares, which in
        # float32 over a few hundred steps loses about four of the seven digits to cancellation.
        # These are thirteen numbers touched once per logging interval, so the slow float64 rate of
        # a consumer card is irrelevant here.
        # Index order: steps, sum norm, sum norm^2, steps clipped, sum predictor norm,
        # sum predictor norm^2, sum predictor squared-norm fraction, the same restricted to clipped
        # steps, sum of the scale factor applied, count of non-finite steps.
        self.sums = torch.zeros(10, dtype=torch.float64, device=device)
        self.maxima = torch.zeros(2, dtype=torch.float64, device=device)
        self.minimum_scale = torch.ones(1, dtype=torch.float64, device=device)

    def clip_and_record(self):
        """Clip the combined gradient exactly as clip_grad_norm_ does, keeping the per-tensor norms.

        `torch.nn.utils.clip_grad_norm_` computes one norm per gradient tensor with
        `torch._foreach_norm`, stacks them, takes the vector norm of that stack, and then throws the
        per-tensor norms away. This does the same three steps by hand and keeps the stack, so the
        predictor's norm is one more vector norm over a 12-element slice rather than a second pass
        over 2.2 million gradient values. The clipping itself is `clip_grads_with_norm_`, the very
        function `clip_grad_norm_` calls, given the very same `total_norm` — so the gradients that
        reach `optimizer.step()` are bit-identical to before.
        """
        # A missing gradient would shift `self.split` and silently mis-attribute the predictor's
        # share, so check once — every minibatch builds the same graph, so once is enough.
        grads = [p.grad for p in self.parameters]
        if not self.checked:
            self._check_grads(grads)

        # before: norms is a list of 32 zero-dim tensors; total_norm = ||stack(norms)||
        # after:  norms shape (32,); total_norm = ||norms||; predictor_norm = ||norms[20:]||
        norms = torch.stack(torch._foreach_norm(grads, 2.0))
        total_norm = torch.linalg.vector_norm(norms, 2.0)
        predictor_norm = torch.linalg.vector_norm(norms[self.split:], 2.0)
        torch.nn.utils.clip_grads_with_norm_(self.parameters, self.clip_threshold, total_norm)

        # Park the two numbers and return. No comparison, no reduction, no host copy in the hot loop.
        torch.stack((total_norm, predictor_norm), out=self.buffer[self.fill])
        self.fill += 1
        if self.fill == self.buffer.shape[0]:
            self._fold()

    def _check_grads(self, grads):
        """Fail loudly if the gradient list is not the fixed 32 float tensors on one device."""
        missing = [i for i, g in enumerate(grads) if g is None]
        if missing:
            raise RuntimeError(
                f"gradient missing for parameter tensors {missing}; the predictor slice index "
                f"{self.split} assumes every parameter receives a gradient on every minibatch"
            )
        dtypes = {g.dtype for g in grads}
        devices = {g.device for g in grads}
        if len(dtypes) != 1 or len(devices) != 1:
            raise RuntimeError(
                f"gradients span dtypes {dtypes} and devices {devices}; the norm ordering is only "
                "bit-identical to clip_grad_norm_ when they form a single group"
            )
        self.checked = True

    def _fold(self):
        """Reduce the filled part of the ring buffer into the running accumulators, on the GPU."""
        # Runs once per logging interval in a normal run. Everything below is a GPU tensor op; the
        # only Python branch reads `self.fill`, a host-side integer.
        if self.fill == 0:
            return
        # The buffer is float32, because that is what the norms are; the reduction is float64, so the
        # moments the standard deviation is built from do not lose digits to cancellation.
        rows = self.buffer[: self.fill].double()
        combined, predictor = rows[:, 0], rows[:, 1]

        # A non-finite norm is only reachable under float16 autocast, where the gradient scaler makes
        # the optimizer skip the step. Such a step is counted separately and excluded from every mean
        # and maximum, rather than poisoning them with inf.
        # before: combined = [0.31, inf, 0.72], threshold 0.5
        # after:  usable = [1, 0, 1]; steps counted 2; clipped 1; non-finite 1
        finite = torch.isfinite(combined) & torch.isfinite(predictor)
        zero = torch.zeros((), dtype=combined.dtype, device=combined.device)
        combined = torch.where(finite, combined, zero)
        predictor = torch.where(finite, predictor, zero)
        usable = finite.to(combined.dtype)

        # The clip fires when the combined norm is strictly greater than the threshold; a norm exactly
        # equal to it leaves the gradient unchanged up to the 1e-6 in clip_grads_with_norm_.
        fired = (combined > self.clip_threshold).to(combined.dtype) * usable
        # The predictor's share of the SQUARED norm, because squared shares partition: the agent's
        # share is one minus this one. The clamp only guards the all-zero-gradient step.
        fraction = (predictor * predictor) / (combined * combined).clamp_min(1e-30)
        # The factor every gradient in the optimizer is actually multiplied by. 1.0 means no clipping.
        scale = torch.clamp(self.clip_threshold / (combined + 1e-6), max=1.0)
        scale = torch.where(finite, scale, torch.ones_like(scale))

        self.sums += torch.stack((
            usable.sum(),
            (combined * usable).sum(),
            (combined * combined * usable).sum(),
            fired.sum(),
            (predictor * usable).sum(),
            (predictor * predictor * usable).sum(),
            (fraction * usable).sum(),
            (fraction * fired).sum(),
            (scale * usable).sum(),
            (1.0 - usable).sum(),
        ))
        self.maxima = torch.maximum(
            self.maxima, torch.stack(((combined * usable).max(), (predictor * usable).max()))
        )
        self.minimum_scale = torch.minimum(self.minimum_scale, scale.min().reshape(1))
        self.fill = 0

    def drain(self):
        """Fold, copy the accumulators to the host in ONE transfer, reset them, and name the numbers.

        This is the only host-device synchronisation in the whole design, and it happens once per
        logging interval — inside the block that already synchronises for `losses/value_loss`.
        """
        self._fold()
        # One transfer, 13 float32 values. Everything after this line is host arithmetic.
        host = torch.cat((self.sums, self.maxima, self.minimum_scale)).cpu().tolist()
        self.sums.zero_()
        self.maxima.zero_()
        self.minimum_scale.fill_(1.0)

        (steps, sum_norm, sum_norm_squared, steps_clipped, sum_predictor,
         sum_predictor_squared, sum_fraction, sum_fraction_clipped, sum_scale,
         nonfinite) = host[:10]
        max_norm, max_predictor, min_scale = host[10], host[11], host[12]
        if steps == 0:
            return {"grad/optimizer_steps_counted": 0}

        # The standard deviation comes from the sum and the sum of squares, so no second pass over
        # the buffer is needed and the buffer can be, and is, reset every interval.
        mean_norm = sum_norm / steps
        variance = max(sum_norm_squared / steps - mean_norm * mean_norm, 0.0)
        mean_predictor = sum_predictor / steps
        variance_predictor = max(sum_predictor_squared / steps - mean_predictor * mean_predictor, 0.0)
        return {
            "grad/clip_threshold": self.clip_threshold,
            "grad/optimizer_steps_counted": int(steps),
            "grad/clip_fired_fraction": steps_clipped / steps,
            "grad/mean_combined_norm_before_clipping": mean_norm,
            "grad/max_combined_norm_before_clipping": max_norm,
            "grad/standard_deviation_combined_norm_before_clipping": variance ** 0.5,
            "grad/mean_predictor_norm_before_clipping": mean_predictor,
            "grad/max_predictor_norm_before_clipping": max_predictor,
            "grad/standard_deviation_predictor_norm_before_clipping": variance_predictor ** 0.5,
            "grad/mean_predictor_fraction_of_squared_norm": sum_fraction / steps,
            "grad/mean_predictor_fraction_of_squared_norm_when_clipped": (
                sum_fraction_clipped / steps_clipped if steps_clipped > 0 else None
            ),
            "grad/mean_scale_applied_to_gradients": sum_scale / steps,
            "grad/min_scale_applied_to_gradients": min_scale,
            "grad/nonfinite_norm_steps": int(nonfinite),
        }
