"""Gradient clipping and its statistics, collected without stalling the GPU.

The question these answer: **how often is the RND predictor's gradient clipped, and is the predictor
the reason?** Two knobs decide what the clipping actually does, and both are recorded in every row:

| knob | meaning |
|---|---|
| `joint_grad_clip` | clip the policy and the predictor together through ONE norm, as CleanRL does |
| `rnd_max_grad_norm` | when the clips are separate, the predictor's own threshold; 0 means the predictor is not clipped at all |

The policy is clipped at `max_grad_norm` in every case. `joint_grad_clip` only decides whether that
one clip call also acts on the predictor.

Why the clipping lives here rather than in the training loop: the predictor's own gradient norm must
be measured BEFORE any clip touches it, and under a joint clip the single `clip_grad_norm_` call
scales the predictor's gradients on the way past. Measuring afterwards silently reports the
post-clip norm, which understates it by the clip's scale factor. Keeping the measurement and the
clip in one function makes that ordering impossible to get wrong.

Why the rest needs care. The update loop runs 16 optimizer steps per policy update, and three forced
host-device synchronisations were already removed from it because each one drains the GPU pipeline.
Calling `.item()` on a gradient norm per optimizer step would put all three back and then some. So
every statistic is accumulated **into GPU tensors** and read to the host **once per logging
interval**.

The headline numbers are free. `torch.nn.utils.clip_grad_norm_` computes the total norm from the
gradients and only then scales them, and returns that pre-clip norm — the trainer already pays for
the computation and was throwing the result away.
"""

import torch
from torch import nn


def clip_gradients(policy_parameters, predictor_parameters, max_grad_norm,
                   joint_grad_clip, rnd_max_grad_norm, statistics=None):
    """Clip one optimizer step's gradients as this arm asks, recording the statistics if kept.

    The order is the point of this function: the predictor's norm is read before anything is
    scaled, so it is the true pre-clip norm under a joint clip as well as a separate one.
    """
    # Measure first. Under a joint clip the call below scales the predictor's gradients, so a
    # measurement taken afterwards would be the post-clip norm — smaller by the clip's scale factor.
    predictor_norm = statistics.predictor_norm() if statistics is not None else None

    # Then clip. Joint: one call over both parameter lists, which is what CleanRL does, and the
    # returned norm is the joint one. Separate: the policy is clipped on its own norm, and the
    # predictor is clipped only if it was given a threshold of its own.
    if joint_grad_clip:
        returned_norm = nn.utils.clip_grad_norm_(policy_parameters + predictor_parameters,
                                                 max_grad_norm)
    else:
        returned_norm = nn.utils.clip_grad_norm_(policy_parameters, max_grad_norm)
        if rnd_max_grad_norm > 0:
            nn.utils.clip_grad_norm_(predictor_parameters, rnd_max_grad_norm)

    if statistics is not None:
        statistics.record(returned_norm, predictor_norm)


class GradientStatistics:
    """Accumulate per-optimizer-step gradient norms on the GPU; read them once per interval."""

    def __init__(self, predictor_parameters, max_grad_norm, device,
                 joint_grad_clip=True, rnd_max_grad_norm=0.0):
        """Prepare the accumulators for a predictor and the two knobs that decide what is clipped.

        The caller passes the norm the single clip call returned, plus the predictor's own pre-clip
        norm, and the third norm is derived from those two exactly:

        - joint clip:    the call returned the JOINT norm, so policy^2 = joint^2 - predictor^2
        - separate clip: the call returned the POLICY norm, so joint^2 = policy^2 + predictor^2

        Both derivations are free, so every arm reports the same three norms and they stay
        comparable across arms.
        """
        self.predictor_parameters = [p for p in predictor_parameters if p.requires_grad]
        self.max_grad_norm = float(max_grad_norm)
        self.joint_grad_clip = bool(joint_grad_clip)
        self.rnd_max_grad_norm = float(rnd_max_grad_norm)
        self.device = device
        # One flat buffer of running sums, so a step adds to a tensor rather than to each field.
        # order: steps, joint, joint^2, predictor, predictor^2, policy,
        #        policy fires, predictor fires, policy scale, predictor scale
        self._sums = torch.zeros(10, dtype=torch.float64, device=device)
        self._max_joint = torch.zeros((), dtype=torch.float64, device=device)
        self._max_predictor = torch.zeros((), dtype=torch.float64, device=device)
        self._min_policy_scale = torch.ones((), dtype=torch.float64, device=device)
        self._min_predictor_scale = torch.ones((), dtype=torch.float64, device=device)
        self._nonfinite = torch.zeros((), dtype=torch.float64, device=device)

    def predictor_norm(self):
        """The predictor's own gradient norm, measured before anything clips it.

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

    def record(self, returned_norm, predictor_norm):
        """Record one optimizer step from the clip's returned norm and the predictor's own norm.

        Everything here stays on the GPU. Nothing is read to the host.
        """
        # Derive the third norm from the two measured ones. clamp at zero because floating-point
        # cancellation can make the difference very slightly negative.
        c = returned_norm.detach().double()
        predictor = predictor_norm.detach().double()
        if self.joint_grad_clip:
            joint = c
            policy = torch.sqrt(torch.clamp(joint * joint - predictor * predictor, min=0.0))
        else:
            policy = c
            joint = torch.sqrt(policy * policy + predictor * predictor)

        # A non-finite norm means the step's gradients were not usable; it is counted and excluded
        # from every average rather than poisoning the sums.
        finite = torch.isfinite(c) & torch.isfinite(predictor)
        one = finite.double()
        zero = torch.zeros_like(joint)
        j = torch.where(finite, joint, zero)
        p = torch.where(finite, predictor, zero)
        pol = torch.where(finite, policy, zero)

        # The policy's clip acts on the joint norm when the clip is joint, and on the policy's own
        # norm when it is separate.
        policy_acted_on = j if self.joint_grad_clip else pol
        policy_fires = (policy_acted_on > self.max_grad_norm).double() * one
        policy_scale = torch.clamp(self.max_grad_norm / (policy_acted_on + 1e-6), max=1.0)

        # The predictor rides the policy's clip when they are joint; otherwise it is clipped at its
        # own threshold, or not at all when that threshold is zero.
        if self.joint_grad_clip:
            predictor_fires, predictor_scale = policy_fires, policy_scale
        elif self.rnd_max_grad_norm > 0:
            predictor_fires = (p > self.rnd_max_grad_norm).double() * one
            predictor_scale = torch.clamp(self.rnd_max_grad_norm / (p + 1e-6), max=1.0)
        else:
            predictor_fires = torch.zeros_like(j)
            predictor_scale = torch.ones_like(j)
        policy_scale = torch.where(finite, policy_scale, torch.ones_like(policy_scale))
        predictor_scale = torch.where(finite, predictor_scale, torch.ones_like(predictor_scale))

        self._sums += torch.stack([one, j, j * j, p, p * p, pol,
                                   policy_fires, predictor_fires, policy_scale, predictor_scale])
        self._max_joint = torch.maximum(self._max_joint, j)
        self._max_predictor = torch.maximum(self._max_predictor, p)
        self._min_policy_scale = torch.minimum(self._min_policy_scale, policy_scale)
        self._min_predictor_scale = torch.minimum(self._min_predictor_scale, predictor_scale)
        self._nonfinite += (1.0 - one)

    def read_and_reset(self):
        """Transfer the interval's statistics to the host once, then zero the accumulators."""
        # The single synchronisation point. One stack means one copy rather than fifteen.
        packed = torch.stack([*self._sums, self._max_joint, self._max_predictor,
                              self._min_policy_scale, self._min_predictor_scale,
                              self._nonfinite]).cpu().tolist()
        steps, s_j, s_jj, s_p, s_pp, s_pol, policy_fires, predictor_fires, s_ps, s_pds = packed[:10]
        max_j, max_p, min_ps, min_pds, nonfinite = packed[10:]
        self._sums.zero_()
        self._max_joint.zero_()
        self._max_predictor.zero_()
        self._min_policy_scale.fill_(1.0)
        self._min_predictor_scale.fill_(1.0)
        self._nonfinite.zero_()

        if steps == 0:
            return {"grad/optimizer_steps_counted": 0}
        mean_j, mean_p = s_j / steps, s_p / steps
        # Variance from the running sums; clamped at zero because floating-point cancellation can
        # make the difference very slightly negative when every sample is identical.
        var_j = max(s_jj / steps - mean_j * mean_j, 0.0)
        var_p = max(s_pp / steps - mean_p * mean_p, 0.0)
        return {
            # The two knobs, in every row, so a record says what its numbers mean without the config.
            "grad/joint_grad_clip": self.joint_grad_clip,
            "grad/policy_clip_threshold": self.max_grad_norm,
            # The predictor has a threshold of its own only when the clips are separate. Under a
            # joint clip it has none: it is bounded through the joint norm, not on its own.
            "grad/rnd_clip_threshold": None if self.joint_grad_clip else self.rnd_max_grad_norm,
            "grad/optimizer_steps_counted": int(steps),
            # Every norm below is measured BEFORE any clipping, including under a joint clip.
            "grad/mean_joint_norm_before_clipping": mean_j,
            "grad/max_joint_norm_before_clipping": max_j,
            "grad/standard_deviation_joint_norm": var_j ** 0.5,
            "grad/mean_policy_norm_before_clipping": s_pol / steps,
            "grad/mean_predictor_norm_before_clipping": mean_p,
            "grad/max_predictor_norm_before_clipping": max_p,
            "grad/standard_deviation_predictor_norm": var_p ** 0.5,
            # A ratio of sums over the interval, not a mean of per-step ratios: the predictor's and
            # the policy's squared norms partition the joint squared norm, so squared shares add to
            # one while plain norm ratios do not.
            "grad/predictor_share_of_squared_norm": (s_pp / s_jj) if s_jj > 0 else None,
            # The question the run was launched to answer: how often the predictor's gradients are
            # scaled down, and by how much. Zero by construction when the predictor is not clipped.
            "grad/predictor_clip_fired_fraction": predictor_fires / steps,
            "grad/mean_scale_applied_to_predictor": s_pds / steps,
            "grad/min_scale_applied_to_predictor": min_pds,
            "grad/policy_clip_fired_fraction": policy_fires / steps,
            "grad/mean_scale_applied_to_policy": s_ps / steps,
            "grad/min_scale_applied_to_policy": min_ps,
            "grad/nonfinite_norm_steps": int(nonfinite),
        }
