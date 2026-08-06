"""Tests for the gradient-norm statistics: known gradients in, known clip fraction and norms out."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from grad_norm_stats import GradNormStats


def build_parameters():
    """Return two 'agent' and two 'predictor' scalar parameters, standing in for the real networks."""
    # Four tensors is enough to exercise the split: the norms vector is (4,) and the predictor is
    # the tail slice [2:], exactly as it is the tail slice [20:] in the real trainer.
    agent = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))]
    predictor = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))]
    return agent, predictor


def set_gradients(agent, predictor, scale):
    """Set gradients (3s, 4s) on the agent and (0, 12s) on the predictor, an exact 5-12-13 triangle.

    Every value is exactly representable in float32 for a power-of-two `scale`, so the resulting
    norms are exact: the agent's is 5s, the predictor's is 12s, and the combined norm is 13s.
    """
    # before: scale = 2  ->  grads 6, 8, 0, 24
    # after:  agent norm 10, predictor norm 24, combined norm sqrt(100 + 576) = 26
    agent[0].grad = torch.tensor([3.0 * scale])
    agent[1].grad = torch.tensor([4.0 * scale])
    predictor[0].grad = torch.tensor([0.0])
    predictor[1].grad = torch.tensor([12.0 * scale])


def test_clipping_is_bit_identical_to_clip_grad_norm():
    """The refactored clip must leave exactly the gradients torch.nn.utils.clip_grad_norm_ leaves."""
    # Golden path: random gradients over tensors of several shapes, clipped both ways.
    torch.manual_seed(0)
    shapes = [(4, 3), (7,), (5, 5), (6, 2), (3,), (9,)]
    reference = [nn.Parameter(torch.zeros(s)) for s in shapes]
    tested = [nn.Parameter(torch.zeros(s)) for s in shapes]
    for r, t in zip(reference, tested):
        g = torch.randn(r.shape) * 0.3
        r.grad, t.grad = g.clone(), g.clone()

    expected_norm = nn.utils.clip_grad_norm_(reference, 0.5)
    stats = GradNormStats(tested[:3], tested[3:], clip_threshold=0.5, capacity=4, device="cpu")
    stats.clip_and_record()

    assert stats.buffer[0, 0].item() == expected_norm.item()
    assert all(torch.equal(r.grad, t.grad) for r, t in zip(reference, tested))


def test_reported_statistics_match_hand_computed_values():
    """Four steps with exactly known norms give exactly the expected clip fraction and moments."""
    # Combined norms 3.25, 13, 26, 52 against a threshold of 13. The clip fires strictly above the
    # threshold, so the step sitting exactly on it does not count: 2 of 4 steps fire.
    agent, predictor = build_parameters()
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=4, device="cpu")
    for scale in (0.25, 1.0, 2.0, 4.0):
        set_gradients(agent, predictor, scale)
        stats.clip_and_record()
    out = stats.drain()

    assert out["grad/optimizer_steps_counted"] == 4
    assert out["grad/clip_fired_fraction"] == pytest.approx(0.5)
    assert out["grad/mean_combined_norm_before_clipping"] == pytest.approx(23.5625)
    assert out["grad/max_combined_norm_before_clipping"] == pytest.approx(52.0)
    assert out["grad/standard_deviation_combined_norm_before_clipping"] == pytest.approx(
        334.69921875 ** 0.5, rel=1e-4
    )
    assert out["grad/mean_predictor_norm_before_clipping"] == pytest.approx(21.75)
    assert out["grad/max_predictor_norm_before_clipping"] == pytest.approx(48.0)
    # The predictor is 12 of the 13 sides, so its share of the squared norm is 144/169 every step.
    assert out["grad/mean_predictor_fraction_of_squared_norm"] == pytest.approx(144 / 169, rel=1e-5)
    assert out["grad/mean_predictor_fraction_of_squared_norm_when_clipped"] == pytest.approx(
        144 / 169, rel=1e-5
    )
    # Scale factors 1, 1, 0.5, 0.25 -> mean 0.6875, harshest 0.25.
    assert out["grad/mean_scale_applied_to_gradients"] == pytest.approx(0.6875, rel=1e-5)
    assert out["grad/min_scale_applied_to_gradients"] == pytest.approx(0.25, rel=1e-5)
    assert out["grad/nonfinite_norm_steps"] == 0


def test_gradients_are_actually_scaled_by_the_reported_factor():
    """The gradient that reaches the optimizer equals the pre-clip gradient times the reported scale."""
    # One step whose combined norm is 26 against a threshold of 13: everything is halved.
    agent, predictor = build_parameters()
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=1, device="cpu")
    set_gradients(agent, predictor, 2.0)
    stats.clip_and_record()
    assert agent[0].grad.item() == pytest.approx(3.0, rel=1e-5)
    assert predictor[1].grad.item() == pytest.approx(12.0, rel=1e-5)
    assert stats.drain()["grad/mean_scale_applied_to_gradients"] == pytest.approx(0.5, rel=1e-5)


def test_folding_at_capacity_matches_a_buffer_that_never_folds():
    """Eight steps through a capacity-2 buffer report the same numbers as a capacity-8 buffer."""
    # Edge case: the ring buffer fills mid-interval, so the reduction happens in four pieces.
    scales = (0.25, 1.0, 2.0, 4.0, 0.5, 8.0, 0.25, 2.0)
    results = []
    for capacity in (2, 8):
        agent, predictor = build_parameters()
        stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=capacity, device="cpu")
        for scale in scales:
            set_gradients(agent, predictor, scale)
            stats.clip_and_record()
        results.append(stats.drain())
    small, large = results
    assert small.keys() == large.keys()
    for key in small:
        if isinstance(small[key], float):
            assert small[key] == pytest.approx(large[key], rel=1e-5), key
        else:
            assert small[key] == large[key], key


def test_draining_resets_the_accumulators():
    """A second drain with no steps in between reports no steps, not the previous interval again."""
    # Edge case: the logging interval that follows a drain must start from zero.
    agent, predictor = build_parameters()
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=4, device="cpu")
    set_gradients(agent, predictor, 4.0)
    stats.clip_and_record()
    assert stats.drain()["grad/optimizer_steps_counted"] == 1
    assert stats.drain() == {"grad/optimizer_steps_counted": 0}


def test_nonfinite_norms_are_counted_separately_and_left_out_of_the_means():
    """A step whose gradient overflows is counted as non-finite and excluded from every mean."""
    # Edge case: only reachable under float16 autocast, where the gradient scaler skips the step.
    agent, predictor = build_parameters()
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=4, device="cpu")
    set_gradients(agent, predictor, 2.0)
    stats.clip_and_record()
    set_gradients(agent, predictor, 2.0)
    agent[0].grad = torch.tensor([float("inf")])
    stats.clip_and_record()
    out = stats.drain()
    assert out["grad/optimizer_steps_counted"] == 1
    assert out["grad/nonfinite_norm_steps"] == 1
    assert out["grad/mean_combined_norm_before_clipping"] == pytest.approx(26.0)
    assert out["grad/max_combined_norm_before_clipping"] == pytest.approx(26.0)


def test_missing_gradient_raises_instead_of_mis_attributing_the_split():
    """A parameter with no gradient would shift the predictor slice, so it must fail loudly."""
    # Edge case: a parameter left out of the loss.
    agent, predictor = build_parameters()
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=4, device="cpu")
    set_gradients(agent, predictor, 1.0)
    agent[1].grad = None
    with pytest.raises(RuntimeError, match="gradient missing"):
        stats.clip_and_record()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to detect a host sync")
def test_recording_performs_no_host_device_synchronisation():
    """With CUDA's synchronisation debugger set to error, recording a step must not raise."""
    # This is the requirement the whole design exists for: no .item(), no .cpu(), no Python branch
    # on a device tensor anywhere in the per-step path.
    agent, predictor = build_parameters()
    agent = [nn.Parameter(p.detach().cuda()) for p in agent]
    predictor = [nn.Parameter(p.detach().cuda()) for p in predictor]
    stats = GradNormStats(agent, predictor, clip_threshold=13.0, capacity=8, device="cuda")
    for p, value in zip(agent + predictor, (3.0, 4.0, 0.0, 12.0)):
        p.grad = torch.tensor([value], device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(4):
            stats.clip_and_record()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert stats.fill == 4
