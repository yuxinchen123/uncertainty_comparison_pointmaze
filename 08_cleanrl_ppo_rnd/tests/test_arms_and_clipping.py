"""Unit tests for the five arms and for what each one actually clips.

The property these protect is the definition of the ablation itself. The policy is clipped at a
global norm of 0.5 in EVERY arm; the only thing arms 2 and 5 change is that the RND predictor is
taken out of that clip. Two knobs express this — `joint_grad_clip` (are the two networks clipped
through one norm, as CleanRL does) and `rnd_max_grad_norm` (the predictor's own threshold when they
are separate, 0 meaning unclipped) — and a silent change to either would make three arms differ from
arm 1 in a way no table records.

The last test is a regression test. The predictor's own gradient norm was once measured after the
clip call rather than before it, so under a joint clip it was reported already scaled down by the
clip's factor. That made arm 1's predictor look several times quieter than arm 2's when in truth
their pre-clip norms are close.
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from grad_stats import GradientStatistics, clip_gradients  # noqa: E402
from ppo_rnd_envpool_shuze import ARMS, Args, apply_arm  # noqa: E402


def make_parameters(norms):
    """Build one parameter per requested gradient norm, so a group's total norm is known exactly."""
    # before: norms = [3.0, 4.0]
    # after:  two 1-element parameters with gradients 3.0 and 4.0, so the group's 2-norm is 5.0
    parameters = []
    for n in norms:
        p = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        p.grad = torch.tensor([float(n)], dtype=torch.float64)
        parameters.append(p)
    return parameters


def group_norm(parameters):
    """The 2-norm of a parameter group's gradients, the same quantity clip_grad_norm_ computes."""
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad, 2) for p in parameters]), 2).item()


def test_only_arms_2_and_5_take_the_predictor_out_of_the_clip():
    """Arms 1, 3 and 4 keep CleanRL's joint clip; only arms 2 and 5 separate it."""
    joint_by_arm = {}
    for arm in ARMS:
        args = Args(arm=arm)
        apply_arm(args)
        joint_by_arm[arm] = args.joint_grad_clip
        # The policy's threshold is untouched by every arm — that is the whole point of the fix.
        assert args.max_grad_norm == 0.5, f"{arm} changed the policy's clip threshold"
    assert joint_by_arm == {
        "arm1_original": True,
        "arm2_no_rnd_grad_clip": False,
        "arm3_update_proportion_1": True,
        "arm4_shallower_predictor": True,
        "arm5_all": False,
    }


def test_arms_2_and_5_leave_the_predictor_with_no_threshold_of_its_own():
    """Separating the clips is only "no RND clip" if the predictor's own threshold is zero."""
    for arm in ("arm2_no_rnd_grad_clip", "arm5_all"):
        args = Args(arm=arm)
        apply_arm(args)
        assert args.rnd_max_grad_norm == 0.0


def test_each_arm_changes_only_its_own_knobs():
    """Arm 5 is exactly arms 2, 3 and 4 combined, so their sum is a real comparison."""
    combined = {}
    for arm in ("arm2_no_rnd_grad_clip", "arm3_update_proportion_1", "arm4_shallower_predictor"):
        combined.update(ARMS[arm])
    assert ARMS["arm5_all"] == combined
    assert ARMS["arm1_original"] == {}


def test_joint_clip_scales_the_policy_and_the_predictor_together():
    """Under a joint clip both groups are scaled by one factor from their combined norm."""
    # before: policy grads [3.0], predictor grads [4.0] -> joint norm 5.0, threshold 1.0
    # after:  both scaled by 1/5, so policy grad 0.6 and predictor grad 0.8
    policy, predictor = make_parameters([3.0]), make_parameters([4.0])
    clip_gradients(policy, predictor, max_grad_norm=1.0, joint_grad_clip=True, rnd_max_grad_norm=0.0)
    assert policy[0].grad.item() == pytest.approx(0.6)
    assert predictor[0].grad.item() == pytest.approx(0.8)


def test_separate_clip_with_no_rnd_threshold_leaves_the_predictor_untouched():
    """Arms 2 and 5: the policy is clipped on its own norm, the predictor not at all."""
    # before: policy grads [3.0] (norm 3.0), predictor grads [4.0], threshold 1.0, rnd threshold 0
    # after:  policy scaled to 1.0, predictor still exactly 4.0
    policy, predictor = make_parameters([3.0]), make_parameters([4.0])
    clip_gradients(policy, predictor, max_grad_norm=1.0, joint_grad_clip=False, rnd_max_grad_norm=0.0)
    assert policy[0].grad.item() == pytest.approx(1.0)
    assert predictor[0].grad.item() == pytest.approx(4.0)


def test_separate_clip_honours_the_predictor_own_threshold():
    """A non-zero rnd_max_grad_norm clips the predictor on its own norm, independently of the policy."""
    policy, predictor = make_parameters([3.0]), make_parameters([4.0])
    clip_gradients(policy, predictor, max_grad_norm=1.0, joint_grad_clip=False, rnd_max_grad_norm=2.0)
    assert policy[0].grad.item() == pytest.approx(1.0)
    assert predictor[0].grad.item() == pytest.approx(2.0)


def test_predictor_norm_is_measured_before_the_joint_clip_scales_it():
    """Regression: under a joint clip the reported predictor norm must be the pre-clip one."""
    # before: predictor grads [4.0] (norm 4.0), joint norm 5.0, threshold 1.0
    # after:  the gradient on disk is 0.8, but the reported norm is 4.0 — what it was before clipping
    policy, predictor = make_parameters([3.0]), make_parameters([4.0])
    stats = GradientStatistics(predictor, 1.0, torch.device("cpu"), joint_grad_clip=True)
    clip_gradients(policy, predictor, 1.0, True, 0.0, stats)
    reported = stats.read_and_reset()
    assert predictor[0].grad.item() == pytest.approx(0.8), "the clip should have scaled the gradient"
    assert reported["grad/mean_predictor_norm_before_clipping"] == pytest.approx(4.0)
    assert reported["grad/mean_joint_norm_before_clipping"] == pytest.approx(5.0)
    assert reported["grad/mean_policy_norm_before_clipping"] == pytest.approx(3.0)


@pytest.mark.parametrize("joint", [True, False])
def test_the_three_norms_are_consistent_in_both_modes(joint):
    """Whichever norm the clip returns, the other two are derived so the squares still add up."""
    policy, predictor = make_parameters([3.0, 4.0]), make_parameters([12.0])
    stats = GradientStatistics(predictor, 100.0, torch.device("cpu"), joint_grad_clip=joint)
    clip_gradients(policy, predictor, 100.0, joint, 0.0, stats)
    r = stats.read_and_reset()
    assert r["grad/mean_policy_norm_before_clipping"] == pytest.approx(5.0)
    assert r["grad/mean_predictor_norm_before_clipping"] == pytest.approx(12.0)
    assert r["grad/mean_joint_norm_before_clipping"] == pytest.approx(13.0)
    assert r["grad/predictor_share_of_squared_norm"] == pytest.approx(144 / 169)
    assert r["grad/joint_grad_clip"] is joint
    # A threshold of 100 is far above the joint norm of 13, so nothing fired anywhere.
    assert r["grad/policy_clip_fired_fraction"] == 0.0
    assert r["grad/predictor_clip_fired_fraction"] == 0.0


def test_predictor_clip_never_fires_when_the_predictor_is_not_clipped():
    """Arms 2 and 5 must report a predictor clip rate of zero however large its gradient grows."""
    policy, predictor = make_parameters([3.0]), make_parameters([400.0])
    stats = GradientStatistics(predictor, 0.5, torch.device("cpu"),
                               joint_grad_clip=False, rnd_max_grad_norm=0.0)
    clip_gradients(policy, predictor, 0.5, False, 0.0, stats)
    r = stats.read_and_reset()
    assert r["grad/predictor_clip_fired_fraction"] == 0.0
    assert r["grad/mean_scale_applied_to_predictor"] == pytest.approx(1.0)
    assert r["grad/policy_clip_fired_fraction"] == 1.0
    assert r["grad/rnd_clip_threshold"] == 0.0


def test_joint_clip_reports_the_predictor_as_clipped_whenever_the_joint_norm_fires():
    """Under a joint clip the predictor is scaled even when the policy caused the excess."""
    # The predictor's own norm is 0.01, far below the 0.5 threshold, but the policy's 3.0 pushes the
    # joint norm over it, so the predictor's gradient is scaled anyway. That is the effect the
    # ablation is testing, and the statistics have to show it.
    policy, predictor = make_parameters([3.0]), make_parameters([0.01])
    stats = GradientStatistics(predictor, 0.5, torch.device("cpu"), joint_grad_clip=True)
    clip_gradients(policy, predictor, 0.5, True, 0.0, stats)
    r = stats.read_and_reset()
    assert r["grad/predictor_clip_fired_fraction"] == 1.0
    assert r["grad/mean_scale_applied_to_predictor"] < 0.2
    assert r["grad/rnd_clip_threshold"] is None


def test_sweep_configs_define_the_same_arms_as_the_trainer():
    """The queue's arm definitions must not drift from the ones the trainer applies."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_runs"))
    from build_ablation_configs import ARMS as SWEEP_ARMS  # noqa: E402
    assert {name: knobs for name, knobs, _ in SWEEP_ARMS} == ARMS
