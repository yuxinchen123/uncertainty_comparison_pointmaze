"""One minibatch of the clipped update against a float64 NumPy transcription of CleanRL.

The reference is `07_reconstruction/ppo_rnd_envpool.py`, lines 463 to 512: the predictor loss, the
ratio, the per-minibatch advantage normalization, the clipped policy loss, the clipped extrinsic
value loss, the intrinsic value loss, the entropy term and the combined loss. The transcription
below repeats those statements in NumPy float64, one policy and one minibatch, with two stated
differences that the platform keeps on purpose:

  1. the predictor mask is the whole batch (the study's one change from CleanRL), so
     `(forward_loss * mask).sum() / max(mask.sum(), 1)` is the plain mean;
  2. the policy is the platform's diagonal Gaussian with a state-independent log standard
     deviation, where CleanRL's Atari script has a categorical one; the log-probability and the
     entropy are those of `torch.distributions.Normal`, summed over the action dimensions.

The networks are the platform's own (actor and critic 64-64 tanh, predictor 256-128-128 with
rectified units, target 256-128), written out again in NumPy so that nothing of the platform's
loss code is reused. The loss must agree, and the platform's gradient must agree with central
differences of the NumPy loss at sampled entries of every parameter tensor.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

CLEANRL = ROOT.parent / "07_reconstruction" / "ppo_rnd_envpool.py"
ROWS = 32
CLIP, ENTROPY, VALUE = 0.1, 0.001, 0.5


def cleanrl_minibatch_loss(params, target, mb) -> float:
    """CleanRL's loss on one minibatch, float64 (lines 463 to 512 of the reference file).

    params: {"agent": {"actor", "critic"}, "bonus": {"predictor"}} of float64 arrays without a
    copy axis. target: the fixed target network. mb: the minibatch fields, [M] or [M, k].
    """
    actor, critic, predictor = params["agent"]["actor"], params["agent"]["critic"], \
        params["bonus"]["predictor"]
    relu = lambda x: np.maximum(x, 0.0)

    # predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
    x = mb["rnd_input"]
    target_feature = relu(x @ target["W0"] + target["b0"]) @ target["W1"] + target["b1"]
    hidden = relu(relu(x @ predictor["W0"] + predictor["b0"]) @ predictor["W1"] + predictor["b1"])
    predict_feature = hidden @ predictor["W2"] + predictor["b2"]
    # forward_loss = F.mse_loss(predict, target.detach(), reduction="none").mean(-1)
    forward_loss = ((predict_feature - target_feature) ** 2).mean(-1)
    # mask = (torch.rand(len(forward_loss)) < args.update_proportion); here: the whole batch
    mask = np.ones(len(forward_loss))
    forward_loss = (forward_loss * mask).sum() / max(mask.sum(), 1.0)

    # _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(...)
    h = np.tanh(np.tanh(mb["obs"] @ actor["W0"] + actor["b0"]) @ actor["W1"] + actor["b1"])
    mean, logstd = h @ actor["W2"] + actor["b2"], actor["logstd"]
    log_two_pi = np.log(2.0 * np.pi)
    newlogprob = (-((mb["actions"] - mean) ** 2) / (2.0 * np.exp(logstd) ** 2) - logstd
                  - 0.5 * log_two_pi).sum(1)
    entropy = np.broadcast_to(0.5 + 0.5 * log_two_pi + logstd, mean.shape).sum(1)
    h = np.tanh(np.tanh(mb["obs"] @ critic["W0"] + critic["b0"]) @ critic["W1"] + critic["b1"])
    new_ext_values = (h @ critic["Wext"] + critic["bext"])[:, 0]
    new_int_values = (h @ critic["Wint"] + critic["bint"])[:, 0]
    # logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()
    ratio = np.exp(newlogprob - mb["old_logprob"])

    # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    advantages = mb["adv"]
    advantages = (advantages - advantages.mean()) / (advantages.std(ddof=1) + 1e-8)
    # pg_loss = torch.max(-adv * ratio, -adv * clamp(ratio, 1 - clip, 1 + clip)).mean()
    pg_loss = np.maximum(-advantages * ratio,
                         -advantages * np.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()

    # ext_v_loss = 0.5 * max((v - R)^2, (v_old + clamp(v - v_old, -clip, clip) - R)^2).mean()
    unclipped = (new_ext_values - mb["ret_ext"]) ** 2
    clipped = mb["vext_old"] + np.clip(new_ext_values - mb["vext_old"], -CLIP, CLIP)
    ext_v_loss = 0.5 * np.maximum(unclipped, (clipped - mb["ret_ext"]) ** 2).mean()
    # int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
    int_v_loss = 0.5 * ((new_int_values - mb["ret_int"]) ** 2).mean()
    v_loss = ext_v_loss + int_v_loss
    entropy_loss = entropy.mean()
    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss
    return pg_loss - ENTROPY * entropy_loss + v_loss * VALUE + forward_loss


def without_copy_axis(tree):
    """A parameter tree of [1, ...] device arrays as float64 host arrays without the copy axis."""
    return jax.tree.map(lambda value: np.asarray(value, np.float64)[0], tree)


def test_reference_statements_are_the_ones_transcribed():
    """The lines the transcription repeats are still at their place in the reference file."""
    lines = CLEANRL.read_text().splitlines()[462:512]
    text = "\n".join(lines)
    for statement in (
            "forward_loss = (forward_loss * mask).sum() / torch.max(",
            "mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)",
            "pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)",
            "pg_loss = torch.max(pg_loss1, pg_loss2).mean()",
            "ext_v_loss = 0.5 * ext_v_loss_max.mean()",
            "int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()",
            "loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss"):
        assert statement in text, f"not in lines 463 to 512 of {CLEANRL}: {statement}"
    print("PASS the transcribed statements are in lines 463 to 512 of the reference file")


def test_loss_and_gradient_of_one_minibatch():
    """The platform's clipped loss and its gradient against the float64 transcription."""
    runner = Runner(PPOConfig(n_copies=1, n_envs=2, num_steps=16, update_style="epoch_minibatch",
                              clip_coef=CLIP, ent_coef=ENTROPY, vf_coef=VALUE))
    rng = np.random.default_rng(7)
    f32 = lambda x: jnp.asarray(x, jnp.float32)

    # the starting weights moved by small random amounts, so no bias or head is at a special value
    params = jax.tree.map(lambda value: f32(np.asarray(value) + 0.05 * rng.normal(
        size=value.shape)), runner.init_trainable())
    target = without_copy_axis(runner.bonus.constants["target"])
    host = without_copy_axis(params)

    # a minibatch whose ratios leave the clip range for some rows and whose value moves exceed
    # the clip for some rows, so every branch of both maxima is taken
    obs = rng.normal(size=(ROWS, 4))
    actions = rng.normal(size=(ROWS, 2))
    mb = {"obs": obs, "actions": actions, "rnd_input": rng.normal(size=(ROWS, 4)),
          "adv": rng.normal(size=ROWS), "ret_ext": rng.normal(size=ROWS),
          "ret_int": rng.normal(size=ROWS), "old_logprob": None, "vext_old": None}
    actor, critic = host["agent"]["actor"], host["agent"]["critic"]
    h = np.tanh(np.tanh(obs @ actor["W0"] + actor["b0"]) @ actor["W1"] + actor["b1"])
    mean = h @ actor["W2"] + actor["b2"]
    logp = (-((actions - mean) ** 2) / (2.0 * np.exp(actor["logstd"]) ** 2) - actor["logstd"]
            - 0.5 * np.log(2.0 * np.pi)).sum(1)
    h = np.tanh(np.tanh(obs @ critic["W0"] + critic["b0"]) @ critic["W1"] + critic["b1"])
    mb["old_logprob"] = logp + 0.15 * rng.normal(size=ROWS)
    mb["vext_old"] = (h @ critic["Wext"] + critic["bext"])[:, 0] + 0.15 * rng.normal(size=ROWS)
    # the float32 values the platform will see are the values the reference uses
    mb = {name: np.asarray(np.asarray(value, np.float32), np.float64) for name, value in mb.items()}
    ratio = np.exp(logp - mb["old_logprob"])
    assert (ratio > 1 + CLIP).any() and (ratio < 1 - CLIP).any() and (
        np.abs(ratio - 1) < CLIP).any(), "the ratios do not cover the clip's three ranges"
    moved = np.abs((h @ critic["Wext"] + critic["bext"])[:, 0] - mb["vext_old"])
    assert (moved > CLIP).any() and (moved < CLIP).any(), "the value moves do not cross the clip"

    # the platform: one copy, the minibatch with a leading copy axis, the clipped style
    batch = {name: f32(value)[None] for name, value in mb.items()}
    loss, grads = jax.value_and_grad(
        lambda p: runner.total_loss(p, batch, style_a=False))(params)
    expected = cleanrl_minibatch_loss(host, target, mb)
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-6)

    # central differences of the reference at six entries of every parameter tensor
    step, worst, checked = 1e-6, 0.0, 0
    paths, _ = jax.tree_util.tree_flatten_with_path(grads)
    for path, grad in paths:
        names = [key.key for key in path]
        tensor = host
        for name in names:
            tensor = tensor[name]
        flat = tensor.reshape(-1)
        for index in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            kept = flat[index]
            flat[index] = kept + step
            above = cleanrl_minibatch_loss(host, target, mb)
            flat[index] = kept - step
            below = cleanrl_minibatch_loss(host, target, mb)
            flat[index] = kept
            numeric = (above - below) / (2 * step)
            actual = float(np.asarray(grad)[0].reshape(-1)[index])
            worst = max(worst, abs(actual - numeric))
            np.testing.assert_allclose(actual, numeric, rtol=2e-3, atol=2e-6,
                                       err_msg="/".join(names))
            checked += 1
    assert checked >= 6 * 15, checked
    print(f"PASS clipped loss {float(loss):.6f} against CleanRL's {expected:.6f}; {checked} "
          f"gradient entries over {len(paths)} tensors, worst difference {worst:.2e}")


if __name__ == "__main__":
    test_reference_statements_are_the_ones_transcribed()
    test_loss_and_gradient_of_one_minibatch()
