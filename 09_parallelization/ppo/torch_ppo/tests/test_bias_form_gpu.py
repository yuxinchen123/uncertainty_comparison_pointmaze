"""GPU-only: adding the bias after the multiplication must compute what folding it in did.

Every layer of every network used to be one `baddbmm` call, which adds the bias by writing it
into the output tensor and asking the multiplication to accumulate on top. The trainer now calls
`bmm` and adds the bias in the same expression as the activation, so the compiler folds the two
into one pass. The arithmetic is the same sum in the same precision, but the two forms are
separate library calls and may select different multiplication algorithms, so this test measures
how far apart they are rather than assuming they are identical.

It checks two things from byte-identical inputs: the loss value, and every one of the twenty-one
parameter gradients the backward pass produces. A test of the loss alone would pass even if the
backward pass had been broken.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_bias_form_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=8, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="eager", compile_opt=False, tf32=True)


def losses_with_baddbmm(t, mb, style_a):
    """The previous form of `_losses`: every layer one `baddbmm`, bias folded into the multiply."""
    cfg = t.cfg
    a = mb["adv"]
    a_n = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, unbiased=True, keepdim=True) + 1e-8)

    # packed actor and critic forward, written the way it was before the change
    W0 = torch.cat([t.actor["W0"], t.critic["W0"]], -1)
    b0 = torch.cat([t.actor["b0"], t.critic["b0"]], -1)
    h = torch.tanh(torch.baddbmm(b0.unsqueeze(1), mb["obs"], W0))
    ha, hc = h[..., :64], h[..., 64:]
    ha = torch.tanh(torch.baddbmm(t.actor["b1"].unsqueeze(1), ha, t.actor["W1"]))
    hc = torch.tanh(torch.baddbmm(t.critic["b1"].unsqueeze(1), hc, t.critic["W1"]))
    mean = torch.baddbmm(t.actor["b2"].unsqueeze(1), ha, t.actor["W2"])
    Wh = torch.cat([t.critic["Wext"], t.critic["Wint"]], -1)
    bh = torch.cat([t.critic["bext"], t.critic["bint"]], -1)
    v = torch.baddbmm(bh.unsqueeze(1), hc, Wh)
    vext, vint = v[..., 0], v[..., 1]

    logstd = t.actor["logstd"]
    newlogp = t._logprob(mean, logstd, mb["actions"])
    ratio = (newlogp - mb["old_logprob"]).exp()
    if style_a:
        pg = (-a_n * ratio).mean(dim=1)
        v_ext = 0.5 * (vext - mb["ret_ext"]).square().mean(dim=1)
    else:
        pg = torch.maximum(-a_n * ratio,
                           -a_n * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean(dim=1)
        vc = mb["vext_old"] + (vext - mb["vext_old"]).clamp(-cfg.clip_coef, cfg.clip_coef)
        v_ext = 0.5 * torch.maximum((vext - mb["ret_ext"]).square(),
                                    (vc - mb["ret_ext"]).square()).mean(dim=1)
    v_int = 0.5 * (vint - mb["ret_int"]).square().mean(dim=1)

    # predictor forward, the previous form
    ph = torch.relu(torch.baddbmm(t.predictor["b0"].unsqueeze(1), mb["rnd_input"],
                                  t.predictor["W0"]))
    ph = torch.relu(torch.baddbmm(t.predictor["b1"].unsqueeze(1), ph, t.predictor["W1"]))
    pf = torch.baddbmm(t.predictor["b2"].unsqueeze(1), ph, t.predictor["W2"])
    fwd = (pf - mb["rnd_tf"]).square().mean(dim=2).mean(dim=1)

    from torch_ppo_rnd import LOG2PI
    ent = (0.5 + 0.5 * LOG2PI + logstd).sum(dim=1)
    return (pg - cfg.ent_coef * ent + cfg.vf_coef * (v_ext + v_int) + fwd).sum()


def main():
    """One loss and one backward pass, computed both ways from byte-identical inputs."""
    t = PPORND(PPOConfig(**CFG), device="cuda")
    torch.manual_seed(11)
    t.prime_obs_rms()
    batch = t.rollout()

    # the shipped form
    loss_new = t._losses(batch, style_a=False)
    grads_new = [g.detach().clone() for g in torch.autograd.grad(loss_new, t.trainable)]

    # the previous form, from the same parameters and the same batch
    loss_old = losses_with_baddbmm(t, batch, style_a=False)
    grads_old = [g.detach().clone() for g in torch.autograd.grad(loss_old, t.trainable)]

    new_val, old_val = float(loss_new.detach()), float(loss_old.detach())
    d_loss = abs(new_val - old_val)
    rel_loss = d_loss / max(abs(old_val), 1e-12)
    worst_rel = 0.0
    moved = 0.0
    for gn, go in zip(grads_new, grads_old):
        d = (gn - go).abs().max().item()
        scale = max(go.abs().max().item(), 1e-12)
        worst_rel = max(worst_rel, d / scale)
        moved = max(moved, go.abs().max().item())
    print(f"loss: new {new_val:.8f} old {old_val:.8f} "
          f"absolute {d_loss:.3e} relative {rel_loss:.3e}")
    print(f"gradients: worst relative difference over all twenty-one tensors {worst_rel:.3e}")
    print(f"largest gradient magnitude {moved:.3e} (a zero here would make the test vacuous)")
    assert moved > 0, "no gradient was produced, so nothing was compared"
    assert rel_loss <= 1e-6, "the two bias forms compute different losses"
    assert worst_rel <= 1e-5, "the two bias forms compute different gradients"
    print("ok test_bias_after_multiply_matches_bias_folded_in")


if __name__ == "__main__":
    main()
