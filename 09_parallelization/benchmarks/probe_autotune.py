"""Would letting the compiler generate the matrix multiplications close the gap to JAX?

At a thousand copies and more the PyTorch trainer's cost is the number of times an intermediate
result is written to memory and read back. The JAX trainer is faster there because its compiler
folds chains of element-wise work into the programs that produce and consume them. PyTorch can be
asked to do the same by letting its compiler generate the multiplication itself, with the bias and
the activation as an epilogue, instead of calling the library's multiplication and following it
with a separate pass.

This probe measures that without changing the trainer: it builds one trainer, times the update
stage with the loss compiled the shipped way, rebuilds the loss with the compiler's own
multiplication templates enabled, and times it again — alternating, so drift between the two
readings is visible. It also reports how far apart the two losses and the two sets of gradients
are, because generating the multiplication changes the order in which it accumulates.

Usage (through the H100 lock wrapper): python probe_autotune.py --n-copies 1024
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def cuda_time(fn, reps=20, warmup=6):
    """Median device time of fn() in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=1024)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from torch_ppo_rnd import PPORND, production_config

    C = args.n_copies
    # one_graph off so the update stage can be called on its own and re-timed after the loss is
    # rebuilt; graph capture would freeze the first version's programs into the recording
    t = PPORND(production_config(C, style=args.style, one_graph=False, capture_update=False),
               device="cuda")
    t.prime_obs_rms()
    t._perm = torch.arange(t.cfg.num_steps * t.cfg.n_envs, device=t.device).expand(
        t.cfg.update_epochs, C, t.cfg.num_steps * t.cfg.n_envs).contiguous()
    t._loss_out = torch.zeros((), device=t.device)
    t._rollout_body()
    t._post_body()

    shipped = t._loss_fn
    generated = torch.compile(t._losses, fullgraph=True, dynamic=False,
                              mode="max-autotune-no-cudagraphs")

    # how far apart the two forms are, from byte-identical inputs, before any timing is believed
    mb = {k: t._U[k][:, :t.cfg.num_steps * t.cfg.n_envs // t.cfg.num_minibatches]
          for k in t._U_KEYS}
    loss_shipped = shipped(mb, style_a=False)
    g_shipped = [g.detach().clone() for g in torch.autograd.grad(loss_shipped, t.trainable)]
    loss_generated = generated(mb, style_a=False)
    g_generated = [g.detach().clone() for g in torch.autograd.grad(loss_generated, t.trainable)]
    rel_loss = (abs(float(loss_shipped.detach()) - float(loss_generated.detach()))
                / max(abs(float(loss_shipped.detach())), 1e-12))
    worst_rel = max((a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
                    for a, b in zip(g_shipped, g_generated))

    # alternate the two forms so a drift in the machine's state shows up as disagreement
    times = {"shipped": [], "generated": []}
    for r in range(args.rounds):
        order = [("shipped", shipped), ("generated", generated)]
        if r % 2:
            order.reverse()
        for name, fn in order:
            t._loss_fn = fn
            times[name].append(cuda_time(lambda: t._update_body_captured()))
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    spread = max(max(v) - min(v) for v in times.values())

    print(f"\n== the update stage at {C} copies, {args.style} ==")
    print(f"  library multiplication, the shipped form : {med['shipped']/1000:8.2f} ms")
    print(f"  compiler-generated multiplication        : {med['generated']/1000:8.2f} ms")
    print(f"  difference {(med['shipped']-med['generated'])/1000:+.2f} ms "
          f"({(med['shipped']/med['generated']-1)*100:+.1f} percent), "
          f"spread within a form {spread/1000:.2f} ms")
    print(f"  loss differs by {rel_loss:.3e} relative; worst gradient {worst_rel:.3e} relative")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_autotune_C{C}{args.tag}.json"
    p.write_text(json.dumps({
        "n_copies": C, "style": args.style, "torch": torch.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "shipped_us": med["shipped"], "generated_us": med["generated"],
        "spread_us": spread, "all_us": times,
        "relative_loss_difference": rel_loss,
        "worst_relative_gradient_difference": worst_rel}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
