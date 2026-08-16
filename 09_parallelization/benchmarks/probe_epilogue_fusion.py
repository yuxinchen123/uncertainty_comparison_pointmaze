"""Does letting the compiler generate the multiplications pay once the matrix units stay on?

Round five measured the compiler-generated form and found it 3 to 4 percent faster on the update
stage, then set it aside because "the compiler's chosen kernels set ALLOW_TF32=False". This probe
asks the question the round-five note left open: WHY they set it, and what the form is worth when
they do not.

The reason is a size rule in this version of the compiler (`template_heuristics/triton.py`):
a generated multiplication is allowed the reduced-precision matrix units only when the number of
rows is at least sixteen AND the smaller of the two inner dimensions is at least 512. Every
multiplication in this trainer has an inner dimension of 4, 64, 128 or 256, so every one of them
is refused. The library's multiplication is under no such rule — round five's alignment probe
showed the four-wide layers changing by 5e-4 when the units were switched on, which is the units
being used at an inner dimension of four.

Four arms, all timed on the same update stage in one process, round-robin, order reversed on
alternate rounds:

  library            the shipped form: the library's multiplication, bias and activation after it
  generated          the compiler's own multiplication, candidates from both backends
  generated_triton   the same with the library backend removed, so a generated kernel is used
                     even where the library's is faster, which is what makes the epilogue fuse
  generated_tf32     the same with the size rule replaced by "use the units whenever the
                     configuration asks for them", which is what the library does

Usage (through the H100 lock wrapper):
  python probe_epilogue_fusion.py --n-copies 4096 --rounds 7
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


def cuda_time(fn, reps=15, warmup=4):
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


def patch_allow_tf32():
    """Replace the compiler's size rule for the reduced-precision units with the config's answer.

    before: ALLOW_TF32 = (fp32_precision == "tf32") and rows >= 16 and min(n, k) >= 512
    after:  ALLOW_TF32 = (fp32_precision == "tf32")
    """
    from torch._inductor.template_heuristics.triton import MMTemplateConfigMixin
    original = MMTemplateConfigMixin.get_extra_kwargs

    def always(self, kernel_inputs, op_name):
        return {"ALLOW_TF32": torch.backends.cuda.matmul.fp32_precision == "tf32"}

    MMTemplateConfigMixin.get_extra_kwargs = always
    return original


def build_arm(trainer, mb, inductor_flags, tf32_everywhere):
    """Compile one form of the loss under its own compiler settings and force it to build now.

    The settings are read when the function is traced, not when it is called, so the first call
    has to happen inside the context that sets them; afterwards the compiled artifact is fixed.
    """
    from torch._inductor.template_heuristics.triton import MMTemplateConfigMixin
    original = MMTemplateConfigMixin.get_extra_kwargs if tf32_everywhere else None
    if tf32_everywhere:
        patch_allow_tf32()
    with torch._inductor.config.patch(**inductor_flags):
        fn = torch.compile(trainer._losses, fullgraph=True, dynamic=False)
        loss = fn(mb, style_a=False)
        torch.autograd.grad(loss, trainer.trainable)     # force the backward to compile too
    if tf32_everywhere:
        MMTemplateConfigMixin.get_extra_kwargs = original
    return fn


def numerics(trainer, mb, reference_fn, fn):
    """How far one form's loss and gradients sit from the reference form's, from equal inputs."""
    loss_ref = reference_fn(mb, style_a=False)
    g_ref = [g.detach().clone() for g in torch.autograd.grad(loss_ref, trainer.trainable)]
    loss_new = fn(mb, style_a=False)
    g_new = [g.detach().clone() for g in torch.autograd.grad(loss_new, trainer.trainable)]
    # scaled by the largest gradient anywhere rather than each tensor's own largest, so a tensor
    # whose gradient is essentially zero cannot report a meaningless ratio
    biggest = max(a.abs().max().item() for a in g_ref)
    worst_abs = max((a - b).abs().max().item() for a, b in zip(g_ref, g_new))
    return {
        "relative_loss_difference": abs(float(loss_ref) - float(loss_new))
                                    / max(abs(float(loss_ref)), 1e-12),
        "worst_absolute_gradient_difference": worst_abs,
        "largest_gradient": biggest,
        "worst_relative_gradient_difference": worst_abs / max(biggest, 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=4096)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--arms", nargs="+",
                    default=["library", "generated", "generated_triton", "generated_tf32"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from torch_ppo_rnd import PPORND, production_config

    C = args.n_copies
    # graph capture off: the recording would freeze whichever form was compiled first
    t = PPORND(production_config(C, style=args.style, one_graph=False, capture_update=False),
               device="cuda")
    t.prime_obs_rms()
    Brows = t.cfg.num_steps * t.cfg.n_envs
    t._perm = torch.arange(Brows, device=t.device).expand(
        t.cfg.update_epochs, C, Brows).contiguous()
    t._loss_out = torch.zeros((), device=t.device)
    t._rollout_body()
    t._post_body()
    mb = {k: t._U[k][:, :Brows // t.cfg.num_minibatches] for k in t._U_KEYS}

    # the four forms, built one at a time so each compiles under its own settings
    specs = {
        "library": (None, False),
        "generated": (dict(max_autotune_gemm=True), False),
        "generated_triton": (dict(max_autotune_gemm=True,
                                  max_autotune_gemm_backends="TRITON"), False),
        "generated_tf32": (dict(max_autotune_gemm=True,
                                max_autotune_gemm_backends="TRITON"), True),
    }
    forms = {}
    for name in args.arms:
        flags, tf32 = specs[name]
        forms[name] = t._loss_fn if flags is None else build_arm(t, mb, flags, tf32)
        print(f"built {name}")

    num = {name: numerics(t, mb, forms["library"], fn)
           for name, fn in forms.items() if name != "library"}

    # round-robin, reversing the arm order on alternate rounds so a drift in the machine cannot
    # systematically favour whichever arm is measured first
    names = list(forms)
    per_round = {n: [] for n in names}
    for rnd in range(args.rounds):
        order = names if rnd % 2 == 0 else names[::-1]
        for name in order:
            t._loss_fn = forms[name]
            per_round[name].append(cuda_time(lambda: t._update_body_captured()))
    t._loss_fn = forms["library"]

    med = {n: sorted(v)[len(v) // 2] for n, v in per_round.items()}
    floor = max(max(v) - min(v) for v in per_round.values())
    print(f"\n== the update stage at {C} copies, {args.style}, {args.rounds} rounds ==")
    rows = []
    for name in names:
        paired = [a - b for a, b in zip(per_round["library"], per_round[name])]
        wins = sum(1 for d in paired if d > 0)
        rows.append({
            "arm": name, "median_us": med[name],
            "change_percent_against_library": (med[name] / med["library"] - 1) * 100.0,
            "rounds_faster_than_library": wins, "rounds": args.rounds,
            "per_round_us": per_round[name], "paired_diff_us": paired,
            **({} if name == "library" else num[name])})
        print(f"  {name:<18s} {med[name]/1000:8.2f} ms  "
              f"{rows[-1]['change_percent_against_library']:+6.2f}%  "
              f"{wins}/{args.rounds} rounds faster than the library form")
        if name in num:
            print(f"      loss differs {num[name]['relative_loss_difference']:.2e} relative; "
                  f"worst gradient {num[name]['worst_absolute_gradient_difference']:.2e} "
                  f"absolute against a largest of {num[name]['largest_gradient']:.2e}")
    print(f"  spread within an arm across rounds: {floor/1000:.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_epilogue_C{C}{args.tag}.json"
    p.write_text(json.dumps({
        "n_copies": C, "style": args.style, "torch": torch.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "noise_floor_us": floor, "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
