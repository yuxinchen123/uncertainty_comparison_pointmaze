"""Does letting the compiler generate the multiplications pay once the matrix units stay on?

Round five measured the compiler-generated form and found it 3 to 4 percent faster on the update
stage, then set it aside because "the compiler's chosen kernels set ALLOW_TF32=False". This probe
asks the question that note left open: WHY they set it, and what the form is worth when they do
not.

The reason is a size rule in this version of the compiler (`template_heuristics/triton.py`):
a generated multiplication is allowed the reduced-precision matrix units only when it has at
least sixteen rows AND the smaller of its two inner dimensions is at least 512. Every
multiplication in this trainer has an inner dimension of 4, 64, 128 or 256, so every one of them
is refused. The library's multiplication is under no such rule — round five's alignment probe
showed the four-wide layers changing by 5e-4 when the units were switched on, which is the units
being used at an inner dimension of four.

The arms, one whole trainer each:

  library            the shipped form: the library's multiplication, bias and activation after it
  generated          the compiler's own multiplication, candidates from both backends
  generated_triton   the same with the library backend removed, so a generated kernel is used
                     even where the library's is faster, which is what makes the epilogue fuse
  generated_tf32     the same with the size rule replaced by "use the units whenever the
                     configuration asks for them", which is what the library does

ONE TRAINER PER ARM is not an implementation detail. The first version of this probe built one
trainer and swapped four compiled functions on it, and all four came out bitwise identical and
the same speed: the compiler caches its work against the FUNCTION being compiled, so the second,
third and fourth compilations were handed the first one's kernels and the settings never applied.
The per-arm numerics printed below are the check that this has not happened again — two arms that
agree to zero are the same kernels, not two forms that agree.

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

# each arm's compiler settings, and whether the size rule for the matrix units is replaced
SPECS = {
    "library": (None, False),
    "generated": (dict(max_autotune_gemm=True), False),
    "generated_triton": (dict(max_autotune_gemm=True,
                              max_autotune_gemm_backends="TRITON"), False),
    # force_disable_caches on this arm alone, for two reasons: its settings are otherwise
    # identical to generated_triton's, so the compiler's stored code would be reused, and the
    # size rule it replaces is a monkeypatch that is not part of the key that code is stored
    # under. Without it this arm silently re-measures the one above — which is what the run of
    # 2026-08-16-01-44 did, and its two arms agreeing to five digits is how that shows.
    "generated_tf32": (dict(max_autotune_gemm=True, max_autotune_gemm_backends="TRITON",
                            force_disable_caches=True), True),
}


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
    Returns the original so the caller can put it back.
    """
    from torch._inductor.template_heuristics.triton import MMTemplateConfigMixin
    original = MMTemplateConfigMixin.get_extra_kwargs
    MMTemplateConfigMixin.get_extra_kwargs = lambda self, kernel_inputs, op_name: {
        "ALLOW_TF32": torch.backends.cuda.matmul.fp32_precision == "tf32"}
    return original


def build_arm(name, n_copies, style):
    """One whole trainer whose loss is compiled under this arm's settings, warmed and ready."""
    from torch_ppo_rnd import PPORND, production_config
    from torch._inductor.template_heuristics.triton import MMTemplateConfigMixin
    flags, tf32_everywhere = SPECS[name]
    original = patch_allow_tf32() if tf32_everywhere else None
    # the same seed for every arm, so every trainer primes its statistics from the same random
    # actions and collects the same rollout. Without this each arm's batch is different and the
    # loss comparison below measures different data rather than different kernels
    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    # graph capture off: the recording would freeze this arm's programs and the probe wants to
    # time the update stage on its own
    t = PPORND(production_config(n_copies, style=style, one_graph=False, capture_update=False),
               device="cuda")
    # the settings are read when the function is traced, so they are passed to the compiler as
    # options rather than set around it: two arms whose settings differ then compile separately
    if flags is not None:
        t._loss_fn = torch.compile(t._losses, fullgraph=True, dynamic=False, options=flags)
    t.prime_obs_rms()
    Brows = t.cfg.num_steps * t.cfg.n_envs
    t._perm = torch.arange(Brows, device=t.device).expand(
        t.cfg.update_epochs, n_copies, Brows).contiguous()
    t._loss_out = torch.zeros((), device=t.device)
    t._rollout_body()
    t._post_body()
    mb = {k: t._U[k][:, :Brows // t.cfg.num_minibatches] for k in t._U_KEYS}
    # forcing the compilation runs a real update, which MOVES the parameters. Every arm would
    # then read its accuracy from weights its own kernels had already changed, and the accuracy
    # columns would compare two different problems rather than two sets of kernels.
    snapshot = [w.detach().clone() for w in t.param_windows]
    t._update_body_captured()                       # force the compilation to happen here
    with torch.no_grad():
        for window, saved in zip(t.param_windows, snapshot):
            window.copy_(saved)
        t._m.zero_()
        t._v.zero_()
        t._adam_t.zero_()
    if tf32_everywhere:
        MMTemplateConfigMixin.get_extra_kwargs = original
    return t, mb


def numerics(trainer, mb, reference):
    """How far this arm's loss and gradients sit from the reference arm's, from equal inputs."""
    loss = trainer._loss_fn(mb, style_a=False)
    grads = [g.detach().clone() for g in torch.autograd.grad(loss, trainer.trainable)]
    if reference is None:
        return {"loss": float(loss.detach()), "grads": grads}
    # scaled by the largest gradient anywhere rather than each tensor's own largest, so a tensor
    # whose gradient is essentially zero cannot report a meaningless ratio
    biggest = max(a.abs().max().item() for a in reference["grads"])
    worst_abs = max((a - b).abs().max().item() for a, b in zip(reference["grads"], grads))
    return {
        "loss": float(loss.detach()),
        "relative_loss_difference": abs(reference["loss"] - float(loss.detach()))
                                    / max(abs(reference["loss"]), 1e-12),
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
                    default=["library", "generated_triton", "generated_tf32"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert args.arms[0] == "library", "the library form is the reference and must come first"

    arms, batches, num = {}, {}, {}
    reference = None
    for name in args.arms:
        arms[name], batches[name] = build_arm(name, args.n_copies, args.style)
        num[name] = numerics(arms[name], batches[name], reference)
        if reference is None:
            reference = num[name]
        print(f"built {name}")

    # round-robin, reversing the arm order on alternate rounds so a drift in the machine cannot
    # systematically favour whichever arm is measured first
    names = list(args.arms)
    per_round = {n: [] for n in names}
    for rnd in range(args.rounds):
        for name in (names if rnd % 2 == 0 else names[::-1]):
            per_round[name].append(cuda_time(lambda: arms[name]._update_body_captured()))

    med = {n: sorted(v)[len(v) // 2] for n, v in per_round.items()}
    floor = max(max(v) - min(v) for v in per_round.values())
    print(f"\n== the update stage at {args.n_copies} copies, {args.style}, "
          f"{args.rounds} rounds ==")
    rows = []
    for name in names:
        paired = [a - b for a, b in zip(per_round["library"], per_round[name])]
        wins = sum(1 for d in paired if d > 0)
        rows.append({
            "arm": name, "median_us": med[name],
            "change_percent_against_library": (med[name] / med["library"] - 1) * 100.0,
            "rounds_faster_than_library": wins, "rounds": args.rounds,
            "per_round_us": per_round[name], "paired_diff_us": paired,
            **{k: v for k, v in num[name].items() if k != "grads"}})
        print(f"  {name:<18s} {med[name]/1000:8.2f} ms  "
              f"{rows[-1]['change_percent_against_library']:+6.2f}%  "
              f"{wins}/{args.rounds} rounds faster than the library form")
        if name != "library":
            print(f"      loss differs {num[name]['relative_loss_difference']:.2e} relative; "
                  f"worst gradient {num[name]['worst_absolute_gradient_difference']:.2e} "
                  f"absolute against a largest of {num[name]['largest_gradient']:.2e}"
                  + ("   <-- ZERO: this arm ran the reference's kernels, not its own"
                     if num[name]['worst_absolute_gradient_difference'] == 0.0 else ""))
    print(f"  spread within an arm across rounds: {floor/1000:.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    p = (RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_epilogue_"
                   f"C{args.n_copies}{args.tag}.json")
    p.write_text(json.dumps({
        "n_copies": args.n_copies, "style": args.style, "torch": torch.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "noise_floor_us": floor, "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
