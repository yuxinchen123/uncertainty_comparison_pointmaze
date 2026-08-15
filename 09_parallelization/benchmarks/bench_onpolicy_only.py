"""Does a build containing ONLY the on-policy path run faster than the full build?

The task statement asks this directly ("check if we only present the on-policy one in the code
would improve performance, because i am afraid an if sentence would break the fusion"). The two
update styles are already separate functions with no runtime branch, so this measures whether the
mere PRESENCE of the epoch/minibatch code costs anything.

Method:
  1. Build a stripped copy of torch_ppo_rnd.py by deleting, with explicit source anchors that fail
     loudly if the code moves: the epoch/minibatch update method, the non-on-policy branch of the
     loss function (and its parameter), the non-on-policy branch of the captured update body, and
     the permutation buffer and its refills.
  2. Verify the stripped build computes the SAME thing: same seed, same iterations, parameters
     compared bitwise against the full build.
  3. Time the two builds in separate processes in ABBA order (full, stripped, stripped, full) so
     process-level drift cancels, and report the paired difference against the spread.

Usage (serval05, under the H100 lock):
  python bench_onpolicy_only.py --abba --n-copies 8 128
  python bench_onpolicy_only.py --variant full --n-copies 128        # single measurement
  python bench_onpolicy_only.py --verify --n-copies 4                # equivalence only
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "ppo" / "torch_ppo" / "torch_ppo_rnd.py"
STRIPPED = Path("/localtmp/sl5nw/torch_ppo_rnd_onpolicy.py")
RESULTS = Path(__file__).resolve().parent / "results"


def cut_block(text, start_anchor, end_anchor, what):
    """Delete from start_anchor up to (not including) end_anchor; both must appear exactly once."""
    i = text.index(start_anchor)
    j = text.index(end_anchor, i)
    assert text.count(start_anchor) == 1, f"anchor not unique for {what}"
    return text[:i] + text[j:]


def make_stripped_source() -> str:
    """Produce the on-policy-only source. Every edit is anchored; a moved anchor raises."""
    s = SRC.read_text()

    # 1. the whole epoch/minibatch update method
    s = cut_block(s, "    def update_epoch_minibatch(self, batch):",
                  "    _U_KEYS = [", "update_epoch_minibatch")

    # 2. the loss function's non-on-policy branch, and the style parameter itself
    old_loss = '''        if style_a:
            pg = (-a_n * ratio).mean(dim=1)
            v_ext = 0.5 * (vext - mb["ret_ext"]).square().mean(dim=1)
        else:
            pg = torch.maximum(-a_n * ratio,
                               -a_n * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean(dim=1)
            vc = mb["vext_old"] + (vext - mb["vext_old"]).clamp(-cfg.clip_coef, cfg.clip_coef)
            v_ext = 0.5 * torch.maximum((vext - mb["ret_ext"]).square(),
                                        (vc - mb["ret_ext"]).square()).mean(dim=1)'''
    new_loss = '''        pg = (-a_n * ratio).mean(dim=1)
        v_ext = 0.5 * (vext - mb["ret_ext"]).square().mean(dim=1)'''
    assert s.count(old_loss) == 1, "loss-branch anchor moved"
    s = s.replace(old_loss, new_loss)
    old_doc = '''    def _losses(self, mb, style_a):
        """Per-copy losses on one (mini)batch dict; returns the scalar sum over copies.

        style_a=True drops the CLIP machinery only (gradient-exact at ratio == 1, where the
        clip branches coincide in value AND derivative). The ratio itself is KEPT: it carries
        the policy gradient -A * grad(log pi); replacing the surrogate by mean(-A) would make
        the policy loss a constant. (This corrects spec section 11 — noted there.)
        """'''
    new_doc = '''    def _losses(self, mb):
        """Per-copy losses on the full batch; returns the scalar sum over copies.

        On-policy-only build: the clip machinery is absent (gradient-exact at ratio == 1,
        where the clip branches coincide in value and derivative). The ratio itself is kept:
        it carries the policy gradient.
        """'''
    assert s.count(old_doc) == 1, "loss docstring anchor moved"
    s = s.replace(old_doc, new_doc)

    # the stripped copy lives outside the source tree, so its relative path lookup for the
    # shared modules must become absolute
    old_base = "BASE = Path(__file__).resolve().parent.parent.parent"
    assert s.count(old_base) == 1, "BASE anchor moved"
    s = s.replace(old_base, f'BASE = Path("{BASE}")')
    s = s.replace("self._loss_fn(self._U, style_a=True)", "self._loss_fn(self._U)")
    s = s.replace("self._loss_fn(batch, style_a=True)", "self._loss_fn(batch)")

    # 3. the captured update body's non-on-policy branch
    i = s.index("        if cfg.update_style == \"full_batch\":")
    j = s.index("        self._loss_out.copy_(loss.detach())", i)
    body = '''        loss = self._loss_fn(self._U)
        loss.backward()
        self._clip_per_copy_and_step()
'''
    s = s[:i] + body + s[j:]

    # 4. the permutation buffer, its two allocations and its two refills
    s = re.sub(r"        self\._perm = torch\.arange\(Brows, device=self\.device\)\.expand\(\n"
               r"            cfg\.update_epochs, C, Brows\)\.contiguous\(\)\n", "", s)
    s = re.sub(r"        if (?:self\.)?cfg\.update_style != \"full_batch\":\n"
               r"(?:            .*\n)+?(?=        self\._(?:update_graph|iteration_graph)\.replay\(\))",
               "", s)
    # 5. the driver's style dispatch (only the on-policy update remains reachable)
    old_dispatch = """        if cfg.capture_update:
            update = self.update_captured
        elif cfg.update_style == "full_batch":
            update = self.update_full_batch
        else:
            update = self.update_epoch_minibatch"""
    new_dispatch = """        update = self.update_captured if cfg.capture_update else self.update_full_batch"""
    assert s.count(old_dispatch) == 1, "train dispatch anchor moved"
    s = s.replace(old_dispatch, new_dispatch)
    # the config knob and the smoke entry point keep no meaning without the other style
    s = s.replace('    update_style: str = "epoch_minibatch"  # or "full_batch"',
                  '    update_style: str = "full_batch"  # on-policy-only build: the only style')
    s = s.replace('update_style=args.style)', 'update_style="full_batch")')

    assert "self._perm" not in s, "a permutation reference survived the strip"
    assert "update_epoch_minibatch" not in s, "an epoch/minibatch reference survived the strip"
    assert "style_a" not in s, "a style parameter survived the strip"
    return s


def load_variant(variant):
    """Import the full or the stripped trainer module under its own module name."""
    import importlib.util
    if variant == "full":
        path = SRC
    else:
        STRIPPED.parent.mkdir(parents=True, exist_ok=True)
        STRIPPED.write_text(make_stripped_source())
        path = STRIPPED
    sys.path.insert(0, str(SRC.parent))
    spec = importlib.util.spec_from_file_location(f"trainer_{variant}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, n_copies, device="cuda"):
    """A trainer in the final production configuration, on-policy style."""
    cfg = mod.PPOConfig(n_copies=n_copies, update_style="full_batch", rollout_mode="capture",
                        capture_update=True, fused_adam=True, one_graph=True, tf32=True)
    return mod.PPORND(cfg, device=device)


def verify(n_copies, device):
    """Both builds must produce bitwise-identical parameters after the same iterations."""
    import torch
    out = {}
    for variant in ("full", "stripped"):
        mod = load_variant(variant)
        torch.manual_seed(11)
        t = build(mod, n_copies, device)
        t.prime_obs_rms()
        upd = t.update_full_batch
        for it in range(3):
            torch.manual_seed(500 + it)
            b = t.rollout()
            upd(b)
        out[variant] = [p.detach().clone() for p in t.trainable]
        del t
        if device == "cuda":
            torch.cuda.empty_cache()
    same = all(torch.equal(a, b) for a, b in zip(out["full"], out["stripped"]))
    worst = max((a - b).abs().max().item() for a, b in zip(out["full"], out["stripped"]))
    print(f"equivalence: {'BITWISE IDENTICAL' if same else 'DIFFERENT'} (worst |d| {worst:.3e})")
    return same


def time_variant(variant, n_copies, iters, warmup):
    """Median seconds per iteration for one build (own process when run via --abba)."""
    import torch
    mod = load_variant(variant)
    t = build(mod, n_copies)
    t.prime_obs_rms()
    t._build_iteration_graph()
    for _ in range(warmup):
        t.iteration_captured()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a = time.perf_counter()
        t.iteration_captured()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - a)
    times.sort()
    return {"variant": variant, "n_copies": n_copies,
            "median_sec_per_iter": times[len(times) // 2],
            "p10": times[len(times) // 10], "p90": times[int(len(times) * 0.9)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "stripped"])
    ap.add_argument("--n-copies", type=int, nargs="+", default=[128])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--abba", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.verify:
        ok = verify(args.n_copies[0], args.device)
        sys.exit(0 if ok else 1)

    if args.variant:
        r = time_variant(args.variant, args.n_copies[0], args.iters, args.warmup)
        print("RESULT " + json.dumps(r))
        return

    # ABBA at the process level: full, stripped, stripped, full — drift cancels in the pairing
    rows = []
    for c in args.n_copies:
        seq = ["full", "stripped", "stripped", "full"]
        got = {"full": [], "stripped": []}
        for v in seq:
            cmd = [sys.executable, __file__, "--variant", v, "--n-copies", str(c),
                   "--iters", str(args.iters), "--warmup", str(args.warmup)]
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
            line = [l for l in out.splitlines() if l.startswith("RESULT ")]
            assert line, f"no result from {v} at C={c}: {out[-2000:]}"
            r = json.loads(line[0][len("RESULT "):])
            got[r["variant"]].append(r["median_sec_per_iter"])
        f = sorted(got["full"])[len(got["full"]) // 2]
        s = sorted(got["stripped"])[len(got["stripped"]) // 2]
        spread = max(max(got["full"]) - min(got["full"]), max(got["stripped"]) - min(got["stripped"]))
        rows.append({"n_copies": c, "full_ms": f * 1e3, "stripped_ms": s * 1e3,
                     "difference_ms": (f - s) * 1e3, "within_variant_spread_ms": spread * 1e3,
                     "full_runs_ms": [x * 1e3 for x in got["full"]],
                     "stripped_runs_ms": [x * 1e3 for x in got["stripped"]]})
        print(f"C={c:>4d}: full {f*1e3:.2f} ms | on-policy-only {s*1e3:.2f} ms | "
              f"difference {(f-s)*1e3:+.2f} ms | within-variant spread {spread*1e3:.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_onpolicy_only_abba.json"
    out.write_text(json.dumps({"rows": rows, "iters": args.iters}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
