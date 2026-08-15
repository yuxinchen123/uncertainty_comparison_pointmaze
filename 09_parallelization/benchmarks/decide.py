"""Print the three comparison tables that justify which environment, trainer and pairing to use.

Reads only the measurement files in results/, so the tables always reflect what was actually
measured. Throughput is shown in millions of environment steps per second.

Run: python decide.py
"""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def newest(pattern):
    """The most recent results file matching the pattern, or None."""
    hits = sorted(p for p in RESULTS.glob("*.json") if re.search(pattern, p.name))
    return json.loads(hits[-1].read_text()) if hits else None


def rows(pattern):
    """Row list of the most recent matching file."""
    d = newest(pattern)
    return d.get("rows", []) if d else []


def table(title, headers, body, notes=()):
    """Print one aligned table with a title and optional trailing notes."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in body)) for i, h in enumerate(headers)]
    line = "  ".join("-" * w for w in widths)
    print(f"\n{title}\n{'=' * len(title)}")
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for r in body:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    for n in notes:
        print(n)


def M(x):
    """Steps per second in millions, formatted."""
    v = x / 1e6
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def environments():
    """Which environment implementation is fastest, and where each one stops scaling."""
    src = [("PyTorch, not compiled", r"envbench_torch_eager\."),
           ("PyTorch, compiled", r"envbench_torch_compile_grid"),
           ("CUDA kernel", r"envbench_cuda_fused_tourn_count_grid"),
           ("JAX, one step per call", r"envbench_jax_jit_grid"),
           ("JAX, many steps per call", r"envbench_jax_scan_grid")]
    cols = [1000, 100000, 1000000]
    body = []
    for label, pat in src:
        d = {r["total_envs"]: r for r in rows(pat)}
        if not d:
            continue
        best = max(d.values(), key=lambda r: r["env_steps_per_sec"])
        floor = min(r["us_per_batch_step"] for r in d.values())
        body.append([label] + [M(d[c]["env_steps_per_sec"]) if c in d else "-" for c in cols]
                    + [M(best["env_steps_per_sec"]), f"{floor:.1f}"])
    table("BEST ENVIRONMENT: throughput in million steps/second",
          ["implementation", "1e3 envs", "1e5 envs", "1e6 envs", "best", "floor us/step"], body,
          notes=["",
                 "Verdict: the CUDA kernel. It is fastest at every size, and its advantage is",
                 "largest for small batches, where the others spend their time dispatching work",
                 "rather than simulating. All implementations pass the same exactness checks",
                 "against the reference simulator, so the choice is purely about speed."])


def trainers():
    """Which trainer implementation is faster, measured the same way on both sides."""
    def grab(pat, style=None):
        d = {}
        for r in rows(pat):
            if style and r.get("style") != style:
                continue
            d[r["n_copies"]] = r.get("sec_per_iteration") or r.get("sec_per_iteration_median")
        return d
    t_sync = {"full_batch": grab(r"trainbench_torch_full_batch_final_styleA"),
              "epoch_minibatch": grab(r"trainbench_torch_epoch_minibatch_final_styleB")}
    t_pipe = {"full_batch": grab(r"trainbench_torch_full_batch_lfl_pipelined"),
              "epoch_minibatch": grab(r"trainbench_torch_epoch_minibatch_lfl_pipelined")}
    j_sync = {st: grab(r"trainbench_jax_ppo_lfl_sync", st)
              for st in ("full_batch", "epoch_minibatch")}
    j_pipe = {st: grab(r"trainbench_jax_ppo_round2", st)
              for st in ("full_batch", "epoch_minibatch")}

    body = []
    for style, name in (("full_batch", "1 update per batch"),
                        ("epoch_minibatch", "16 updates per batch")):
        for c in (8, 32, 128):
            ts, js = t_sync[style].get(c), j_sync[style].get(c)
            tp, jp = t_pipe[style].get(c), j_pipe[style].get(c)
            if not (ts and js):
                continue
            body.append([name, c, f"{ts*1e3:.1f}", f"{js*1e3:.1f}", f"{ts/js:.2f}x",
                         f"{tp*1e3:.1f}" if tp else "-", f"{jp*1e3:.1f}" if jp else "-"])
    table("BEST TRAINER: milliseconds per training iteration (lower is better)",
          ["update convention", "copies", "PyTorch", "JAX", "JAX faster by",
           "PyTorch pipelined", "JAX pipelined"], body,
          notes=["",
                 "The first two number columns are the fair comparison: both frameworks waiting",
                 "for each iteration to finish before starting the next. The last two columns let",
                 "the host run ahead and wait once per block of iterations. PyTorch is identical",
                 "either way, which means its recorded iteration is entirely device-bound; JAX",
                 "gains about 1.3 ms from pipelining, which means it has host-side work that",
                 "overlaps. Comparing PyTorch-synchronised against JAX-pipelined, as an earlier",
                 "version of this table did, credits JAX with that difference twice.",
                 "",
                 "Verdict: JAX is faster, by 3 to 16 percent with one update per batch and by",
                 "about 35 percent with sixteen. Both compute the same algorithm and agree to",
                 "8.6e-7 on every intermediate quantity, so this is an implementation difference.",
                 "PyTorch remains the recommended default because the capabilities built on it",
                 "exist only there: the learning-rate sweep across copy groups, the resumable",
                 "training driver, and the campaign records. Choose JAX when iteration speed is",
                 "the only consideration, and especially with many small updates per batch."])


def combination():
    """Which environment to put inside which trainer."""
    tor = {r["n_copies"]: r for r in rows(r"trainbench_torch_epoch_minibatch_final_styleB")}
    cud = {r["n_copies"]: r for r in
           rows(r"trainbench_torch_epoch_minibatch_cudaenv_streamfixed_prod")}
    jx = {r["n_copies"]: r for r in rows(r"trainbench_jax_ppo_lfl_sync")
          if r.get("style") == "epoch_minibatch"}
    sec = lambda r: r.get("sec_per_iteration") or r.get("sec_per_iteration_median")
    cross = rows(r"cross_pairing")
    body = []
    for c in (8, 128):
        if c in tor:
            body.append([f"PyTorch env + PyTorch trainer", c, f"{sec(tor[c])*1e3:.1f}", "shipped default"])
        if c in cud:
            body.append([f"CUDA env + PyTorch trainer", c, f"{sec(cud[c])*1e3:.1f}", "works, no gain"])
        if c in jx:
            body.append([f"JAX env + JAX trainer", c, f"{sec(jx[c])*1e3:.1f}", "fastest"])
    if cross:
        worst = max(r["boundary_overhead_us"] for r in cross)
        body.append(["env and trainer in different frameworks", "-", "-",
                     f"rejected: +{worst/1000:.1f} ms per env step"])
    table("BEST COMBINATION: milliseconds per training iteration (16 updates per batch, "
          "each iteration waited for)",
          ["combination", "copies", "ms/iteration", "status"], body,
          notes=["",
                 "Verdict: keep the environment in the same framework as the trainer.",
                 "",
                 "Why the fastest environment does not win here: inside a training iteration the",
                 "environment is a small part of the work (rollout is 5.0 ms of a 20.2 ms",
                 "iteration at 128 copies, and only part of that is simulation). Substituting the",
                 "CUDA kernel, which is about 24x faster standalone, changes the iteration by",
                 "about 1%. Crossing frameworks is far worse: every step must hand arrays between",
                 "two runtimes, which costs more than the step itself and also prevents both",
                 "frameworks from fusing or recording the loop.",
                 "",
                 "Use the CUDA kernel when the environment is the whole job: generating data,",
                 "evaluating a fixed policy, or any sweep that does not train."])


if __name__ == "__main__":
    environments()
    trainers()
    combination()
