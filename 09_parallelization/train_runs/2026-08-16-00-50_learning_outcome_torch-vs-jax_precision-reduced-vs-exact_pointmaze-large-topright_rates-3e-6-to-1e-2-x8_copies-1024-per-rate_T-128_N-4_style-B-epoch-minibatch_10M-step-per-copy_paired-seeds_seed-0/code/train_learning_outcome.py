"""One configuration of the learning-outcome campaign: {torch, jax} x {reduced, exact} precision.

Trains a learning-rate sweep (8 rates x 1,024 copies) to a fixed step budget and records, every
`--history-every` iterations, each copy's extrinsic reward, intrinsic reward and maze coverage.

Three properties this driver exists to provide, none of which the trainers have on their own:

1. **The requested precision is verified before any training happens.** A knob that was silently
   ignored would produce two identical curves, which reads exactly like the finding "precision
   does not matter". The probe multiplies matrices of the trainer's own shapes against a double
   precision reference and refuses to continue if the measured error does not match the mode
   that was asked for.
2. **Resumable in chunks.** State is written every `--checkpoint-every` iterations, so a kill
   costs one chunk rather than the run. Re-running the same command continues where it stopped.
3. **Per-item progress on disk while it runs**, flushed, so the run is observable from another
   machine without touching the card.

The learning-rate annealing runs on the ABSOLUTE iteration index, so a resumed run continues one
schedule rather than restarting it.

Run (on serval05, under locks/gpu_run.sh):
  PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python train_learning_outcome.py \
      --framework torch --precision reduced --outdir <run folder>
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]      # .../09_parallelization
RATES = (3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


# ---------------- small file helpers (append-only, flushed, idempotent on re-run) ----------

def append_jsonl(path: Path, row: dict):
    """Append one record and flush it, so progress is visible while the run continues."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def truncate_jsonl_after(path: Path, last_iteration: int):
    """Drop rows written past the checkpoint, so a resumed run re-runs them exactly once.

    before: history.jsonl holds iterations 200, 400, ..., 2600 but the checkpoint is at 2000
    after:  it holds 200 .. 2000, and the resumed run appends 2200 onwards
    """
    if not path.exists():
        return
    kept = [ln for ln in path.read_text().splitlines()
            if ln.strip() and json.loads(ln)["iteration"] <= last_iteration]
    path.write_text("".join(ln + "\n" for ln in kept))


def write_json(path: Path, obj: dict):
    """Write a JSON file atomically (temporary file, then rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def git_hash():
    """Short hash of the code that is about to run."""
    return subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ---------------- precision: set it, then prove it took ----------------

REDUCED_MIN_ERROR = 1e-4     # the card's reduced matrix mode keeps 10 mantissa bits: ~1e-3 here
EXACT_MAX_ERROR = 1e-5       # exact single precision on these shapes: ~1e-7


def check_probe(probe: dict, precision: str):
    """Refuse to train when the measured matrix error does not match the requested mode."""
    err = probe["relative_error_against_float64"]
    if precision == "reduced":
        assert err > REDUCED_MIN_ERROR, (
            f"asked for reduced precision but the matrix multiplication is exact to {err:.2e} — "
            "the knob did not reach the card")
    else:
        assert err < EXACT_MAX_ERROR, (
            f"asked for exact single precision but the matrix multiplication is only accurate to "
            f"{err:.2e} — the knob did not reach the card")


def probe_torch(precision: str) -> dict:
    """Measure what a float32 batched matrix multiplication actually does on this card."""
    import torch
    # the trainer's own update-stage shape: one copy's 512 rows against a 64x64 layer
    # before: x [64, 512, 64] and W [64, 64, 64] in float32
    # after:  the same product in float64, and the largest disagreement between them
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(64, 512, 64, device="cuda", generator=g)
    w = torch.randn(64, 64, 64, device="cuda", generator=g)
    y32 = torch.bmm(x, w)
    y64 = torch.bmm(x.double(), w.double())
    err = float((y32.double() - y64).abs().max() / y64.abs().max())
    return {"framework": "torch", "requested": precision,
            "declared_setting": torch.get_float32_matmul_precision(),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "relative_error_against_float64": err}


def probe_jax(precision: str) -> dict:
    """The same measurement in JAX (x64 is already enabled by the trainer module)."""
    import jax
    import jax.numpy as jnp
    k = jax.random.PRNGKey(7)
    x = jax.random.normal(jax.random.fold_in(k, 0), (64, 512, 64), jnp.float32)
    w = jax.random.normal(jax.random.fold_in(k, 1), (64, 64, 64), jnp.float32)
    y32 = jnp.matmul(x, w)
    y64 = jnp.matmul(x.astype(jnp.float64), w.astype(jnp.float64))
    err = float(jnp.abs(y32.astype(jnp.float64) - y64).max() / jnp.abs(y64).max())
    return {"framework": "jax", "requested": precision,
            "declared_setting": str(jax.config.jax_default_matmul_precision),
            "relative_error_against_float64": err}


# ---------------- history rows ----------------

def history_row(iteration, global_step, seconds, reward, rint, coverage):
    """One recorded iteration, rounded so the file stays a readable size.

    before: reward is a float32 sum of 0/1 rewards over 512 steps, e.g. 22.000000476837158
    after:  the integer count of steps that copy spent inside the goal radius, e.g. 22
    """
    return {"iteration": int(iteration), "global_step": int(global_step),
            "seconds": round(float(seconds), 2),
            "reward_ext_sum_per_copy": [int(round(float(v))) for v in reward],
            "rint_mean_per_copy": [float(f"{float(v):.6g}") for v in rint],
            "coverage_per_copy": [round(float(v), 5) for v in coverage]}


# ---------------- the two framework drivers ----------------

def run_torch(args, outdir: Path, ckpt_dir: Path, log):
    """Train one PyTorch configuration, chunked and resumable."""
    import torch
    sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
    from torch_ppo_rnd import PPORND, sweep_config

    # precision first: the trainer sets it from its own config, so build the config, then probe
    cfg = sweep_config(RATES, args.copies_per_rate, style=args.style,
                       base_seed=args.base_seed, tf32=(args.precision == "reduced"))
    torch.manual_seed(args.base_seed)          # the action noise and shuffles are otherwise
    torch.cuda.manual_seed_all(args.base_seed)  # seeded at random by torch itself
    torch.cuda.reset_peak_memory_stats()
    # timed from before the trainer is constructed: building 8,192 copies' weights is itself
    # minutes of work, and a "build" figure that starts after it understates what a resume costs
    t_build = time.time()
    trainer = PPORND(cfg, device="cuda")
    probe = probe_torch(args.precision)
    check_probe(probe, args.precision)
    log(f"precision probe: {json.dumps(probe)}")
    trainer.prime_obs_rms()
    trainer._build_iteration_graph()
    build_seconds = time.time() - t_build
    log(f"built in {build_seconds:.1f} s")

    # the tensors that carry everything one iteration reads and writes
    state_tensors = {
        "flat": trainer._flat, "m": trainer._m, "v": trainer._v, "adam_t": trainer._adam_t,
        "S_pos": trainer._S_pos, "S_vel": trainer._S_vel, "S_goal": trainer._S_goal,
        "S_sc": trainer._S_sc, "S_rc": trainer._S_rc, "S_obs": trainer._S_obs,
        "obs_mean": trainer.obs_rms.mean, "obs_var": trainer.obs_rms.var,
        "obs_count": trainer.obs_rms.count, "int_mean": trainer.int_rms.mean,
        "int_var": trainer.int_rms.var, "int_count": trainer.int_rms.count,
        "int_filter": trainer.int_filter, "visited": trainer._visited,
    }

    def save(iteration):
        """Write every tensor the next iteration reads, plus the RNG, to the local disk."""
        blob = {k: t.detach().cpu() for k, t in state_tensors.items()}
        blob["iteration"] = iteration
        blob["global_step"] = trainer.global_step
        blob["cpu_rng"] = torch.get_rng_state()
        blob["cuda_rng"] = torch.cuda.get_rng_state()
        tmp = ckpt_dir / "latest.pt.tmp"
        torch.save(blob, tmp)
        tmp.replace(ckpt_dir / "latest.pt")

    def load():
        """Restore into the SAME storages the captured graph holds, and return the iteration."""
        blob = torch.load(ckpt_dir / "latest.pt", map_location="cuda", weights_only=False)
        for k, t in state_tensors.items():
            t.copy_(blob[k])
        trainer.global_step = int(blob["global_step"])
        torch.set_rng_state(blob["cpu_rng"].cpu())
        torch.cuda.set_rng_state(blob["cuda_rng"].cpu())
        return int(blob["iteration"])

    start_iter = load() if (ckpt_dir / "latest.pt").exists() else 0
    if start_iter:
        log(f"resuming at iteration {start_iter + 1} of {args.iterations}")
        truncate_jsonl_after(outdir / "history.jsonl", start_iter)
        truncate_jsonl_after(outdir / "progress.jsonl", start_iter)

    def read_metrics():
        """The three per-copy quantities recorded at the history cadence."""
        cov = trainer._visited[:, trainer._open_cells].float().mean(dim=1)
        return (trainer._log_rext_sum.tolist(), trainer._log_rint_mean.tolist(), cov.tolist())

    # --stop-after ends the loop early WITHOUT changing the annealing denominator
    last = args.stop_after or args.iterations
    t0 = time.time()
    last_log = t0
    for it in range(start_iter + 1, last + 1):
        trainer._lr_scale.fill_(1.0 - (it - 1.0) / args.iterations)
        trainer.iteration_captured()
        now = time.time()
        record = it % args.history_every == 0 or it == last
        checkpoint = it % args.checkpoint_every == 0 or it == last
        status = now - last_log >= args.log_every or it == last
        # one device-to-host read serves all three, so a recorded iteration syncs once
        if record or checkpoint or status:
            rew, rint, cov = read_metrics()
        if record:
            append_jsonl(outdir / "history.jsonl",
                         history_row(it, trainer.global_step, now - t0, rew, rint, cov))
        if checkpoint:
            save(it)
            append_jsonl(outdir / "progress.jsonl", {
                "iteration": it, "global_step": trainer.global_step,
                "seconds": round(now - t0, 1),
                "sec_per_iteration": round((now - t0) / (it - start_iter), 5),
                "reward_mean": round(sum(rew) / len(rew), 4),
                "coverage_mean": round(sum(cov) / len(cov), 5)})
        if status:
            last_log = now
            log(f"iter {it}/{args.iterations} step {trainer.global_step} "
                f"reward/copy mean {sum(rew) / len(rew):.3f} max {max(rew):.0f} "
                f"coverage mean {sum(cov) / len(cov):.3f} elapsed {now - t0:.0f}s")
    seconds = time.time() - t0

    rew, rint, cov = read_metrics()
    return {"train_seconds": seconds, "build_seconds": build_seconds,
            "iterations_run": last - start_iter,
            "sec_per_iteration": seconds / max(1, last - start_iter),
            "global_step": trainer.global_step,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "learning_rate_per_copy": trainer.lr_per_copy.tolist(),
            "group_index": trainer.copy_group.tolist(),
            "final_reward_per_copy": [int(round(v)) for v in rew],
            "final_coverage_per_copy": [round(v, 5) for v in cov],
            "precision_probe": probe, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "config": str(trainer.cfg)}


def run_jax(args, outdir: Path, ckpt_dir: Path, log):
    """Train one JAX configuration, chunked and resumable."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    if args.precision == "exact":
        # set before anything is traced, so every compiled program is built for this mode
        jax.config.update("jax_default_matmul_precision", "highest")
    sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
    from jax_ppo_rnd import JaxPPORND, sweep_config

    probe = probe_jax(args.precision)
    check_probe(probe, args.precision)
    log(f"precision probe: {json.dumps(probe)}")

    cfg = sweep_config(RATES, args.copies_per_rate, style=args.style,
                       base_seed=args.base_seed, track_coverage=True)
    # timed from before the trainer is constructed: the per-copy weight initialisation dispatches
    # about a dozen small operations per copy and takes minutes at 8,192 copies, so a "build"
    # figure that starts after it understates what a resume costs
    t_build = time.time()
    trainer = JaxPPORND(cfg)
    key = jax.random.PRNGKey(args.base_seed + 1_000_003 * cfg.base_seed)
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    jax.block_until_ready(state.params)
    build_seconds = time.time() - t_build
    log(f"built and primed in {build_seconds:.1f} s")

    # the whole TrainState is one pytree, so its leaves save and restore as one array list
    _, treedef = jax.tree.flatten(state)

    def save(st, iteration):
        """Write every leaf of the training state, plus the iteration it belongs to."""
        leaves = [np.asarray(x) for x in jax.tree.leaves(st)]
        tmp = ckpt_dir / "latest.npz.tmp"
        # written through an open file object: np.savez appends ".npz" to a path that lacks it,
        # which would put the temporary file somewhere the rename below does not look
        with open(tmp, "wb") as f:
            np.savez(f, iteration=np.asarray(iteration),
                     **{f"leaf_{i}": a for i, a in enumerate(leaves)})
        tmp.replace(ckpt_dir / "latest.npz")

    def load():
        """Rebuild the training state from the saved leaves; returns (state, iteration)."""
        z = np.load(ckpt_dir / "latest.npz")
        n = len(jax.tree.leaves(state))
        leaves = [jnp.asarray(z[f"leaf_{i}"]) for i in range(n)]
        return jax.tree.unflatten(treedef, leaves), int(z["iteration"])

    start_iter = 0
    if (ckpt_dir / "latest.npz").exists():
        state, start_iter = load()
        log(f"resuming at iteration {start_iter + 1} of {args.iterations}")
        truncate_jsonl_after(outdir / "history.jsonl", start_iter)
        truncate_jsonl_after(outdir / "progress.jsonl", start_iter)

    steps_per_iter = cfg.num_steps * cfg.n_copies * cfg.n_envs
    # --stop-after ends the loop early WITHOUT changing the annealing denominator
    last = args.stop_after or args.iterations
    t0 = time.time()
    last_log = t0
    metrics = None
    for it in range(start_iter + 1, last + 1):
        state, metrics = trainer._iterate(state, jax.random.fold_in(key, it),
                                          trainer.lr_argument(it, args.iterations))
        now = time.time()
        record = it % args.history_every == 0 or it == last
        checkpoint = it % args.checkpoint_every == 0 or it == last
        if record or checkpoint or now - last_log >= args.log_every:
            rew = np.asarray(metrics["reward_ext_sum"])
            rint = np.asarray(metrics["rint_mean"])
            cov = trainer.coverage(state)
        if record:
            append_jsonl(outdir / "history.jsonl",
                         history_row(it, it * steps_per_iter, now - t0, rew, rint, cov))
        if checkpoint:
            save(state, it)
            append_jsonl(outdir / "progress.jsonl", {
                "iteration": it, "global_step": it * steps_per_iter,
                "seconds": round(now - t0, 1),
                "sec_per_iteration": round((now - t0) / (it - start_iter), 5),
                "reward_mean": round(float(rew.mean()), 4),
                "coverage_mean": round(float(cov.mean()), 5)})
        if now - last_log >= args.log_every or it == last:
            last_log = now
            log(f"iter {it}/{args.iterations} step {it * steps_per_iter} "
                f"reward/copy mean {rew.mean():.3f} max {rew.max():.0f} "
                f"coverage mean {cov.mean():.3f} elapsed {now - t0:.0f}s")
    jax.block_until_ready(state.params)
    seconds = time.time() - t0

    rew = np.asarray(metrics["reward_ext_sum"])
    cov = trainer.coverage(state)
    peak = jax.local_devices()[0].memory_stats()
    return {"train_seconds": seconds, "build_seconds": build_seconds,
            "iterations_run": last - start_iter,
            "sec_per_iteration": seconds / max(1, last - start_iter),
            "global_step": last * steps_per_iter,
            "peak_vram_mb": peak["peak_bytes_in_use"] / 2 ** 20 if peak else None,
            "learning_rate_per_copy": np.asarray(trainer.lr_per_copy).tolist(),
            "group_index": trainer.copy_group.tolist(),
            "final_reward_per_copy": [int(round(float(v))) for v in rew],
            "final_coverage_per_copy": [round(float(v), 5) for v in cov],
            "precision_probe": probe, "jax": jax.__version__,
            "gpu": str(jax.local_devices()[0]), "config": str(cfg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=["torch", "jax"])
    ap.add_argument("--precision", required=True, choices=["reduced", "exact"])
    ap.add_argument("--outdir", required=True, help="the run folder")
    ap.add_argument("--ckpt-root", default="/localtmp/sl5nw/rnd09_learning_outcome",
                    help="local disk for the (multi-gigabyte) checkpoints")
    ap.add_argument("--copies-per-rate", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=19531)   # 9,999,872 steps per copy
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--history-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=300, help="seconds between status lines")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop at this iteration WITHOUT finishing the run, to test the resume. "
                         "--iterations keeps its value, so the annealing schedule is the one the "
                         "full run would have followed; lowering --iterations instead would "
                         "anneal faster and the two runs would differ for that reason alone")
    args = ap.parse_args()

    tag = f"{args.framework}_{args.precision}"
    outdir = Path(args.outdir) / "data" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_root) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.outdir) / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        """One status line, to stdout and to the run folder's log, flushed both ways."""
        line = f"[{tag}] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
            f.flush()

    if (outdir / "record.json").exists():
        log("record.json exists — this configuration is already finished, skipping")
        return

    log(f"start: {args.iterations} iterations, {args.copies_per_rate} copies per rate, "
        f"{len(RATES)} rates, style {args.style}")
    runner = run_torch if args.framework == "torch" else run_jax
    out = runner(args, outdir, ckpt_dir, log)
    if args.stop_after:
        log(f"stopped early at iteration {args.stop_after} (resume test); no record written")
        return
    out.update({"framework": args.framework, "precision": args.precision,
                "rates": list(RATES), "copies_per_rate": args.copies_per_rate,
                "n_copies": args.copies_per_rate * len(RATES), "style": args.style,
                "iterations": args.iterations, "base_seed": args.base_seed,
                "steps_per_copy": args.iterations * 128 * 4, "git": git_hash()})
    write_json(outdir / "record.json", out)
    log(f"DONE in {out['train_seconds']:.0f} s "
        f"({out['sec_per_iteration'] * 1000:.1f} ms per iteration)")


if __name__ == "__main__":
    main()
