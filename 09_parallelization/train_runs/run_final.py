"""Final-deliverable driver: sequential PPO+RND trainings at n_copies in {8,16,32,64,128,...}.

Resumable by construction: one JSON per copy-count under <outdir>/data/, written on
completion; a re-run skips counts whose JSON already exists. Live progress streams to
<outdir>/logs/copies_<C>.log (per-line flush). Run on serval05 under the H100 lock.

Usage:
  python run_final.py --outdir <run_folder> --n-copies 8 16 32 64 128 \
      --iterations 20000 --style epoch_minibatch
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))


def run_one(n_copies, args, outdir):
    """Train one copy-count to completion and write its JSON record."""
    import torch
    from torch_ppo_rnd import PPOConfig, PPORND

    data = outdir / "data" / f"copies_{n_copies}_{args.style}.json"
    if data.exists():
        print(f"[resume] {data.name} exists — skipping", flush=True)
        return
    torch.cuda.reset_peak_memory_stats()
    log_path = outdir / "logs" / f"copies_{n_copies}_{args.style}.log"
    log_f = open(log_path, "a")

    def log_fn(msg):
        # per-line flush so the run is observable live (live-progress rule)
        print(f"[C={n_copies}] {msg}", flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    from torch_ppo_rnd import production_config
    cfg = production_config(n_copies, style=args.style, one_graph=args.one_graph,
                            base_seed=args.base_seed)
    t_build = time.time()
    trainer = PPORND(cfg, device="cuda")
    stats = trainer.train(args.iterations, log_every_seconds=args.log_every,
                          log_fn=log_fn, history_every=args.history_every)
    cov = trainer._visited[:, trainer._open_cells].float().mean(dim=1)
    record = {
        "n_copies": n_copies, "style": args.style, "one_graph": args.one_graph,
        "iterations": args.iterations, "base_seed": args.base_seed,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "build_plus_train_seconds": time.time() - t_build,
        "train_seconds": stats["seconds"],
        "global_step": stats["global_step"],
        "env_steps_per_sec": stats["global_step"] / stats["seconds"],
        "final_coverage_per_copy": cov.tolist(),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "history": stats["history"],
    }
    data.parent.mkdir(parents=True, exist_ok=True)
    tmp = data.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(data)
    log_fn(f"DONE C={n_copies}: {record['env_steps_per_sec']:.3e} env-steps/s, "
           f"coverage mean {cov.mean():.3f}, vram {record['peak_vram_mb']:.0f} MB")
    log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-copies", type=int, nargs="+", required=True)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--one-graph", action="store_true")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--history-every", type=int, default=50)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    if len(args.n_copies) == 1:
        run_one(args.n_copies[0], args, outdir)
        return
    # one child process per copy-count: captured graphs hold memory pools for the process
    # lifetime, so separate processes keep the per-count peak-VRAM numbers honest
    for c in args.n_copies:
        cmd = [sys.executable, __file__, "--outdir", str(outdir), "--n-copies", str(c),
               "--iterations", str(args.iterations), "--style", args.style,
               "--base-seed", str(args.base_seed), "--log-every", str(args.log_every),
               "--history-every", str(args.history_every)]
        if args.one_graph:
            cmd.append("--one-graph")
        subprocess.run(cmd, check=True)
    print("ALL_RUNS_DONE", flush=True)


if __name__ == "__main__":
    main()
