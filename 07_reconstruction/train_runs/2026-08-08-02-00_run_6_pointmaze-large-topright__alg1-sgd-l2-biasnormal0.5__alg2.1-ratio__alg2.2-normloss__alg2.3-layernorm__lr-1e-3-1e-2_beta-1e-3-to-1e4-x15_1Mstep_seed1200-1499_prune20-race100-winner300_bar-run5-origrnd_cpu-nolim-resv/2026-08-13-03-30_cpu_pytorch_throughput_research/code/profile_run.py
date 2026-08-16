#!/usr/bin/env python
"""One instrumented ext4m training run for the CPU/PyTorch throughput research.

Three modes, all on the REAL train4m pipeline (train4m.run4m with a short budget):
- --mode gate    : 6,000 steps, then print "GATEHASH <sha256>" over the run's record histories
                   and every model tensor — the bit-exactness gate. Two runs whose GATEHASH
                   match produced identical training numbers.
- --mode time    : 30,000 steps with two full-size evals; print the per-phase wall-time table
                   (env stepping / SAC gradient updates / buffer sampling / RND compute+update /
                   eval / record IO) and total wall -> the A/B measurement.
- --mode profile : like time but under cProfile; dumps data/<tag>.prof + a top-60 text.

--candidate applies one torch-level change: polyak (opt_polyak_foreach), torchreward
(opt_torch_reward), interop (torch.set_num_interop_threads(1)), all (every one), none.
--config picks the workload: alg23 / run5rnd / gt (the three ext4m configurations, verbatim).
Everything runs single-thread (OMP_NUM_THREADS=1), the final fleet's per-worker shape.
"""
import argparse
import cProfile
import functools
import hashlib
import io
import json
import os
import pstats
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
RUN_DIR = os.path.dirname(RESEARCH)
PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))

import torch  # noqa: E402

# the three ext4m configurations, imported from the sweep's own builder (verbatim by construction)
import ext4m_build_queue as ebq  # noqa: E402

CONFIG_ARGS = {
    "alg23": next(c for c in ebq.CONFIGS if c["arm"] == "alg2.3"),
    "run5rnd": next(c for c in ebq.CONFIGS if c["arm"] == "run5-origrnd"),
    "gt": next(c for c in ebq.CONFIGS if c["arm"] == "gt-position-velocity"),
}

TIMERS = {}   # name -> [cumulative seconds, calls]


def timed(name, fn):
    """Wrap fn so its wall time accumulates under `name` (nested timers overlap by design)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            slot = TIMERS.setdefault(name, [0.0, 0])
            slot[0] += time.perf_counter() - t0
            slot[1] += 1
    return wrapper


def install_timers():
    """Patch the six phase boundaries with accumulating timers."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import SAC
    from rnd_exploration.buffers.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
    from rnd_exploration.methods.rnd import RND
    from rnd_exploration.callbacks import WandbEvalLoggingCallback
    import train as t
    DummyVecEnv.step_wait = timed("env_step", DummyVecEnv.step_wait)
    SAC.train = timed("sac_train (incl. buffer+rnd)", SAC.train)
    VectorIntrinsicReplayBuffer.sample = timed("buffer_sample (incl. rnd)",
                                               VectorIntrinsicReplayBuffer.sample)
    RND.compute = timed("rnd_compute", RND.compute)
    RND.update = timed("rnd_update", RND.update)
    WandbEvalLoggingCallback._on_step = timed("eval_callback", WandbEvalLoggingCallback._on_step)
    t._write_local_log = timed("record_io", t._write_local_log)


def apply_candidate(name, argv):
    """Apply one candidate: append its Config switch to argv and/or set its torch knob."""
    if name in ("polyak", "all"):
        argv.append("--opt_polyak_foreach=True")
    if name in ("torchreward", "all"):
        argv.append("--opt_torch_reward=True")
    if name in ("interop", "all"):
        torch.set_num_interop_threads(1)
    return argv


def build_argv(cfg_spec, steps, eval_freq, n_eval, out_dir, tag):
    """The train4m-shaped argv for one short run of one ext4m configuration."""
    fixed = dict(ebq.FIXED_COMMON)
    fixed.update({"total_timesteps": steps, "eval_freq": eval_freq,
                  "n_eval_episodes": n_eval})
    argv = [
        f"--ckpt_dir={out_dir}/{tag}_ckpt", f"--ckpt_every={steps * 2}",   # never fires
        "--suspend_end_epoch=0", "--buffer_tail=1000",
        f"--algorithm={cfg_spec['algorithm']}", f"--beta={cfg_spec['beta']}",
        "--a_seed=42", f"--env_setup={cfg_spec['env_setup']}",
        "--z_logging_mode=local", "--use_wandb=False",
        f"--local_log_dir={out_dir}/{tag}_data", "--run_id=0", "--run_total=1",
    ]
    argv += [f"--{k}={v}" for k, v in cfg_spec["params"].items()]
    argv += [f"--{k}={v}" for k, v in fixed.items()]
    return argv


def gate_hash(out_dir, tag):
    """sha256 over the record's histories AND every model tensor written by the run's final
    checkpoint-free state — the record alone pins rewards/episodes; tensors pin the weights.
    The trainer deletes checkpoints on completion, so tensors come from a state file the run
    mode leaves behind via TRAIN4M_GATE_STATE (set below)."""
    h = hashlib.sha256()
    rec = json.load(open(f"{out_dir}/{tag}_data/local/0_of_1.json"))
    for key in ("eval_history", "train_history", "train_episode_history"):
        h.update(json.dumps(rec.get(key, []), sort_keys=True).encode())
    state = torch.load(f"{out_dir}/{tag}_state.pt", map_location="cpu", weights_only=False)
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].numpy().tobytes())
    return h.hexdigest()


def run_once(args, out_dir, tag):
    """Run one short training with the requested candidate; return wall seconds."""
    import train4m
    import train as t
    spec = CONFIG_ARGS[args.config]
    n_eval = 5 if args.mode == "gate" else 100
    eval_freq = 2000 if args.mode == "gate" else 15000
    steps = 6000 if args.mode == "gate" else 30000
    argv = build_argv(spec, steps, eval_freq, n_eval, out_dir, tag)
    argv = apply_candidate(args.candidate, argv)
    sys.argv = [sys.argv[0]] + argv
    ext = train4m.parse_ext_args()
    cfg = t.parse_config()
    if args.mode != "gate":
        install_timers()
    t0 = time.perf_counter()
    train4m.run4m(cfg, ext)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gate", "time", "profile"), required=True)
    p.add_argument("--config", choices=tuple(CONFIG_ARGS), required=True)
    p.add_argument("--candidate", choices=("none", "polyak", "torchreward", "interop", "all"),
                   default="none")
    p.add_argument("--tag", required=True, help="output name inside the research data/ dir")
    args = p.parse_args()
    out_dir = os.path.join(RESEARCH, "data")
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(1)

    # gate mode: capture the final model tensors before run4m deletes its state — patch the
    # checkpoint-cleanup boundary: run4m's completion path removes state files, so instead hook
    # torch.save is not called at completion; grab tensors via a callback patched into SAC.learn
    if args.mode == "gate":
        import train4m
        from stable_baselines3 import SAC
        orig_learn = SAC.learn

        def learn_and_dump(self, *a, **k):
            out = orig_learn(self, *a, **k)
            tensors = {f"policy.{k2}": v.detach().clone() for k2, v in
                       self.policy.state_dict().items()}
            buf = self.replay_buffer
            model = getattr(buf, "intrinsic_reward_model", None)
            if model is not None and hasattr(model, "predictor"):
                tensors.update({f"rnd.{k2}": v.detach().clone() for k2, v in
                                model.predictor.state_dict().items()})
            torch.save(tensors, os.path.join(out_dir, f"{args.tag}_state.pt"))
            return out
        SAC.learn = learn_and_dump

    if args.mode == "profile":
        prof = cProfile.Profile()
        prof.enable()
    wall = run_once(args, out_dir, args.tag)
    if args.mode == "profile":
        prof.disable()
        prof.dump_stats(os.path.join(out_dir, f"{args.tag}.prof"))
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(60)
        with open(os.path.join(out_dir, f"{args.tag}_top60.txt"), "w") as fh:
            fh.write(s.getvalue())

    print(f"\n[RESULT] tag={args.tag} mode={args.mode} config={args.config} "
          f"candidate={args.candidate} wall={wall:.1f}s")
    if args.mode == "gate":
        print(f"[GATEHASH] {gate_hash(out_dir, args.tag)}")
    if TIMERS:
        print(f"{'phase':34s} {'seconds':>9s} {'calls':>9s} {'% wall':>7s}")
        for name, (sec, calls) in sorted(TIMERS.items(), key=lambda t: -t[1][0]):
            print(f"{name:34s} {sec:9.1f} {calls:9d} {100 * sec / wall:6.1f}%")


if __name__ == "__main__":
    main()
