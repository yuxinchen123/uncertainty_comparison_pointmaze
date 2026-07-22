"""Convergence-rate run 1 trainer: pure RND distillation on a fixed point set (no RL loop).

One run = one (point set, initialization, optimizer configuration, seed). Build the point tensor
once; then repeat --n_steps times: forward ALL points through target + predictor in one batch and
take one optimizer step via RND.update() — the exact training code path of the 3.2.x runs (same
squared distillation objective; the readout never enters training). At log-spaced checkpoint steps,
store the per-point l2 bonus b_i(n) = ||g(x_i) - f_hat_n(x_i)||_2 (the exact norm, without
compute()'s 1e-8 numerical clamp, which would floor deeply converged values at 1e-4).

Observation normalization is OFF (raw [x, y, 0, 0] inputs), so given the seed the whole run is
deterministic: the initialization realization is the only randomness. Design + tables:
development_document/main.tex, Section "Convergence rate runs" (convergence run 1).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from rnd_exploration.methods.rnd import RND

# PointMaze_Large-v3 geometry (verified 2026-07-19 by executing the env): 9 cells high, 12 wide,
# cell size 1x1, x in [-6, 6], y in [-4.5, 4.5]; cell (row, col) center = (col - 5.5, 4 - row),
# row 0 at top, y up. 46 of the 108 cells are free.
MAZE_ROWS = 9
MAZE_COLS = 12

POINT_SET_CHOICES = ("center_square", "top_right_cell", "cell_midpoints")


def checkpoint_steps(n_steps: int) -> list:
    """Log-spaced checkpoint steps: unique values of round(2^(j/8)) for j = 0, 1, ... up to n_steps."""
    # quarter splits of the half-exponent 2^{k/2} grid (user request 2026-07-19: 3 extra log points
    # between each previous pair whenever the integers allow; small-end duplicates collapse).
    # before (j=0..12): 1.00, 1.09, 1.19, 1.30, 1.41, 1.54, 1.68, 1.83, 2.00, 2.18, 2.38, 2.59, 2.83
    # after round + dedup: 1, 2, 3   (for n_steps=4096 the full list has 79 entries, ending 4096)
    steps = []
    j = 0
    while round(2 ** (j / 8.0)) <= n_steps:
        s = round(2 ** (j / 8.0))
        if not steps or s != steps[-1]:
            steps.append(s)
        j += 1
    return steps


def point_set(name: str) -> np.ndarray:
    """Build one fixed evaluation point set as a (P, 4) float32 array of [x, y, vx=0, vy=0] rows."""
    # 10x10 sub-cell midpoints of a 1x1 square: offsets -0.45, -0.35, ..., +0.45 (spacing 0.1),
    # the same midpoint convention as Figure 12's heatmap grid.
    # before: j = 0..9 -> after: -0.5 + (j + 0.5) * 0.1 = -0.45 .. +0.45
    offs = -0.5 + (np.arange(10) + 0.5) * 0.1
    if name == "center_square":
        # env 1.1: the 1x1 square centered at (0, 0) (user decision: kept although its left half
        # overlaps wall cell (row 4, col 5); the network is defined at wall points too)
        xs, ys = np.meshgrid(offs, offs, indexing="ij")
    elif name == "top_right_cell":
        # env 1.2: the top-right free cell (row 1, col 10) = the fixed top_right goal cell,
        # world center (4.5, 3.0), spanning [4, 5] x [2.5, 3.5]
        xs, ys = np.meshgrid(4.5 + offs, 3.0 + offs, indexing="ij")
    elif name == "cell_midpoints":
        # env 2: the midpoint of EVERY maze cell of the 9 x 12 grid, walls included (user
        # decision; 46 free + 62 wall cells).
        # before: (row, col) = (0, 0) .. (8, 11) -> after: x = col - 5.5 in {-5.5 .. 5.5},
        # y = 4 - row in {4 .. -4}; 108 points
        cols, rows = np.meshgrid(np.arange(MAZE_COLS), np.arange(MAZE_ROWS), indexing="ij")
        xs, ys = cols - 5.5, 4.0 - rows
    else:
        raise ValueError(f"point_set must be one of {POINT_SET_CHOICES}; got {name!r}")
    zeros = np.zeros(xs.size)
    return np.stack([xs.ravel(), ys.ravel(), zeros, zeros], axis=1).astype(np.float32)


def readout_l2(model: RND, x: torch.Tensor) -> np.ndarray:
    """Per-point l2 bonus ||g(x_i) - f_hat(x_i)||_2 as float64 numpy (exact norm, no clamp)."""
    # same forward as compute() with use_obs_norm=False (normalization is the identity); dropping
    # only the (2*B_mse).clamp(min=1e-8) floor so a deeply converged curve is not flattened at 1e-4
    with torch.no_grad():
        e = model.target(x) - model.predictor(x)
        return e.pow(2).sum(dim=1).sqrt().cpu().numpy().astype(np.float64)


def run_name(args) -> str:
    """JSON filename stem: sweep runs are id-named (sortable), standalone runs descriptive."""
    if args.run_total > 0:
        return f"{args.run_id:0{len(str(args.run_total))}d}_of_{args.run_total}"
    return (f"standalone_{args.point_set}_w-{args.rnd_weight_init}_b-{args.rnd_bias_init}"
            f"_{args.rnd_optimizer}_lr{args.rnd_lr:g}_eta{args.rnd_sgd_eta0:g}"
            f"_t{args.rnd_sgd_t0:g}_seed{args.a_seed}")


def run(args) -> dict:
    """Run one distillation job and write its checkpointed JSON record; returns the final record."""
    # honor the Slurm arm's thread budget exactly like train.py (workers export OMP_NUM_THREADS=1)
    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    # seed every RNG together, the same call sequence as train.py run() (cpu-only here)
    random.seed(args.a_seed)
    np.random.seed(args.a_seed)
    torch.manual_seed(args.a_seed)
    t_start = time.time()

    # fixed points, built once and reused for all updates (and stored in the record)
    points = point_set(args.point_set)
    x = torch.as_tensor(points)
    samples = {"next_observations": points}

    # the exact run-3.2.x model: 4 -> 256 -> ReLU -> 128 target + predictor, obs norm OFF (raw
    # inputs, stationary), squared training objective, l2 readout for reference cross-checks
    model = RND(
        obs_shape=(points.shape[1],),
        output_dim=128,
        lr=args.rnd_lr,
        batch_size=256,
        device="cpu",
        use_obs_norm=False,
        distance="mse",
        n_predictors=1,
        beta_std=0.0,
        feature="rnd_next_state",
        linear_rnd=False,
        optimizer=args.rnd_optimizer,
        bonus_readout="l2",
        sgd_eta0=args.rnd_sgd_eta0,
        sgd_t0=args.rnd_sgd_t0,
        weight_init=args.rnd_weight_init,
        bias_init=args.rnd_bias_init,
        bias_seed=args.a_seed,
    )

    cps = checkpoint_steps(args.n_steps)
    record = {
        # --- identity / config group ---
        "completed": False,
        "diverged": False,
        "trainer": "convergence_distill",
        "run_id": args.run_id,
        "run_total": args.run_total,
        "sweep_id": args.sweep_id,
        "point_set": args.point_set,
        "algorithm": "rnd_next_state",
        "rnd_weight_init": args.rnd_weight_init,
        "rnd_bias_init": args.rnd_bias_init,
        "rnd_optimizer": args.rnd_optimizer,
        "rnd_lr": args.rnd_lr,
        "rnd_sgd_eta0": args.rnd_sgd_eta0,
        "rnd_sgd_t0": args.rnd_sgd_t0,
        "a_seed": args.a_seed,
        "n_steps": args.n_steps,
        "n_points": int(points.shape[0]),
        "checkpoint_steps": [0] + cps,
        "points_xy": [[float(px), float(py)] for px, py in points[:, :2]],  # vx = vy = 0 for all
        "runtime_seconds": 0.0,
        # --- per-checkpoint group: one row per stored step, per-point raw l2 values ---
        "checkpoints": [],
    }

    # atomic checkpointed flush (project logging convention): tmp + os.replace at every checkpoint
    # with completed=False, once at the end with completed=True; a killed run leaves partial curves
    out_dir = os.path.join(args.local_log_dir, "local") if args.local_log_dir else ""

    def _flush(completed: bool) -> None:
        """Atomically rewrite this run's JSON with everything recorded so far."""
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        record["completed"] = completed
        record["runtime_seconds"] = time.time() - t_start
        path = os.path.join(out_dir, run_name(args) + ".json")
        with open(path + ".tmp", "w") as fh:
            json.dump(record, fh)
        os.replace(path + ".tmp", path)

    # step 0 = at initialization, before any update: the per-point normalization baseline b_i(0)
    b0 = readout_l2(model, x)
    record["checkpoints"].append({"step": 0, "b_l2": b0.tolist()})
    _flush(False)

    # the training loop: one full-batch RND.update per step; per-point readout at checkpoints only
    cp_set = set(cps)
    for t in range(1, args.n_steps + 1):
        model.update(samples)
        if t in cp_set:
            b = readout_l2(model, x)
            # divergence guard: a non-finite readout means the optimizer blew up — stop early,
            # keep the finite checkpoints, and mark the record diverged (excluded from fits)
            if not np.isfinite(b).all():
                record["diverged"] = True
                break
            record["checkpoints"].append({"step": t, "b_l2": b.tolist()})
            _flush(False)
    _flush(True)
    print(f"convergence run {run_name(args)}: {len(record['checkpoints'])} checkpoints, "
          f"diverged={record['diverged']}, runtime={record['runtime_seconds']:.1f}s")
    return record


def parse_args() -> argparse.Namespace:
    """Argparse for one convergence-distillation run (defaults match the sweep's fixed knobs)."""
    p = argparse.ArgumentParser(description="RND convergence-rate distillation on a fixed point set")
    p.add_argument("--point_set", type=str, required=True, choices=list(POINT_SET_CHOICES))
    p.add_argument("--rnd_weight_init", type=str, default="orthogonal",
                   choices=["orthogonal", "pytorch_default"])
    p.add_argument("--rnd_bias_init", type=str, default="zero",
                   help="zero | pytorch_default | normal_<sigma> (e.g. normal_0.5)")
    p.add_argument("--rnd_optimizer", type=str, required=True, choices=["adam", "sgd1t"])
    p.add_argument("--rnd_lr", type=float, default=1e-3,
                   help="Adam learning rate (ignored by sgd1t, whose rate is eta0/(1 + t/t0)).")
    p.add_argument("--rnd_sgd_eta0", type=float, default=1e-2)
    p.add_argument("--rnd_sgd_t0", type=float, default=1e3)
    p.add_argument("--a_seed", type=int, required=True)
    p.add_argument("--n_steps", type=int, default=4096)
    p.add_argument("--run_id", type=int, default=-1)
    p.add_argument("--run_total", type=int, default=0)
    p.add_argument("--sweep_id", type=str, default="")
    p.add_argument("--local_log_dir", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
