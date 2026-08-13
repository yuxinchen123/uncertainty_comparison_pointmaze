# CPU/PyTorch throughput research for the ext4m 96-hour training (2026-08-13)

Ten-hour budget, ordered by the user before the 4M fleet's final launch: profile the real
training on puma01 (one job at a time for clean numbers), apply only changes that keep previous
results VALID (bit-exact: same seed, same record, same weights), park meaningful-but-breaking
ideas in `future_improvements/`, then submit the final fleet (no puma01). Methodology follows
the karpathy/autoresearch loop the user's prompt references (measure -> change -> gate ->
measure -> document), adapted to one module (the CPU trainer) and one framework (PyTorch).

## Harness

- `code/profile_run.py` — one instrumented run of the REAL `train4m` pipeline per invocation:
  - gate mode (6k steps): prints a sha256 over the record's histories AND every policy/RND
    tensor. Two runs match iff their training numbers are identical — the validity gate.
  - time mode (30k steps, two full-size evals, fixed seed): per-phase wall-time table
    (env stepping / SAC gradient updates / buffer sampling / RND compute + update / eval /
    record IO) plus total wall — the A/B measurement.
  - profile mode: time mode under cProfile (`data/*.prof`, top-60 text).
- `code/run_research.sh` — the whole serial sequence inside ONE puma01 job
  (data/research_job_<jobid>.log): determinism pair, then gates for every configuration x
  candidate, then timings, then the deep profile.

## Candidates (torch-level, validity-preserving by intent; every one must pass the gate on all
three configurations before its timing counts)

1. `polyak` — `--opt_polyak_foreach=True`: the codebase's own `torch._foreach_` polyak target
   update (documented bit-exact; both perf switches default to False, so the 1M sweep and the
   first ext4m fleet ran without them).
2. `torchreward` — `--opt_torch_reward=True`: reward combine stays in torch at buffer-sample
   time (the default path round-trips torch -> numpy -> torch on EVERY gradient step). Risk to
   gate: the gt configuration's VisitCount.compute may not accept torch inputs.
3. `interop` — `torch.set_num_interop_threads(1)`: the inter-op pool has nothing to do in a
   single-thread worker; shrinking it removes scheduler overhead. No numeric surface.
4. `all` — the passing candidates combined (the fleet setting if gates hold).

## Trials

(source of truth: data/research_job_6536784.log)

### Stage 1 — determinism (03:33–03:37)

Two identical baseline gates (alg23, seed 42, 6k steps) hashed identically
(`934650de…`), so the pipeline is bit-deterministic on CPU and hash equality is a valid
bit-exactness test.

### Stage 2 — bit-exactness gates (03:37–04:01): ALL 15 PASSED

Per-configuration baseline hashes: alg23 `934650de…`, run5rnd `631d55ed…`, gt `eab27c59…`.
Every candidate — polyak, torchreward, interop, and all three combined — reproduced its
configuration's baseline hash exactly. Notes:

- `torchreward` on gt was the flagged risk (VisitCount receiving torch tensors instead of
  numpy); it neither crashed nor diverged — `observation_to_count` handles the torch rows and
  the combine reproduces the numpy path bit for bit.
- Every gate ran 87–97 s; the gate suite cost ~28 minutes total.

Verdict: the full candidate set satisfies the results-validity constraint. Whether each is
WORTH applying is decided by stage 3's timings.

### Stage 3 — A/B timings (04:01–04:55; 30k steps, fixed seed, single thread, idle puma01)

| configuration | candidate | wall (s) | vs baseline |
|---|---|---|---|
| alg2.3 | none (baseline) | 467.4 | — |
| alg2.3 | polyak | 452.9 | −3.1% |
| alg2.3 | torchreward | 465.4 | −0.4% (noise) |
| alg2.3 | interop | 466.8 | −0.1% (noise) |
| alg2.3 | all three | 452.3 | **−3.2%** |
| run-5 original RND | none (baseline) | 451.7 | — |
| run-5 original RND | all three | 436.8 | **−3.3%** |

### Stage 4 — where the time goes (phase table, alg2.3 baseline; cProfile in data/)

| phase | seconds | % of wall |
|---|---|---|
| SAC gradient updates (incl. the two rows below) | 428.6 | 91.7% |
| — buffer sampling incl. RND bonus | 99.6 | 21.3% |
| — RND predictor update | 56.0 | 12.0% |
| RND forward computes (logging + sampling) | 50.2 | 10.8% |
| env stepping (whole wrapper stack) | 18.6 | 4.0% |
| record IO | 0.5 | 0.1% |
| eval callback | 0.0 | 0.0% |

cProfile: backward passes 131 s, module forwards 170 s of the 467 s — the wall is the SAC
update math itself. Two findings that corrected prior beliefs:

1. Evaluation costs nothing here: the production configs run `eval_standalone=False` (scores
   come from training episodes), so the imagined 100-episode eval overhead does not exist.
2. The python env stack is only 4% — env-side rewrites would buy almost nothing for THIS
   workload (they matter for the AntMaze/vision families, not a 4-d PointMaze).

## Decision

Apply all three candidates to the ext4m fleet (`opt_polyak_foreach=True` and
`opt_torch_reward=True` via the worker's train4m argv; `torch.set_num_interop_threads(1)`
inside train4m.py): bit-exact on every configuration, a consistent −3.2/−3.3% wall on the two
net-bearing configurations, zero risk. Per-run effect ≈ 2 h saved on a 60–76 h 4M run. The
larger levers (compile, foreach Adam, denormal flushing, logging-forward removal) all change
numerics and stay in FUTURE_CHANGES.md / future_improvements/. Fleet swap executed at the first
monitoring tick where the running fleet's markers have checkpoints (~13:30), so the switch
costs minutes, not hours.
