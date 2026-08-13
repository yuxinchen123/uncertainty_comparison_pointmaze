# 4M-step extension sweep (ext4m) — design

Ordered by the user 2026-08-13: extend train run 6 with an additional sweep that runs three
configurations for 4,000,000 steps each on 300 fresh seeds, with a model checkpoint every
500,000 steps (only the newest checkpoint kept), no replay buffer in the checkpoint (storage),
resumable across job walltimes, collaborator submission enabled, results added as ROWS to
Table 60 and LINES to Figure 18 (no new subsection). Submission cpu → nolim → reservation
puma01 only (no jaguar03).

## The three configurations (3 × 300 seeds = 900 runs, `a_seed` 1500–1799)

| # | configuration | source of the exact parameters |
|---|---|---|
| 1 | run-5 original RND: `rnd_next_state`, Adam 10⁻⁴, mse-mean readout, reward norm ON (γ=0.99), β=1000 | run 5's own queue markers (verbatim) |
| 2 | algorithm 2.3: alg1 params + frozen-init ratio readout + LayerNorm, plain SGD lr 0.01, β=30 | run-6 `build_queue.py` ALG1_PARAMS + ARM_EXTRAS["alg2.3"] (imported, byte-identical) |
| 3 | best ground-truth bonus: `gt_position_velocity`, bonus min(1, 1/√n) (`visit_count_decay=-0.5`), β=1 | run-3.2.2 oracle re-run markers (verbatim); the best row of the writeup's oracle table (66.24 ± 1.30, n=397) |

Why β=30 for algorithm 2.3 and not the phase-2 winner β=10: at the decision date the two are
statistically indistinguishable (β=30: 43.75 ± 2.45, n=197; β=10: 42.73 ± 1.88, n=296), β=30 is
the arm's best configuration by mean — the rule Table 60 and Figure 18 already use — and the 4M
line must extend the line the figure shows.

Environment: `initial_single_large_pointmaze_max_400` (run-5's task verbatim, same as the 1M
sweep). Everything else identical to the 1M sweep except `total_timesteps=4000000`.

## What a checkpoint stores (the brainstormed list), and what it deliberately omits

One `state_step<N>.pt` (torch.save) per run under `checkpoints/<sweep_id>/<run_id>/`, written
atomically (tmp + rename) every 500,000 steps at the eval boundary; the previous file is deleted
after the new one lands. Contents:

1. SAC (SB3): policy state dict (actor, critic, critic target), actor optimizer, critic
   optimizer, entropy-coefficient tensor + its optimizer, `num_timesteps`, `_n_updates`,
   `_episode_num`.
2. RND (configurations 1–2): predictor net, frozen target net, frozen init-copy net (alg2.3),
   predictor optimizer (Adam moments matter for configuration 1; SGD is stateless but saved
   uniformly), observation running mean/var/count (`obs_rms`), reward running std + per-env
   forward-filter state `_rff_return` (configuration 1 only). Configuration 3 (`gt_*`) has no
   nets; its bonus state IS the visit-count arrays below.
3. Visit-count arrays of both wrappers (position 9×12 and position-velocity 9×12×10×10) — they
   feed the `visit_counts/*` logged metrics and are the entire state of the `gt_*` bonus. The
   eval stack reads the same arrays by reference, so they are restored in place.
4. RNG states: python `random`, numpy, torch CPU.
5. A replay-buffer TAIL: the most recent 100,000 transitions (float32 arrays; ~5 MB). The user
   ruled out saving the full 1M-transition buffer; an empty buffer after resume would make SAC
   overfit the first few post-resume samples, so the tail is the compromise — bounded size,
   much closer to the pre-kill sampling distribution.
6. Meta: run id, config key, step, write time.

Known, accepted infidelities on resume (logged in the record):
- the replay buffer restarts with only the last 100k transitions (the user's storage decision);
- the training env starts a fresh episode (mid-episode env state is not serialized);
- envs are reseeded with a keyed seed derived from (a_seed, "resume", step) — never the original
  seed again, so no episode sequence is replayed.

Checkpoint size ≈ 25 MB/run (SAC nets + Adam moments ≈ 18 MB, RND ≈ 2 MB, tail ≈ 5 MB).
Live ceiling ≈ 900 × 25 MB ≈ 23 GB if every run were mid-flight at once; in practice bounded by
the worker fleet (≤ ~500 concurrent) ≈ 13 GB. On completion a run's checkpoint directory is
DELETED (the durable record is the JSON; final models are not kept). Records: 4M histories ≈
2.2 MB/run × 900 ≈ 2 GB. Both fit the 227 GB free on /p/rlprojects (checked 2026-08-13).

## Resume protocol (tested before launch)

- The trainer is a NEW entry `train4m.py` beside `train.py` (train.py is untouched — the 1M
  sweep still runs from it). It rebuilds everything exactly as a fresh run, then overwrites the
  mutable state from the newest checkpoint, pre-seeds the callback histories from the run's own
  record JSON (rows with step ≤ checkpoint step), sets `_last_obs=None`, and continues with
  `learn(remaining, reset_num_timesteps=False)` — all cadences key on `num_timesteps`, so eval,
  logging and checkpoint boundaries stay aligned.
- Walltime suspend: at every checkpoint boundary the trainer measures the last 0.5M-step chunk's
  wall time; if the job's remaining walltime is under 1.15 × that + 15 min, it checkpoints,
  flushes the record (`completed=false`), and exits with code 3. The worker moves the marker
  BACK TO PENDING (not failed): the next claimer resumes from the checkpoint. Exit 0 = done;
  everything else = failed.
- The worker's claim guard needs only ONE chunk of walltime (default 12 h), not a whole run.
- Orphans (job killed between checkpoints): requeue moves the marker back to pending; the resume
  loses at most the steps since the last checkpoint (≤ 0.5M).

## Sweep mechanics

- Same run folder, same git branch, NEW sweep id `2026-08-13-*_run6ext4m` — the folder machinery
  already namespaces queue/, data/, checkpoints/, id files and logs by sweep id, so the live 1M
  sweep is untouched and there is nothing to merge later (the user asked for a single
  folder/branch at the end: this is it from day one).
- No truncation race: all 900 runs go to completion. Sentinel `SWEEP4M_COMPLETE` when
  done == 900.
- Seed OUTERMOST (seed index i owns ids 3i..3i+2) as everywhere in this project.
- Submission: cpu → nolim, NO reservation (revised by the user 2026-08-13 03:20, replacing the
  earlier cpu → nolim → puma01 plan; the first fleet, submitted 02:38 with loop workers and a
  puma01 reservation job, was cancelled at 03:2x before any checkpoint existed — its ~487
  claimed runs lost under an hour each and re-pended).
- ONE-SHOT workers (user rule 2026-08-13): each worker claims exactly one run and exits; the job
  ends when all its workers finish (srun --wait=0), so every claim owns a full 96-hour cpu
  walltime and no run straddles a job boundary. The trainer's suspend stays as a safety net.
  Replenishment is a ladder of unpinned PENDING jobs (`filler` jobs in ext4m_plan_jobs.py) that
  start as the running wave's nodes free.
- WORKLOAD SHARES: the owner submits at most 600 runs' worth of slots (2/3), a collaborator 300,
  enforced by the append-only slots ledger `ext4m_slots_<sweep>_<user>.txt` next to each id file
  (cancelled jobs' slots return to the budget).
- Collaborator: cpu + nolim buckets via her own scripts and id file; README_ext4m.md in
  for_collaborator/ explains the one-shot worker and her 300-run share.
