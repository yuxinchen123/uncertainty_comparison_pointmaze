# Operational manual: launching a packed GPU sweep on this cluster

Everything below is from `/p/rlprojects/.claude/skills/submit-gpu-sweep/` (SKILL.md, `references/file_formats.md`, `references/packing_validation/`, the four scripts, `server_introduction/`), plus two real launched sweeps used as worked examples.

**Skill root** (constant, hardcoded inside the scripts): `/p/rlprojects/.claude/skills/submit-gpu-sweep`

| path | what it is |
|---|---|
| `/p/rlprojects/.claude/skills/submit-gpu-sweep/SKILL.md` | the procedure (8 steps + safety rules) |
| `.../references/file_formats.md` | the authoritative spec of every file format |
| `.../references/packing_validation/packing_validation.md` | one-time proof that W slots share one GPU under one job id + measured overhead. **Never re-profile packing.** |
| `.../scripts/build_queue.py` | writes queue markers from `configs.jsonl` |
| `.../scripts/make_submission_scripts.py` | packing math → one `.slurm` per (node class × packing group) + `PACKING.md` |
| `.../scripts/availability.py` | read-only live cluster picture + printed sbatch suggestions (never submits) |
| `.../scripts/worker_manager.py` | runs INSIDE each job: claims runs, pins them to GPUs, samples usage, moves markers |
| `.../scripts/monitor.py` | one-shot bookkeeping per wake-up (infra table, estimate-vs-actual, requeue) |
| `.../server_introduction/server_introduction.json` | node-class table (`schema_version: 2`, 27 classes) + per-partition per-user caps |
| `.../server_introduction/server_introduction.md` | same, human readable (Table 1 node classes, Table 2 limits) |
| `.../server_introduction/code/generate_server_introduction.py` | regenerates both files from live Slurm |
| `.../server_introduction/code/gpu_catalog.json` | static per-GPU-type facts (memory, arch, year, compute capability, capability rank) |

**Worked example A (multi-profile torch sweep, has a collaborator packet):**
`/p/rlprojects/RLforOR/anti_concentrated_confidence_bonuses/train_runs/2026-07-15-21-11_fig5-6_acb-rnd_8games_intrinsic-only_steps-2e7_numaux-128_replicates-10_gpu`

**Worked example B (JAX sweep, first use of the skill):**
`/p/rlprojects/RLforOR/distributional_rnd/train_runs/2026-07-11-19-05_tables3-10-11-12-13-14-15-16-17_sac-drnd_offline_hopper-halfcheetah-walker2d-antmaze_target-count-3-5-10-20_alpha-0-0.1-0.5-0.9-1_seed-10-1-2-3-4-5-6-7-8-9-11-12-13-14-15-16-17-18-19-20_new-skill_gpu`

---

## 1. Exact run-folder layout

Sweep folder = `<project>/train_runs/<YYYY-MM-DD-HH-MM>_<sweep-name>/`, with every load-bearing knob spelled into the name (both examples above do this).

```
<sweep folder>/
├── experiment_background.md        # YOU write, at folder creation (experiment-background skill)
├── configs.jsonl                   # YOU write: the fixed run list, one JSON line per run
├── infra_history.md                # monitor.py APPENDS one dated block per wake-up (never edited)
├── queue/                          # created by build_queue.py
│   ├── pending/                    #   marker here = run waiting
│   ├── running/                    #   marker here = run claimed by a worker slot (has "claim" block)
│   ├── done/                       #   marker here = run finished ok
│   └── failed/                     #   marker here = run exited non-zero / completion flag not true
├── canary/                         # created by build_queue.py --canary and worker_manager --canary
│   ├── queue/{pending,running,done,failed}/
│   ├── data/                       #   throwaway canary outputs (NEVER data/)
│   ├── logs/                       #   canary per-run logs run_<run_id>.log
│   ├── resource_usage/{per_run/,node_status/,estimate_vs_actual.md,.monitor_state.json}
│   └── infra_history.md            #   written by monitor.py --canary
├── data/                           # the TRAINING PIPELINE writes run outputs here (paths come from argv)
│   └── killed_attempts_<YYYY-MM-DD-HH-MM>/   # monitor.py --requeue moves partial outputs here
├── for_collaborator/               # only if collaborators submit (collab-handbook skill)
│   ├── README.md, packet_env.sh, plan_submissions.py, launch_workers_collaborator.sh,
│   │   monitor_collaborator.sh, refresh_my_ids.sh, resource_facts.md, smoke_test.sh, smoke_run.slurm
│   ├── worker_<node-class>_collab.slurm   (one per node class)
│   ├── logs/                       #   collaborator job stdout
│   ├── problems/{open,resolved}/   #   the problem channel; open/ is checked EVERY wake-up
│   └── submitted_jobids_<sweep_id>_<user>.txt   # ONE per collaborator; monitor.py merges these
└── slurm/
    ├── worker_env.sh               # YOU write: sourced by every submission script before worker_manager
    ├── resource_estimate/
    │   ├── resource_estimate.json  # YOU write: machine-readable estimate (drives packing + canary)
    │   └── resource_estimate.md    # YOU write: each number with the code lines it came from
    ├── submission_script/
    │   ├── <class>__<group>.slurm  # generated by make_submission_scripts.py
    │   └── PACKING.md              # generated: per class×group, W / c_used / G_used / cpus / mem / why-capped
    ├── resource_usage/
    │   ├── per_run/<run_id>.jsonl      # worker_manager sampler, one line per sample per run
    │   ├── node_status/<node>.jsonl    # worker_manager sampler, one line per sample per node
    │   ├── estimate_vs_actual.md       # monitor.py rebuilds atomically each pass
    │   └── .monitor_state.json         # monitor.py internal (last-seen job states)
    ├── logs/                       # sbatch %x_%j.out + worker_manager's per-run run_<run_id>.log
    └── submitted_jobids.txt        # YOU append EVERY sbatch id the moment sbatch returns
```

You author exactly five things: `experiment_background.md`, `configs.jsonl`, `slurm/worker_env.sh`, `slurm/resource_estimate/resource_estimate.{json,md}`, and the lines you append to `slurm/submitted_jobids.txt`. Everything else is generated.

Before the canary/submission batches the OWNER commits and pushes the run code (`/p/rlprojects/.claude/CLAUDE.md` "Commit-before-submit"); collaborators adding workers are exempt. Record the commit hash in `experiment_background.md`'s "Git state" line.

---

## 2. Sweep-config / queue-entry formats

### 2a. `configs.jsonl` (input to `build_queue.py`) — one JSON object per line

Spec fields: `run_id`, `run_total`, `seed`, `group`, `profile`, `argv`, `completed_marker` (optional). Extra fields are allowed and preserved.

- `run_id` is **gap-free 0..run_total−1 and seed-OUTERMOST** (all configs of seed A precede seed B), so a cancelled sweep leaves complete early-seed coverage.
- `group` = packing group name (must match a key in `packing_inputs`); `profile` = resource profile name.
- `argv[0]` is the **absolute path to the shared env's python** (never a private `/u/<user>` env — collaborators cannot reach it).
- `completed_marker` accepts BOTH shapes; `worker_manager.completed_marker_ok` handles both: a dict `{"path": ..., "flag_field": ...}` or a plain string path (flag_field then defaults to `"completed"`). When absent, exit code alone decides done/failed.

Real line (example A, `run_total=160`, torch/Atari), abbreviated only in the long repeated path:

```json
{"run_id": 0, "run_total": 160, "seed": 0, "group": "acb-short", "profile": "acb-short",
 "argv": ["/p/rlprojects/RLforOR/.venvs/anti_concentrated_confidence_bonuses/bin/python", "-u",
          "/p/rlprojects/RLforOR/anti_concentrated_confidence_bonuses/acb/replication/atari/run_one.py",
          "--run-dir", "<sweep>/data/runs/0000_breakout_acb_numaux-128_dummy-0",
          "--out", "<sweep>/data/local/0000_of_160.json",
          "--env", "breakout", "--intrinsic", "acb", "--numAux", "128", "--dummy", "0",
          "--max-global-step", "20000000", "--run-id", "0", "--run-total", "160", "--snapshot-init"],
 "completed_marker": "<sweep>/data/local/0000_of_160.json"}
```

Real line (example B, `run_total=1380`, JAX/D4RL) uses the dict shape:

```json
{"run_id": 0, "run_total": 1380, "seed": 10, "group": "mujoco+antmaze", "profile": "mujoco",
 "argv": ["/p/rlprojects/RLforOR/.venvs/distributional_rnd/bin/python3.10", "-u",
          "/p/rlprojects/RLforOR/distributional_rnd/DRND/replication/offline/run_sac_drnd.py",
          "--config_path", ".../hopper/hopper_random.yaml", "--seed", "10", "--seed_order_index", "0",
          "--run_id", "0", "--run_total", "1380", "--sweep_id", "2026-07-11-19-05_new-skill",
          "--local_log_dir", "<sweep>/data/2026-07-11-19-05_new-skill", "--paper_setting", "main"],
 "completed_marker": {"path": "<sweep>/data/2026-07-11-19-05_new-skill/local/0000_of_1380.json",
                      "flag_field": "completed"}}
```

### 2b. Queue markers (output of `build_queue.py`, mutated by `worker_manager.py`)

- Filename: `<run_id zero-padded to len(str(run_total))>_of_<run_total>.json`. `run_total=160` → `000_of_160.json`; `run_total=1380` → `0042_of_1380.json`. Width is constant within a sweep.
- Content = the configs.jsonl line verbatim, plus fields the worker adds:
  - at claim time, rewritten in place inside `running/`:
    `{"claim": {"job_id": "6490000", "node": "serval06", "gpu_index": 1, "slot": 3, "claim_ts": "..."}}`
  - at exit: `{"exit": {"rc": 1, "signal": null, "end_ts": "..."}}`, plus `"watchdog_killed": true` if the watchdog ended it.
- **The marker's DIRECTORY is the run's state.** Claim = `os.rename(pending/x, running/x)` (atomic on one filesystem; a lost rename just means another slot won — pick again).
- Claim order: sort pending filenames ascending, filter by `--group` if set, choose **uniformly among the first 32** (`CLAIM_WINDOW = 32`) so seeds finish roughly in order while rename collisions stay rare.

Real done canary marker (example A) — note `"canary": true`, the appended override args, and the dropped `completed_marker`:

```json
{"run_id": 0, "run_total": 160, "seed": 0, "group": "acb-short", "profile": "acb-short",
 "argv": [... "--run-id", "0", "--run-total", "160",
          "--canary-root", "<sweep>/canary/data", "--max-global-step", "32768"],
 "canary": true,
 "claim": {"job_id": "6504354", "node": "cheetah02", "gpu_index": 3, "slot": 0,
           "claim_ts": "2026-07-15T21:18:40"}}
```

### 2c. `slurm/resource_estimate/resource_estimate.json` (input to `make_submission_scripts.py`, `availability.py`, `build_queue.py --canary`, `monitor.py`)

```json
{"profiles": {
   "mujoco":  {"device": "gpu", "gpu_mem_mb": 3072, "host_mem_mb": 4096, "cpu_threads_busy": 2},
   "antmaze": {"device": "gpu", "gpu_mem_mb": 3072, "host_mem_mb": 4096, "cpu_threads_busy": 2}},
 "packing_groups": [["mujoco", "antmaze"]],
 "packing_inputs": {"mujoco+antmaze":
   {"gpu_mem_mb": 3072, "host_mem_mb": 4096, "c_min": 2, "c_max": 8, "w_max": 8}},
 "compatible_gpu_types": {"min_compute_capability": 6.1, "max_compute_capability": 9.0,
   "reason": "Empirically tested 2026-07-12 (job 6489171, ai09 gtx_1080_ti, sm_61): JAX 0.4.x initializes and trains on Pascal ... ceiling 9.0 = Hopper, Blackwell (12.0) still postdates the build"},
 "canary": {"override_args": ["--canary_epochs", "60",
                              "--local_log_dir", "<sweep>/canary/data/2026-07-11-19-05_new-skill"],
            "expected_minutes": 12,
            "note": "override_args MUST carry the output redirect: argparse last-wins ..."},
 "correction_history": [ {"when": "...", "source": "canary wave 1 ...", "before": {...}, "observed_peaks": {...}, "note": "..."} ]}
```

Rules for the fields:
- `v = gpu_mem_mb`, `m = host_mem_mb` are **FIXED by the training code's hyperparameters — never changed to make a run fit**. Size `m` against the LARGEST dataset in the profile; size `cpu_threads_busy` against the CPU-heaviest phase (evaluation bursts), not the training average.
- `device: "cpu"` profiles do **not** run under this skill — they get their own sweep folder under `/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md`.
- Group name = profile names joined with `+`. Profiles within ~2× on every resource share one group; further apart → separate groups. `packing_inputs` holds the per-resource MAXIMUM within the group, **RAW** (the ×1.25 is applied by `make_submission_scripts.py`).
- `w_max` (optional) caps worker slots per GPU until higher packing has live evidence.
- `compatible_gpu_types.min_compute_capability` / `max_compute_capability` gate node classes (a class below the min or above the max gets no script). `max_compute_capability` matters: without it a too-new GPU (cc 12.0 on a cuda-11 build) gets suggested and every run dies at XLA setup (live incident 2026-07-12, job 6489199).
- `canary.override_args` are appended to argv **in canary mode only** and **MUST include the pipeline's output-directory redirect** (argparse last-wins), so canary outputs land under `canary/`. A canary output reaching `data/` blocks the real run and, if it carries `completed: true`, silently poisons results (happened 2026-07-11).
- `correction_history` is optional but is how example B recorded the canary-driven correction.

Example A shows the multi-group shape: five profiles (`acb-short`, `acb-capride`, `rnd-short`, `rnd-capride`, `acb-short18`), each its own single-profile packing group, with per-group `w_max` of 1 or 2 and a `canary.note` recording every live correction.

---

## 3. Exact command lines, in order

All scripts are stdlib-only python3.9+; run them with `python3`.

**Step 0 — pre-knowledge (only after cluster changes, never per submission):**
```bash
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/code/generate_server_introduction.py
# self-check without writing:
python3 .../generate_server_introduction.py --check
# other flags: --catalog, --json-output, --md-output, --node-json <saved scontrol json fixture>
```
Extend `server_introduction/code/gpu_catalog.json` when a new GPU type appears — the generator refuses to guess memory for unknown types.

**Step 1 — build the queue** (after writing `configs.jsonl`):
```bash
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/build_queue.py \
  --sweep_dir /abs/path/to/<sweep folder>
# prints: wrote <N> pending markers to <sweep>/queue/pending
```
Flags: `--canary` (build under `canary/queue/` instead), `--count N` (required with `--canary`), `--force` (rebuild over an already-populated queue; without it, it refuses).

**Step 2 — generate submission scripts:**
```bash
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/make_submission_scripts.py \
  --sweep_dir /abs/path/to/<sweep folder> \
  --codename acbfig5    # job-name base; each generated script gets this plus a running number
# optional: --server_introduction <path>  (defaults to the skill's copy)
```
Writes `slurm/submission_script/<class>__<group>.slurm` for every non-skipped pair plus `PACKING.md`. Review `PACKING.md` before submitting.

**Step 3 — read live availability (every time, before every batch):**
```bash
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/availability.py \
  --sweep_dir /abs/path/to/<sweep folder> \
  --need <pending worker slots>
# add --json for machine output; offline replay: --nodes_json / --reservations_text / --squeue_text
```
It runs only `scontrol show node --json`, `scontrol show reservation -o`, and `squeue -u $USER -h -o "%i|%P|%T|%C|%D|%b|%N|%m"`. It prints five sections: (i) free GPUs per node by class, (ii) our usage vs per-user caps for `gpu` and `gnolim`, (iii) the H100 allowance, (iv) next maintenance + recommended `--time` per partition + our reservation ends, (v) a suggested ~20-GPU batch **printed only, never executed**. Its last line is: `After each sbatch, append the returned id to <sweep>/slurm/submitted_jobids.txt. This tool never submits.`

**Gotcha:** `suggest_batch` plans for `packing_groups[0]` only (`group = groups[0]`). For a multi-group sweep (like example A's five groups) the printed commands cover one group; size the other groups yourself from `PACKING.md`.

**Step 4 — submit.** Take the suggestion (or write it yourself from `PACKING.md`), and always `--parsable` + append the id in the same command:

```bash
id=$(sbatch --parsable --nodelist=cheetah08 \
       --gres=gpu:nvidia_rtx_a4000:3 \
       --cpus-per-task=15 --mem=150000M --time=4-00:00:00 \
       <sweep>/slurm/submission_script/cheetah08-09__rnd-short.slurm 3)
echo "$id cheetah08-09 cheetah08 3 1 5 rnd-short $(date -Is)" >> <sweep>/slurm/submitted_jobids.txt
```

`submitted_jobids.txt` line format (spec): `<jobid> <node_class> <node> <G> <W> <c_used> <group> <submit_ts>`. `monitor.py` tolerates short lines and skips any line whose first token is not digits (annotation/comment lines are safe — a non-numeric token used to poison the whole batched squeue query, fixed 2026-07-16). Real lines from example B:

```
6489023 serval06-09 serval06 1 1 4 import_check 2026-07-11T19:18:54
6489028 serval06-09 serval06 2 7 4 mujoco+antmaze 2026-07-11T19:20:33
6489032 jaguar03 jaguar03 5 2 8 mujoco+antmaze 2026-07-11T19:20:33
```
Real annotated lines from example A (comments and free text are fine):
```
6504703 rnd-short jaguar03 8gpu W1 c5 RESERVATION qos=csresnolim
# cancelled 6504704 6504705 (gnolim CPU cap ~68/80, 15-cpu jobs do not fit)
```

**Reservation jobs** additionally carry `--reservation=<name> --qos=csresnolim`; discover the name live, never hardcode:
```bash
scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" | grep -oP 'ReservationName=\K\S+'
```

### What a generated submission script actually contains

Verbatim, `<sweep>/slurm/submission_script/cheetah08-09__rnd-short.slurm` from example A:

```bash
#!/bin/bash
#SBATCH --job-name=acbfig5648
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a4000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200000M
#SBATCH --time=4-00:00:00
#SBATCH --output=<sweep>/slurm/logs/%x_%j.out
#
# Generated by make_submission_scripts.py: node class cheetah08-09 x packing group rnd-short.
# Sized for a fully free node (4 GPU(s), W=1 slots/GPU, c_used=5 threads/slot, capped by cpu).
# For a partly free node, override on the sbatch command line:
#   sbatch --parsable --nodelist=<node> --gres=gpu:nvidia_rtx_a4000:<G_free> \
#          --cpus-per-task=<1*G_free*5> --mem=<1*G_free*50000>M \
#          <this script> <G_free>
# The trailing <G_free> positional is informational only; worker_manager reads the
# GPUs it was actually given from CUDA_VISIBLE_DEVICES.
# Exported BEFORE worker_env.sh so the env file can derive per-worker settings
# (e.g. a JAX per-process GPU memory fraction from the packing count).
export GPU_SWEEP_SLOTS_PER_GPU=1
export GPU_SWEEP_CPUS_PER_RUN=5
source <sweep>/slurm/worker_env.sh
# GPU_SWEEP_EXTRA_ARGS lets one script serve both modes: the canary wave submits with
# --export=ALL,GPU_SWEEP_EXTRA_ARGS=--canary (plus a short --time); full runs leave it unset.
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/worker_manager.py \
  --sweep_dir <sweep> \
  --slots_per_gpu 1 \
  --cpus_per_run 5 \
  --group rnd-short ${GPU_SWEEP_EXTRA_ARGS:-}
```

The default `--time` a generated script carries is its partition's max (`gpu` → `4-00:00:00`, `gnolim` → `20-00:00:00`); override on the command line with availability.py's recommendation.

### Environment variables

**Set by the submission script before sourcing `worker_env.sh`:**
- `GPU_SWEEP_SLOTS_PER_GPU` = W
- `GPU_SWEEP_CPUS_PER_RUN` = c_used
- `GPU_SWEEP_EXTRA_ARGS` — passed in at submit via `--export=ALL,GPU_SWEEP_EXTRA_ARGS=--canary`; empty for full runs.

**Set by `worker_manager.py` for every child run process:**
- `CUDA_VISIBLE_DEVICES` = the single GPU token that slot is pinned to
- `GPU_SWEEP_RUN_ID`, `GPU_SWEEP_W`, `GPU_SWEEP_CPUS_PER_RUN`, `GPU_SWEEP_SWEEP_DIR`

**Read by `worker_manager.py` from Slurm:** `CUDA_VISIBLE_DEVICES` (its GPU list — accepts index or UUID tokens), `SLURMD_NODENAME`, `SLURM_JOB_ID`, `SLURM_MEM_PER_NODE`.

**What `worker_env.sh` must cover** (Step 3 of SKILL.md): the project conda env (argv[0] is already the full python path, plus activation if the code needs it); framework allocator settings for PACKED GPUs; and any library-path fixes (`LD_LIBRARY_PATH` additions such as `/usr/lib/nvidia`).

For JAX, the required pair (unbounded per-slot growth shows up as cuDNN initialization failures). Example B, verbatim:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
W_SLOTS="${GPU_SWEEP_SLOTS_PER_GPU:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$(awk -v w="$W_SLOTS" 'BEGIN{ if (w < 1) w = 1; printf "%.4f", 0.72 / w }')"
```
plus `export PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`, `D4RL_DATASET_DIR=...`, `MUJOCO_PY_MUJOCO_PATH=/p/rlprojects/RLforOR/distributional_rnd/mujoco/mujoco210` (a shared path, because a `~/.mujoco` fallback only exists for the owner and breaks collaborator jobs).

For torch (example A) no preallocation fraction is needed — torch allocates on demand; `w_max` bounds co-location instead. Its whole file:
```bash
umask 002
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_CACHE_PATH=/tmp/cuda_cache_acb_$USER   # PTX JIT cached per node on local disk
mkdir -p "$CUDA_CACHE_PATH" 2>/dev/null || true
```

### How the worker claims and finishes work

`worker_manager.py --sweep_dir <abs> --slots_per_gpu W --cpus_per_run c [--group NAME] [--canary] [--watchdog_seconds N] [--sample_seconds N]`

1. Reads `CUDA_VISIBLE_DEVICES`; empty ⇒ hard exit (`CUDA_VISIBLE_DEVICES is empty; the manager needs at least one GPU token`).
2. Starts **one sampler thread** for the whole job, then **W threads per GPU token** = W×G_used worker slots. Slurm sees `--ntasks=1`; slots are subprocesses, not Slurm tasks.
3. Each slot loops: list `pending/` sorted ascending → keep up to 32 candidates matching `--group` → `random.choice` → `os.rename` into `running/` → write the `claim` block → `subprocess.Popen(cfg["argv"], env=…, start_new_session=True)` with stdout+stderr appended to `<logs>/run_<run_id>.log`.
4. On child exit it **SIGKILLs the whole process group** (survivor sweep — added after 2026-07-16 on ai02, where a dead wrapper left its trainer squatting on the GPU and every next claim OOMed against it).
5. `finalize`: `done/` if rc==0 **and** the completion flag is true (or there is no completion file); otherwise `failed/` with the `exit` block. A watchdog kill counts as **done** in canary mode and **failed** in normal mode.
6. On SIGTERM/SIGINT (walltime, preemption) it forwards SIGTERM to every child group and **leaves markers in `running/`** for a later wake-up to requeue. Stubborn children are SIGKILLed after `KILL_GRACE_SECONDS = 30`.
7. A slot exits when no matching pending run is left; the manager exits when every slot has exited, printing `manager done on <node> (job <id>)`.

Sampling cadence: `--canary` ⇒ 60 s and a 1200 s watchdog; normal ⇒ 120 s and no watchdog; `--sample_seconds` / `--watchdog_seconds` override either.

---

## 4. The packing rule

Per node class (G = GPUs/node, V = `gpu_mem_mb`, C = `cpu_alloc_threads` = Slurm CPUEfctv, M = `sys_mem_mb`) × packing group (v, m raw; c_min = busy threads, c_max = 8):

```
W_mem  = min( floor(V / 1.25v),  floor(M / (1.25m·G)) )    # the two fixed memory bounds
         (then W_mem = min(W_mem, w_max) if w_max is set)
c_fit  = floor(C / (W_mem·G))                              # threads per slot if memory sets W
c_used = min(c_max, max(c_fit, c_min))                     # CPU is the one adjustable knob
W      = min(W_mem, floor(C / (c_used·G)))
if W == 0 and the RAW estimate fits (v ≤ V and c_min·G ≤ C and m·G ≤ M): W = 1, inflation removed
   else: use the largest G_used in [1, G−1] with c_min·g ≤ C and m·g ≤ M; if none, SKIP the class
   in the W=1 branch, c_used = min(c_max, max(floor(C / (W·G_used)), c_min))
```

Job request per node:
```
--ntasks=1  --cpus-per-task = W · G_used · c_used  --mem = W · G_used · (1.25m, or raw m when the W=1 branch fired) MB
```

**The 25% memory inflation** is exactly the `INFLATION = 1.25` multiplier applied to BOTH memory estimates (`gpu_mem_mb` and `host_mem_mb`) before dividing into capacity. It is dropped only in the W=1 escape branch — the inflation must never block one run per GPU.

**The CPU-threads knob (c).** v and m are never adjusted; **only W and c are.** If GPU or host memory caps W, give each slot MORE threads (up to c_max=8) so CPUs do not idle. If CPU caps W, never cut c below `cpu_threads_busy` — cutting busy threads slows every run and buys no throughput. When the per-user CPU pool is what binds new submissions, lower c toward c_min on the NEW jobs (never below c_min) to buy more GPUs.

**Class skipped** when `compute_capability < min_compute_capability`, `> max_compute_capability`, `v > V`, or a single run does not fit on CPU/host memory. `PACKING.md` lists every skip with its reason.

**Script naming:** `slurm/submission_script/<node class name>__<packing group name>.slurm`, where the class name is the compressed node list from `server_introduction.json` (`serval06-09`, `ai01-04_lynx10`, `cheetah08-09`, `jaguar03`) and the group name is profile names joined with `+` (`mujoco+antmaze`, `rnd-short`). **Job names:** at most two base codenames per launch, numbered suffixes per node class (`acbfig5648` = codename `acbfig5` + running number).

Real `PACKING.md` rows (example A) showing the four `why-capped` values:

```
| node class  | group        | W | c_used | G_used | cpus/job | mem/job MB | why-capped |
| serval06-09 | rnd-short    | 2 | 8      | 2      | 32       | 200000     | cpu        |
| cheetah04   | rnd-capride  | 2 | 8      | 4      | 64       | 960000     | host-mem   |
| cheetah01   | acb-capride  | 1 | 8      | 2      | 16       | 192000     | W=1-branch |
| nekomata01  | acb-short    | 1 | 8      | 2      | 16       | 100000     | gpu-mem    |
```

**Packing feasibility is already proven — do not re-profile it.** `references/packing_validation/packing_validation.md`: on h100_nvl, rtx_a4000, quadro_rtx_4000, gtx_1080_ti (and p100 partially), W worker slots ran concurrently on one GPU under one job id through the real job shape, with throughput× at W=4 of 0.92–0.97 under a deliberately compute-saturating mock workload. Reading: the sharing mechanism costs only 3–8%; **packing pays only for runs that under-use the GPU** — a run using fraction u of the GPU gains roughly min(W, 1/u).

---

## 5. The canary phase

The first submission wave. It fills the GPU allowance with SHORT runs of the **real entry point** (canary-only step limit, hard-killed at 20 minutes), samples every 1 minute, and writes everything under `canary/`. Its purpose: prove the setup and measure true peaks before the full run.

1. **Import check, one run per node class** (single process, e.g. the pipeline's `--check_import`), so first-import builds happen once instead of as a W×G race. Example B recorded these in the id file with group `import_check` (jobs 6489023–6489027).
2. **Build the canary queue and submit:**
   ```bash
   python3 .../scripts/build_queue.py --sweep_dir <sweep> --canary --count 32
   ```
   `--count` = the worker slots you will fill. Selection is **round-robin across resource profiles, keeping run_id (seed) order inside each profile**; each selected marker gets `canary.override_args` appended to argv, `completed_marker` dropped, `"canary": true` added.
   **Run at least 30 canary runs at the same time (rule of 2026-07-15).** A couple per packing group misses per-config outliers; 30+ short runs spread across the sweep's distinct configs (every game/dataset × every resource profile) and across candidate node classes catch slow configs, memory outliers and node-class quirks in one pass, for the same wall clock as a single canary. Fewer than 30 distinct configs ⇒ repeat configs across node classes rather than submitting fewer.
   Canary runs are **NEVER full-size**. Submit with the normal submission rules but:
   ```bash
   sbatch --parsable --nodelist=<node> --gres=gpu:<type>:<G> \
          --cpus-per-task=<W*G*c> --mem=<...>M --time=00:40:00 \
          --export=ALL,GPU_SWEEP_EXTRA_ARGS=--canary \
          <sweep>/slurm/submission_script/<class>__<group>.slurm <G>
   ```
   `--time=00:40:00` is the one deliberate exception to "always submit at the maximum allowed `--time`". `worker_manager --canary` hard-kills any run at 1200 s and samples every 60 s.
3. **Read the results and correct once:**
   ```bash
   python3 .../scripts/monitor.py --sweep_dir <sweep> --canary
   ```
   It writes `canary/infra_history.md` and `canary/resource_usage/estimate_vs_actual.md`. Verdicts per profile × quantity: ratio `> 1.0` → "over the estimate - lower W for new submissions now"; `< 0.6` → "room to raise the packing count W (recompute)"; otherwise "packing holds".
   **Fix W and c per node class ONCE**, edit `resource_estimate.json` (record the correction in `canary.note` or `correction_history`), then re-run `make_submission_scripts.py`.
4. Only then start the full submission batches — sampling drops to 120 s, reporting to 20 minutes.

**What the canary must prove before the main submission:** the entry point imports and trains on each candidate node class; the real GPU-memory and host-memory peaks and busy-thread counts per profile; and that canary outputs landed only under `canary/`. Example A's canary produced exactly the corrections that mattered: ACB jacobian transients OOMed W=2 on a 20 GB card ⇒ `w_max=1`; sm_89 PTX JIT works on cheetah02; `rnd-short` GPU memory had to be raised 3000 → 4300 MB from LIVE peaks after 7 GPU-OOM failures on 8 GB jinx cards; an 18-action-game profile `acb-short18` (v=11300) was split out after a collaborator report of a Gravitar OOM on 11 GB cards.

Note the sampler caveat visible in both `estimate_vs_actual.md` files: host-RSS is summed over the process tree, so a fork-heavy pipeline overcounts copy-on-write pages (example A shows host-memory ratios of 6–16× that are an artifact, not a real overrun). GPU memory and CPU threads are the trustworthy columns there.

---

## 6. Submission batching, ordering, etiquette, cap accounting

Read availability **first, every time**. Then four rules:

1. **H100 allowance.** With F free H100 GPUs cluster-wide, hold at most `max(2, F − 2)`, and never more than are free. availability.py prints `allowed extra = max(0, min(F, max(2, (F+H)-2) - H))` where H is what we already hold.
2. **Pack worker slots, one job id per node.** W slots per GPU; ALL claimed GPUs of a node under ONE job id; `--ntasks=1`, subprocesses.
3. **Capability order, ~20-GPU submission batches.** Fill from the highest-capability node class downward (`capability_rank` 1 = H100 NVL … 16 = Titan X). A batch may end above or below 20 GPUs because jobs carry 1–8 GPUs. **Wait until the batch RUNS** before the next one; a job stuck PENDING ~5 minutes ⇒ cancel that exact id and skip that node this round (example A's id file: `# cancelled 6504367 (pending Resources >5min, jaguar02)`). Then re-read availability and submit the next batch. The per-user **CPU and memory** caps often bind BEFORE the GPU cap under heavy packing — availability.py checks all three.
4. **Partition order: `gpu` → `gnolim` → reservation LAST.** Reservation jobs are admitted above the per-user partition caps but their usage still counts INTO those caps, so reserved-first shrinks the open-partition ceiling. Reservation jobs add `--reservation=<name>` (discovered live) **plus `--qos=csresnolim`** — without the qos the job is charged to the open-partition caps and pends on QOSMaxCpuPerU even though the reservation would admit it (observed 2026-07-15; example A's id file records exactly this: `# cancelled 6504698 (needs qos=csresnolim for reservation cap override)`).

**`--time` for every job** = min(partition limit, time to next maintenance window − 30 min, reservation end when applicable). availability.py prints the current value per partition. **Always submit at this MAXIMUM allowed `--time` (rule of 2026-07-15)** — never a shorter, guessed walltime. A job that outlives a short guess is killed and wastes its whole segment; a job that finishes early releases the node immediately, so the long request costs nothing. Long runs that cannot finish inside even the maximum segment checkpoint + resume across segments. (An over-limit `--time` is NOT rejected on this cluster — it pends forever with `StartTime=Unknown`.)

**Per-user cap accounting** (`server_introduction.json` → `partition_limits`, Table 2):

| pool | GPUs in partition | per-user GPU cap | per-user CPU cap | per-user memory cap | time limit | qos |
|---|---|---|---|---|---|---|
| `gpu` | 158 | 40 | 400 | 4,194,304 MB | 4-00:00:00 | cspartgpu |
| `gnolim` | 25 | 20 | 80 | 1,048,576 MB | 20-00:00:00 | cspartgnolim |
| reservation | read live | 1024 | 16384 | 1,073,741,824 MB | min(partition limit, reservation end) | csresnolim |

(Also present for CPU-only work: `cpu` 400 cpu / 4 TB / 4 days / cspartcpu, and `nolim` 80 cpu / 1 TB / 20 days / cspartnolim.)

Read the live counters with `scontrol show assoc_mgr qos=<qos> flags=qos` → `MaxTRESPU=cpu=<cap>(<current>)`. availability.py section (ii) does the same accounting from `squeue` (running AND pending both count — they commit the caps).

**Fill-the-quota checklist (rule of 2026-07-15, run after EVERY batch)** — do not stop early with GPUs on the table:
1. GPU counters below cap on `gpu` AND `gnolim` while pending runs remain and compatible free GPUs exist ⇒ submit more. If the CPU pool binds, LOWER c toward c_min on the NEW submissions (never below the busy-thread count).
2. The reserved node comes LAST but is NEVER skipped while it has free GPUs and runs are pending. Check its ACTUAL free GPUs live (`scontrol show node <node> | grep AllocTRES` — CPU-only jobs can occupy all its CPUs while every GPU stays free; a stale morning briefing is not evidence).
3. Reservation jobs MUST carry `--qos=csresnolim` with `--reservation=<name>`; with the qos, reservation capacity rides ON TOP of the open-pool caps.

Stop submitting only when worker slots ≥ pending runs, or every pool cap AND the reservation are genuinely exhausted.

**Collaborator capacity (rule of 2026-07-16).** If collaborators are requested: read `/p/rlprojects/.claude/skills/collab-handbook/SKILL.md` including its `references/common_problems.md`, generate the `for_collaborator/` packet, and then **send an independent subagent to verify the packet** before handing it over — re-run the permission gates, `bash -n` every script, check every `.slurm` locates packet files by ABSOLUTE path (never `dirname "$0"` — sbatch spools the script), confirm the uid guard, queue-depth guard and `DRY=1` preview, and walk the common-problems list as a checklist. A packet bug otherwise surfaces only when the collaborator's first job dies (2026-07-16: every packet job failed at t=1 s on the spool trap).

---

## 7. The 20-minute monitoring cycle

### What `monitor.py` does (one pass)

```bash
python3 /p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/monitor.py --sweep_dir <sweep>            # bookkeeping only
python3 .../monitor.py --sweep_dir <sweep> --requeue                                                    # + act on orphans
python3 .../monitor.py --sweep_dir <sweep> --canary                                                     # canary tree instead
python3 .../monitor.py --selftest                                                                       # pure-helper assertions, no sweep needed
# testing-only: --squeue_override <file> --sacct_override <file>; --server_introduction <path>
```

Per invocation, six steps:

1. Looks up the live state of **ONLY** the ids in `slurm/submitted_jobids.txt` (owner, tagged `owner`) merged with every `for_collaborator/submitted_jobids_*.txt` (each tagged with the filename's trailing token, e.g. `…_yuxinchen.txt` → `yuxinchen`). `squeue` for running/pending, `sacct` for ended. It never selects jobs any other way. Terminal prefixes: COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED, BOOT_FAIL, DEADLINE, REVOKED.
2. Counts queue markers in pending/running/done/failed.
3. **Appends one block to `infra_history.md`** — header `## <YYYY-MM-DD HH:MM>  (jobs: N running (owner a, name b), M pending | runs: R running, P pending, D done, F failed)` then a per-node table, ranked by GPU capability, columns: `node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs`. Every physical GPU on its own line inside the memory/usage cells; `*` marks GPUs running our job. Collaborator rows are tinted grey (`<span style="color:gray">`) with an owner tag in the node cell (`cheetah01 (yuxinchen)`) as the fallback; purely visual — every aggregate still counts them. The table ends with THREE total rows: `TOTAL (owner)` black, `TOTAL (collaborators)` grey, grand `TOTAL` black. A node with a live job but stale samples (older than `STALE_SECONDS = 600`) prints `not recorded`, never a fake 0. **Existing blocks are never edited.**
4. **Rebuilds `slurm/resource_usage/estimate_vs_actual.md`** atomically: per profile × quantity (GPU memory MB / host memory MB / CPU threads busy), raw estimate vs observed peak vs 95th-percentile per-run peak vs ratio vs verdict vs runs sampled.
5. Reports failed runs (broken down by exit code and by node class) and running-marker orphans whose owning job is terminal. With `--requeue` it archives each orphan's partial output into `data/killed_attempts_<YYYY-MM-DD-HH-MM>/` (moved, not copied), strips the `claim` block, and writes the marker back to `pending/`. **Failed markers are only listed, never auto-requeued.**
6. Prints a plain-text summary (jobs line, newly ended ids, marker counts, failed breakdown, orphan/requeue actions, the three worst estimate/observed ratios, and the two paths it wrote).

### The per-tick order (SKILL.md Step 8)

1. **Top-up** — availability.py; if other users released GPUs, our jobs ended at their limit, or cap room opened, and pending runs remain: submit under the same four rules, appending every id at submit time.
2. **Requeue** — `monitor.py --requeue`, then classify `failed/` markers by the **failure-cause rule**: requeue ONLY failures whose cause is outside the run — out of memory (and also lower that node class's W or raise the memory-fraction guard), an incompatible or broken node (GPU abort signatures like rc −6), or an environment problem after `worker_env.sh` is fixed. **NEVER requeue a deterministic application error** (same config + seed fails identically every retry, e.g. a training divergence): it stays in `failed/` as the truthful record and is reported to the user.
3. **Collaborator channel** — every wake-up checks `for_collaborator/problems/open/`; a new report is handled THIS tick (diagnose, fix owner-side, append the resolution, move to `problems/resolved/`, and add the incident to collab-handbook's `references/common_problems.md`). Also glance at new collaborator id-file lines and `for_collaborator/logs/`. (2026-07-16: a t=1 s packet-bug report sat ~9.5 h in `open/`.)
4. **Bookkeeping** — `monitor.py` (no flags); refresh `submitted_jobids.txt` with any new ids.
5. **Packing changes** — raising W or changing c at most once per ~2 hours; **lowering W after a real out-of-memory applies immediately** to new submissions. v and m are never adjusted.
6. **Close** — when the queue is drained and all jobs ended, write a closing summary into `experiment_background.md` and stop the loop.

**Monitoring roster — exactly three things run:** (1) the in-job usage sampler (60 s canary / 120 s after); (2) this 20-minute wake-up; (3) ad hoc sub-agents to diagnose a single failure. Nothing runs between sessions.

### Arming the loop as its own Slurm job

Per `/p/rlprojects/.claude/skills/sweep-monitoring/SKILL.md`, the standing loop is submitted as its OWN Slurm job so it survives the launching session; the Claude session only supervises (reads the latest report, confirms the job is alive via `squeue` on the id from the id file, intervenes only to advance the queue, diagnose failures, or resubmit the monitor job if it died).

- One 1-task job on an open long partition (e.g. `nolim`): `--ntasks=1 --cpus-per-task=1 --gpus-per-node=0`, generous `--mem` (16G+; a 4G monitor was SIGKILLed at ~3000 records), a long `--time` capped by the next maintenance window, output to `logs/monitor_loop_%j.log`.
- **ABSOLUTE paths only** — sbatch spools the script, so `dirname "$0"` points at the spool directory.
- Its id goes into the run's own id file like any other job.
- It cycles every ~1200 s and stops when the completion sentinel (`SWEEP_COMPLETE`, written when no pending and no running markers remain) appears.
- Reduce the report's record loader to scalars at read time so peak memory scales with record COUNT, not episode count.

Working `monitor_loop.slurm` in this repo (RND run 8.1.2, `.../2026-08-01-01-44_run_8_1_2_…/slurm/monitor_loop.slurm`) — the skeleton, verbatim in structure:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=32G
#SBATCH --output=<RUN_DIR>/logs/monitor_loop_%j.log
set -u
RUN_DIR="<absolute run dir>"
MONITOR="$RUN_DIR/slurm/monitor.sh"
SENTINEL="$RUN_DIR/SWEEP_COMPLETE"
INTERVAL="${INTERVAL:-1200}"   # 20 min between ticks
while true; do
  if [[ -f "$SENTINEL" ]]; then echo "SWEEP_COMPLETE present; monitor loop exiting"; break; fi
  bash "$MONITOR" "$SWEEP_ID"
  ... completion test ...
  sleep "$INTERVAL"
done
```
and `slurm/monitor.sh` is one tick: refresh + dedup the own-id file (`sort -u -o "$JOBIDS" "$JOBIDS"`, then `squeue -u "$USER" -j "$ids"`), requeue orphans, run controllers + checkers, disk guard, then write the one combined Markdown report; all output `tee -a`'d to `slurm/monitor_history.log`. For a GPU sweep, the requeue/bookkeeping step of that tick is `monitor.py --sweep_dir <sweep> --requeue` followed by `monitor.py --sweep_dir <sweep>`.

Reports go to `train_runs/<run>/20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_monitoring.md`, combining, in order: a running-status table, a per-node run-count table covering every submitter, and the interim metrics tables (best value per metric column bold, second best underlined).

---

## 8. Hard rules about job ids, id files, and cancellation — quoted verbatim

**From `SKILL.md`, "Safety rules (non-negotiable)":**

> - Cancel only exact job ids from THIS sweep's `slurm/submitted_jobids.txt`. Never `scancel -u`,
>   `-t`, `-n`, or any other blanket form — the uid is shared with other sessions and manual jobs.
> - A completed run output is never deleted, and its run is never executed again.
> - A `running/` marker returns to `pending/` only after squeue AND sacct show its owning job
>   terminal, and only after its partial output is archived to `data/killed_attempts_<timestamp>/`.
> - Never archive or move any output whose run id has a marker in `queue/running/` — re-check
>   `running/` immediately before every archive operation; a moved checkpoint directory crashes the
>   live trainer that owns it.
> - Never cancel a job because progress is invisible — check the usage samples, the output file
>   count, and the queue markers first; low CPU on a network- or GPU-bound job is normal.
> - CPU-only helper work (import checks, monitors) never runs on A100 or H100 nodes.
> - Every job requests its `--time` from the current caps (partition, maintenance, reservation) —
>   an over-limit request silently pends forever on this cluster.

**From `SKILL.md`, Step 7:**

> Submission always uses `sbatch --parsable` and appends the id to `slurm/submitted_jobids.txt` in the same command.

**From `references/file_formats.md`, "submitted_jobids.txt (slurm/)":**

> One line per sbatch: `<jobid> <node_class> <node> <G> <W> <c_used> <group> <submit_ts>`.
> Cancellation may use ONLY ids from this file. Never any blanket scancel.

**From `references/file_formats.md`, "Queue markers":**

> A running/ marker may be moved back to pending/ ONLY after squeue/sacct show its `claim.job_id`
> terminal, and after archiving any partial run output to `data/killed_attempts_<YYYY-MM-DD-HH-MM>/`.
> **Never archive or move an output whose run_id has a marker in `running/`** — check `running/`
> IMMEDIATELY BEFORE every archive operation. Moving a live run's checkpoint directory crashes its
> trainer mid-write (this happened once, 2026-07-11: seven live runs lost their checkpoint dirs to
> a cleanup pass that iterated a stale id list).

**From `/p/rlprojects/.claude/CLAUDE.md`:**

> Never blanket-cancel: `scancel -u <user>`, `scancel -t PD`/`-t R`, and `scancel -n <name>` are
> forbidden (multiple people and sessions run jobs on this cluster; a blanket cancel has destroyed
> other people's running work). Record every id `sbatch` returns in YOUR OWN id file and cancel only ids from that file. The
> sweep owner's file is `slurm/submitted_jobids_<sweep_id>.txt`; every ADDITIONAL submitter uses
> `for_collaborator/submitted_jobids_<sweep_id>_$USER.txt` — one writer per file, never shared.

**From `/p/rlprojects/.claude/rules/cluster-slurm.md`:**

> **Never run `scancel -u <user>`, `scancel -t PD`/`-t R`, or any blanket cancel.** A job *name* can be
> reused by another session, so `scancel -n <name>` (or `--name`) is **also unsafe** — do not use it.
> …
> **Do NOT ask the user for job ids** — the user leaves right after submitting, so the run-folder id file is
> the source of truth. Never blanket-cancel by user / state / name.

**From the user's global CLAUDE.md:**

> **Never cancel a Slurm job I did not submit in THIS session.**

Note on naming: this GPU skill's own file is `slurm/submitted_jobids.txt` (that is the name `monitor.py` reads via `resolve_paths`); the shared CLAUDE.md rule writes it with the sweep-id suffix `slurm/submitted_jobids_<sweep_id>.txt`. Both real GPU sweeps use the unsuffixed `slurm/submitted_jobids.txt` for the owner and `for_collaborator/submitted_jobids_<sweep_id>_<user>.txt` for collaborators — keep that, because `monitor.py` hardcodes the owner path and globs collaborator files by the `submitted_jobids_*.txt` prefix.