# RK4 AntMaze: where one training iteration spends its time, and speed ideas that keep results identical

Measurements of 2026-11-13, 13:40 to 14:23 PT, by a helper of the main session; measurement and code
behind options only. No configuration file, queue, fleet table, `STATE.md`, frozen code or running job was
changed. Times are Pacific, converted from machines running Eastern. Tables come from `scripts/tables.py`
reading `records/`.

## Result

1. **The iteration is the physics.** At 128 policies x 16 environments on the H100 NVL (serval05) one
   iteration takes 1.855 s, and the 640 batched RK4 physics steps of the rollout alone take 1.816 s
   (97.9%). Everything else together is about 2%: the environment's own work (fault tests, reward, reset,
   observation) 21 ms, the update 17 ms, the rest under 5 ms each.
2. **Most of it is a fixed cost per physics step.** A line through 32 and 128 policies gives 1.401 s
   fixed per iteration and 3.55 ms per policy, so the fixed part is 76% of the iteration at 128 policies.
   Natively, one RK4 physics step of MuJoCo Warp is 422 kernel launches (4 forward passes of about 102
   launches each, the solver's loop body counted once, and the Runge-Kutta bookkeeping); it takes 1.93 ms
   at zero worlds plus 0.34 ms per 1,000 worlds, 2.63 ms at 2,048 worlds. That is about 4 microseconds per
   kernel, the time a chain of small dependent kernels needs on this card. Within one forward pass the
   constraint solve is 54%, the position stage 16% and the velocity stage 11%.
3. **No idea that keeps results identical gains 5% per iteration.** Measured in the whole iteration, built
   in one process with the unit's program and timed in alternating rounds: the rollout loop unrolled 1 or
   4 steps per body instead of 16 (gain 1.002), the frame loop over the 5 physics steps unrolled (1.003
   with the rollout at 4 steps per body), and MuJoCo Warp's solver loop as a plain loop of 4 iterations
   instead of its conditional graph loop (gain 1.043 at 128 policies and 1.033 at 32: the largest found,
   but under 5%). All are bit-identical on processors. The update loop's layout is not: it changes numbers.
4. **Start-up has two savings that keep results identical, and both are now options.** A unit restart
   compiles the iteration three times: on its first call, again on its second call, and again after the
   first window end, because the state made outside the compiled program (restore, reset, the window's
   counters set back to zero) is not yet placed on the card. On serval05 that costs 12.3 s and 12.1 s
   beyond the first compile. Putting the state on the card before those calls (`--put-state-on-card`)
   leaves one compile; JAX's compilation cache (`--compilation-cache-dir`) cuts that first compile from
   15.4 s to 5.4 s. From process start to the end of the third iteration, at 128 policies on serval05:
   56.4 s today, and 12.1 s more at the first window end; 46.4 s and nothing more with the state on the
   card; 35.4 s with both options. On the training nodes, where the first iteration took 87 to 106 s today,
   the saving should be larger in proportion, but it is not measured there.
5. **A hazard found on the way.** More than about 20 physics calls in one compiled program (the frame loop
   unrolled inside a rollout loop of 16 or 32 steps per body: 80 or 160 calls) makes MuJoCo Warp capture
   its CUDA graphs again on every call: 12.7 s per iteration instead of 1.85 s. The unit's program has 16
   calls and captured nothing in steady state.
6. **No A100 row.** No A100 was free during the measurement: all four cards of cheetah01 and of cheetah04
   were held by other users' jobs with 10 to 21 hours left (13:55 PT). The A100 figure of the integrator
   check (2.168 s at 128 policies) is the reference there.

## The workload

The unit's trainer as `scripts/run_training.py` builds it from
`configs/experiments/antmaze_umaze_warp-physics_rk4_time-step-0.01_frame-skip-5_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml`
(through `device/series.py`, as the doubling series and the integrator check build it), at 16 environments
per policy and the policy counts given, with 2 priming rollouts instead of the file's 50 (priming only sets
the input statistics):

- MuJoCo Warp inside MJX 3.11.0 (warp-lang 1.14.0), RK4 at 0.01 s x 5 physics steps per environment step,
  4 solver and 8 line-search iterations, the fault guard at 30 m/s and 100 rad/s, 48 contact slots and
  512 constraint rows per environment, graph mode `WARP`;
- 128-step rollouts, the CleanRL PPO update with random network distillation (4 epochs x 4 minibatches),
  the episode log and the coverage map on, the rollout loop 16 steps per body (`scan_unroll = 0` picks 16
  above 8 policies), the update loop 2;
- JAX 0.11.0, driver 580.159.04, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, Warp's kernels on `/localtmp`.

Interpreter `/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python` with `PYTHONNOUSERSITE=1`. Card units:
the serval05 queue, kind `profile`, each under 15 minutes, each with a watcher that ends the unit with exit
code 90 when a process it did not start appears on the card (none did). Processor checks: puma01 through
the reservation `sl5nw_168`.

## Throughput of the unit's program

Define $s$ as the median seconds of one whole iteration in chained calls (as the training loop calls it),
$C$ the policies (copies) and $e = 16$ the environments per policy. Total steps per second
$\dfrac{128\thinspace e\thinspace C}{s}$, steps per second per copy $r = \dfrac{128\thinspace e}{s}$,
hours per million steps per copy $\dfrac{10^{6}}{3600\thinspace r}$.

| card | copies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per copy (thousands) | hours per million<br>steps per copy |
|---|---:|---:|---:|---:|---:|
| H100 NVL, serval05 | 32 | 1.514 | 0.0433 | 1.352 | 0.205 |
| H100 NVL, serval05 | 128 | 1.855 | 0.1413 | 1.104 | 0.252 |

At 128 policies, 134,217,728 steps per policy take $0.252 \times 134.2 = 33.8$ hours of iterations on this
card (1.4 days).

## Where the time goes in one iteration

Each piece is its own compiled function that returns all it computes (pieces of
`benchmarks/harness/profile_study_iteration_pieces.py`, `build_pieces`), timed after two warm-up calls. The
environment step and the bare physics are timed as the iteration calls them: compiled with the state
donated and chained, so every call gets the buffers the previous call returned, at the same 128 random
torques per environment. Define $t(C)$ as a piece's seconds at $C$ policies. The line through the two
policy counts gives the part per policy $b = \dfrac{t(128) - t(32)}{96}$ and the fixed part
$f = t(32) - 32\thinspace b$. Shares are of the whole iteration at the same policy count. No piece's timed
calls captured a CUDA graph.

| part | 128 policies (s) | share | 32 policies (s) | share | fixed part (s) | per policy (ms) |
|---|---:|---:|---:|---:|---:|---:|
| whole iteration (chained calls, as the training loop calls it) | 1.8552 | 100.0% | 1.5144 | 100.0% | 1.4008 | 3.550 |
| environment step, 128 steps (chained, state donated) | 1.8372 | 99.0% | 1.4974 | 98.9% | 1.3842 | 3.539 |
| &nbsp;&nbsp;bare physics: 640 batched RK4 physics steps | 1.8160 | 97.9% | 1.4898 | 98.4% | 1.3810 | 3.398 |
| &nbsp;&nbsp;the environment's own work: fault tests, reward, reset, observation | 0.0212 | 1.1% | 0.0077 | 0.5% | 0.0032 | 0.141 |
| policy forward, 128 steps | 0.0006 | 0.0% | 0.0003 | 0.0% | 0.0002 | 0.003 |
| action noise draw | 0.0001 | 0.0% | 0.0001 | 0.0% | 0.0001 | 0.000 |
| critic pass after the rollout | 0.0006 | 0.0% | 0.0003 | 0.0% | 0.0002 | 0.003 |
| distillation forward, input statistics, whitening | 0.0010 | 0.1% | 0.0004 | 0.0% | 0.0002 | 0.006 |
| reward filter, its statistics, both advantage sums | 0.0039 | 0.2% | 0.0037 | 0.2% | 0.0037 | 0.001 |
| rows put in per-policy order | 0.0002 | 0.0% | 0.0001 | 0.0% | 0.0001 | 0.001 |
| coverage map | 0.0003 | 0.0% | 0.0002 | 0.0% | 0.0002 | 0.001 |
| episode log sums | 0.0001 | 0.0% | 0.0001 | 0.0% | 0.0001 | -0.000 |
| update: shuffle, gather, 16 gradients, clip, Adam | 0.0168 | 0.9% | 0.0072 | 0.5% | 0.0041 | 0.099 |
| &nbsp;&nbsp;16 gradients | 0.0141 | 0.8% | 0.0056 | 0.4% | 0.0027 | 0.089 |
| &nbsp;&nbsp;clip | 0.0006 | 0.0% | 0.0009 | 0.1% | 0.0009 | -0.003 |
| &nbsp;&nbsp;shuffle, gather and Adam | 0.0020 | 0.1% | 0.0008 | 0.1% | 0.0004 | 0.013 |
| not explained by the pieces (joins between pieces, host work) | -0.0054 | -0.3% | 0.0045 | 0.3% | 0.0077 | -0.103 |
| host time between calls: one call with a wait minus chained | 0.0036 | 0.2% | 0.0079 | 0.5% | 0.0094 | -0.045 |

- The bare physics is the same 128 x 5 batched `mjx.step` calls with the torques set per environment step
  and nothing else: no fault test, reward, reset, observation or counters. The environment's own work is
  the environment step minus the bare physics.
- The physics' cost depends on the torques (how often the ant touches the floor and the walls). With one
  constant torque per environment for all 128 steps, the bare physics took 1.583 s at 128 policies and
  1.244 s at 32 instead of 1.816 s and 1.490 s; a first reading of the environment's own work against
  that constant-torque physics (0.24 s) was wrong for this reason.
- The pieces timed alone add up to slightly more than the iteration at 128 policies (the row "not
  explained" is -5 ms): each piece waits for the card on its own, and its physics runs on other torques.
- The host is not a bottleneck: one call with a wait takes 3.6 ms more than a chained call at 128 policies.

## The physics step, stage by stage

MuJoCo Warp run natively (no JAX) on the unit's C model at 2,048 worlds (128 policies x 16) and 512
worlds, from the unit's spawn after 60 physics steps of random torques (torque redrawn every 5 steps).
Each stage is captured as its own CUDA graph and replayed 5 rounds of 20 times. Kernel launches are those
made while the graph is captured, so the solver's loop body counts once although up to 4 iterations run.
A line through the two world counts gives the fixed part (at zero worlds) and the part per 1,000 worlds.
The share is of the whole physics step at 2,048 worlds; the first eight rows are one forward pass, and one
RK4 physics step holds four forward passes.

| stage (one call) | kernel launches | 2,048 worlds (ms) | share of the step | 512 worlds (ms) | fixed part (ms) | per 1,000 worlds (ms) |
|---|---:|---:|---:|---:|---:|---:|
| position stage: kinematics, centre of mass, composite inertia | 26 | 0.095 | 3.6% | 0.086 | 0.084 | 0.005 |
| collision detection | 2 | 0.023 | 0.9% | 0.018 | 0.017 | 0.003 |
| constraint build | 13 | 0.043 | 1.6% | 0.041 | 0.040 | 0.001 |
| transmission | 1 | 0.008 | 0.3% | 0.007 | 0.007 | 0.001 |
| velocity stage | 21 | 0.068 | 2.6% | 0.063 | 0.061 | 0.003 |
| actuation | 3 | 0.013 | 0.5% | 0.013 | 0.012 | 0.000 |
| acceleration with the mass-matrix factor | 3 | 0.019 | 0.7% | 0.016 | 0.015 | 0.002 |
| constraint solve | 22 | 0.329 | 12.5% | 0.278 | 0.261 | 0.033 |
| forward (every stage above, once) | 102 | 0.609 | 23.1% | 0.523 | 0.495 | 0.056 |
| Runge-Kutta part: stages 2 to 4 (three more forwards) and the combination | 320 | 2.023 | 76.8% | 1.673 | 1.556 | 0.228 |
| whole physics step (forward and the Runge-Kutta part) | 422 | 2.633 | 100.0% | 2.108 | 1.933 | 0.342 |

- One forward pass is 0.609 ms at 2,048 worlds; four make 2.44 ms of the step's 2.63 ms, and the
  Runge-Kutta combination (copies of the state, the weighted sums and the final advance) the other
  0.20 ms. Within a forward pass: solve 54%, position stage 16%, velocity stage 11%, constraint build 7%,
  collision 4%.
- At these states the solver used 1.08 iterations per world on average, but 11 of the 2,048 worlds used
  all 4, so the conditional loop runs 4 iterations in nearly every physics step.
- The earlier native split of this study started the ant at the model's own origin, inside a wall of the
  U-maze (38 contacts and 160 rows per world); it is kept in `records/breakdown_h100-nvl_serval05_copies-128.json`
  and not used.

The same native step with one setting changed, timed in 8 alternating rounds of 10 replays:

| setting | changes numbers | 2,048 worlds: time against<br>the unit's setting | 512 worlds |
|---|---|---:|---:|
| solver loop as a plain loop of 4 iterations (no conditional graph node) | no | 0.958 | 0.982 |
| 128 constraint rows per environment instead of 512 | no while nothing is dropped | 0.978 | 1.024 |
| 16 contact slots per environment instead of 48 | no while nothing is dropped | 1.023 | 1.043 |
| 16 contact slots and 128 rows | no while nothing is dropped | 0.961 | 1.027 |
| 2 solver iterations instead of 4 | yes | 0.833 | 0.838 |
| 4 line-search iterations instead of 8 | yes | 1.001 | 0.989 |

Smaller pools gain nothing consistent, and the training has already reached 161 rows once; they are not
proposed. Two solver iterations save 17% of the physics but change the physics; reported only.

## Ideas measured in the whole iteration

The unit's program and each variant in one process at 128 policies, each with its own copy of the same
primed state, two warm-up calls each, then 8 alternating rounds of 2 chained calls (order reversed every
second round). The gain is the median over rounds of the unit's seconds per iteration divided by the
variant's, paired by round. "First call" is tracing, compiling, Warp's graph captures and one iteration.

| program | copies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per copy (thousands) | hours per million<br>steps per copy | first call (s) | graph captures<br>while timed | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the unit's program: rollout loop 16 steps per body, frame loop not unrolled | 128 | 1.8520 | 0.1415 | 1.106 | 0.2512 | 14.2 | 16 | 1.000 |
| frame loop unrolled (5 physics steps per body), rollout loop 16 steps per body | 128 | 12.7381 | 0.0206 | 0.161 | 1.7277 | 26.5 | 10240 | 0.145 |
| rollout loop not unrolled (1 step per body) | 128 | 1.8478 | 0.1419 | 1.108 | 0.2506 | 7.6 | 1 | 1.002 |
| rollout loop 4 steps per body, frame loop unrolled | 128 | 1.8453 | 0.1421 | 1.110 | 0.2503 | 9.7 | 0 | 1.003 |
| rollout loop 32 steps per body, frame loop unrolled | 128 | 12.6441 | 0.0207 | 0.162 | 1.7150 | 39.2 | 10240 | 0.146 |

The solver loop, in its own unit at 128 and then 32 policies, 10 alternating rounds each:

| program | copies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per copy (thousands) | hours per million<br>steps per copy | first call (s) | graph captures<br>while timed | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the unit's program: rollout loop 16 steps per body, frame loop not unrolled | 128 | 1.8644 | 0.1406 | 1.098 | 0.2529 | 14.3 | 0 | 1.000 |
| MuJoCo Warp's solver loop as a plain loop of 4 iterations | 128 | 1.7879 | 0.1466 | 1.145 | 0.2425 | 12.1 | 0 | 1.043 |
| the unit's program: rollout loop 16 steps per body, frame loop not unrolled | 32 | 1.5205 | 0.0431 | 1.347 | 0.2062 | 13.0 | 0 | 1.000 |
| MuJoCo Warp's solver loop as a plain loop of 4 iterations | 32 | 1.4720 | 0.0445 | 1.391 | 0.1997 | 10.9 | 0 | 1.033 |

- The plain loop was faster in 10 of 10 paired rounds at 128 policies and in 9 of 10 at 32. It saves
  77 ms per iteration at 128 policies and 48 ms at 32; a line through the two gives 39 ms fixed and
  0.29 ms per policy. My reading: at these states some world needs all 4 iterations in nearly every
  physics step, so the conditional loop runs 4 iterations anyway and its graph conditions are pure cost
  (2,560 forward passes per iteration). It is not proposed under the 5% rule; as an option it would be a
  few lines (`graph_conditional` off in the Warp model that `WarpAntMaze` builds), not added here.

- The two programs with 80 and 160 physics calls (frame loop unrolled inside a rollout loop of 16 or 32
  steps per body) captured a CUDA graph on every physics call (10,240 captures in 16 iterations): those
  programs never reach a steady state. A program with 20 calls (rollout loop 4, frame loop unrolled)
  captured none.
- The unit's program captured 16 graphs in its first timed round after the other programs had run; in a
  process that runs only the unit's program (the breakdown units) it captured none in steady state.

### Identity on processors

`scripts/identity_check.py` (puma01, job 6909304): the unit's trainer at 2 policies x 2 environments with
32-step rollouts, the rollout loop given explicitly as 16 steps per body, one priming rollout, then 3
iterations of each program from the same primed state. All 247 arrays of the state (the 14 island arrays
of the Warp data that a physics step returns unwritten left out) and the 3 returned metrics are compared
bit for bit with the unit's program.

| program | state arrays<br>bit-identical | metrics<br>bit-identical |
|---|---:|---:|
| the unit's program, built a second time | 247 of 247 | 3 of 3 |
| frame loop unrolled (5 per body) | 247 of 247 | 3 of 3 |
| rollout loop 1 step per body | 247 of 247 | 3 of 3 |
| rollout loop 4 steps per body | 247 of 247 | 3 of 3 |
| rollout loop 4 steps per body, frame loop unrolled | 247 of 247 | 3 of 3 |
| solver loop as a plain loop of 4 iterations | 247 of 247 | 3 of 3 |
| update loop 1 gradient step per body (the unit: 2) | 128 of 247 | 1 of 3 |
| update loop 4 gradient steps per body | 111 of 247 | 1 of 3 |

The update loop's layout changes the rounding of the gradients, which then changes the physics (the
largest differences are in the constraint forces); it is "same distribution" at best and is not proposed.
On processors MuJoCo Warp runs its solver loop as a host loop that stops when every world is done, and on
the card as a conditional graph node; in both, a world that is done skips the remaining iterations, so the
plain loop changes no number (code: `mujoco_warp/_src/solver.py`, `_solve` and `_solve_done`).

## Start-up of a resumed unit

`scripts/startup.py`, four processes in a row on serval05 (one per row), each as a resumed unit starts:
imports, build, no priming, three iterations each waited for, then a window end as the training command
makes it (the episode log's sums and the fault record set back to zero outside the program) and a fourth
iteration. "Programs compiled" is the jit cache of the iteration after each call.

| process | build (s) | iteration 1 (s) | iteration 2 (s) | iteration 3 (s) | process start to<br>end of iteration 3 (s) | iteration after<br>a window end (s) | programs compiled<br>after calls 1 to 4 | graph captures<br>in calls 1 to 3 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| as the units start today: no cache, state as built | 21.0 | 15.3 | 12.3 | 1.85 | 56.4 | 12.09 | 1, 2, 2, 3 | 16, 16, 0 |
| no cache, state put on the card | 21.3 | 15.4 | 1.9 | 1.86 | 46.4 | 1.86 | 1, 1, 1, 1 | 16, 0, 0 |
| cache, first process (fills it), state put on the card | 20.8 | 18.0 | 1.9 | 1.86 | 47.1 | 1.86 | 1, 1, 1, 1 | 16, 0, 0 |
| cache, second process (reads it), state put on the card | 20.4 | 5.4 | 1.9 | 1.87 | 35.4 | 1.87 | 1, 1, 1, 1 | 16, 0, 0 |

- As units start today the iteration compiles three times. With the state put on the card, once.
- The cache folder held 384 programs after the first process (every program of the process, the small
  ones of the build included); reading it cut the first iteration from 15.4 s to 5.4 s. The build itself
  (20 to 21 s) did not get shorter.
- The 16 graph captures of the first call are MuJoCo Warp's and happen in every new process.
- On the training nodes today the first iteration took 87 to 106 s and building and restoring 38 to 64 s
  (unit logs of the AntMaze folder), with several processes compiling on one host at once; the three
  compiles there are correspondingly longer, but this was not measured.

## Options added (default: as before)

| option | where | what | resumes a checkpointed unit |
|---|---|---|---|
| `--put-state-on-card` | `scripts/run_training.py` | `jax.device_put` of the state before priming, before the first iteration and after every window end | yes: nothing is added to the checkpoint or its identity |
| `--compilation-cache-dir <folder>` | `scripts/run_training.py` | JAX's persistent compilation cache in that folder, every program in it, switched on before anything is compiled | yes: nothing is added to the checkpoint or its identity |
| `frame_unroll` of `WarpAntMaze` | `src/exploration_platform/envs/antmaze/warp_antmaze.py` | the frame loop's layout; used by the measurement scripts only (no gain, and a hazard above 20 physics calls) | not a training option |

Tests (`tests/envs/test_antmaze_warp_speed_options.py`, processors, puma01): the frame loop unrolled 2 and 5
is bit-identical to the loop over 8 RK4 environment steps with a step-limit reset; the state put on the
card compiles the iteration once through its second call and a window end, and the state equals the state
without it bit for bit; a unit stopped after its first iteration without options and continued in a new
process with both options writes the same window records, number for number, as the unit run in one go
without them, and the cache folder is filled. All pass (job 6909350: this file, the configuration-file
tests and the Warp environment tests, 28 passed; job 6909352: the Warp training, composition and
checkpoint tests, 18 passed). Commit 7f32e6cd.

Using the options for the running units: they need the new code in a unit's frozen copy and the two flags
on its command line; a unit's results do not change. A cache folder serves one card model and library
version (JAX keys its entries on both); a folder on node-local `/localtmp` helps only a restart on the
same node, a folder on `/p/rlprojects` helps every node of that card model. Not decided here: the main
session's choice.

## Records, scripts and jobs

- `records/breakdown_h100-nvl_serval05_copies-128_second.json`, `records/breakdown_h100-nvl_serval05_copies-32.json`:
  the iteration breakdown; `records/native-split-from-spawn_h100-nvl_serval05.json`: the native split and
  its settings; `records/ideas_*.json`: the ideas; `records/startup_h100-nvl_serval05.jsonl`: start-up;
  `records/identity_check_processors.json`: identity.
- Scripts: `scripts/common.py` (the trainer as the unit builds it), `breakdown.py`, `ideas.py`,
  `identity_check.py`, `startup.py`, `tables.py`; units in `units/`, logs in `logs/`.
- serval05 queue units (kind `profile`), all exit code 0 with no other process on the card: `2026-11-13-13-39_profile-rk4-ant-iteration-breakdown-128-policies` (13:40 to 13:43 PT), `2026-11-13-13-44_profile-rk4-ant-ideas-loop-layout-128-policies` (13:44 to 13:57), `2026-11-13-13-45_profile-rk4-ant-iteration-breakdown-32-policies` (13:56 to 14:00), `2026-11-13-13-45_profile-rk4-ant-startup-compilation-cache` (14:00 to 14:03), `2026-11-13-13-53_profile-rk4-ant-iteration-breakdown-128-policies-second` (14:03 to 14:07), `2026-11-13-13-56_profile-rk4-ant-native-split-from-spawn` (14:07), `2026-11-13-14-07_profile-rk4-ant-ideas-solver-plain-loop` (14:18 to 14:23).
- Slurm jobs (puma01, processors), ids in the AntMaze folder's `slurm/submitted_jobids_speed-rk4_sl5nw.txt`:
  6909304 (identity), 6909330 (tests of an earlier form of the options, 28 passed), 6909350 and 6909352
  (tests of the options as committed, 28 and 18 passed). No job is left running.
