# montezuma_jaxatari — round ledger

Montezuma's Revenge — the hard-exploration game the original random-network-distillation paper
solved — fused into the platform through JAXAtari's pure-JAX reimplementation of the game, as
the platform's first discrete-action environment. Specification and deviations:
`src/exploration_platform/envs/atari_montezuma/spec.md` (`montezuma_oc4500_sticky@1`).

Times are Pacific with a `PT` marker; the machines run on Eastern and the times were converted
for display. Result files are not committed (the repository ignores `*.json`); each round's
result JSON lives under `benchmarks/*/results/` on disk and the tables below carry what it said.

## The research question this family answers

Can the platform train PPO with an exploration bonus end to end on a game — discrete actions,
stochastic dynamics (sticky actions), multi-room structure — with the game logic compiled into
the same single program as the agent and the bonus, and what does one such environment step
cost next to the maze families?

## The gate this family had to pass

1. **Correctness before speed.** The suite's Montezuma gates: adapter contract, keyed
   determinism and per-environment diversity, copy isolation, truncation and auto-reset, the
   x64 boundary; the discrete actor's sampling distribution against its softmax; end-to-end
   composition for `rnd_next_state` and `none`, with the PointMaze-only visit-count family
   refused. The PointMaze golden-parity gate and the whole AntMaze set must stay green beside
   them (the discrete path is chosen at compose time; the continuous path is untouched).
2. **Measured throughput** on the H100 NVL of serval05, env step alone and whole training
   iteration, recorded below.

## Rounds

| round | when | what was tried | outcome |
|---|---|---|---|
| 1 — fusing through JAXAtari | 2026-08-18, from 02:30 PT | JAXAtari commit `035fc46b` installed into `platform_jax` (jax/numpy/scipy pins unchanged); replacement asset package (no ROM ownership declared); the platform adapter over `AtariWrapper`; the categorical actor added to the agent | end-to-end training runs on CPU at the smoke scale (intrinsic reward falls as the predictor learns; coverage counts the start room). Two integration hazards found and handled: JAXAtari's integer game logic breaks under the platform's global x64 (fixed with the same trace-time `jax.enable_x64(False)` boundary the MJX AntMaze uses), and a compiled reset can hand two all-zero state fields one device buffer, which the donated training state refuses (fixed by copying the reset tree once on the host) |

## Where this family stands — environment step alone (H100 NVL, serval05)

Measured 2026-08-18 04:35–04:50 PT by `benchmarks/env/bench_montezuma_env.py` (medians over 11
rounds of 10 env steps, random actions, the state donated, two-call warm-up; result JSON
`benchmarks/env/results/2026-08-18_montezuma_env_h100nvl_serval05.json`). One env step is 4
game frames plus the observation stack and the explicit auto-reset. n_envs = 1, so copies =
environments:

| copies | seconds per env step | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory |
|---|---|---|---|---|---|
| 512 | 0.000664 | 0.77 M | 1,505.9 | 0.18 | 0.07 GiB |
| 1,024 | 0.000672 | 1.52 M | 1,487.1 | 0.19 | 0.07 GiB |
| 2,048 | 0.000643 | 3.18 M | 1,554.9 | 0.18 | 0.08 GiB |
| 4,096 | 0.000781 | 5.24 M | 1,280.3 | 0.22 | 0.10 GiB |

The game logic scales almost perfectly with the copy count — per-copy speed is flat at about
1,300–1,550 steps per second while the total climbs to 5.2 M at 4,096 copies — because the
state is a few hundred integers per environment and the whole step is elementwise arithmetic;
compare the ant's contact solve, which costs about 25 times more per step and saturates the
card far earlier. Compile time is 9–10 s per shape.

## Where this family stands — whole training iteration (H100 NVL, serval05)

(to be filled by `benchmarks/end_to_end/bench_montezuma_train.py` in round 1)

## Correctness, after round 1

All green on the processor, 2026-08-18 02:00–02:30 PT: the discrete actor's Gumbel-argmax
sampling frequencies match its softmax to 5e-3 over 200,000 draws and its log-probability is
the log-softmax gather on a real keyed network; the adapter's contract, keyed determinism,
per-environment sticky diversity, copy isolation, truncation-and-auto-reset (fresh episode
with 5 lives, wrapper clock at zero, keys advanced) and the x64 boundary all hold; and the
fused composition trains end to end for `rnd_next_state` (intrinsic reward falls as the
predictor learns) and `none` (intrinsic reward exactly zero), with the PointMaze-only
visit-count family refused at composition time. The PointMaze golden-parity gate and the
whole AntMaze set stay green beside them.

## Recorded future work

1. **Pixel observations and convolutional networks.** The original RND setup is 84x84
   grayscale pixel stacks through convolutional actor, critic and bonus networks; the platform
   currently feeds JAXAtari's object-centric observation to its multilayer perceptrons.
   JAXAtari's `PixelObsWrapper` provides the pixels; the agent and bonus need the
   convolutional variants.
2. **Cheaper auto-reset.** The adapter re-derives a fresh reset state every step and selects
   it where an episode ended; with no-op starts disabled the reset state is a constant except
   for its key, so a cached template with a re-split key would remove that work if profiling
   shows it matters.
