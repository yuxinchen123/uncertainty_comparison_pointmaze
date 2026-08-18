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

(to be filled by `benchmarks/env/bench_montezuma_env.py` in round 1)

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
