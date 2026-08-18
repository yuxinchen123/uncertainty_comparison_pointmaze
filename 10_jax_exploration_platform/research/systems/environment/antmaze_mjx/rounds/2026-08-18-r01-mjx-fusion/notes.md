# Round 1 — fusing AntMaze through MJX

Times are Pacific with a `PT` marker; the machines run on Eastern and were converted for
display. Card: the H100 NVL of serval05, held for this session under the lock, one job at a
time. Environment: `platform_jax` (jax 0.11.0) with `mujoco` / `mujoco-mjx` 3.11.0 added
2026-08-18 00:25 PT (registered in `.venvs/ENVS.md`; jax, jaxlib, numpy, scipy unchanged).

## What the round had to decide

The reference model — the Gymnasium `ant.xml` plus one box geom per wall cell, RK4 at MuJoCo's
default Newton budget — loads through MJX as-is, but does not run at a usable speed. The round
measured its way to settings that are fast AND provably faithful, in four probes.

## Probe 1 — the model as shipped (2026-08-18 ~00:30 PT)

umaze, float32, H100. **2,134 ms per env step at 1,024 copies; 2,789 ms at 4,096** (an env step
is 5 physics steps). About 400x too slow to be worth fusing. Also established: under the
platform's global `jax_enable_x64=True` MJX builds a float64 program; `jax.enable_x64(False)`
as a trace-time context gives float32 (the boundary the accepted stepper uses).

## Probe 2 — the MJX example-model settings (~00:50 PT)

Wall runs merged into single boxes; `iterations=1, ls_iterations=4` (the setting of MJX's own
humanoid/barkour models); Euler with `eulerdamp` disabled against `implicitfast`; the
`max_geom_pairs` / `max_contact_points` custom numerics.

1. Speed arrives immediately: 2.4–6.8 ms per env step at 1,024 copies depending on variant —
   about 400x over probe 1.
2. **The contact-cap numerics are broken in MJX 3.11's JAX backend**: ANY setting — including
   `(-1, 48)` and values far above the true pair count — silently builds a model with ZERO
   contact slots (`ncon-max=0`), and the ant free-falls. Do not use them.
3. `implicitfast` beats Euler-with-eulerdamp-off at the same budget (2.80 against 4.84 ms at
   1,024 copies).

## Probe 3 — is the cheap solver still the reference's physics? (~01:00 PT, CPU)

C MuJoCo (the reference implementation itself), passive settle, 400 env steps FROM THE MAP
ORIGIN — which on the umaze is inside a wall box, a flaw found later; the numbers below carried
wall interference and were superseded by probe 5:

| settings | settled torso z | contacts |
|---|---|---|
| RK4, full solver budget (the reference) | 0.0427 m | 38 |
| implicitfast, iterations 1, ls 4 | **19.0 m — flies away, 0 contacts** | 0 |
| implicitfast, iterations 4, ls 8 | 0.0427 m | 38 |

What survives from this probe: the MJX example-model budget (`iterations=1`) is genuinely
unstable for this ant (in float32 it NaNs), and `iterations=4, ls_iterations=8` is stable.

## Probe 5 — the committed gates, from the real spawn (~01:50–02:20 PT, CPU)

Three findings, now the standing gates in `tests/envs/test_antmaze_parity_cpu_mujoco.py`:

1. **Away from contacts, MJX IS the C algorithm.** One env step from 50 states along a
   random-torque trajectory: at the 25 states whose 5-substep window holds no contact, MJX
   float64 equals C float64 to 4.4e-16, and the fused float32 stepper is within 1.76e-6
   (float32 against float64 MJX: at most 1.8e-6, so the float32 choice costs nothing).
2. **At contact events the difference is centimetre-level and algorithmic.** Worst one-step
   |dqpos| 2.75e-2 m at the 25 in-contact states — MJX's collision functions differ from C's
   by documented design. Over a horizon this seeds ordinary chaotic divergence: 20 shared-
   torque env steps agree to 1.67e-6 until the first contact (state 4), then diverge to
   0.47 m — two samples of the same dynamics, not two different dynamics.
3. **The tuned solver shares the reference's equilibria exactly.** A first settling
   comparison at 400 env steps disagreed (tuned z 0.563 against reference 0.382) and looked
   like different physics; the controls showed the tuned transient was simply unfinished
   (|qvel| 1.4e-2 at 400 steps). Settled 1,200 steps, BOTH integrators rest at z = 0.38248 m,
   and a state settled under either stays settled under the other to 3.7e-11, in both
   directions.

## Probe 4 — abandoned

A speed re-measure of the accepted settings with the ant embedded at the origin (the probe-2
harness's flaw); it also tripped repeated 11.9 GiB device-memory allocation failures because it
rebuilt donated buffers outside a scan. Killed and replaced by the committed benchmark
harnesses, which measure the real environment class from its real spawn:
`benchmarks/env/bench_antmaze_env.py` and `benchmarks/end_to_end/bench_antmaze_train.py`.
Their numbers are the ledger's standing tables.

## What round 1 accepted

1. `implicitfast`, Newton `iterations=4`, `ls_iterations=8`, float32 behind a trace-time
   x64-off boundary (`spec.md` deviations 3 and 6; held to the reference by probe 5's three
   gates).
2. Horizontal wall runs merged into single boxes — an identical union of blocks, fewer geom
   pairs (`spec.md` deviation 4).
3. No contact caps (broken in this MJX version, see probe 2).
4. Deterministic resets (reference's joint noise is already zero; cells pinned), so auto-reset
   is a `where` against the spawn and the environment holds no RNG.
