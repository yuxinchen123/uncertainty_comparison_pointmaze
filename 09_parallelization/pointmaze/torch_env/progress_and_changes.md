# torch_env — experiment ledger (autoresearch style)

Goal metric (correctness phase): fixture checker results (`../common/code/check_against_fixtures.py`).
Goal metric (speed phase): env steps/second on the serval05 H100 at n_envs in {1e2..1e6}
(benchmark JSONs in `../../benchmarks/results/`). One change per row; keep or revert.

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: closed-form integrator + hard wall clamp (faces per axis, exposed corners radial), keyed-hash reset RNG, branch-free auto-reset | free-space fixtures EXACT (<=5e-15). One-step teacher-forced errors at contact steps: head-on 1.8 mm, impact 7.8 mm, slide 11.8 mm, corner 47.7 mm (hard clamp vs MuJoCo soft contact). No wall penetration anywhere (<=4e-16). | superseded by 1-2 |
| 1 | exact MuJoCo soft-contact force law (solref/solimp/margin, extracted from efc internals; single contact per axis + exposed corners) | one-step exact (<=9e-16) everywhere EXCEPT coplanar-box-run interior corners (wall_slide 3.35 mm): MuJoCo emits a second oblique contact with the neighboring box's corner. episode rollouts already matched to 1e-13. | superseded by 2 |
| 2 | contacts per neighboring wall BOX (nearest-point normal, unifies faces+corners) + exact 2-contact QP by case enumeration (measured max simultaneous contacts = 2) | fixture checker ALL PASS: one-step <=4.4e-16 (float64) / <=4.6e-7 (float32) on every case; closed-loop 400-step rollouts match to 1e-13; only knife-edge corner_graze separates in closed loop (physical instability). | KEEP — correctness baseline |
| 3 | fused contact pipeline: unrolled 8-candidate loop + two-smallest tournament (no topk/gather), geometry via 1-byte neighbor bitmask + arithmetic rectangles | compile @1M envs: 2242 -> 248 us/step = 4.03e9 env-steps/s (9.0x). @100k: 259 -> 171 us (1.5x). Small batch slightly worse (117 -> 153 us at 1k; python-call floor). Fixtures + unit tests still ALL PASS (float64 exact). Profile after: 8 kernels/step, contact pipeline still split into 3 kernels of 58/57/46 us + integration 37 us + RNG 15 us + cat 10+10 us. | KEEP |
| 4 | frozen cross-impl RNG: two fmix32 rounds, uniform = (h >> 8) * 2^-24 (float32-exact) | resets bit-identical torch vs jax (test_cross_impl_rng.py PASS, incl. generation-1 respawns); draws changed once (pre-production) | KEEP (RNG spec now FROZEN) |
| 5 | raise inductor realize thresholds to force one-kernel fusion (compile-bigfuse mode) | Triton compile ran >40 min at 0% GPU without producing a result — compile cost explosion; killed | DISCARD (mode kept in bench script for reference, do not use) |


## Notes

- 2026-08-15: one-step (teacher-forced) comparison added to the checker — endpoint comparison
  of multi-contact rollouts is chaotic and not a fidelity metric (divergence horizon reported
  instead). Free-space fixture cases are now generated with a verified no-contact property.
- Next: extract MuJoCo's exact 1-dof soft-contact law (solref/solimp/margin) from the solver
  internals (efc_* arrays) and implement it; contacts here are decoupled (axis-aligned faces,
  radial corners), so exactness should be achievable.
