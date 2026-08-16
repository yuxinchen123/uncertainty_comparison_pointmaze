# ppo

`jax_ppo_rnd.py` is the trainer copied from `09_parallelization/ppo/jax_ppo/` (see `../../../../ORIGIN.md`).
It is still ONE module: it holds the PPO agent, the RND bonus, the rollout, the update and the driver
together, exactly as 09 left it, with only its import paths changed.

**The split is pending.** The next stage separates this file into the agent (configuration, losses,
update), the bonus family (its own configuration, networks, reference and fused implementation) and
the composition that picks a bonus before the program is compiled. Until that is done, treat this
file as frozen baseline code: the golden-parity gate in `../../../../tests/golden_09/` compares it
against the original in `09_parallelization/`, and every commit of the split has to keep that gate
green.

`SWEEP.md` is 09's description of the learning-rate sweep across copy groups, copied unchanged. The
sweep grows a second swept axis (the intrinsic-reward weight) in the same stage.
