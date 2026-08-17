# exploration_platform

The accepted implementations only — code that passed its gates; research rounds stay under
`research/`.

```
exploration_platform/
├── __init__.py       double precision on for the running statistics; the F32 and LOG2PI constants
├── statistics.py     per-copy running mean and variance, used by the agent AND by a bonus
├── networks.py       the keyed per-copy weight initialisation every network draws from
├── envs/pointmaze/   the batched PointMaze: shared physics constants and the JAX stepper
├── agents/ppo/       proximal policy optimisation: config, networks, losses, optimizer
├── bonuses/          one folder per intrinsic-reward family, behind one set of hooks
├── training/         state, composition, the training step, the sweep, the runner
└── evaluation/       measurements taken alongside training
```

The dependency direction is one way: `statistics` and `networks` depend on nothing, `agents` and
`bonuses` depend on those, and `training` depends on everything and is the only place the three
pieces meet. No module imports `training`.

Nothing on the hot path is an object with methods. The state is one immutable tree, every step is
a pure function of it, and which functions those are is decided in Python before `jax.jit` sees
anything — so the compiled program holds one environment, one agent and one bonus, and no branch
that chooses between them.
