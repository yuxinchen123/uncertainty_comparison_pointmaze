"""Every knob of the proximal-policy-optimisation agent and of the loop that drives it.

Knobs that belong to a bonus family (the feature width of random network distillation, the decay
of a visit-count bonus) live with that family, not here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """Agent and loop knobs (algorithm specification section 15)."""
    n_copies: int = 8
    n_envs: int = 4
    num_steps: int = 128
    update_style: str = "epoch_minibatch"  # or "full_batch"
    # hold every parameter in ONE array of shape [copies, total parameters per copy] instead of
    # twenty-one named arrays, so the gradient clip is one reduction and Adam is a handful of
    # elementwise operations rather than twenty-one of each. The networks are unchanged: the
    # array is cut back into the named tensors before every forward pass.
    flat_params: bool = False
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    anneal_lr: bool = True
    gamma_ext: float = 0.999
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    int_coef: float = 1.0            # weight on the intrinsic advantage; a beta sweep replaces
                                     # this scalar with one value per copy
    ext_coef: float = 2.0
    prime_iterations: int = 10       # warm-up rollouts of random actions before training, so a
                                     # bonus that needs statistics of the observations has them
                                     # before it scores anything. A bonus with no warm-up hook
                                     # skips them entirely
    bootstrap_on_truncation: bool = True
    base_seed: int = 0
    hoist_rollout: bool = True       # compute the critic values, the log-probability and the
                                     # intrinsic reward AFTER the rollout scan, in one wide pass
                                     # each, instead of once per step inside it (round 2, J1)
    scan_unroll: int = 0             # unroll factor for the rollout scan; 0 picks by copy count
                                     # (round 5, J5): 32 at 8 copies, 16 above, each winning
                                     # 11 of 11 paired rounds against the previous value of 4
    update_unroll: int = 2           # unroll factor for the sixteen-step update scan (round 5,
                                     # J6): 2 emits two steps per loop body, which gives the
                                     # compiler one step's optimizer and the next step's
                                     # gradient to overlap. 4 is not better than 2. Unlike the
                                     # rollout scan this is NOT bit-neutral — it changes the
                                     # order float32 accumulates in — but the same update in
                                     # double precision agrees to 3.7e-16, so it computes the
                                     # same function; single precision differs by 3.6e-07
    batch_stats_f32: bool = False    # reduce batch mean/variance in float32 before promoting
                                     # (round 2, J5) — changes the last bits of the statistics,
                                     # so it also breaks bit-agreement with the torch twin
    learning_rates: tuple = ()       # sweep: the learning rates to try; empty means every copy
                                     # uses `learning_rate` (round 4)
    copies_per_rate: tuple = ()      # sweep: copies each rate gets (must sum to n_copies)
    sweep_seed_mode: str = "paired"  # "paired": copy k of every group shares one seed stream,
                                     # so groups differ ONLY by the swept value;
                                     # "distinct": every copy is its own seed
    track_coverage: bool = False     # maintain a per-copy visited-cell map on the device, so
                                     # per-copy exploration curves can be recorded (round 4)
