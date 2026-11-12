"""Every knob of the proximal-policy-optimisation agent and of the loop that drives it.

Knobs that belong to a bonus family (the feature width of random network distillation, the decay
of a visit-count bonus) live with that family, not here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """Agent and loop knobs (algorithm specification section 15)."""
    # Resolve these choices from the environment's named configuration before compiling.
    # Empty choices are never passed into the trainer; the resolved run record names them.
    configuration_preset: str = "environment_default"
    intrinsic_bootstrap_observation: str = ""
    no_bonus_advantage_stream: str = ""
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
    betas: tuple = ()                # sweep: the weights on the intrinsic advantage to try;
                                     # empty means every copy uses `int_coef`
    parameter_settings: tuple = ()   # explicit cells, one row per cell holding one value per
                                     # name in `setting_names` (by default a learning rate and
                                     # an intrinsic weight); selects only the requested cells
                                     # instead of a cross product
    copies_per_group: int = 0        # sweep: copies each (learning rate, beta) cell gets; the
                                     # groups are the cross product of the two lists
    sweep_seed_mode: str = "paired"  # "paired": copy k of every group shares one seed stream,
                                     # so groups differ ONLY by the swept value;
                                     # "distinct": every copy is its own seed
    copy_seed_offset: int = 0        # shift added to every copy's seed index, so one logical run
                                     # of many copies can be CUT INTO CHUNKS that run on separate
                                     # cards: chunk j of k asks for offset j x (copies / k) and
                                     # its copies are then exactly copies j x (copies/k) onward of
                                     # the whole run, with no seed shared with another chunk.
                                     # 0 is the whole-run case and leaves every existing run's
                                     # draws bit-identical
    track_coverage: bool = False     # maintain a per-copy visited-cell map on the device, so
                                     # per-copy exploration curves can be recorded (round 4)

    # ---- the reward-decline study (2026-10-19). Every default reproduces the arithmetic above,
    # so a configuration that sets none of these compiles exactly the program it did before ----
    diagnostics: bool = False        # extra per-iteration metrics (policy log std, update KL,
                                     # gradient norms, value fit, advantage sizes) and a per-copy
                                     # episode tracker (episodes ended, goal reached, steps inside)
    total_iterations: int = 0        # the run's length in iterations; read only by the schedules
                                     # below and by a bonus's reward divisor held after a
                                     # fraction of the run, which need to know where in the run
                                     # an iteration is
    int_coef_schedule: str = "constant"  # "linear_decay": the intrinsic weight times
                                     # 1 - (k - 1) / total_iterations at iteration k, reaching
                                     # nearly zero at the end, the same shape as the rate anneal
    success_gate_rate: float = 0.0   # > 0: the intrinsic weight of each copy is multiplied by
                                     # 1 - s, where s is that copy's running goal-reaching rate
                                     # over ended episodes, moved by this step size per iteration
    normalize_ext_reward: bool = False   # divide the task reward by the running standard
                                     # deviation of its discounted sum, as the intrinsic reward is
    clip_per_network: bool = False   # clip the actor's, the critic's and the predictor's gradients
                                     # to max_grad_norm separately instead of jointly
    logstd_floor: float = float("-inf")  # after every update, raise the policy's log standard
                                     # deviation to at least this; -inf leaves it free
    critic_lr_scale: float = 1.0     # the critic's update step is this fraction of the step Adam
                                     # computed; 1 leaves it as the actor's
    normalize_agent_obs: bool = False  # feed the actor and critic the observation centred and
                                     # scaled by its running per-copy statistics, clipped to +-5,
                                     # instead of the raw position and velocity

    # ---- the PointMaze and AntMaze study (2026-11-11). Every default reproduces the arithmetic
    # above, so a configuration that sets none of these compiles exactly the program it did ----
    setting_names: tuple = ("learning_rate", "int_coef")  # what each row of `parameter_settings`
                                     # holds, in order. Allowed names: learning_rate, int_coef,
                                     # ent_coef, clip_coef, vf_coef, ext_reward_scale,
                                     # ext_reward_offset. A named value is one number per copy;
                                     # every other value stays the scalar of this configuration
    ext_reward_scale: float = 1.0    # the extrinsic reward the learner sees is
    ext_reward_offset: float = 0.0   # scale x (the environment's reward) + offset. Scale 1 and
                                     # offset 0 add no arithmetic. With a reward of 1 on arrival
                                     # and 0 elsewhere, scale 1/limit and offset -1/limit give a
                                     # cost of 1/limit per step and 0 on arrival
    episode_log: bool = False        # per copy, count the episodes that end in each rollout and
                                     # add up their returns, lengths and steps to the goal, and
                                     # keep each copy's first training step with an arrival
    gather_minibatches_per_epoch: bool = False  # epoch-minibatch updates gather one epoch's
                                     # minibatches at a time, so one shuffled copy of the batch
                                     # is held instead of one per epoch

    # ---- speed keys of the PointMaze and AntMaze study (2026-11-12). Every default keeps the
    # program above. What each one gained on which card is recorded in the ledger of
    # benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_... ----
    target_features_once_per_iteration: bool = False  # random network distillation: the fixed
                                     # target's features of the update batch are computed once
                                     # after the rollout and gathered with the minibatches,
                                     # instead of once in every gradient step (idea 20)
    permutation_draw: str = "argsort_uniform"  # how each epoch's row order is drawn for every
                                     # copy. "argsort_uniform": the positions that sort one
                                     # uniform number per row, one sort per copy and epoch.
                                     # "sort_packed_keys": one sort of 64-bit keys holding
                                     # (epoch and copy, 32 random bits, row). Both read the same
                                     # random bits, so the orders differ only between rows whose
                                     # uniform numbers are equal (idea 19)
    next_values_from_shifted_values: bool = False  # the value of a step's next observation is
                                     # read from the value of the following step's observation
                                     # instead of a second critic pass over the final
                                     # observations; only an environment's truncated step gets
                                     # its final observation's value (idea 14)
    skip_reset_draws_at_zero_noise: bool = False  # PointMaze: with position_noise 0 the reset
                                     # places the ball and the goal at their cells' centres
                                     # without drawing the four keyed uniform numbers per
                                     # environment and step that the zero noise multiplies
                                     # (idea 15)
    intrinsic_sums_by_matrix_product: bool = False  # the forward filter of the intrinsic reward
                                     # and the intrinsic advantage are each one product with a
                                     # fixed [T, T] matrix of discount powers instead of a
                                     # 128-step loop; the extrinsic advantage, which is cut at
                                     # episode ends, keeps its loop (idea 17)
    advantage_scan_unroll: int = 1   # how many steps one body of the reward-filter loop and of
                                     # the advantage loop holds; 1 is the loop as it always was
                                     # (an addition to idea 17, not in the plan's list)
    action_noise_generator: str = "threefry"  # where the rollout's action noise comes from.
                                     # "threefry": JAX's default counter-based generator, the
                                     # same numbers on every machine. "rbg": the compiler's own
                                     # bit generator, keyed from the iteration's key; another
                                     # random scheme, whose numbers may differ between card
                                     # models and library versions (idea 16)
    actor_critic_joint_trunk: bool = False  # in the update's loss, the two hidden layers of the
                                     # actor and of the critic run as one batched network of
                                     # twice the copies, instead of two networks one after the
                                     # other (idea 21)
