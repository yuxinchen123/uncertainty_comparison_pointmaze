# Implementation specification — five ablation arms for PPO + RND on Atari

Files to edit:

- `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py` (trainer)
- `/p/rlprojects/RND/08_cleanrl_ppo_rnd/tests/test_arms.py` (new test file)

Reference sources read: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py` (upstream, commit `fe8d8a0`, gitignored clone recorded in `UPSTREAM.md`) and the trainer above. Environment: `/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python` (Python 3.10, torch 2.6.0+cu124), which is the canonical env for `08_cleanrl_ppo_rnd/` per `/p/rlprojects/RND/.venvs/ENVS.md`.

---

## 0. Two facts that must be settled before any code is written

**0.1 "Original CleanRL RND" cannot mean the upstream file, or arm 1 differs from arms 2–5 by more than the ablated knob.** The trainer is not upstream: it fixes the envpool auto-reset bug (`fix_envpool_autoreset`, default on), drops wandb and tensorboard, checkpoints, and carries ten throughput options. If arm 1 ran upstream's file and arms 2–5 ran this trainer, every arm-versus-arm difference would also contain the auto-reset fix and the option set.

**Decision: arm 1 is this trainer with every default unchanged.** Its values for the three ablated knobs are CleanRL's, quoted in §1. Every arm runs the same trainer, the same `fix_envpool_autoreset=True`, the same `opt_*` defaults, and differs only in the three knobs of §6. This is an arm invariant and is pinned by a test (§7, test 18).

**0.2 There is no gradient-statistics logging in this trainer today.** The premise that "the number comes from `clip_grad_norm_`'s return value" does not hold: `clip_grad_norm_` is called at lines 811 and 817 and its return value is discarded, no gradient norm reaches `eval_entry` (lines 837–863), and the project's own logging audit (`shuze_experiment/_reports/audit_logging-inventory.md:186`) lists "no gradient norm" among the absent fields. So arm 2 does not have to preserve an existing number — the logging has to be **added**, for all five arms, in a form that works with and without clipping. §2 gives that code.

---

## 1. Arm 1 — the reference configuration

### 1.1 The three knobs the other arms vary

| knob | upstream value | upstream line | same line in the trainer |
|---|---|---|---|
| `max_grad_norm` | `0.5` | `ppo_rnd_envpool.py:71` | `ppo_rnd_envpool_shuze.py:89` |
| `update_proportion` | `0.25` | `ppo_rnd_envpool.py:77` | `ppo_rnd_envpool_shuze.py:95` |
| predictor depth | three `Linear` layers after `Flatten`, i.e. **two** `[ReLU, Linear(512,512)]` blocks after `Linear(3136,512)` | `ppo_rnd_envpool.py:202–206` | `ppo_rnd_envpool_shuze.py:313–317` |

Upstream, quoted:

```python
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
```
```python
    # RND arguments
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
```

The clip site (`ppo_rnd_envpool.py:514–521`) — note it clips the **policy and the predictor jointly**, through one parameter list and one optimizer:

```python
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
```
```python
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )
```

The mask site (`ppo_rnd_envpool.py:463–472`):

```python
                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
```

### 1.2 The predictor and the target, layer by layer, with initialisations

Initialisation is one function applied to **every** `Conv2d` and `Linear` in both towers, with its default arguments — orthogonal weights at gain `sqrt(2)`, zero bias (`ppo_rnd_envpool.py:132–135`, identical at `ppo_rnd_envpool_shuze.py:225–229`):

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```

The two towers (`ppo_rnd_envpool.py:194–223`; the trainer's copy at `:305–332` is character-for-character the same modules):

```python
        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False
```

Facts measured on the constructed modules (torch 2.6.0, `RNDModel(4, 18)`):

| quantity | value |
|---|---|
| predictor index order | `Conv2d(1,32,k8,s4)`, `LeakyReLU(0.01)`, `Conv2d(32,64,k4,s2)`, `LeakyReLU(0.01)`, `Conv2d(64,64,k3,s1)`, `LeakyReLU(0.01)`, `Flatten`, `Linear(3136,512)`, `ReLU`, `Linear(512,512)`, `ReLU`, `Linear(512,512)` |
| target index order | same conv tower, `Flatten`, `Linear(3136,512)` |
| flattened width | `7*7*64 = 3136` |
| output width, both towers | 512 |
| predictor parameter count | 2,203,296 |
| target parameter count | 1,677,984 (all `requires_grad=False`) |
| LeakyReLU negative slope | 0.01 (torch's default; the original RND used 0.2) |

### 1.3 Everything else arm 1 fixes (unchanged in all five arms)

`env_id=MontezumaRevenge-v5`, `total_timesteps=2000000000`, `learning_rate=1e-4` with linear anneal, `num_envs=128`, `num_steps=128`, `num_minibatches=4`, `update_epochs=4`, `gamma=0.999`, `int_gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.1`, `clip_vloss=True`, `norm_adv=True`, `ent_coef=0.001`, `vf_coef=0.5`, `int_coef=1.0`, `ext_coef=2.0`, `target_kl=None`, `num_iterations_obs_norm_init=50`, Adam `eps=1e-5`, envpool `episodic_life=True, reward_clip=True, repeat_action_probability=0.25`. Trainer-only: `fix_envpool_autoreset=True`, `opt_*` at their defaults, `minibatch_drop_allowance=1024`, giving `batch_size=16384`, a constant post-drop batch of 15360, `minibatch_size=3840`, and 16 optimizer steps per policy update.

---

## 2. Arm 2 — no gradient clipping

### 2.1 What "no clipping" must mean in this code, and which representation is cleaner

`max_grad_norm` is read at exactly three places in the trainer: the `Args` field (line 89), the AMP branch (lines 810–811) and the plain branch (lines 816–817). Nothing else reads it — not `checkpointing.py`, not `run_record.py`. It reaches the JSON record only through the config splat at line 486 (`**{k: v for k, v in vars(args).items() ...}`), so whatever value it holds is recorded automatically.

Both `0.0` and `None` skip the clip under `if args.max_grad_norm:`. **Use `0.0`.** Reasons:

1. The field stays a plain `float`, so `tyro` needs no `Optional` annotation and `--max_grad_norm 0` is valid without touching the type.
2. The JSON record keeps one schema across 150 runs — a number in every record, never `null`. An analysis can compute over the column without a null branch.
3. `0.0` is the convention the original RND code itself used for "off" (`max_grad_norm=0.0` in `run_atari.py`, recorded in `shuze_experiment/_reports/atari-rnd-domain.md:369`).
4. `target_kl: float = None` already shows the mixed-type style in this dataclass; do not add a second instance of it.

The truthiness guard must be replaced by an explicit comparison, so that a later edit to `is not None` cannot turn `0.0` into "clip to zero norm", which would erase every gradient silently:

```python
    # `if args.max_grad_norm:` treats 0.0 and None alike; spell the intent out instead, because a
    # later change to `is not None` would read 0.0 as "clip to norm zero" and wipe every gradient.
    clip_gradients = args.max_grad_norm > 0
```

Place this once, before the update loop (next to `combined_parameters`, or at the top of each iteration — it is a constant for the run).

### 2.2 The pre-clip gradient norm without clipping

`torch.nn.utils.clip_grad_norm_` is, in torch 2.6.0, literally `get_total_norm` followed by `clip_grads_with_norm_`, and it returns the value the first half computed:

```python
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
```

So arm 2 calls the first half alone. Verified on this env: `nn.utils.get_total_norm(grads, 2.0)` returned `0.2936650216579437` against `clip_grad_norm_`'s `0.2936650216579437` on a deepcopy of the same module — bitwise equal — and left every gradient untouched. Do **not** emulate it with `clip_grad_norm_(params, float("inf"))`: that still runs a `_foreach_mul_` by 1.0 over 3.9 million gradient elements on every one of the 16 minibatch steps per update, for nothing. `get_total_norm` requires torch ≥ 2.4; the canonical env has 2.6.0.

### 2.3 Exact code

Helper, next to `layer_init` in the trainer:

```python
def gradient_norm_before_clipping(parameters, max_grad_norm):
    """Return the global gradient norm BEFORE clipping, clipping in place only when it is enabled.

    torch's clip_grad_norm_ is get_total_norm followed by clip_grads_with_norm_, and it returns what
    the first half computed. So the arm that does not clip gets the identical number from the first
    half alone — verified bitwise equal on torch 2.6.0 — without touching a single gradient.
    """
    # A zero or negative max_grad_norm means "do not clip" (arm 2 and arm 5); every other arm clips
    # the policy and the predictor jointly, which is what CleanRL does.
    if max_grad_norm > 0:
        return nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    return nn.utils.get_total_norm(
        [p.grad for p in parameters if p.grad is not None], norm_type=2.0
    )
```

Replace the trainer's lines 804–818 with:

```python
                optimizer.zero_grad()
                if args.opt_amp_fp16:
                    scaler.scale(loss).backward()
                    # The gradients must be unscaled before the norm is either measured or clipped:
                    # on the scaled gradients the norm is the loss scale times the true norm, and the
                    # clip threshold would mean nothing.
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # One number per minibatch step, kept on the device. Calling .item() here would put a
                # host-device synchronisation back into the loop that opt_no_sync_update removed.
                gradient_norm = gradient_norm_before_clipping(combined_parameters, args.max_grad_norm)
                gradient_norm_sum += gradient_norm
                gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm)
                gradient_norm_over_reference += (gradient_norm > REFERENCE_CLIP_NORM).float()
                gradient_steps += 1

                if args.opt_amp_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
```

Module-level constant and per-iteration accumulators (the accumulators are reset at the top of each policy update, just before `for epoch in range(args.update_epochs):`):

```python
REFERENCE_CLIP_NORM = 0.5
"""arm 1's clip threshold, kept as a fixed yardstick so an arm that does not clip still reports how
often it would have been clipped"""
```
```python
        # Gradient statistics for this update. Accumulated on the device and read once, at logging
        # time, so the update loop keeps the sync-free property opt_no_sync_update gives it.
        gradient_norm_sum = torch.zeros((), device=device)
        gradient_norm_max = torch.zeros((), device=device)
        gradient_norm_over_reference = torch.zeros((), device=device)
        gradient_steps = 0
```

New rows in `eval_entry` (trainer lines 837–863), present in every arm:

```python
                    "losses/gradient_norm_before_clipping_mean": float(gradient_norm_sum.item() / gradient_steps),
                    "losses/gradient_norm_before_clipping_max": float(gradient_norm_max.item()),
                    "losses/fraction_of_steps_above_reference_clip_norm": float(
                        gradient_norm_over_reference.item() / gradient_steps),
                    "losses/gradient_clipping_applied": args.max_grad_norm > 0,
```

The third row is the number that makes arm 2 readable: in arm 1 it is the fraction of steps that were actually clipped; in arm 2 it is the fraction that would have been.

---

## 3. Arm 3 — update proportion 1.0

### 3.1 What `update_proportion` controls

It is the keep probability of a per-sample Bernoulli mask on the RND predictor's distillation loss, and nothing else. In the trainer (lines 753–768) the per-sample loss is the mean over the 512 feature dimensions of the squared predictor-target error, one value per minibatch row; the mask keeps each row with probability `update_proportion` and the loss is the mean over the kept rows. It does not touch the intrinsic reward, the advantage, the policy loss, or which rows the policy trains on — only how much of the batch the predictor is fitted to. Upstream's value 0.25 matches Burda et al. Table 5 for the 128-environment setting.

### 3.2 What 1.0 does to the mask

`torch.rand` samples from the half-open interval `[0, 1)`, so `rand < 1.0` is true for every element — verified over 10^6 draws, no element reached 1.0. At `update_proportion=1.0` the mask is therefore exactly all ones, deterministically, `mask.sum()` is exactly the row count, and the `torch.clamp(..., min=1.0)` denominator guard is inert. The loss becomes the plain mean of the per-sample loss over all 3,840 rows of the minibatch.

### 3.3 Can the mask be skipped at 1.0, and does that move the random stream

It can be skipped arithmetically — the result is the same number — but it **changes the random number stream**, and the stream must be preserved.

`torch.rand(n, device=device)` and the rollout's `Categorical(...).sample()` draw from the **same** generator: the device generator (CUDA when `device.type == "cuda"`, the CPU generator otherwise). Skipping the mask draw therefore shifts every subsequent action sample. Demonstrated on this env: three successive `Categorical.sample()` draws from the same seed gave `[[5,2,1,2],[3,4,0,1],[1,0,0,4]]` with a preceding 3,840-element `torch.rand`, and `[[3,0,0,4],[4,3,2,3],[4,5,1,0]]` without it.

**The stream must be preserved.** Arm 3 exists to isolate one knob. If it also skipped the draw, then from the very first minibatch onward arm 3 and arm 1 would take different actions in the environment for reasons that have nothing to do with the update proportion, and the two arms would lose their common random numbers before the policies had diverged at all. Keeping the draw costs one uniform sample of 3,840 elements per minibatch step, 16 per policy update — negligible against the two convolutional towers on the same minibatch.

Note the opposite precedent in the project's own RND (`/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/methods/rnd.py:635–671`), which **does** skip the draw at 1.0. It skips for the same reason we keep: there, 1.0 is the historical default and skipping is what keeps the old runs' stream byte-identical. Here 1.0 is the new setting and 0.25 is the reference, so keeping the draw is what preserves the reference stream. The rule is "do not move the stream relative to the baseline", and it points in opposite directions in the two codebases.

### 3.4 Exact code

Extract the mask into a helper so it is testable and so the "always draw" property lives in one place. Both existing branches are preserved exactly (the `opt_no_sync_update=False` path is the literal upstream expression):

```python
def masked_forward_loss(per_sample_loss, update_proportion, sync_free=True):
    """Average the predictor's per-sample loss over a Bernoulli(update_proportion) keep-mask.

    The mask is drawn at EVERY proportion, 1.0 included. torch.rand and the rollout's
    Categorical.sample() draw from the same device generator, so dropping the draw at 1.0 would shift
    every later action sample and the arms would stop sharing their random numbers for a reason that
    has nothing to do with the update proportion. At 1.0 the draw is free arithmetically: torch.rand
    samples [0, 1), so `rand < 1.0` is every element and the mask is exactly all ones.
    """
    # before: per_sample_loss of 3840 rows, proportion 0.25 -> about 960 rows kept, mean over those
    # after:  per_sample_loss of 3840 rows, proportion 1.00 -> all 3840 kept, mean over all of them
    device = per_sample_loss.device
    if sync_free:
        # .type(torch.FloatTensor) is the CPU type, so the literal form round-trips the mask to the
        # host and back on every minibatch; this form keeps it on the device.
        mask = (torch.rand(len(per_sample_loss), device=device) < update_proportion).float()
        return (per_sample_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    mask = torch.rand(len(per_sample_loss), device=device)
    mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
    return (per_sample_loss * mask).sum() / torch.max(
        mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
    )
```

Trainer lines 758–768 become:

```python
                    forward_loss = masked_forward_loss(
                        forward_loss, args.update_proportion, sync_free=args.opt_no_sync_update
                    )
```

The rejected alternative, for the record — do not implement it:

```python
                    # REJECTED: skips the torch.rand draw at 1.0, which shifts every subsequent
                    # action sample on the shared device generator and breaks the common random
                    # numbers between arm 3 and arm 1.
                    if args.update_proportion >= 1.0:
                        forward_loss = forward_loss.mean()
                    else:
                        forward_loss = masked_forward_loss(forward_loss, args.update_proportion)
```

---

## 4. Arm 4 — one 512 layer removed from the predictor

### 4.1 The readings, enumerated

1. **Drop the middle `Linear(512,512)` together with the `ReLU` before it.** Predictor head becomes `Linear(3136,512), ReLU, Linear(512,512)`. Output width 512.
2. **Drop the last `Linear(512,512)` together with its preceding `ReLU`.** Predictor head becomes `Linear(3136,512), ReLU, Linear(512,512)`. Output width 512.
3. **Drop the `Linear(3136,512)`.** Not implementable as stated: it is the only layer that maps the 3,136 flattened convolution features down to 512. Removing it leaves the head taking 3,136 inputs, so a replacement `Linear(3136,512)` has to be reintroduced and the result is reading 1 again.
4. **Narrow a layer from 512 to something else** ("remove one 512" read as a width). If applied to the output layer the predictor emits a width other than 512 and the loss against the 512-wide target no longer type-checks; if applied to a hidden layer it is a width ablation, not a layer removal, and the owner said "remove one layer". Rejected.
5. **Remove a 512 layer from the target.** The target has exactly one `Linear`, and removing it leaves a 3,136-wide output against a 512-wide predictor. The loss breaks. Rejected.
6. **Remove one `Linear(512,512)` but keep both `ReLU`s.** Two consecutive `ReLU`s, which is a no-op pair. Rejected as certainly unintended.

**Readings 1 and 2 are the same module, and under a fixed seed they are the same weights.** `layer_init` draws in construction order, so the surviving `Linear(512,512)` takes the first of the two draws either way. Verified: with `torch.manual_seed(7)`, the layer at index 9 of the shallow predictor is bitwise equal to index 9 of the deep one. So the apparent ambiguity between "middle" and "last" is not an ambiguity at all in the constructed network.

### 4.2 The recommended reading

**Remove one `[ReLU, Linear(512,512)]` block from the predictor head, leaving `Conv, Conv, Conv, Flatten, Linear(3136,512), ReLU, Linear(512,512)`.** Reasons:

1. It is the only reading that removes a layer, keeps a working network, and leaves the loss untouched.
2. The output width stays 512, so `F.mse_loss(predict, target.detach(), reduction="none").mean(-1)` keeps its exact shape and meaning, the intrinsic reward `(target - predict).pow(2).sum(1) / 2` keeps its 512-term sum, and no other line of the trainer changes. Verified: predictor and target both emit `(batch, 512)` at one block and at two.
3. It preserves the structural property that distinguishes an RND predictor from its target: the predictor stays strictly deeper. At one block the predictor has two `Linear` layers after the flatten against the target's one — exactly one `[activation, Linear]` block deeper.
4. Parameter counts: predictor 2,203,296 → 1,940,640, a reduction of 262,656 (11.9%); target unchanged at 1,677,984.

**The project's own RND supports this reading.** `/p/rlprojects/RND/07_reconstruction/src/rnd_exploration/methods/rnd.py` carries `predictor_extra_layers`, which appends exactly that many `[activation, Linear(out, out)]` blocks to the predictor and none to the target, and the real runs use `rnd_predictor_extra_layers=1` — the run-5 and run-8.1.x slug reads `predextra1`, and its `experiment_background.md` states "predictor deeper than target by one 128-wide block". So the shape this project has been running for months, in the PointMaze widths, is precisely the arm-4 shape in the Atari widths: predictor = target + one block. Arm 4 is not an exotic architecture, it is the depth asymmetry this project already treats as standard, and arm 1 is the deeper-still variant that CleanRL and Burda et al. use.

### 4.3 Exact code

Replace the `RNDModel.__init__` predictor block (trainer lines 305–318). The loop form is bitwise identical to the current explicit form at two blocks — verified with `torch.manual_seed(7)`: every predictor and target tensor equal, module reprs equal — so arm 1 is unchanged by the refactor:

```python
class RNDModel(nn.Module):
    """The Random Network Distillation pair: a trained predictor and a frozen random target."""

    def __init__(self, input_size, output_size, predictor_hidden_blocks=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.predictor_hidden_blocks = int(predictor_hidden_blocks)
        if self.predictor_hidden_blocks < 1:
            raise ValueError(
                "predictor_hidden_blocks must be at least 1, so the predictor stays deeper than the "
                f"target; got {predictor_hidden_blocks}"
            )
        feature_output = 7 * 7 * 64

        # The predictor is the target's tower plus `predictor_hidden_blocks` [ReLU, Linear(512,512)]
        # blocks. CleanRL and Burda et al. use 2; arm 4 uses 1, which leaves the predictor exactly one
        # block deeper than the target — the same asymmetry this project's own RND runs with
        # rnd_predictor_extra_layers=1. The output width is 512 either way, so the distillation loss
        # and the intrinsic reward are unchanged.
        # before (2 blocks): Linear(3136,512), ReLU, Linear(512,512), ReLU, Linear(512,512)
        # after  (1 block):  Linear(3136,512), ReLU, Linear(512,512)
        predictor_layers = [
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        ]
        for _ in range(self.predictor_hidden_blocks):
            predictor_layers += [nn.ReLU(), layer_init(nn.Linear(512, 512))]
        self.predictor = nn.Sequential(*predictor_layers)

        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        for param in self.target.parameters():
            param.requires_grad = False
```

Construction site (trainer line 433):

```python
    rnd_model = RNDModel(4, envs.single_action_space.n,
                         predictor_hidden_blocks=args.rnd_predictor_hidden_blocks).to(device)
```

### 4.4 One consequence to record, not to fix

The predictor is constructed before the target, and `layer_init` draws from the CPU generator, so removing a predictor block shifts the target's draws: **arm 4's frozen target is a different random function from arm 1's at the same seed.** Verified — with `torch.manual_seed(1)` the target's first convolution and last linear both differ between the two depths, while the predictor's shared prefix is bitwise identical. The agent is unaffected (it is built first), and on a CUDA device the rollout's action sampling is unaffected (initialisation runs on the CPU generator, sampling on the device generator); on a CPU device both would shift.

Do not reorder the construction to "fix" this. Reordering would change arm 1 away from the upstream reference, which is the one thing arm 1 must not do. The target is a random draw by design and the sweep averages over 30 seeds, so the shift is a nuisance variable already being averaged; but it means arm 4 versus arm 1 at a fixed seed is not a paired comparison of the same target function, and the analysis must not claim it is. Test 12 in §7 pins this so nobody silently changes it.

---

## 5. Arm 5 — do the three changes compose

**They compose at the code level, without conflict.** The three edits touch disjoint sites: the clip call at the optimizer step, the mask inside the loss, and the predictor's constructor. No shape, assertion, or invariant couples them. `combined_parameters` is built from `rnd_model.predictor.parameters()` and picks up the shallower tower automatically; the optimizer sees 262,656 fewer parameters and needs no change; the mask length is the minibatch row count and does not depend on the predictor; the loss shape is unchanged because the output width is still 512; the AMP, `torch.compile` and `channels_last` options are all off in every arm. Arm 5 is arms 2, 3 and 4 applied together, with no fourth change.

**They do not compose in effect, and the interaction has a name: the single global clip couples the predictor's gradient to the policy's.** One `loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss + forward_loss`, one `backward()`, one parameter list, one `clip_grad_norm_` over policy and predictor jointly. Whenever the combined norm exceeds 0.5, the whole vector is rescaled — the policy's part included. Therefore:

1. **Arm 3 changes the policy's step size, not only the predictor's.** Raising the keep proportion to 1.0 changes the predictor gradient's magnitude and variance, which changes the combined norm, which changes how often and how hard the shared clip rescales the policy gradient. Arm 3 is not a pure predictor-side change while clipping is on.
2. **Arm 4 does the same by a different route.** A shallower predictor contributes a different gradient norm, so it too moves the shared clip factor and thus the effective policy step.
3. **Arm 2 is exactly the removal of that coupling channel.** With no clip, the predictor's gradient magnitude no longer affects the policy update at all.
4. **So arm 5's effect is not the sum of the arm-2, arm-3 and arm-4 effects.** Arms 3 and 4 measure their knob *plus* its clip-mediated effect on the policy; arm 5 measures the same knobs with that channel removed. Any additivity check ("does arm 5 equal 2 + 3 + 4?") will fail for this reason and not because of noise. The writeup must state it.

A second, smaller point: arm 5 inherits arm 4's shifted target initialisation (§4.4), and it is the only arm with both no clipping and a full-batch predictor loss, so it is the arm where gradient norms are most likely to be large. The gradient logging of §2 is what makes that visible rather than a guess.

---

## 6. Flag design

**Use one `--arm` flag taking a name, with the three knobs derived from it.** Not three independent flags.

For a 150-run sweep (5 arms x 30 seeds) where the arm has to be recoverable from the record, three independent flags fail in a specific way: the five arms are five points in a three-dimensional grid of 8 possible combinations, so a wrong or missing token does not produce an error, it produces a valid configuration that is not one of the arms — clipping off with proportion 0.25 and a shallow predictor, say — which then sits in the sweep and is analysed as whichever arm the queue label claimed. One flag per run means one token to get wrong instead of three, `tyro` rejects a misspelled arm name at submit time because the field is a `Literal`, and the record carries a single string that names the arm rather than a triple that has to be reverse-matched. The three knobs are still written to the record by the existing config splat, so the derived values remain auditable and the reverse mapping is available as a cross-check.

An escape hatch is still needed for one-off exploratory runs, so `"custom"` is a legal arm value that leaves the three knobs exactly as passed. A named arm refuses hand-set knobs, so the two mechanisms can never disagree silently.

### 6.1 Dataclass fields

Add to `Args` (`from typing import Literal` at the top of the file):

```python
    # The ablation arm. One flag decides all three ablated knobs, so a sweep command line carries one
    # token per run and the record carries one name. "custom" leaves the three knobs as passed, for a
    # one-off run that is not part of the sweep.
    arm: Literal[
        "arm1_cleanrl_reference",
        "arm2_no_gradient_clipping",
        "arm3_update_proportion_one",
        "arm4_predictor_one_hidden_block",
        "arm5_no_clipping_proportion_one_one_hidden_block",
        "custom",
    ] = "arm1_cleanrl_reference"
    """the ablation arm; it sets max_grad_norm, update_proportion and rnd_predictor_hidden_blocks"""

    rnd_predictor_hidden_blocks: int = 2
    """[ReLU, Linear(512,512)] blocks in the predictor after its Linear(3136,512); CleanRL uses 2"""
```

`max_grad_norm: float = 0.5` and `update_proportion: float = 0.25` stay exactly as they are — they are already arm 1's values.

### 6.2 The arm table and its application

```python
ARM_SETTINGS = {
    "arm1_cleanrl_reference":
        {"max_grad_norm": 0.5, "update_proportion": 0.25, "rnd_predictor_hidden_blocks": 2},
    "arm2_no_gradient_clipping":
        {"max_grad_norm": 0.0, "update_proportion": 0.25, "rnd_predictor_hidden_blocks": 2},
    "arm3_update_proportion_one":
        {"max_grad_norm": 0.5, "update_proportion": 1.0, "rnd_predictor_hidden_blocks": 2},
    "arm4_predictor_one_hidden_block":
        {"max_grad_norm": 0.5, "update_proportion": 0.25, "rnd_predictor_hidden_blocks": 1},
    "arm5_no_clipping_proportion_one_one_hidden_block":
        {"max_grad_norm": 0.0, "update_proportion": 1.0, "rnd_predictor_hidden_blocks": 1},
}
"""the five arms, as the three knobs each one sets; arm 1 holds CleanRL's own values"""

ARM_BASELINE = "arm1_cleanrl_reference"


def apply_arm(args):
    """Set the three ablated knobs from the named arm, refusing a knob that was also set by hand.

    A named arm is the single source of truth for its three knobs. If the command line also carried
    one of them, the two could disagree and the record would name an arm the run did not execute, so
    that is an error rather than a precedence rule. "custom" is the escape hatch for a one-off run.
    """
    # before: args.arm="arm5_...", args.max_grad_norm=0.5, args.update_proportion=0.25, blocks=2
    # after:  args.max_grad_norm=0.0, args.update_proportion=1.0, blocks=1
    if args.arm == "custom":
        return args
    settings = ARM_SETTINGS[args.arm]
    for knob, baseline_value in ARM_SETTINGS[ARM_BASELINE].items():
        given = getattr(args, knob)
        if given != baseline_value:
            raise ValueError(
                f"--arm {args.arm} sets {knob} itself, but --{knob} was passed as {given}. Pass the "
                f"arm alone, or pass --arm custom to set the knobs by hand."
            )
        setattr(args, knob, settings[knob])
    return args
```

In `main()`, immediately after `args = tyro.cli(Args)`:

```python
    args = apply_arm(args)
```

### 6.3 What lands in the record

The config splat at trainer line 486 excludes only `run_id`, `run_total`, `seed`, `env_id`, so `arm`, `max_grad_norm`, `update_proportion` and `rnd_predictor_hidden_blocks` all reach the JSON with no extra code. Add two entries that make the architecture checkable from the record alone, next to `"implementation"` in the `config=` dict:

```python
            "rnd_predictor_layers": [type(m).__name__ for m in rnd_model.predictor],
            "rnd_predictor_parameter_count": sum(p.numel() for p in rnd_model.predictor.parameters()),
```

Also add the arm to the checkpoint payload in `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/checkpointing.py` and check it on load. `rnd_model.load_state_dict` is strict, so an arm-4 checkpoint loaded into an arm-1 model already raises — but on a key-shape error, not a readable message. Store `"arm": args.arm` and raise `ValueError(f"checkpoint {path} was written by arm {saved} and this run is arm {current}")` before touching the state dicts.

### 6.4 Sweep ids

Per `/p/rlprojects/RND/.claude/rules/run-id-and-logging.md`, seed is outermost and the inner axis follows a fixed order. Here: `run_total = 150 = 30 seeds x 5 arms`, arms in the `ARM_SETTINGS` insertion order, so seed *s* owns ids `5s` through `5s+4` and `run_id % 5` names the arm. Pin that in the queue builder and in test 19.

---

## 7. Unit tests

New file `/p/rlprojects/RND/08_cleanrl_ppo_rnd/tests/test_arms.py`, following the existing test convention in that folder (`sys.path.insert` of `../src`, run with `/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python -m pytest`). The trainer module imports cleanly without a GPU or an env — verified — and `RNDModel` is constructible standalone, so every test below runs on the login node.

**Settings, so an arm cannot be redefined silently**

1. `test_arm_table_is_exactly_the_five_arms` — compare `ARM_SETTINGS` against a literal dict written out in the test: five names, three knobs each, the exact values of §6.2. Any edit to any arm's knob fails here first.
2. `test_arm5_is_the_union_of_arms_2_3_4` — assert `ARM_SETTINGS["arm5_..."]["max_grad_norm"] == ARM_SETTINGS["arm2_..."]["max_grad_norm"]`, likewise `update_proportion` from arm 3 and `rnd_predictor_hidden_blocks` from arm 4. Changing arm 2, 3 or 4 without mirroring it into arm 5 fails.
3. `test_arm_settings_are_distinct` — the five triples are pairwise different, so the record's triple identifies the arm and the reverse mapping in test 19 is a function.
4. `test_named_arm_overwrites_the_knobs` — for each of the five names, `apply_arm` on a default `Args` yields the table's values.
5. `test_named_arm_refuses_a_hand_set_knob` — `pytest.raises(ValueError)` for `arm="arm3_..."` with `max_grad_norm=0.2`, and for each of the other two knobs.
6. `test_custom_arm_keeps_hand_set_knobs` — `arm="custom"` with `max_grad_norm=0.2, update_proportion=0.7, rnd_predictor_hidden_blocks=1` passes all three through unchanged.
7. `test_arm1_defaults_match_upstream` — `ast`-parse `cleanrl/cleanrl/ppo_rnd_envpool.py`, read the `Args` dataclass defaults, and compare every field name the two dataclasses share (`learning_rate`, `num_envs`, `num_steps`, `gamma`, `gae_lambda`, `num_minibatches`, `update_epochs`, `norm_adv`, `clip_coef`, `clip_vloss`, `ent_coef`, `vf_coef`, `max_grad_norm`, `target_kl`, `update_proportion`, `int_coef`, `ext_coef`, `int_gamma`, `num_iterations_obs_norm_init`, `anneal_lr`, `total_timesteps`, `env_id`). `pytest.skip` with an explicit message if the clone is absent — it is gitignored (`UPSTREAM.md`) and will not exist on a fresh checkout; that is the one legitimate skip in this file.
8. `test_every_arm_holds_the_numeric_options_off` — for each arm: `opt_amp_fp16`, `opt_channels_last`, `opt_matmul_tf32`, `opt_torch_compile` are all `False`, `fix_envpool_autoreset` is `True`, and `minibatch_drop_allowance == 1024`. This is the arm invariant of §0.1: arms differ in three knobs and nothing else.

**Constructed modules, so an architecture cannot drift**

9. `test_predictor_and_target_module_sequence` — for `predictor_hidden_blocks` 2 and 1, assert the exact list of `(type name, in_features/in_channels, out_features/out_channels, kernel_size, stride)` against the literal sequence of §1.2 and §4.3, and assert the target's sequence is identical in both cases.
10. `test_predictor_is_exactly_one_block_deeper_than_target_at_one` — count `nn.Linear` instances: target 1, predictor `1 + predictor_hidden_blocks`; at blocks 1 the difference is exactly 1.
11. `test_output_width_is_512_and_the_loss_shape_is_unchanged` — forward a `(4, 1, 84, 84)` tensor through predictor and target at both depths, assert `(4, 512)` each, and assert `F.mse_loss(pred, tgt.detach(), reduction="none").mean(-1).shape == (4,)` and `(tgt - pred).pow(2).sum(1).shape == (4,)` — the intrinsic reward's shape.
12. `test_parameter_counts` — predictor 2,203,296 at two blocks, 1,940,640 at one, target 1,677,984 in both, and every target parameter has `requires_grad is False`.
13. `test_two_block_predictor_is_bitwise_the_upstream_construction` — under `torch.manual_seed(7)`, build the explicit twelve-module `nn.Sequential` of §1.2 and compare every tensor of `RNDModel(4, 18).predictor.state_dict()` and `.target.state_dict()` with `torch.equal`. This is what stops the loop refactor of §4.3 from changing arm 1.
14. `test_shallow_predictor_shifts_the_target_stream` — under one seed, assert `RNDModel(4, 18, predictor_hidden_blocks=2).target` and `...=1).target` are **not** equal, and that the predictors' shared prefix (conv layers and `Linear(3136,512)`) **is** equal. Docstring states this is documented behaviour of §4.4, not a bug, and that reordering the construction to hide it would change arm 1.
15. `test_predictor_hidden_blocks_below_one_is_rejected` — `pytest.raises(ValueError)` at 0 and at -1.

**Gradient clipping and the norm it reports**

16. `test_gradient_norm_without_clipping_equals_the_clip_return` — build a small module twice by deepcopy, run the same backward on both, call `gradient_norm_before_clipping(params, 0.5)` on one and `(params, 0.0)` on the other, assert the two returned norms are equal by `==` on `.item()` (bitwise, as measured), and assert the second module's gradients are bitwise unchanged from a pre-call copy.
17. `test_gradient_norm_with_clipping_is_the_preclip_number` — construct gradients whose norm exceeds the threshold, assert the returned value is the pre-clip norm (matching an independently computed `get_total_norm` on a copy) and that the post-call gradient norm equals the threshold.
18. `test_clip_enabled_per_arm` — `ARM_SETTINGS[name]["max_grad_norm"] > 0` is `True` for arms 1, 3, 4 and `False` for arms 2, 5.
19. `test_gradient_statistics_reach_the_record` — build a `RunRecord`, call `add_update` with an `eval_entry` built the way the trainer builds it, flush, reload the JSON, and assert `losses/gradient_norm_before_clipping_mean`, `..._max`, `losses/fraction_of_steps_above_reference_clip_norm` and `losses/gradient_clipping_applied` are present and are finite numbers / a bool, for both a clipping and a non-clipping configuration.

**The random stream**

20. `test_mask_draw_is_consumed_at_every_proportion` — seed, call `masked_forward_loss(loss, 0.25)`, capture `torch.get_rng_state()`; reseed, call `masked_forward_loss(loss, 1.0)`, capture again; assert the two states are equal. Then assert that a variant which skips the draw at 1.0 leaves a different state — the property arm 3 must not have.
21. `test_mask_is_all_ones_at_proportion_one` — over 10^5 rows, assert every mask element is 1.0 and that the returned loss equals the per-sample mean to `torch.allclose` with `rtol=0`, `atol=0` where the reduction order permits, otherwise `atol=1e-7` with the reason in the docstring.
22. `test_masked_loss_denominator_is_the_kept_count` — with a monkeypatched `torch.rand` returning a fixed vector, assert the loss equals the mean over exactly the kept rows, and that an all-false mask gives a finite result (the `clamp(min=1.0)` guard).

**Recoverability from the record**

23. `test_arm_is_recoverable_from_the_record` — build the trainer's `config=` dict from an `Args` with each arm applied, assert `config["arm"]` is the arm name, and assert the reverse lookup of `(max_grad_norm, update_proportion, rnd_predictor_hidden_blocks)` in `ARM_SETTINGS` returns that same name.
24. `test_run_id_maps_to_arm` — for `run_total=150`, assert `run_id % 5` indexes `list(ARM_SETTINGS)` and `run_id // 5` is the seed index, over all 150 ids.