# Continuous-action PPO + RND — algorithm spec for the torch and jax implementations

This document is the single definition of the algorithm that
`09_parallelization/ppo/torch_ppo/` and `09_parallelization/ppo/jax_ppo/` must both compute. Every
number, tensor shape, formula, and ordering below is binding on both. Where it departs from
CleanRL, the departure is listed in §16 with the CleanRL expression next to it, so either behaviour
can be reproduced on purpose.

Sources read to write it:

- `reference_repo/cleanrl/cleanrl/ppo_rnd_envpool.py` (the PPO+RND base, discrete Atari).
- `reference_repo/cleanrl/cleanrl/ppo_continuous_action.py` (the Gaussian policy head and the
  observation/reward normalization wrappers).
- `08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py` (this project's modified copy of the base).
- `07_reconstruction/src/rnd_exploration/methods/rnd.py` and `methods/base.py` (this project's RND:
  encoder sizes, observation normalization, bonus readout, predictor update).
- `pointmaze/common/physics_spec.md` (the environment this trainer drives).
- `gymnasium/wrappers/utils.py` `RunningMeanStd`, `stateful_observation.py`, `stateful_reward.py`
  (the exact running-statistics update the wrappers use).

## 1. What "the same algorithm" means here

Two implementations in two frameworks will not produce the same random numbers, so equality is
defined at three levels and each level has its own test (§17).

- **Structural equality.** Same layers, same shapes, same formulas, same order of operations, same
  hyperparameters. Checked by reading the code against this document.
- **Exact numerical equality under supplied randomness.** With identical initial weights, identical
  observations, identical action noise, and identical keep-masks loaded from a file, every
  intermediate quantity (log-probabilities, intrinsic rewards, advantages, each loss term, the
  post-step parameters) must agree to a relative tolerance of `1e-5` in float32.
- **Distributional equality under own randomness.** Running both with their own random draws over
  many copies, per-copy learning curves and the summary statistics of every logged quantity must be
  indistinguishable.

## 2. Symbols, sizes, and tensor layout

| symbol | meaning | default |
|---|---|---|
| `C` | independent training copies (independent seeds, own networks, own envs) | knob: 8–128 |
| `N` | parallel environments inside one copy | 4 |
| `T` | rollout length (`num_steps`) | 128 |
| `B` | rows per copy per rollout, `B = T * N` | 512 |
| `obs_dim` | observation width | 4 |
| `act_dim` | action width | 2 |
| `M` | rows per copy in a network forward call | `N`, `B`, or 128 |
| `F` | RND feature width (`output_dim`) | 128 |

Rollout buffers carry the time axis first, then the copy axis, then the environment axis:
`(T, C, N, ...)`. The update phase works on `(C, B, ...)` obtained by moving the copy axis to the
front and flattening time and environment together:

```
buf: (T, C, N, k)  --permute(1, 0, 2, 3)-->  (C, T, N, k)  --reshape-->  (C, B, k)
```

Both implementations use this exact flatten, so row index `b = t * N + n` refers to the same
transition in both. Network inputs are always `(C, M, obs_dim)`; the `C` axis is a batched-GEMM
axis, never a reduction axis.

Every one of `C`, `N`, `T`, `B`, the minibatch size, and the number of minibatches is a
compile-time constant. Nothing in the traced/compiled step may depend on a runtime value.

## 3. Networks

Four parameter groups per copy. All parameters are stored with a leading copy axis, and every
forward pass is a batched matrix multiply over that axis:

```
h = einsum('cmi,cio->cmo', x, W) + b[:, None, :]      # x: (C, M, i), W: (C, i, o), b: (C, o)
```

### 3.1 Actor

| layer | shape (per copy) | activation | init |
|---|---|---|---|
| `W1, b1` | `(4, 64), (64,)` | tanh | orthogonal, gain `sqrt(2)`; bias 0 |
| `W2, b2` | `(64, 64), (64,)` | tanh | orthogonal, gain `sqrt(2)`; bias 0 |
| `Wmu, bmu` | `(64, 2), (2,)` | none | orthogonal, gain `0.01`; bias 0 |
| `logstd` | `(2,)` | none | constant 0 |

Batched shapes: `W1 (C,4,64)`, `W2 (C,64,64)`, `Wmu (C,64,2)`, `logstd (C,2)`.

`logstd` is a free parameter, not a function of the observation — this is CleanRL's
`actor_logstd` (`ppo_continuous_action.py`, line 129), and it is trained by the policy gradient
like any other parameter. It is not clamped. If a run drives the standard deviation to a degenerate
value, both implementations add the identical clamp at the same time and it is recorded here.

### 3.2 Critic with two value heads

| layer | shape (per copy) | activation | init |
|---|---|---|---|
| `Wc1, bc1` | `(4, 64), (64,)` | tanh | orthogonal, gain `sqrt(2)`; bias 0 |
| `Wc2, bc2` | `(64, 64), (64,)` | tanh | orthogonal, gain `sqrt(2)`; bias 0 |
| `Wext, bext` | `(64, 1), (1,)` | none | orthogonal, gain `1.0`; bias 0 |
| `Wint, bint` | `(64, 1), (1,)` | none | orthogonal, gain `1.0`; bias 0 |

One trunk, two heads reading the same 64-d features: `Vext` for the extrinsic stream and `Vint` for
the intrinsic stream. The actor has its own separate trunk (CleanRL's continuous script keeps actor
and critic separate; the Atari RND script shares one convolutional trunk between actor and both
critics and adds a residual `extra_layer`). The `extra_layer` residual is **not** reproduced: it
belongs to the shared-trunk design and there is no shared trunk here.

### 3.3 RND target (frozen) and predictor (trained)

RND input is the 4-d observation, so both nets are the project's `ObservationEncoder`
(`07_reconstruction/.../rnd.py`) with `output_dim = 128`:

| net | layers | trainable |
|---|---|---|
| target | `Linear(4,256)`, ReLU, `Linear(256,128)` | no |
| predictor | `Linear(4,256)`, ReLU, `Linear(256,128)`, ReLU, `Linear(128,128)` | yes |

Batched: target `(C,4,256) (C,256) (C,256,128) (C,128)`; predictor adds `(C,128,128) (C,128)`.

The predictor is one block deeper than the target. That asymmetry is the original RND design and
this project's `predictor_extra_layers = 1`; CleanRL's Atari version uses two extra blocks. All
weights use orthogonal init with gain `sqrt(2)` and zero bias (`layer_init`'s default in both
CleanRL and this project). The target's parameters never receive a gradient and are never passed to
the optimizer.

Hidden activation is ReLU in both RND nets, this project's default. The original RND paper's
LeakyReLU with slope 0.2 is the recorded alternative; if it is ever switched on, both
implementations switch together (note that torch's `nn.LeakyReLU` default slope is 0.01, not 0.2).

### 3.4 Sizes

Trainable parameters per copy: actor 4,612 + critic 4,610 + predictor 50,688 = **59,910**. Frozen
target: 34,176. At `C = 128` that is 7.7 M trainable parameters. Rollout buffers hold under 50 KB
per copy per iteration (about 6 MB at `C = 128`). Neither memory nor arithmetic is the constraint;
the constraint is kernel-launch count, which is why the copy axis exists.

### 3.5 Per-copy initialization

Each copy's weights are drawn independently, keyed by `(base_seed, copy_index, net_name, layer_index)`
per `~/.claude/rules/rng-seeding.md`, so copy `i` gets the same initial weights whether the run has
`C = 8` or `C = 128`. Initialization happens once, so a Python loop over `C` is acceptable in torch;
JAX uses `jax.random.fold_in` on the copy index and vmaps the initializer.

The two frameworks' orthogonal initializers do not produce identical matrices from identical seeds.
That is expected and is covered by level 3 of §1; the exact-equality tests of §17 load weights from
a shared `.npz`.

## 4. Observations

### 4.1 Policy and critic input: raw observations

The actor and the critic consume the raw 4-d state $(x, y, v_x, v_y)$ with no normalization and no
scaling. Reasons, both specific to this environment:

- The observation is already bounded and well scaled: position within about $\pm 6$, velocity
  within $\pm 5$ by the environment's own velocity clip.
- A running observation normalizer makes the policy's input distribution move as the agent explores
  new regions. Under an exploration algorithm, that couples the meaning of every learned weight to
  the exploration progress, which is the thing being measured.

CleanRL's continuous script normalizes with `NormalizeObservation` (running mean/std) followed by a
clip to $\pm 10$, because MuJoCo observations have wildly different per-dimension scales. That path
stays available as a compile-time constant `normalize policy obs = False`. If it is ever switched
on, the statistic is **per copy** and the formula is the one in §4.2 with a $\pm 10$ clip and its
own running statistics object, never shared with the RND normalizer.

### 4.2 RND input: per-copy running mean and variance, clipped

The RND target and predictor see the whitened next observation:

$$\tilde{o} = \mathrm{clip}\!\left(\dfrac{o - \mu}{\sqrt{\sigma^2 + 10^{-8}}},\, -5,\, +5\right)$$

where $\mu, \sigma^2$ are the running per-dimension mean and variance of the observations this copy
has seen. Shapes: `mean (C, 4)`, `var (C, 4)`, `count (C, 1)`. The clip bound of 5 and the whitening
without any epsilon are CleanRL's (`ppo_rnd_envpool.py`, lines 365–370); the $10^{-8}$ inside the
square root is this project's RND (`_normalize_obs`) and is kept so the expression is defined when
the variance is zero.

Update rule — the parallel-variance formula from `gymnasium.wrappers.utils.RunningMeanStd`, applied
independently per copy, with the **population** (biased, `ddof = 0`) batch variance:

```
delta      = batch_mean - mean                                  # (C, 4)
tot        = count + batch_count                                # (C, 1)
new_mean   = mean + delta * batch_count / tot
M2         = var * count + batch_var * batch_count + delta**2 * count * batch_count / tot
new_var    = M2 / tot
new_count  = tot
```

Initial state: `mean = 0`, `var = 1`, `count = 1e-4` (the gymnasium default epsilon). These three
arrays are float64 in both implementations; every other array in the trainer is float32. In float32,
`delta * batch_count / tot` stops changing the mean at a total count near $1.7\times10^{7}$, which a
long run reaches. JAX must therefore run with `jax_enable_x64 = True` and annotate every other array
`float32` explicitly.

**Update cadence and ordering.** Exactly CleanRL's, and it matters because the same batch is
whitened twice with different statistics:

1. During the rollout, the bonus is computed with the statistics as they stood at the **end of the
   previous iteration**.
2. After the rollout and after the advantages are computed, the statistics are updated with all
   `T*N` next observations of this rollout (per copy).
3. The RND input used by the update phase is rebuilt with the **new** statistics.

### 4.3 Priming the RND normalizer

Before the first training iteration, run `obs norm init iterations = 10` rollouts of `T` steps with
uniform random actions in $[-1, 1]^2$, drawn independently per copy and per environment, updating
each copy's observation statistics from the next observations. These steps train nothing, and are
not counted in the environment-step counter. CleanRL uses 50 iterations over an 84×84 image; 10
iterations here is 5,120 samples per copy for a 4-d statistic.

### 4.4 Extrinsic reward is not normalized

The extrinsic reward is the environment's sparse reward (1.0 at the goal, 0.0 otherwise, plus the
configured constant `reward shift`), passed through unchanged. CleanRL's continuous script wraps the
environment in `NormalizeReward` (divide by the running standard deviation of the discounted return)
and clips to $\pm 10$; that is not used here. With a sparse reward that is zero for long stretches,
the running standard deviation of the return is near zero early in training and the division
amplifies the first non-zero reward by an arbitrary factor that depends on how long the agent
searched. The Atari RND base does not normalize the extrinsic reward either — envpool's
`reward_clip` is a sign clip, which is the identity on a 0/1 reward.

## 5. Intrinsic reward

### 5.1 Bonus

For a whitened next observation $\tilde{o}$, with target features $f(\tilde{o})$ and predictor
features $\hat{f}(\tilde{o})$, both of width `F = 128`:

$$r^{\mathrm{int}} = \tfrac{1}{2}\,\lVert \hat{f}(\tilde{o}) - f(\tilde{o}) \rVert_2^2
= \tfrac{1}{2}\sum_{j=1}^{128}\bigl(\hat{f}_j - f_j\bigr)^2$$

This is CleanRL's `(target - predict).pow(2).sum(1) / 2` and this project's `_dist_ensemble` `mse`
readout — the same expression. It is computed under no-gradient during the rollout.

Note that the bonus and the predictor's training loss use different reductions over the feature
axis: the bonus is $\tfrac12\sum_j e_j^2$ and the loss is $\tfrac1F\sum_j e_j^2$ (§9.4), so
$r^{\mathrm{int}} = \tfrac{F}{2}\times$ per-sample loss. Both come straight from CleanRL and both are
kept. The bonus scale is irrelevant downstream because §5.2 divides it by its own running standard
deviation; the loss scale is not, because the loss enters the total objective with coefficient 1.

### 5.2 Normalization by the running standard deviation of the discounted intrinsic return

Two pieces of state per copy, neither ever reset:

- The forward filter $R \in \mathbb{R}^{C \times N}$, one accumulator per environment, float32.
- A scalar running mean/variance/count per copy for the filtered values.

Per iteration, walk the rollout forward in time (`t = 0 ... T-1`):

$$R_t = \gamma_{\mathrm{int}}\, R_{t-1} + r^{\mathrm{int}}_t$$

with $\gamma_{\mathrm{int}} = 0.99$, giving `(T, C, N)` filtered values. Update the per-copy running
statistics from all `T*N` of them, using the §4.2 formula with `batch_count = T*N` and the
population variance. Then normalize the whole rollout in place:

$$\hat{r}^{\mathrm{int}}_t = \dfrac{r^{\mathrm{int}}_t}{\sqrt{\sigma^2_{\mathrm{int}} + 10^{-8}}}$$

The mean is deliberately not subtracted (Burda et al.'s choice, and CleanRL's). The current
rollout's own values are included in the statistics before the division, so a rollout is normalized
by statistics that contain it. That ordering is CleanRL's and is kept. Everything downstream — the
intrinsic value head, the intrinsic GAE — uses $\hat{r}^{\mathrm{int}}$.

**This differs from CleanRL's code and the difference is intentional.** CleanRL writes

```python
curiosity_reward_per_env = np.array(
    [discounted_reward.update(reward_per_step)
     for reward_per_step in curiosity_rewards.cpu().data.numpy().T])
mean, std, count = np.mean(...), np.std(...), len(curiosity_reward_per_env)
```

with `curiosity_rewards` of shape `(num_steps, num_envs)`. Transposing gives `(num_envs, num_steps)`,
so the loop runs over the **environment** axis and the filter state is a vector of length
`num_steps`: the discount is applied across environments as though the environment index were time.
A discounted return has to accumulate along time, one accumulator per environment, which is what
§5.2 above does. CleanRL's default configuration has `num_steps = num_envs = 128`, so the wrong axis
never raises a shape error. The same lines also pass `count = len(...)` — the number of rows, not the
number of values — understating the sample count by a factor of `T`. Both are corrected here.
Correcting them is also what makes the step a single vectorized scan instead of a Python loop over
environments.

## 6. Rollout

Per copy, the environment returns at each step:

| field | shape | meaning |
|---|---|---|
| `obs` | `(C, N, 4)` | observation to act on next (post-reset if a reset fired) |
| `real next obs` | `(C, N, 4)` | true post-step observation, before any auto-reset |
| `reward ext` | `(C, N)` | sparse extrinsic reward |
| `terminated` | `(C, N)` | reached the goal (only when `continuing task = False`) |
| `truncated` | `(C, N)` | hit the 400-step cap |

`real next obs` is what makes correct bootstrapping possible at a time-limit reset; the environment
module already commits to exposing it.

One rollout step `t`:

1. `Vext[t], Vint[t] = critic(obs)` — shapes `(C, N)` each, no gradient.
2. `mean = actor(obs)` `(C, N, 2)`; `std = exp(logstd)[:, None, :]` `(C, 1, 2)`;
   `action = mean + std * z` with `z` standard normal `(C, N, 2)`, drawn independently per copy.
3. `logprob[t] = sum over the 2 action dims of` $-\tfrac12 z^2 - \log \sigma - \tfrac12\log(2\pi)$,
   shape `(C, N)`. This is the log-density of the **unclipped** Gaussian sample.
4. Store `obs[t]`, `action[t]`, `logprob[t]`, `terminated[t]`, `truncated[t]`.
5. Step the environment with `action` (the environment itself clips to $[-1,1]$ per the physics
   spec, so no separate clip wrapper exists and the stored action stays unclipped).
6. `rnd input = whiten(real next obs)` with the previous iteration's statistics;
   `r_int[t] = bonus(rnd input)` under no gradient.
7. Store `real next obs[t]`, `reward ext[t]`; set `obs = obs next`.

After the loop, one batched call computes the bootstrap values on every stored next observation:

```
Vext_next, Vint_next = critic(real_next_obs.reshape(C, T*N, 4))   -> (C, T*N) -> (T, C, N)
```

This is one large GEMM rather than `T` small ones, and it is mathematically equal to reading
`Vext[t+1]` on every non-reset step, because the parameters have not changed since the rollout. That
equality is a useful self-check in the test harness.

The rollout consumes `T*C*N` environment steps per iteration.

## 7. Two-stream GAE

Both streams use $\lambda = 0.95$. The extrinsic stream is episodic; the intrinsic stream is not.
The recursion runs backwards from `t = T-1` to `0` with `A[T] = 0`, with no `if t == T-1` branch
anywhere (the bootstrap values from §6 already cover the last step).

Extrinsic, $\gamma_{\mathrm{ext}} = 0.999$:

```
delta_ext[t] = r_ext[t] + gamma_ext * Vext_next[t] * (1 - terminated[t]) - Vext[t]
A_ext[t]     = delta_ext[t] + gamma_ext * lam * (1 - done[t]) * A_ext[t+1]
done[t]      = max(terminated[t], truncated[t])
```

Intrinsic, $\gamma_{\mathrm{int}} = 0.99$, no masks at all:

```
delta_int[t] = r_int_hat[t] + gamma_int * Vint_next[t] - Vint[t]
A_int[t]     = delta_int[t] + gamma_int * lam * A_int[t+1]
```

Returns for the value targets: `ret_ext = A_ext + Vext`, `ret_int = A_int + Vint`.

Two masks appear in the extrinsic stream and they are different on purpose:

- The **value bootstrap** is cut only by a true termination. A time-limit truncation is not the end
  of the world for the agent, so its continuation value is still `Vext(real next obs)`.
- The **advantage recursion** is cut by either cause, because the next transition belongs to a
  different episode.

CleanRL does neither: it carries one `dones` buffer holding "was this environment done before step
`t`", uses `1 - dones[t+1]` for both roles, and bootstraps from `values[t+1]`, which after an
auto-reset is the value of the reset state rather than of the final state. With the default
PointMaze setup (`continuing task = True`, truncation every 400 steps, never terminated) CleanRL's
treatment would make the value function learn a 400-step episodic task under $\gamma = 0.999$.
Setting `bootstrap on truncation = False` reproduces CleanRL exactly, and it is a compile-time
constant.

The intrinsic stream carries across episode boundaries, which is what "non-episodic intrinsic
reward" means in Burda et al. and what CleanRL implements with `int_nextnonterminal = 1.0`.

## 8. Combining the two advantage streams

Define $c_{\mathrm{int}} = 1.0$ and $c_{\mathrm{ext}} = 2.0$ as the intrinsic and extrinsic
advantage coefficients (`int coef` and `ext coef`). The combined advantage is

$$A = c_{\mathrm{int}}\, A^{\mathrm{int}} + c_{\mathrm{ext}}\, A^{\mathrm{ext}}$$

One combined advantage drives the single policy loss; the two value heads are trained separately on
their own returns. (CleanRL's argument docstrings for these two coefficients are swapped —
`int_coef` is documented as "coefficient of extrinsic reward" — but the arithmetic at line 442 is
the one written above.)

Advantage normalization (`norm adv = True`) is applied to `A` inside each minibatch, **per copy**:

```
A_mb = (A_mb - mean(A_mb, axis=rows)) / (std(A_mb, axis=rows) + 1e-8)      # reduce only over rows
```

with the standard deviation computed over that copy's rows only. Torch's `Tensor.std` defaults to
the unbiased estimator (`ddof = 1`), JAX's `jnp.std` to the biased one (`ddof = 0`); both
implementations must use **unbiased, `ddof = 1`**, matching CleanRL.

## 9. Losses

All per-copy losses reduce over that copy's rows only, producing a `(C,)` vector. The scalar handed
to the optimizer is the **sum over copies**, never the mean:

$$\mathcal{L} = \sum_{c=1}^{C} \mathcal{L}_c$$

so that each copy's gradient is exactly the gradient of its own loss. Averaging over copies would
scale every gradient by $1/C$, which changes what `max grad norm = 0.5` means and, through Adam's
`eps`, changes the update itself.

### 9.1 Policy loss

```
logratio = newlogprob - old_logprob            # (C, m)
ratio    = exp(logratio)
pg_loss  = mean_rows( max( -A_mb * ratio,
                           -A_mb * clip(ratio, 1 - clip_coef, 1 + clip_coef) ) )
```

`clip_coef = 0.2` (CleanRL's continuous value; the Atari RND script uses 0.1).

`newlogprob` is recomputed from the stored action:

$$\log \pi(a) = \sum_{d=1}^{2}\left[-\tfrac12\left(\dfrac{a_d - \mu_d}{\sigma_d}\right)^2
- \log \sigma_d - \tfrac12 \log (2\pi)\right]$$

Written out because JAX has no distributions library in the dependency set and the two
implementations must agree term by term.

### 9.2 Extrinsic value loss (clipped)

```
v_unclipped = (Vext_new - ret_ext)**2
v_clipped   = (Vext_old + clip(Vext_new - Vext_old, -clip_coef, +clip_coef) - ret_ext)**2
ext_v_loss  = 0.5 * mean_rows( max(v_unclipped, v_clipped) )
```

The clip range is `clip_coef` in raw value units, which is CleanRL's choice and is kept.

### 9.3 Intrinsic value loss (never clipped)

```
int_v_loss = 0.5 * mean_rows( (Vint_new - ret_int)**2 )
```

CleanRL clips the extrinsic value loss and not the intrinsic one. That asymmetry is deliberate in
the source and is reproduced.

`v_loss = ext_v_loss + int_v_loss`, entering the total with `vf_coef = 0.5`.

### 9.4 Entropy

$$H = \sum_{d=1}^{2}\left[\tfrac12 + \tfrac12\log(2\pi) + \log \sigma_d\right]$$

State-independent, because `logstd` is. `ent coef = 0.0` (CleanRL's continuous default), so the term
contributes nothing to the gradient and may be dropped from the compiled graph entirely — an exact
simplification, not an approximation. It is still computed and logged. The Atari RND script's
`ent_coef = 0.001` applies to a categorical entropy over 18 actions and does not carry over: here
the entropy bonus would only inflate a state-independent standard deviation, and exploration is
RND's job.

### 9.5 RND predictor loss

```
per_sample   = mean over the 128 feature dims of (predict - stop_gradient(target))**2   # (C, m)
forward_loss = mean_rows(per_sample)                                    # update_proportion = 1.0
```

`update proportion = 1.0` is the default, following this project's own RND. With a proportion below
1 the masked form is used:

```
keep         = (uniform((C, m)) < update_proportion)                    # independent per copy
forward_loss = sum_rows(per_sample * keep) / max(sum_rows(keep), 1.0)
```

CleanRL uses 0.25, sized for its 16,384-row batch. At 512 rows, dropping three quarters of them
leaves 128 rows per predictor step. The keep-mask draw must never be broadcast across the copy axis.

### 9.6 Total

```
loss_c = pg_loss - ent_coef * entropy + vf_coef * (ext_v_loss + int_v_loss) + forward_loss
loss   = sum over copies of loss_c
```

The RND forward loss carries coefficient 1.0, as in CleanRL.

## 10. Optimizer, gradient clipping, learning rate

- **One Adam** over the actor, the critic, and the RND predictor together (the frozen target is not
  passed to it). `lr = 3e-4`, `eps = 1e-5`, betas `(0.9, 0.999)`. CleanRL's Atari RND run uses
  `1e-4`; `3e-4` is CleanRL's continuous-control value and the right scale for a 64-unit MLP.
- **Adam is elementwise**, so one Adam over parameters carrying a leading copy axis is exactly `C`
  independent Adams. No special handling is needed, and none may be added.
- **Gradient clipping is per copy.** `max grad norm = 0.5` applies to the norm of one copy's own
  gradients:

```
g_norm[c] = sqrt( sum over all parameter tensors of sum over that copy's slice of grad**2 )   # (C,)
scale[c]  = min(1.0, max_grad_norm / (g_norm[c] + 1e-6))
grad      = grad * scale[c] broadcast along the copy axis
```

  A single global norm over the stacked tensors — the direct translation of
  `nn.utils.clip_grad_norm_` — makes one copy's large gradient shrink every other copy's update.
  This is the single most damaging coupling in a naive batched port.
- **Learning-rate annealing** is linear from `lr` to 0 over the planned iteration count, the same
  scalar for every copy (`lr_t = lr * (1 - (iteration - 1) / num_iterations)`). It is passed into
  the compiled step as a traced scalar argument, never baked in as a Python float, or the step
  recompiles every iteration.
- **`target_kl` is `None` and the early-stop branch does not exist.** It is a data-dependent branch
  that would both break the compiled step and couple the copies (one copy's KL would stop every
  copy's update).

## 11. Update style A — one full-batch gradient step, then discard the data

One optimizer step per rollout, over all `B = 512` rows per copy.

```
# inputs, all per copy:
#   obs        (C, 512, 4)      actions   (C, 512, 2)     old_logprob (C, 512)
#   A          (C, 512)         ret_ext   (C, 512)        ret_int     (C, 512)
#   Vext_old   (C, 512)         rnd_input (C, 512, 4)
def update_full_batch(params, opt_state, batch, lr):
    def loss_fn(params):
        A_n  = (batch.A - mean(batch.A, rows)) / (std(batch.A, rows, ddof=1) + 1e-8)   # (C, 512)

        mu   = actor(params, batch.obs)                     # (C, 512, 2)
        sd   = exp(params.logstd)[:, None, :]               # (C, 1, 2)
        lp   = gaussian_logprob(batch.actions, mu, sd)      # (C, 512)
        ent  = gaussian_entropy(sd)                         # (C,)

        # ratio is identically 1 here: the parameters have not moved since the rollout.
        pg   = mean_rows(-A_n)                              # (C,)

        Vext, Vint = critic(params, batch.obs)              # (C, 512) each
        v_ext = 0.5 * mean_rows((Vext - batch.ret_ext)**2)  # clipping is a no-op, see below
        v_int = 0.5 * mean_rows((Vint - batch.ret_int)**2)

        pred  = predictor(params, batch.rnd_input)          # (C, 512, 128)
        targ  = stop_gradient(target(params, batch.rnd_input))
        fwd   = mean_rows(mean_features((pred - targ)**2))  # (C,)

        loss_c = pg - ent_coef * ent + vf_coef * (v_ext + v_int) + fwd
        return sum(loss_c)                                  # scalar

    grads = grad(loss_fn)(params)
    grads = clip_per_copy(grads, max_grad_norm)             # per-copy norm, never global
    params, opt_state = adam_step(params, grads, opt_state, lr)
    return params, opt_state
```

**CORRECTION (2026-08-15, found during the torch implementation).** The pseudocode above writes
`pg = mean_rows(-A_n)`. That is value-correct but GRADIENT-WRONG: at `ratio == 1` the surrogate's
value equals `-A_n`, but its derivative is `-A_n * grad(newlogprob)` — the policy gradient — and
`mean_rows(-A_n)` has derivative zero, so style A would never train the actor. The correct
simplification keeps the ratio and drops only the clip machinery:
`pg = mean_rows(-A_n * exp(newlogprob - old_logprob))`. Dropping the clips IS gradient-exact
(both max branches coincide in value and derivative in a neighborhood of `ratio == 1`, since the
clip is inactive there). Both implementations must use the corrected form.

**Why the clipping terms are dropped rather than computed.** At the single point where the loss is
evaluated the parameters are the rollout parameters, so `newlogprob == old_logprob` and `ratio == 1`
exactly (up to GEMM-shape rounding). Then

- `max(-A * 1, -A * clip(1, 0.8, 1.2)) = -A`, so the clipped policy surrogate equals `-A`;
- `Vext_new == Vext_old`, so `clip(Vext_new - Vext_old, ±0.2) = 0`, the "clipped" value prediction
  equals the unclipped one, and the `max` of the two squared errors is the plain squared error.

Both simplifications are algebraic identities, not approximations, so style A's kernel may omit the
clip machinery and still compute the same function. In floating point the two forms can differ in the
last bits if `Vext_new` is recomputed with a different GEMM shape than the rollout used; passing the
stored `Vext_old` through makes them bit-equal.

Style A takes one Adam step per 512 rows and trains the RND predictor with one pass over each row.

## 12. Update style B — 4 epochs x 4 minibatches of 128, reshuffled each epoch

16 sequential optimizer steps per rollout. The permutations for all four epochs are drawn once, up
front, so the compiled scan carries no random state:

```
perm = argsort(uniform((4, C, 512)), axis=-1)      # (4, C, 512), independent per copy AND per epoch

def update_epoch_minibatch(params, opt_state, batch, lr):
    for epoch in range(4):                          # unrolled or lax.scan over a (4,) axis
        for k in range(4):                          # unrolled or lax.scan over a (4,) axis
            idx = perm[epoch][:, k*128:(k+1)*128]   # (C, 128)
            mb  = gather_rows(batch, idx)           # each field -> (C, 128, ...)

            def loss_fn(params):
                A_n = (mb.A - mean(mb.A, rows)) / (std(mb.A, rows, ddof=1) + 1e-8)

                mu  = actor(params, mb.obs)                          # (C, 128, 2)
                sd  = exp(params.logstd)[:, None, :]
                lp  = gaussian_logprob(mb.actions, mu, sd)           # (C, 128)
                ratio = exp(lp - mb.old_logprob)
                pg  = mean_rows(maximum(-A_n * ratio,
                                        -A_n * clip(ratio, 0.8, 1.2)))

                Vext, Vint = critic(params, mb.obs)
                vc    = mb.Vext_old + clip(Vext - mb.Vext_old, -0.2, 0.2)
                v_ext = 0.5 * mean_rows(maximum((Vext - mb.ret_ext)**2,
                                                (vc   - mb.ret_ext)**2))
                v_int = 0.5 * mean_rows((Vint - mb.ret_int)**2)

                pred  = predictor(params, mb.rnd_input)
                targ  = stop_gradient(target(params, mb.rnd_input))
                fwd   = mean_rows(mean_features((pred - targ)**2))

                ent   = gaussian_entropy(sd)
                return sum(pg - ent_coef * ent + vf_coef * (v_ext + v_int) + fwd)

            grads = grad(loss_fn)(params)
            grads = clip_per_copy(grads, max_grad_norm)
            params, opt_state = adam_step(params, grads, opt_state, lr)
    return params, opt_state
```

Style B takes 16 Adam steps per 512 rows and trains the RND predictor with four passes over each row
(one per epoch). That is a 16x difference in optimizer steps and a 4x difference in predictor
training against style A; the two styles are different algorithms and their results are not
comparable except as a deliberate comparison.

### 12.1 The two styles are two separate compiled functions

`update_full_batch` and `update_epoch_minibatch` are built and compiled separately, and which one
runs is chosen in Python when the trainer is constructed. There is no `if style == ...` inside any
traced or compiled region, and no shared function with a branch. The reasons are concrete: under
`jax.jit` a Python branch on a traced value is an error and a branch on a static value forces a
second compilation; under `torch.compile` a data-dependent branch is a graph break, and with CUDA
graphs a branch makes the captured graph invalid.

The same rule covers every other knob in this document: `norm adv`, `clip vloss`, `bootstrap on
truncation`, `update proportion == 1.0`, `normalize policy obs`, `ent coef == 0.0`. Each is resolved
at build time into the graph that is actually compiled.

## 13. What changes when everything is batched over `C` copies

CleanRL runs one seed per process, so nothing in it is written to keep seeds apart. Every shared
piece of state below becomes a coupling the moment `C` copies share one process.

| item in CleanRL | what it must become | if it is not fixed |
|---|---|---|
| `obs_rms`, one `RunningMeanStd` | per copy: `mean (C,4)`, `var (C,4)`, `count (C,1)` | one copy's exploration rescales every copy's RND input |
| `reward_rms`, one scalar `RunningMeanStd` | per copy: `mean (C,)`, `var (C,)`, `count (C,)` | intrinsic reward scale is shared, so copies see each other's novelty |
| `RewardForwardFilter.rewems` | per copy and per env: `(C, N)` | same as above, plus a wrong discount axis |
| `nn.utils.clip_grad_norm_` over one flat parameter list | per-copy norm and per-copy scale, `(C,)` | one copy's gradient spike shrinks every copy's step |
| `mb_advantages.mean()/.std()` over the whole minibatch | reduce over rows only, keep the copy axis | advantages are centred by other copies' data |
| loss reduced to one scalar by `.mean()` | sum over copies of per-copy means | every gradient scaled by `1/C`, changing what the clip norm means |
| `np.random.shuffle(b_inds)`, one index array | `(C, 512)` permutation, one per copy | harmless in principle, but it makes minibatch groupings shared |
| `torch.rand(len(forward_loss))` keep-mask | `(C, m)`, never broadcast along `C` | copies train the predictor on the same rows |
| `approx_kl` early break | removed | one copy's KL stops every copy's update |
| per-episode `print` / logging branch | reduction into preallocated buffers | host synchronization per episode, and a data-dependent branch |
| Adam over one parameter list | unchanged — Adam is elementwise | nothing; this one is already safe |
| linear learning-rate anneal | unchanged — one scalar for all copies | nothing; it is deterministic and copy-independent |

Two further requirements:

- **Never broadcast a random draw along the copy axis.** Drawing `(C, ...)` from one generator gives
  independent values per copy and is correct. Drawing `(1, ...)` and broadcasting makes the copies'
  updates dependent random variables even though each copy's marginal distribution is right.
- **Copy `i`'s trajectory should not depend on `C`.** Initial weights and environment reset noise are
  keyed by `(base seed, copy index, quantity)` and satisfy this exactly. Per-step action noise and
  keep-masks drawn from one shared stream do not: their values shift when `C` changes. That is
  accepted for now and stated in the run record; the exact-across-`C` version needs a counter-based
  generator keyed by `(base seed, copy index, iteration, step, quantity)` in both frameworks, which
  JAX gets from `fold_in` and torch would need built by hand.

## 14. Rollout size: `T * N = 512` per copy

| `T` | `N` | envs at `C = 128` | GAE horizon | sequential steps per iteration |
|---|---|---|---|---|
| **128** | **4** | **512** | **128** | **128** |
| 64 | 8 | 1,024 | 64 | 64 |
| 32 | 16 | 2,048 | 32 | 32 |
| 16 | 32 | 4,096 | 16 | 16 |

Default is `T = 128, N = 4`: closest to CleanRL's `num_steps = 128` and the longest GAE horizon.
The throughput consequence is real — the rollout is `T` sequential steps of very small matrix
multiplies, so `T = 32, N = 16` cuts the sequential step count fourfold and widens every GEMM by the
same factor. That alternative is blessed on the condition that both implementations move together and
the change is recorded, because it changes the numbers: it truncates the extrinsic GAE at 32 steps
(the $\lambda = 0.95$ weighting is effectively spent after about 20 steps, so the truncation bias is
modest, and the $\gamma = 0.999$ bootstrap covers the rest).

`C` and `N` multiply into the environment batch `C * N`; only `N` is constrained by the 512-row
budget, so raising `C` is the way to fill the GPU without touching the algorithm.

## 15. Hyperparameters

| knob | CleanRL Atari RND | CleanRL continuous | this spec |
|---|---|---|---|
| learning rate | 1e-4 | 3e-4 | **3e-4** |
| Adam eps | 1e-5 | 1e-5 | 1e-5 |
| anneal learning rate | yes, linear | yes, linear | yes, linear |
| `num steps` | 128 | 2048 | **128** |
| `num envs` | 128 | 1 | **4** |
| rows per update | 16,384 | 2,048 | **512** |
| `gamma` extrinsic | 0.999 | 0.99 | **0.999** |
| `gamma` intrinsic | 0.99 | n/a | 0.99 |
| `gae lambda` | 0.95 | 0.95 | 0.95 |
| `update epochs` | 4 | 10 | **4** (style B), 1 (style A) |
| `num minibatches` | 4 | 32 | **4** (style B), 1 (style A) |
| `clip coef` | 0.1 | 0.2 | **0.2** |
| `clip vloss` | yes (extrinsic only) | yes | yes (extrinsic only) |
| `norm adv` | yes | yes | yes |
| `ent coef` | 0.001 | 0.0 | **0.0** |
| `vf coef` | 0.5 | 0.5 | 0.5 |
| `max grad norm` | 0.5 | 0.5 | 0.5, **per copy** |
| `target kl` | none | none | none, branch removed |
| `int coef` | 1.0 | n/a | 1.0 |
| `ext coef` | 2.0 | n/a | 2.0 |
| `update proportion` | 0.25 | n/a | **1.0** |
| RND feature width | 512 | n/a | **128** |
| predictor extra blocks | 2 | n/a | **1** |
| obs-norm init iterations | 50 | n/a | **10** |
| policy obs normalized | no (just `/255`) | yes, running, clip ±10 | **no** |
| extrinsic reward normalized | no (sign clip) | yes, running, clip ±10 | **no** |

## 16. Every deviation from CleanRL, in one place

1. **Continuous Gaussian head instead of a categorical head.** Actor outputs a mean and holds a
   state-independent `logstd`; log-probability and entropy are summed over the two action
   dimensions. Taken from `ppo_continuous_action.py`.
2. **MLPs instead of the Nature CNN.** 64-unit tanh trunks for actor and critic; the shared
   convolutional trunk and its `extra_layer` residual do not apply to a 4-d observation.
3. **RND nets sized from this project's `ObservationEncoder`** (`4-256-128`, predictor one block
   deeper) instead of CleanRL's convolutional `512`-wide pair.
4. **The intrinsic reward filter runs over time, not over environments** (§5.2). CleanRL's transpose
   iterates the wrong axis, and its `count` is the row count rather than the value count.
5. **The bonus, the observation statistics, and the predictor's training input are all the same
   tensor**, the true next observation. CleanRL computes the rollout bonus on the post-auto-reset
   observation and then trains the predictor and updates `obs_rms` on `b_obs`, the observations at
   time `t` — an off-by-one set of rows.
6. **Bootstrapping distinguishes truncation from termination** (§7). CleanRL treats both as terminal.
   Reproduce CleanRL by setting `bootstrap on truncation = False`.
7. **Extrinsic reward and policy observations are not normalized** (§4.1, §4.4), unlike
   `ppo_continuous_action.py`'s wrapper stack.
8. **Gradient clipping, advantage normalization, and every running statistic are per copy** (§13).
9. **`target_kl`, the per-episode logging branch, and the `update proportion` shortcut branch are
   removed**, so the update has no data-dependent control flow.
10. **No envpool, so no auto-reset "burned rows".** The whole `fix_envpool_autoreset` machinery of
    `08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py` — dropping the fabricated rows, passing the GAE
    carry through them, and the fixed-minibatch-shape allowance that repaired the shape drift they
    caused — has no counterpart here. The batched environment auto-resets inside its own step and
    reports `real next obs`, so no fabricated transition ever enters the buffer.
11. **Running statistics in float64, everything else float32** (§4.2).
12. **The two update styles are two compiled functions**, not one function with a flag (§12.1).

## 17. Verifying that the two implementations agree

Three tests, in increasing strength. All three belong in
`ppo/torch_ppo/tests/` and `ppo/jax_ppo/tests/`, reading the same fixture files.

**Test 1 — shared-weight forward agreement.** Load one `.npz` of initial parameters into both, feed
one fixed `(C, M, 4)` observation batch, and compare: action means, `logstd`, both value heads,
target features, predictor features, and the intrinsic bonus. Tolerance `1e-5` relative.

**Test 2 — one full iteration under supplied randomness.** Fixture supplies initial parameters, a
`(T, C, N, 4)` observation stream, the `(T, C, N, 2)` standard-normal action noise, the extrinsic
rewards, the termination and truncation flags, the `(4, C, 512)` permutations, and the keep-masks.
Both implementations run one complete iteration — rollout arithmetic, intrinsic filter, both running
statistics, both GAE streams, the combined advantage, every loss term, the clipped gradients, one
Adam step — and every listed intermediate is compared at `1e-5`. This test is what makes §1 level 2
concrete, and it is the only cheap way to catch an axis or ordering mistake in the running
statistics.

**Test 3 — independence of copies.** Run `C = 4` with all four copies given identical initial
weights and identical observations except copy 2, whose observations are perturbed. After one
iteration, copies 0, 1, and 3 must be bit-identical to each other and unchanged from a `C = 3` run
that omits copy 2. Any failure points straight at a shared statistic, a global gradient norm, or a
mean over the copy axis. Run it in both frameworks.

Beyond the tests: a short learning check on the same seed set (extrinsic return and the fraction of
the maze visited over a fixed step budget) must give the same curves within seed noise. That is the
only evidence that the spec was implemented as written rather than merely implemented consistently.
