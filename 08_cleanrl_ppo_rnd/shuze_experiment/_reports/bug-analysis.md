All claims verified. Here is the report.

---

# EnvPool next-step autoreset in `cleanrl/ppo_rnd_envpool.py` — engineering analysis

**Repo under analysis:** `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl` @ `fe8d8a0`
**File:** `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py` (539 lines)

Everything below marked **VERIFIED** was reproduced by running code on this machine, on **two** envpool versions: `0.6.6` (the version cleanrl pins) with `gym==0.23.1` + `numpy==1.26.4`, and `1.2.5` (newest release). Claims I could not test are marked **NOT VERIFIED**.

---

## 1. The rollout loop, step by step

The storage writes (lines 344–364):

```python
344	        for step in range(0, args.num_steps):
345	            global_step += 1 * args.num_envs
346	            obs[step] = next_obs
347	            dones[step] = next_done
...
358	            actions[step] = action
359	            logprobs[step] = logprob
360	
361	            # TRY NOT TO MODIFY: execute the game and log data.
362	            next_obs, reward, done, info = envs.step(action.cpu().numpy())
363	            rewards[step] = torch.tensor(reward).to(device).view(-1)
364	            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
```

Write `o_k, r_k, d_k` for what the *k*-th `envs.step()` call returns, with `o_0 = envs.reset()` (line 320). The loop body at index `t` gives exactly:

| slot | holds |
|---|---|
| `obs[t]` | `o_t` |
| `dones[t]` | `d_t` (and `d_0 = 0`) |
| `actions[t]`, `logprobs[t]` | `a_t`, sampled from `π(o_t)`; consumed by step call `t+1` |
| `rewards[t]` | `r_{t+1}` |
| `curiosity_rewards[t]` | novelty of `o_{t+1}` (lines 365–373 run *after* the step) |
| `ext_values[t]`, `int_values[t]` | `V(o_t)` |

### Timeline — envpool (next-step autoreset)

Let step call **K** be the one that ends the episode.

| call | returns | stored at index | contents |
|---|---|---|---|
| K−1 | `o_{K-1}`, real | t = K−1 | `obs=o_{K-1}`, `act=a_{K-1}` real, `rew=r_K` real terminal reward, `done=d_{K-1}=0` |
| **K** | `o_K = s_last`, `d_K = True` | t = **K** | `obs=s_last`, `dones[K]=1`, `act=a_K` **discarded by envpool**, `rew=r_{K+1}=0`, next state `= s_new` |
| K+1 | `o_{K+1} = s_new`, `r=0`, `d=False` | t = K+1 | `obs=s_new`, `dones=0`, real again |

**Index K is a transition that never happened**: `(s_last, a_K, 0, s_new)`. The action was thrown away, the reward is a constant 0, and `s_new` is not a consequence of `s_last`.

### Timeline — gym `<=0.23.1` (same-step autoreset)

Call K returns `o_K = s_new` **with** `d_K = True`; `s_last` is dropped (or put in `info["terminal_observation"]`). There is no extra call, so no fabricated row. Index K is immediately the new episode's first real state.

**So the difference is not merely "which observation" — envpool consumes one extra `step()` call per episode end, and that call produces a whole fabricated row in the PPO batch.**

### VERIFIED

`test_autoreset_is_next_step_and_the_burned_step_is_inert`, on Breakout with `max_episode_steps=4`:

```
elapsed_step == [1, 2, 3, 4, 0] * 3
done        == [False, False, False, True, False] * 3
rows 4, 9, 14: reward exactly 0.0, done False
```

`test_the_action_on_the_burned_step_is_discarded` (CartPole, differential test): two pools fed identical actions except at the burned index stay **byte-identical for all steps**; the same injection one index earlier diverges immediately. The action is genuinely discarded.

**`dones[t] == 1` marks exactly the fabricated rows** — verified on Breakout: burned indices `[24, 49, 74, …]`, all with `rewards[t] == 0.0`, all with `obs[t]` equal to the previous call's returned observation.

**How much of the batch:** 3.5% (Breakout, `episodic_life=True`, random policy); **1.16%** on `MontezumaRevenge-v5` — the script's default (line 41) — with `episodic_life=True`, `num_envs=8`, random policy over 16,000 transitions.

---

## 2. envpool's documented semantics, and the newest release

The envpool docs state it exactly (quoted verbatim from https://envpool.readthedocs.io/en/latest/content/python_interface.html):

> "EnvPool enables auto-reset by default. Let's suppose an environment that has a `max_episode_steps = 3`. When we call `env.step(action)` five consecutive times, the following would happen:
> 1. the first call would trigger `env.reset()` and return with `done = False` and `reward = 0`, i.e., **the action will be discarded**;
> …
> 4. the fourth call would trigger `env.step(action)` and elapsed step is 3. At this time it returns `truncated = True`;
> 5. the fifth call would trigger `env.reset()` since the last episode has finished, and return with `done = False` and `reward = 0`, i.e., **the action will be discarded**."

**sail-sg/envpool#194** — "[BUG] Vectorized Environment Autoreset Incompatible with openai/baselines' API" — reports the same: gym returns `obs=0` (initial state of the next episode) at the done step; envpool returns `obs=4` (terminal) at the done step and `obs=0` on the following step. The issue shows as **Closed**, and I found **no fix commit or resolution comment** in it.

### Newest release and whether behaviour changed — VERIFIED

- **Newest envpool release: `1.2.5`** (GitHub releases page dates it 20 May 2026; PyPI's JSON metadata omits a date for it and lists `1.2.2` at 9 May 2026 — minor inconsistency in the sources, not in the behaviour).
- **The autoreset behaviour is unchanged from `0.6.6` to `1.2.5`.** I ran the same six-test suite against both and got identical results, including byte-identical CartPole observations across the two versions. The release notes for 1.0.0 → 1.2.5 mention gymnasium namespace aliases, removing the gym dependency, new env families — **nothing about autoreset or step-return semantics**.
- **`env_type="gymnasium"` has identical timing to `env_type="gym"`** — VERIFIED on both CartPole and Breakout. It changes only the tuple arity, not when the reset happens.
- **The upstream fix cleanrl is waiting for is not coming, because the standard moved the other way.** Gymnasium 1.0 changed its own vector autoreset *to* next-step, explicitly to align with EnvPool and SampleFactory. The doc note at `docs/rl-algorithms/ppo-rnd.md:44` — "we take a note here and await for the upstream fix" — is now obsolete; envpool's behaviour is the standard.

### Two API breaks in envpool 1.x — VERIFIED, separate from the bug

Under `1.2.5`, `env_type="gym"` returns a **5-tuple** from `step()` and a **2-tuple** from `reset()`. Line 362 (`next_obs, reward, done, info = envs.step(...)`) and line 320 (`torch.Tensor(envs.reset())`) both break outright. Under `0.6.6` with `gym==0.23.1` the 4-tuple/1-tuple form is what you get (envpool 0.6.6 dispatches on the installed gym version — with `gym==0.26.2` it already returns 5-tuples). So "just upgrade envpool" is not a drop-in.

---

## 3. How the other envpool scripts in cleanrl handle it — they don't

Six scripts import envpool:

```
cleanrl/ppo_atari_envpool.py
cleanrl/ppo_rnd_envpool.py
cleanrl/pqn_atari_envpool.py
cleanrl/pqn_atari_envpool_lstm.py
cleanrl/ppo_atari_envpool_xla_jax_scan.py
cleanrl/ppo_atari_envpool_xla_jax.py
```

**None of them work around the off-by-one.** `ppo_atari_envpool.py:83-114` and `pqn_atari_envpool.py:77-108` contain a `RecordEpisodeStatistics` wrapper that is character-for-character the same as `ppo_rnd_envpool.py:97-128`, and their rollout loops store rows the same way. `docs/rl-algorithms/ppo.md:508` and `docs/rl-algorithms/pqn.md:130,221` carry the identical `???+ bug` admonition, so the maintainers know it affects those scripts too.

The XLA/JAX variants (`ppo_atari_envpool_xla_jax.py:222-237`) differ only in episode-statistics bookkeeping — they gate on `info["terminated"] + info["TimeLimit.truncated"]` so the *logged return* is the last completed episode's rather than a running total:

```python
227	            episode_returns=(new_episode_return) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
```

That is a logging improvement, not a fix — the fabricated row is still in the training batch.

Nothing in `cleanrl/*.py` references `elapsed_step` or `terminal_observation`. Grep confirms: zero hits.

### `git log --oneline --all -- cleanrl/ppo_rnd_envpool.py`

```
35896b1 Refactor to use tyro (#424)
e92cf57 quick fix
5f3f716 fix
6220645 fix pre-commit
896f346 refactor
42d21bd PPO + JAX + EnvPool + Atari (#227)
58318ec Merge branch 'master' into jax-ppo-envpool-atari
f0bbf49 Remove the unnecessary regular advantage code in PPO (#287)
55c9a74 use the latest envpool interface
0284d74 Add rnd_ppo.py documentation and refactor (#151)
67e6db6 Format with pre_submit.
0b8eaae Merge ppo_rnd_envpool.
9f2fa70 Fix a bug that model uses old forward method.
54e1ddf Refactor code and add implementation details.
f19a07a minor refactor
21839a8 quick fix
65b8a45 don't use frames argument
a6a7ef1 Remove video stuff (which are not used)
2d38f9e Fix typos.
1bd0d18 Switch to use envpool and rename it to ppo_end_envpool.py.
```

**No commit addresses the autoreset off-by-one.** `19cc1fe add a note one envpool` (in the docs history) is where the `???+ bug` admonition was added — the note *is* the response.

Issue search `rnd envpool` on vwxyzjn/cleanrl returns `#552` (a tensor-copy performance PR), `#151` (docs refactor), and `#416` "Potential bug in PPO+RND?" — which is about a missing `torch.no_grad()` around `rnd_model.predictor` during rollout, **not** the autoreset. **No open or closed issue proposes a fix for the off-by-one.**

---

## 4. Concrete consequences for RND, by array index

Let index **K** be a row with `dones[K] == 1`.

### Wrong at index K

| tensor | line | value stored | why it is wrong |
|---|---|---|---|
| `obs[K]` | 346 | `s_last` | genuine observation — *not* wrong by itself |
| `actions[K]`, `logprobs[K]` | 358–359 | `a_K ~ π(·|s_last)` | envpool **discarded** this action (VERIFIED); no consequence exists for it |
| `rewards[K]` | 363 | **always exactly 0.0** (VERIFIED) | fabricated |
| `dones[K]` | 347 | 1 | correct, and it is the fix's handle |
| `curiosity_rewards[K]` | 373 | novelty(`s_new`) | the new episode's initial state's novelty, credited to `(s_last, a_K)` in the *old* episode |
| `ext_values[K]`, `int_values[K]` | 351–355 | `V(s_last)` | legitimate predictions, but trained toward fabricated targets |

Everything at indices K−1 and K+1 is correct. In particular `curiosity_rewards[K-1] = novelty(s_last)` is right — the terminal state *is* the next state of the real transition at K−1.

### GAE — the extrinsic stream self-heals, the intrinsic stream does not

```python
417	                    ext_nextnonterminal = 1.0 - dones[t + 1]
418	                    int_nextnonterminal = 1.0
...
421	                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
422	                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
423	                ext_advantages[t] = ext_lastgaelam = (
424	                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
425	                )
426	                int_advantages[t] = int_lastgaelam = (
427	                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
428	                )
```

- **Extrinsic:** at `t = K-1`, `ext_nextnonterminal = 1 - dones[K] = 0`, which zeroes the `ext_lastgaelam` term at line 424. The fabricated advantage cannot propagate backward.
- **Intrinsic:** `int_nextnonterminal` is hardcoded `1.0` (line 418). This is *deliberate* — the original `openai/random-network-distillation` does the same (`orig_ppo_agent.py:274`: `nextnew = 0.0 #No dones for intrinsic reward.`). But it means **nothing stops the fabricated row from leaking backward**.

**VERIFIED** with the script's own GAE code (`probe_gae_leak.py`, T=64, burn at index 40, `int_gamma=0.99`, `gae_lambda=0.95`): perturbing only the burned row's intrinsic reward by +1.0 moves

- extrinsic advantages at indices **`[40]`** — the burned row only;
- intrinsic advantages at indices **`0..40`** — 41 of 64 rows, magnitude `1.0000` at t=40, `0.5415` at t=30, `0.0860` at t=0.

So one fabricated row per episode contaminates the intrinsic advantage of **every preceding row in the rollout**, with a decay of `(int_gamma * gae_lambda)^Δ ≈ 0.94^Δ` — a ~1/16-step half-life over ~11 steps, reaching the head of a 128-step rollout at ~0.1% but hitting the 10 rows before the boundary at 50%+.

### Downstream of GAE

- Line 429–430: `ext_returns`/`int_returns` inherit the corruption at K, and `int_returns` inherit it at all `t <= K`.
- Line 400 `curiosity_rewards /= np.sqrt(reward_rms.var)` — the normalizer statistics (lines 390–398) are computed over all rows including the burned ones, so `novelty(s_new)` samples bias the intrinsic reward scale. Small (1–4% of samples) but nonzero.
- Line 447 `b_inds = np.arange(args.batch_size)` — every row, burned included, enters the policy loss (490–492), both value losses (496–509), and the RND predictor loss (463–472).
- Line 444 `obs_rms.update(b_obs[...])` — includes `s_last`, which is a real observation, so this is fine.

### A separate pre-existing shift, worth flagging

Line 449 builds the RND predictor's training input from `b_obs`, i.e. the **current** observations `{o_0 … o_{T-1}}`:

```python
449	        rnd_next_obs = (
450	            (
451	                (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
```

but `curiosity_rewards[t]` at line 373 was computed on the **next** observations `{o_1 … o_T}`. The variable is even named `rnd_next_obs` while holding current observations. The two sets differ only at the window endpoints, so the practical effect is small — but it is a genuine inconsistency independent of the envpool bug, and it is *not* what I am proposing to fix.

### Minor bookkeeping consequences

- `global_step += 1 * args.num_envs` (line 345) counts burned rows as environment steps, overstating throughput by the burn fraction (~1–4%).
- `RecordEpisodeStatistics.step` (lines 113–128) does `self.episode_lengths += 1` on every call including burned ones, so `info["l"]` overstates episode length by the number of intra-episode burns. **VERIFIED** on Breakout: a game-over at call 124 had `elapsed_step=120` with 4 prior life-loss burns; the wrapper would report 124. `info["r"]` is unaffected because the burned row's reward is 0.

---

## 5. The fix

### Ranked candidates

| # | Candidate | Verdict |
|---|---|---|
| **1** | **Mask burned rows out of the batch + stop the intrinsic GAE carry chaining through them** | **Recommended** |
| 2 | Mask the batch only, leave GAE alone | Partial — leaves the intrinsic backward leak |
| 3 | Per-environment compaction of the rollout before GAE | Most faithful, far too invasive |
| 4 | Detect the reset step via `info["elapsed_step"] == 0` | **Broken** — VERIFIED, see below |
| 5 | Switch to `env_type="gymnasium"` | **No effect** — VERIFIED identical timing |
| 6 | Upgrade envpool | **No effect** — VERIFIED identical through 1.2.5, and it breaks the API |

**Why #4 is broken, concretely.** With `episodic_life=True` — the script's own setting at line 281 — a life-loss `done` burns a step but does **not** reset `elapsed_step`, because the underlying game did not reset. VERIFIED on Breakout over 600 steps:

```
burned rows (ground truth, dones[t]==1):            24
  detected by info['elapsed_step']==0 on that row:  4   (misses 20)
  false positives from elapsed_step==0:             0
  rows where info['terminated']==1 (true game over): 4  <- only game overs, not life losses
```

`elapsed_step == 0` finds **4 of 24** burned rows. `info["terminated"]` is the true-game-over flag and finds the same 4. `info["players"]` carries only `env_id` and is useless here. Neither info field can do this job. **`dones[t]` is the only correct detector**, and it costs nothing because the script already stores it.

**What #3 would buy and why I am not recommending it.** Dropping burned rows *before* GAE, per environment, would let row K−1 chain directly to row K+1 with the exact non-episodic intrinsic return. But rows-per-env then vary, so the rectangular `(num_steps, num_envs)` buffers must become ragged — a rewrite of the storage, GAE, and flattening. Not worth it for 1–4% of rows.

**What could go wrong with #1.** (a) The minibatch size becomes update-dependent (`valid_batch_size // num_minibatches`), so gradient-noise scale varies slightly between updates; for `batch_size=16384` and a ~1% drop this is a ±160-sample wobble. (b) The last partial minibatch is dropped when `valid_batch_size` is not divisible by `num_minibatches` — the same as upstream behaviour when it *is* divisible, but now up to `num_minibatches-1` extra rows can go unused per epoch. Both are acceptable; if you need fixed shapes (JIT/XLA), use the variant in the note at the end.

---

### Edit 1 of 2 — GAE loop

**BEFORE** (lines 410–428):

```python
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
```

**AFTER:**

```python
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                # the intrinsic return is deliberately non-episodic (Burda et al.), so no terminal mask here
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues - int_values[t]
                ext_advantages[t] = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_delta + args.int_gamma * args.gae_lambda * int_lastgaelam
                # envpool spends one extra step call auto-resetting after every episode end; that row's
                # action was discarded, so it is not a transition. dones[t] == 1 marks exactly those rows.
                # Pass the incoming carry straight through them, so row t-1 chains to row t+1 as if the
                # burned row were not there -- otherwise the fabricated advantage propagates backward
                # through the whole rollout in the intrinsic stream, which has no terminal mask to stop it.
                burned = dones[t]
                ext_lastgaelam = burned * ext_lastgaelam + (1.0 - burned) * ext_advantages[t]
                int_lastgaelam = burned * int_lastgaelam + (1.0 - burned) * int_advantages[t]
```

`int_nextnonterminal` is deleted because it was the constant `1.0`. The extrinsic carry-through line is a no-op in effect (row K−1 already multiplies by `ext_nextnonterminal = 0`) and is kept only so the two streams read the same.

### Edit 2 of 2 — batch index construction

**BEFORE** (lines 446–447):

```python
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
```

**AFTER:**

```python
        # Optimizing the policy and value network.
        # Drop the burned auto-reset rows from the batch: envpool discarded their action, so
        # (b_obs, b_actions, b_logprobs, b_*_returns, b_*_advantages) at those indices describe a
        # transition that never happened.
        # before: b_inds = [0, 1, 2, 3, 4, ...]                      (every row of the flattened batch)
        # after:  b_inds = [0, 1, 3, 4, ...] with the burned rows removed (~1-4% of the batch)
        b_inds = torch.nonzero(dones.reshape(-1) == 0, as_tuple=False).squeeze(-1).cpu().numpy()
        valid_batch_size = len(b_inds)
        minibatch_size = valid_batch_size // args.num_minibatches
```

and the loop bounds (lines 459–460):

```python
            for start in range(0, args.batch_size, args.minibatch_size):    # BEFORE
                end = start + args.minibatch_size
```
```python
            for start in range(0, valid_batch_size, minibatch_size):        # AFTER
                end = start + minibatch_size
```

`rnd_next_obs` (line 449) is built over the full flattened batch and indexed by `mb_inds`, which remain **global** indices — so it needs no change. `obs_rms.update` at line 444 keeps all rows, which is correct: `s_last` is a genuine observation.

### Semantic note, stated honestly

The original `openai/random-network-distillation` runs on baselines' same-step-reset VecEnv, where `s_last` is discarded and the terminal transition's intrinsic reward is `novelty(s_new)`. After this fix, cleanrl keeps `novelty(s_last)` at row K−1 and discards `novelty(s_new)` entirely. That is a deliberate difference from the original, and I would argue the better one — `s_last` is a state reached by a real action, `s_new` is reached by no action at all — but it is a difference, not exact parity, and I am not claiming it reproduces the original's numbers.

### VERIFIED end to end

Patched file: `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/ppo_rnd_envpool_fixed.py`

Both the upstream script and the patched script were run on CPU under the exact pinned stack (`/u/sl5nw/.conda/envs/cleanrl_uv`: torch 2.4.1, envpool 0.6.6, gym 0.23.1, tyro 1.0.5). Both complete four updates. On `Breakout-v5` (`--num-envs 16 --num-steps 128`) an instrumented build reports:

```
[fix] batch_size=2048 kept=2016 dropped=32 (1.56%) minibatch=504
```

---

## 6. How to verify the fix — CPU-only deterministic test

Test file: `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/test_envpool_autoreset.py`
**6 tests, all passing on envpool 0.6.6 and 1.2.5. No GPU, no torch, no network, 1.8 s.**

The substrate is `Breakout-v5` with `max_episode_steps=4`, `noop_max=1`, `repeat_action_probability=0.0`, `episodic_life=False` — fully deterministic, so every reset returns the *same* observation and the whole sequence is exactly periodic.

> Gotcha found the hard way: **`noop_max=0` segfaults envpool 1.2.5** on `reset()`. Use `noop_max=1`.

1. **`test_autoreset_is_next_step_and_the_burned_step_is_inert`** — asserts `elapsed_step == [1,2,3,4,0]*3` and `done == [F,F,F,T,F]*3`, and that rows 4/9/14 carry reward exactly `0.0`.
2. **`test_exact_observation_sequence_repeats_with_the_burn_included`** — the exact-observation-sequence assertion. Hashes the newest frame of each observation and asserts the sequence is periodic with period `max_episode_steps + 1` (**not** `+0`), that `s_last != s_new`, and that `s_new` equals the observation a fresh `reset()` returns. This is the off-by-one made directly visible.
3. **`test_the_action_on_the_burned_step_is_discarded`** — CartPole (an action moves the cart on the very next observation). Two same-seed pools fed identical actions except at the burned index must stay identical; a control injecting at a real index must diverge. Without the control the test would pass vacuously.
4. **`test_dones_mask_selects_exactly_the_burned_rows`** — replays cleanrl's storage layout and asserts the burned rows are exactly `[4, 9, 14, …]`, that each has `rewards[t] == 0.0`, and that `obs[t]` equals the previous call's returned observation.
5. **`test_carry_through_stops_the_burned_row_contaminating_the_intrinsic_advantages`** — pure numpy, no env. Perturbs the burned row's intrinsic reward and asserts: unfixed, indices `0..burn_at` all move; fixed, **only** `[burn_at]` moves — and that row is dropped from the batch.
6. **`test_batch_mask_drops_every_burned_row`** — asserts the `b_inds` construction keeps exactly the `dones == 0` rows in flattened `(step, env)` order.

Beyond the unit tests, the empirical check I would run before trusting a training result: log `dropped / batch_size` per update. It should sit at roughly `num_envs * num_steps / mean_episode_length`; if it is 0 the mask is not firing, and if it exceeds ~10% something is wrong with the env config.

---

## 7. Artifacts

All under `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/`:

- `ppo_rnd_envpool_fixed.py` — the patched script (diff shown above)
- `test_envpool_autoreset.py` — the 6-test suite
- `probe_both_versions.py` — the three-claim proof, runs on either envpool version
- `probe_episodic_life.py` — the `episodic_life` / `elapsed_step` finding
- `probe_gae_leak.py` — the intrinsic-advantage leak measurement
- `envpool_test_venv/` — python 3.11 + envpool 1.2.5
- `ep066_venv/` — python 3.10 + envpool 0.6.6 + gym 0.23.1 + numpy 1.26.4 (cleanrl's pinned stack)

**What I did not verify:** whether the bug measurably changes final MontezumaRevenge return. The upstream doc claims "it does not seem to impact performance"; confirming or refuting that needs a ~250-hour benchmark run, which I did not do. Everything else above is reproduced from running code.