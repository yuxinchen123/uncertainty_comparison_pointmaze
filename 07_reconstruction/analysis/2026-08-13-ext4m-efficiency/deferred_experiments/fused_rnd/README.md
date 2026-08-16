# Deferred: fuse the RND bonus forward with the predictor-update forward

**Status: implemented here, deliberately NOT applied.** It changes the trained predictor, so every
number recorded by run 5, run 6 and the 1M sweep would stop being comparable. It belongs to the
first sweep that is allowed to start a fresh baseline.

## What it does

Per gradient step the sweep runs the RND nets over the same 256-row batch twice:

```
VectorIntrinsicReplayBuffer.sample()
   -> RND.compute(samples)   # _raw_bonus: no-grad forward of predictor + target  (the reward)
   -> RND.update(samples)    # obs_rms.update(x); forward of predictor + target;  (the loss)
                             # backward; opt.step()
```

`fused_rnd.py` computes the forward **once**, with grad, and reads the bonus off the same tensors
(`.detach()`), then backpropagates the loss through them. One predictor forward and one target
forward per step disappear.

Measured slice to attack: the RND compute+update block is ~5.0 ms of a ~30.5 ms step (16%,
`analysis/2026-06-25-run-profiling/analysis.md` §0); the redundant half of it is worth roughly 3-5%
end to end for the two RND configurations (it does nothing for `gtposvel`, which has no nets).

## Why it cannot be applied under the current constraint

The two forwards are not the same computation. `RND.update()` refreshes the observation statistics
**before** normalizing its input:

```python
# methods/rnd.py:630-633
x = self._get_feature_tensor(samples)
if self.use_obs_norm and self.obs_rms is not None:
    self.obs_rms.update(x.detach().cpu().numpy())     # <-- statistics move here
x = self._normalize_obs(x)                            # <-- so this normalization differs from compute()'s
```

So `compute()` sees the pre-update statistics and `update()` the post-update ones. Fusing forces one
of two choices, and both change the predictor that gets trained:

| choice | consequence |
|---|---|
| **(a) bonus-time normalization for both** (what `fused_rnd.py` implements) | the predictor trains on the pre-update normalization — a one-batch lag versus today |
| (b) move `obs_rms.update()` ahead of the bonus | the *bonus* changes instead, and with it the reward the agent sees |

Choice (a) is the smaller change (the statistics lag by exactly one batch, and they converge within
the first few thousand steps anyway), which is why it is the one implemented.

## How to try it in a future run

```bash
EFF_PATCHES=fused_rnd PYTHONPATH=<this dir> \
  python analysis/code/patched_entry.py <the usual train4m args>
```

`patched_entry.py` does not know this patch — that is intentional. Wiring it in is the first step of
adopting it, and it must happen together with a fresh baseline, not inside a running sweep.

Expect `run_conditions.py` to report `DIFFERS at record.train_episode_history[...]` — that is the
correct outcome here, not a bug.
