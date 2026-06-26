#!/usr/bin/env python3
"""
Fully-JAX RND (rnd_state) — flax predictor/target MLPs + optax Adam + running obs-norm, jitted.

Mirrors the torch RND in src/rnd_exploration/methods/rnd.py for feature="rnd_state", n_predictors=1,
distance="mse", beta_std=0 (the Train-run-2 config), so Train run 4 is the SAME algorithm with the RND
folded into JAX (no torch in the per-step intrinsic path). The math, matched line-for-line:
  net: Linear(in,256) -> ReLU -> Linear(256,128), orthogonal init std=sqrt(2)  (== ObservationEncoder)
  normalize: clip((x-mean)/sqrt(var+1e-8), -5, 5)                               (== _normalize_obs)
  intrinsic = 0.5 * sum((target(nx) - predictor(nx))^2, axis=-1)               (== _dist_ensemble mse, n=1)
  update loss = mean(intrinsic); Adam on predictor only; target frozen          (== update())
  obs_rms = running mean/var over the visited features                         (== RunningMeanStd)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as fnn
import optax
import numpy as np


class _RNDNet(fnn.Module):
    """2-layer MLP encoder (== torch ObservationEncoder, non-linear mode): in -> 256 -> relu -> output_dim."""
    output_dim: int = 128

    @fnn.compact
    def __call__(self, x):
        ortho = fnn.initializers.orthogonal(np.sqrt(2.0))
        x = fnn.relu(fnn.Dense(256, kernel_init=ortho)(x))
        x = fnn.Dense(self.output_dim, kernel_init=ortho)(x)
        return x


class JaxRND:
    """Fully-JAX RND matching rnd.py(rnd_state). compute(x)->intrinsic; update(x) trains the predictor."""

    def __init__(self, input_dim: int, output_dim: int = 128, lr: float = 1e-3, seed: int = 0):
        # two independently-initialized MLPs: target (frozen) + predictor (trained)
        net = _RNDNet(output_dim)
        kt, kp = jax.random.split(jax.random.PRNGKey(seed))
        dummy = jnp.zeros((1, input_dim), jnp.float32)
        self.target_params = net.init(kt, dummy)
        self.pred_params = net.init(kp, dummy)
        opt = optax.adam(lr)
        self.opt_state = opt.init(self.pred_params)
        # running obs-norm state (numpy, like gymnasium RunningMeanStd: mean/var/count over visited features)
        self.input_dim = input_dim
        self.mean = np.zeros(input_dim, np.float64)
        self.var = np.ones(input_dim, np.float64)
        self.count = 1e-4

        # jitted intrinsic + predictor-update steps (net + opt are closed over -> static)
        def _compute(pred_params, target_params, nx):
            src = net.apply(pred_params, nx)
            tgt = net.apply(target_params, nx)
            return 0.5 * jnp.sum((tgt - src) ** 2, axis=-1)  # (batch,) == _dist_ensemble mse, n_predictors=1

        def _update(pred_params, opt_state, target_params, nx):
            def loss_fn(pp):
                src = net.apply(pp, nx)
                tgt = net.apply(target_params, nx)
                return jnp.mean(0.5 * jnp.sum((tgt - src) ** 2, axis=-1))
            loss, grads = jax.value_and_grad(loss_fn)(pred_params)
            updates, opt_state = opt.update(grads, opt_state)
            return optax.apply_updates(pred_params, updates), opt_state, loss

        self._compute = jax.jit(_compute)
        self._update = jax.jit(_update)

    def _normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        # (x - mean) / sqrt(var + 1e-8), clipped to [-5, 5] (== torch _normalize_obs)
        mean = jnp.asarray(self.mean, jnp.float32)
        std = jnp.sqrt(jnp.asarray(self.var, jnp.float32) + 1e-8)
        return jnp.clip((x - mean) / std, -5.0, 5.0)

    def _update_rms(self, x: np.ndarray) -> None:
        # parallel-variance (Chan) running mean/var update, matching gymnasium RunningMeanStd.update
        # before: self.mean/var/count summarize all features seen; x = this batch (batch, input_dim)
        # after:  mean/var/count include this batch
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = M2 / tot
        self.count = tot

    def _compute_array(self, x_np: np.ndarray) -> np.ndarray:
        """Intrinsic reward for already-extracted features x_np (batch, input_dim) -> (batch,) numpy."""
        nx = self._normalize(jnp.asarray(x_np, jnp.float32))
        return np.asarray(self._compute(self.pred_params, self.target_params, nx))

    def _update_array(self, x_np: np.ndarray) -> None:
        """Update obs-norm + train the predictor one Adam step on already-extracted features x_np."""
        self._update_rms(np.asarray(x_np, np.float64))
        nx = self._normalize(jnp.asarray(x_np, jnp.float32))
        self.pred_params, self.opt_state, _ = self._update(self.pred_params, self.opt_state, self.target_params, nx)

    # ---- samples-dict interface: EXACTLY the torch RND's compute(samples)/update(samples) (feature=rnd_state
    #      uses the 'observations' field), so SACWithIntrinsic, the env wrapper, AND the distance metric
    #      (compute_intrinsic_vector_distance -> intrinsic_reward_model.compute(samples)) all call it the same way.
    def compute(self, samples: dict) -> np.ndarray:
        """Intrinsic for a samples dict; rnd_state uses the 'observations' field (== torch RND.compute)."""
        return self._compute_array(np.asarray(samples["observations"], np.float32))

    def update(self, samples: dict) -> None:
        """Train obs-norm + predictor on the samples dict's 'observations' field (== torch RND.update)."""
        self._update_array(np.asarray(samples["observations"], np.float32))

    def warmup_obs_rms(self, observation_space, n: int = 200) -> None:
        """Seed obs_rms from n sampled observations (== run-2's _warmup_obs_rms for feature=observations).

        Run-2's torch RND seeds its running obs-normalization at construction by drawing 200
        observation_space.sample() and folding them into obs_rms; without this the first few hundred gradient
        steps normalize differently (and, scaled by beta, shape early exploration differently). Match it.
        before: obs_rms = mean 0 / var 1 / count 1e-4. after: obs_rms = the obs-space box statistics."""
        # draw n observations from the env's flat observation space and fold them into the running obs stats
        obs = np.stack([np.asarray(observation_space.sample(), np.float32) for _ in range(n)], axis=0)
        self._update_rms(obs.astype(np.float64))
