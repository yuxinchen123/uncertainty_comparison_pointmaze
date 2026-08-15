"""Batched multi-copy PPO+RND in JAX — the exact algorithmic twin of
../torch_ppo/torch_ppo_rnd.py per ../research/ppo_rnd_algorithm_spec.md (including the
section-11 correction: style A keeps the ratio and drops only the clip machinery).

C independent copies advance in lockstep: every parameter carries a leading copy axis, every
forward is a batched matmul, every reduction keeps the copy axis, the scalar loss is the SUM
over copies, gradient clipping is per copy, and Adam (hand-rolled, torch-formula-exact) is
elementwise so one Adam is C independent Adams. Running statistics are float64 (x64 enabled);
everything else is explicitly float32. The two update styles are two separately jitted
functions with no traced branch.
"""
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)   # float64 running statistics (spec 4.2)

import jax.numpy as jnp  # noqa: E402

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "jax_env"))
from pm_common import EnvConfig  # noqa: E402
from jax_pointmaze import EnvState, JaxPointMaze  # noqa: E402

LOG2PI = 1.8378770664093453
F32 = jnp.float32


@dataclass(frozen=True)
class PPOConfig:
    """All trainer knobs (spec section 15) — mirrors the torch PPOConfig field for field."""
    n_copies: int = 8
    n_envs: int = 4
    num_steps: int = 128
    update_style: str = "epoch_minibatch"  # or "full_batch"
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
    int_coef: float = 1.0
    ext_coef: float = 2.0
    rnd_feature_dim: int = 128
    rnd_hidden: int = 256
    obs_norm_init_iters: int = 10
    bootstrap_on_truncation: bool = True
    base_seed: int = 0
    hoist_rollout: bool = True       # compute the critic values, the log-probability and the
                                     # RND bonus AFTER the rollout scan, in one wide pass each,
                                     # instead of once per step inside it (round 2, J1)
    scan_unroll: int = 1             # unroll factor for the rollout scan (round 2, J2)


class RMSState(NamedTuple):
    """Per-copy running mean/var/count, float64 (gymnasium parallel-variance formula)."""
    mean: jnp.ndarray   # [C, dim]
    var: jnp.ndarray    # [C, dim]
    count: jnp.ndarray  # [C, 1]


class TrainState(NamedTuple):
    """Everything one training iteration reads and writes."""
    params: dict          # trainable pytree: actor / critic / predictor
    opt_m: dict           # Adam first moments, same tree
    opt_v: dict           # Adam second moments, same tree
    opt_t: jnp.ndarray    # Adam step counter, int32 scalar
    env_state: EnvState
    obs: jnp.ndarray      # [C, N, 4] current observation
    obs_rms: RMSState     # RND input statistics, dim 4
    int_rms: RMSState     # intrinsic-return statistics, dim 1
    int_filter: jnp.ndarray  # [C, N] forward filter accumulator


def rms_init(n_copies, dim):
    """Fresh statistics: mean 0, var 1, count 1e-4 (gymnasium epsilon)."""
    return RMSState(jnp.zeros((n_copies, dim), jnp.float64),
                    jnp.ones((n_copies, dim), jnp.float64),
                    jnp.full((n_copies, 1), 1e-4, jnp.float64))


def rms_update(rms: RMSState, batch) -> RMSState:
    """Parallel-variance update per copy; batch [C, B, dim] float32, population variance."""
    b = batch.astype(jnp.float64)
    batch_mean = b.mean(axis=1)
    batch_var = b.var(axis=1, ddof=0)
    batch_count = float(b.shape[1])
    delta = batch_mean - rms.mean
    tot = rms.count + batch_count
    new_mean = rms.mean + delta * batch_count / tot
    m2 = rms.var * rms.count + batch_var * batch_count + delta ** 2 * rms.count * batch_count / tot
    return RMSState(new_mean, m2 / tot, tot)


def _orthogonal(key, rows, cols, gain):
    """Orthogonal [rows, cols] like torch nn.init.orthogonal_ (QR with sign fix), float32."""
    flat = jax.random.normal(key, (max(rows, cols), min(rows, cols)), F32)
    q, r = jnp.linalg.qr(flat)
    q = q * jnp.sign(jnp.diagonal(r))
    if rows < cols:
        q = q.T
    return (gain * q)[:rows, :cols].astype(F32)


class JaxPPORND:
    """Static config + pure jitted functions; mutable state lives in a TrainState."""

    # stable integer ids for the keyed per-copy weight init (rng-seeding rule)
    _NET_IDS = {"actor": 1, "critic": 2, "critic_heads": 3, "rnd_target": 4, "rnd_predictor": 5}

    def __init__(self, cfg: PPOConfig, env_cfg: EnvConfig = None):
        self.cfg = cfg
        self.env = JaxPointMaze(env_cfg or EnvConfig(), cfg.n_copies, cfg.n_envs,
                                base_seed=cfg.base_seed)
        C, F, Hh = cfg.n_copies, cfg.rnd_feature_dim, cfg.rnd_hidden

        def stack(net, layers):
            # per-copy keyed init: copy c's weights identical whether C=8 or C=128
            out = {}
            for li, (rows, cols, gain) in enumerate(layers):
                ws = []
                for c in range(C):
                    k = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
                        jax.random.PRNGKey(cfg.base_seed), self._NET_IDS[net]), li), c)
                    # stored [in, out] for x @ W; orthogonal drawn on [out, in] then transposed
                    ws.append(_orthogonal(k, rows, cols, gain).T)
                out[f"W{li}"] = jnp.stack(ws)
                out[f"b{li}"] = jnp.zeros((C, rows), F32)
            return out

        actor = stack("actor", [(64, 4, 2 ** 0.5), (64, 64, 2 ** 0.5), (2, 64, 0.01)])
        actor["logstd"] = jnp.zeros((C, 2), F32)
        critic = stack("critic", [(64, 4, 2 ** 0.5), (64, 64, 2 ** 0.5)])
        heads = stack("critic_heads", [(1, 64, 1.0), (1, 64, 1.0)])
        critic["Wext"], critic["bext"] = heads["W0"], heads["b0"]
        critic["Wint"], critic["bint"] = heads["W1"], heads["b1"]
        self.target = stack("rnd_target", [(Hh, 4, 2 ** 0.5), (F, Hh, 2 ** 0.5)])
        predictor = stack("rnd_predictor",
                          [(Hh, 4, 2 ** 0.5), (F, Hh, 2 ** 0.5), (F, F, 2 ** 0.5)])
        self.init_params = {"actor": actor, "critic": critic, "predictor": predictor}

        # jitted entry points (style chosen HERE, never branched on inside a trace)
        update = self._update_full_batch if cfg.update_style == "full_batch" \
            else self._update_epoch_minibatch
        # the WHOLE iteration (rollout scan + statistics + GAE + update) is one XLA program;
        # the TrainState argument is donated so params/opt/env buffers are updated in place
        self._iterate = jax.jit(partial(self._iterate_impl, update), donate_argnums=(0,))
        self._prime = jax.jit(self._prime_impl)

    def init_state(self) -> TrainState:
        """Fresh TrainState: keyed weights, zero Adam moments, reset envs."""
        cfg = self.cfg
        zeros_like_tree = jax.tree.map(jnp.zeros_like, self.init_params)
        return TrainState(
            params=self.init_params, opt_m=zeros_like_tree,
            opt_v=jax.tree.map(jnp.zeros_like, self.init_params),
            opt_t=jnp.zeros((), jnp.int32),
            env_state=self.env.reset(),
            obs=jnp.zeros((cfg.n_copies, cfg.n_envs, 4), F32),
            obs_rms=rms_init(cfg.n_copies, 4), int_rms=rms_init(cfg.n_copies, 1),
            int_filter=jnp.zeros((cfg.n_copies, cfg.n_envs), F32),
        )

    # ---- batched forwards (x always [C, M, in]) ----

    @staticmethod
    def _actor_mean(actor, x):
        """Action mean [C, M, 2]."""
        h = jnp.tanh(jnp.matmul(x, actor["W0"]) + actor["b0"][:, None, :])
        h = jnp.tanh(jnp.matmul(h, actor["W1"]) + actor["b1"][:, None, :])
        return jnp.matmul(h, actor["W2"]) + actor["b2"][:, None, :]

    @staticmethod
    def _critic_values(critic, x):
        """(Vext, Vint) each [C, M]."""
        h = jnp.tanh(jnp.matmul(x, critic["W0"]) + critic["b0"][:, None, :])
        h = jnp.tanh(jnp.matmul(h, critic["W1"]) + critic["b1"][:, None, :])
        vext = jnp.matmul(h, critic["Wext"]) + critic["bext"][:, None, :]
        vint = jnp.matmul(h, critic["Wint"]) + critic["bint"][:, None, :]
        return vext[..., 0], vint[..., 0]

    def _rnd_features(self, predictor, x):
        """(target_features stop-grad by construction, predictor_features), each [C, M, F]."""
        th = jax.nn.relu(jnp.matmul(x, self.target["W0"]) + self.target["b0"][:, None, :])
        tf = jnp.matmul(th, self.target["W1"]) + self.target["b1"][:, None, :]
        ph = jax.nn.relu(jnp.matmul(x, predictor["W0"]) + predictor["b0"][:, None, :])
        ph = jax.nn.relu(jnp.matmul(ph, predictor["W1"]) + predictor["b1"][:, None, :])
        pf = jnp.matmul(ph, predictor["W2"]) + predictor["b2"][:, None, :]
        return jax.lax.stop_gradient(tf), pf

    @staticmethod
    def _whiten(obs, obs_rms):
        """RND input whitening, clip +-5 — float64 stats cast to float32 first (torch twin)."""
        mean = obs_rms.mean.astype(F32)[:, None, :]
        std = jnp.sqrt(obs_rms.var + 1e-8).astype(F32)[:, None, :]
        return jnp.clip((obs - mean) / std, -5.0, 5.0)

    @staticmethod
    def _logprob(mean, logstd, action):
        """Diagonal-Gaussian log-density summed over action dims -> [C, M]."""
        z = (action - mean) / jnp.exp(logstd)[:, None, :]
        return (-0.5 * z * z - logstd[:, None, :] - 0.5 * LOG2PI).sum(-1)

    # ---- Adam (torch.optim.Adam formula, elementwise => C independent Adams) ----

    def _adam_step(self, params, grads, m, v, t, lr):
        """One Adam step: m,v update, bias correction, p -= lr/bc1 * m / (sqrt(v/bc2)+eps)."""
        b1, b2, eps = 0.9, 0.999, self.cfg.adam_eps
        t = t + 1
        tf_ = t.astype(F32)
        bc1 = 1.0 - b1 ** tf_
        bc2 = 1.0 - b2 ** tf_
        m = jax.tree.map(lambda mm, g: b1 * mm + (1 - b1) * g, m, grads)
        v = jax.tree.map(lambda vv, g: b2 * vv + (1 - b2) * g * g, v, grads)
        params = jax.tree.map(
            lambda p, mm, vv: p - (lr / bc1) * mm / (jnp.sqrt(vv / bc2) + eps),
            params, m, v)
        return params, m, v, t

    def _clip_per_copy(self, grads):
        """Per-copy gradient-norm clip: norm over each copy's own slice of every tensor."""
        C = self.cfg.n_copies
        leaves = jax.tree.leaves(grads)
        g2 = sum(leaf.reshape(C, -1).astype(F32).__pow__(2).sum(axis=1) for leaf in leaves)
        scale = jnp.minimum(1.0, self.cfg.max_grad_norm / (jnp.sqrt(g2) + 1e-6))
        return jax.tree.map(
            lambda g: g * scale.reshape((C,) + (1,) * (g.ndim - 1)), grads)

    # ---- losses (shared by both styles; the style flag is a PYTHON constant) ----

    def _losses(self, params, mb, style_a):
        """Scalar sum-over-copies loss on one (mini)batch dict (spec 9 + 11 correction)."""
        cfg = self.cfg
        a = mb["adv"]
        a_n = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, ddof=1, keepdims=True) + 1e-8)

        mean = self._actor_mean(params["actor"], mb["obs"])
        logstd = params["actor"]["logstd"]
        newlogp = self._logprob(mean, logstd, mb["actions"])
        vext, vint = self._critic_values(params["critic"], mb["obs"])
        ratio = jnp.exp(newlogp - mb["old_logprob"])

        if style_a:
            pg = (-a_n * ratio).mean(axis=1)
            v_ext = 0.5 * ((vext - mb["ret_ext"]) ** 2).mean(axis=1)
        else:
            pg = jnp.maximum(-a_n * ratio,
                             -a_n * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                             ).mean(axis=1)
            vc = mb["vext_old"] + jnp.clip(vext - mb["vext_old"], -cfg.clip_coef, cfg.clip_coef)
            v_ext = 0.5 * jnp.maximum((vext - mb["ret_ext"]) ** 2,
                                      (vc - mb["ret_ext"]) ** 2).mean(axis=1)
        v_int = 0.5 * ((vint - mb["ret_int"]) ** 2).mean(axis=1)

        tf, pf = self._rnd_features(params["predictor"], mb["rnd_input"])
        fwd = ((pf - tf) ** 2).mean(axis=2).mean(axis=1)

        ent = (0.5 + 0.5 * LOG2PI + logstd).sum(axis=1)
        loss_c = pg - cfg.ent_coef * ent + cfg.vf_coef * (v_ext + v_int) + fwd
        return loss_c.sum()

    # ---- update styles (two separate functions, chosen at build time) ----

    def _update_full_batch(self, state: TrainState, batch, lr, key):
        """Style A: one gradient step on all T*N rows per copy (spec 11 + correction)."""
        loss, grads = jax.value_and_grad(
            lambda p: self._losses(p, batch, style_a=True))(state.params)
        grads = self._clip_per_copy(grads)
        params, m, v, t = self._adam_step(state.params, grads, state.opt_m, state.opt_v,
                                          state.opt_t, lr)
        return state._replace(params=params, opt_m=m, opt_v=v, opt_t=t), loss

    def _update_epoch_minibatch(self, state: TrainState, batch, lr, key):
        """Style B: epochs x minibatches shuffled steps; permutations drawn up front,
        independent per copy AND per epoch; the 16 steps run as one lax.scan."""
        cfg = self.cfg
        C = cfg.n_copies
        Brows = cfg.num_steps * cfg.n_envs
        mb_size = Brows // cfg.num_minibatches
        n_steps = cfg.update_epochs * cfg.num_minibatches

        # permutations (E, C, B) -> minibatch row indices (E*K, C, mb)
        perm = jnp.argsort(
            jax.random.uniform(key, (cfg.update_epochs, C, Brows), F32), axis=-1)
        idx = perm.reshape(cfg.update_epochs, C, cfg.num_minibatches, mb_size)
        idx = idx.transpose(0, 2, 1, 3).reshape(n_steps, C, mb_size)

        # gather all minibatches up front: field (C,B,k) -> (S,C,mb,k); (C,B) -> (S,C,mb)
        def gather(t):
            if t.ndim == 3:
                return jnp.take_along_axis(t[None], idx[..., None], axis=2)
            return jnp.take_along_axis(t[None], idx, axis=2)

        mbs = {k: gather(batch[k]) for k in ["obs", "actions", "old_logprob", "adv",
                                             "ret_ext", "ret_int", "vext_old", "rnd_input"]}

        def body(carry, mb):
            params, m, v, t = carry
            loss, grads = jax.value_and_grad(
                lambda p: self._losses(p, mb, style_a=False))(params)
            grads = self._clip_per_copy(grads)
            params, m, v, t = self._adam_step(params, grads, m, v, t, lr)
            return (params, m, v, t), loss

        (params, m, v, t), losses = jax.lax.scan(
            body, (state.params, state.opt_m, state.opt_v, state.opt_t), mbs)
        return state._replace(params=params, opt_m=m, opt_v=v, opt_t=t), losses[-1]

    # ---- one full iteration (rollout + statistics + GAE + update), jitted once ----

    def _iterate_impl(self, update_fn, state: TrainState, key, lr):
        """rollout T steps -> intrinsic filter -> GAE -> batch -> update (spec 6, 7, 11/12)."""
        cfg = self.cfg
        C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
        params = state.params
        obs_rms_old = state.obs_rms
        z_all = jax.random.normal(jax.random.fold_in(key, 0), (T, C, N, 2), F32)

        logstd = params["actor"]["logstd"]
        flat = lambda x: x.transpose(1, 0, 2, *range(3, x.ndim)).reshape(
            C, T * N, *x.shape[3:])
        unflat = lambda x: x.reshape(C, T, N).transpose(1, 0, 2)

        if cfg.hoist_rollout:
            # The scan carries only what is genuinely sequential: the action (which the
            # environment consumes) and the environment step. The critic values, the
            # log-probability and the RND bonus are pure functions of data the scan already
            # stores and of parameters that do not change during a rollout, so they are
            # computed once afterwards over the whole [C, T*N, ...] batch.
            def body(carry, z):
                env_state, obs = carry
                mean = self._actor_mean(params["actor"], obs)
                action = mean + jnp.exp(logstd)[:, None, :] * z
                env_state, next_obs, r_ext, term, trunc, final_obs = self.env.step(
                    env_state, action)
                out = (obs, action, final_obs, r_ext,
                       term.astype(F32), (term | trunc).astype(F32))
                return (env_state, next_obs), out

            (env_state, obs_last), outs = jax.lax.scan(
                body, (state.env_state, state.obs), z_all, unroll=cfg.scan_unroll)
            obs_buf, act_buf, nobs_buf, rext_buf, term_buf, done_buf = outs

            # log-probability of the sampled actions depends only on the noise and on logstd
            logp_buf = (-0.5 * z_all * z_all - logstd[None, :, None, :] - 0.5 * LOG2PI).sum(-1)
            # one critic pass covers the on-step values and the bootstrap values together
            obs_flat, nobs_flat = flat(obs_buf), flat(nobs_buf)
            vall_e, vall_i = self._critic_values(
                params["critic"], jnp.concatenate([obs_flat, nobs_flat], axis=1))
            vext_buf, vint_buf = unflat(vall_e[:, :T * N]), unflat(vall_i[:, :T * N])
            vext_next, vint_next = unflat(vall_e[:, T * N:]), unflat(vall_i[:, T * N:])
            # intrinsic bonus with the statistics as they stood at the START of the iteration
            tf_old, pf_old = self._rnd_features(
                params["predictor"], self._whiten(nobs_flat, obs_rms_old))
            rint_buf = unflat(0.5 * ((pf_old - tf_old) ** 2).sum(-1))
        else:
            # round-1 form, kept so the hoist can be measured and checked against it
            def body(carry, z):
                env_state, obs = carry
                vext, vint = self._critic_values(params["critic"], obs)
                mean = self._actor_mean(params["actor"], obs)
                action = mean + jnp.exp(logstd)[:, None, :] * z
                logp = (-0.5 * z * z - logstd[:, None, :] - 0.5 * LOG2PI).sum(-1)
                env_state, next_obs, r_ext, term, trunc, final_obs = self.env.step(
                    env_state, action)
                rnd_in = self._whiten(final_obs, obs_rms_old)
                tf, pf = self._rnd_features(params["predictor"], rnd_in)
                r_int = 0.5 * ((pf - tf) ** 2).sum(-1)
                out = (obs, action, logp, vext, vint, final_obs, r_ext,
                       term.astype(F32), (term | trunc).astype(F32), r_int)
                return (env_state, next_obs), out

            (env_state, obs_last), outs = jax.lax.scan(
                body, (state.env_state, state.obs), z_all, unroll=cfg.scan_unroll)
            (obs_buf, act_buf, logp_buf, vext_buf, vint_buf, nobs_buf, rext_buf,
             term_buf, done_buf, rint_buf) = outs
            nobs_flat = flat(nobs_buf)
            vext_next, vint_next = self._critic_values(params["critic"], nobs_flat)
            vext_next, vint_next = unflat(vext_next), unflat(vint_next)

        # intrinsic filter forward in time, then per-copy normalization (spec 5.2)
        def filt_body(f, r):
            f = cfg.gamma_int * f + r
            return f, f
        int_filter, filt = jax.lax.scan(filt_body, state.int_filter, rint_buf)
        int_rms = rms_update(state.int_rms, filt.transpose(1, 0, 2).reshape(C, T * N, 1))
        int_std = jnp.sqrt(int_rms.var + 1e-8).astype(F32).reshape(1, C, 1)
        rint_hat = rint_buf / int_std

        # two-stream GAE backwards (spec 7)
        boot_mask = term_buf if cfg.bootstrap_on_truncation else done_buf

        def gae_body(carry, xs):
            aext, aint = carry
            r_e, v_e, vn_e, r_i, v_i, vn_i, bm, dn = xs
            d_ext = r_e + cfg.gamma_ext * vn_e * (1 - bm) - v_e
            aext = d_ext + cfg.gamma_ext * cfg.gae_lambda * (1 - dn) * aext
            d_int = r_i + cfg.gamma_int * vn_i - v_i
            aint = d_int + cfg.gamma_int * cfg.gae_lambda * aint
            return (aext, aint), (aext, aint)

        zero = jnp.zeros((C, N), F32)
        _, (aext_buf, aint_buf) = jax.lax.scan(
            gae_body, (zero, zero),
            (rext_buf, vext_buf, vext_next, rint_hat, vint_buf, vint_next,
             boot_mask, done_buf), reverse=True)

        adv = cfg.int_coef * aint_buf + cfg.ext_coef * aext_buf
        ret_ext = aext_buf + vext_buf
        ret_int = aint_buf + vint_buf

        # update RND observation statistics, then whiten the update batch with NEW statistics
        obs_rms = rms_update(obs_rms_old, nobs_flat)
        batch = {
            "obs": flat(obs_buf), "actions": flat(act_buf), "old_logprob": flat(logp_buf),
            "adv": flat(adv), "ret_ext": flat(ret_ext), "ret_int": flat(ret_int),
            "vext_old": flat(vext_buf), "rnd_input": self._whiten(nobs_flat, obs_rms),
        }
        state = state._replace(env_state=env_state, obs=obs_last, obs_rms=obs_rms,
                               int_rms=int_rms, int_filter=int_filter)
        state, loss = update_fn(state, batch, lr, jax.random.fold_in(key, 1))
        metrics = {"loss": loss, "reward_ext_sum": rext_buf.sum(axis=(0, 2)),
                   "rint_mean": rint_buf.mean(axis=(0, 2))}
        return state, metrics

    def _prime_impl(self, state: TrainState, key):
        """Spec 4.3: one priming iteration — T random-action steps, update obs stats only."""
        cfg = self.cfg
        C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
        acts = jax.random.uniform(key, (T, C, N, 2), F32, -1.0, 1.0)

        def body(carry, a):
            env_state = carry
            env_state, _, _, _, _, final_obs = self.env.step(env_state, a)
            return env_state, final_obs

        env_state, nobs = jax.lax.scan(body, state.env_state, acts)
        obs_rms = rms_update(state.obs_rms, nobs.transpose(1, 0, 2, 3).reshape(C, T * N, 4))
        return state._replace(env_state=env_state, obs_rms=obs_rms)

    # ---- driver ----

    def prime_obs_rms(self, state: TrainState, key) -> TrainState:
        """Run obs_norm_init_iters priming iterations, then respawn envs at the carried
        reset generation with step counts zeroed (mirrors torch env.reset semantics)."""
        for i in range(self.cfg.obs_norm_init_iters):
            state = self._prime(state, jax.random.fold_in(key, i))
        rc = state.env_state.reset_count
        pos, vel, goal = self.env._spawn(rc)
        env_state = EnvState(pos, vel, goal, jnp.zeros_like(state.env_state.step_count), rc)
        return state._replace(env_state=env_state,
                              obs=jnp.concatenate([pos, vel], -1))

    def train(self, num_iterations, run_seed=0, log_every_seconds=1200, log_fn=print):
        """Full loop with sparse logging; returns final stats."""
        cfg = self.cfg
        state = self.init_state()
        key = jax.random.PRNGKey(run_seed + 1_000_003 * cfg.base_seed)
        state = self.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
        t0 = time.time()
        last_log = t0
        for it in range(1, num_iterations + 1):
            lr = cfg.learning_rate * (1.0 - (it - 1.0) / num_iterations) \
                if cfg.anneal_lr else cfg.learning_rate
            state, metrics = self._iterate(state, jax.random.fold_in(key, it),
                                           jnp.asarray(lr, F32))
            now = time.time()
            if now - last_log >= log_every_seconds or it == num_iterations:
                jax.block_until_ready(metrics)
                last_log = now
                r = np.asarray(metrics["reward_ext_sum"])
                log_fn(f"iter {it}/{num_iterations} loss {float(metrics['loss']):.4f} "
                       f"ext-reward/copy mean {r.mean():.3f} min {r.min():.3f} "
                       f"max {r.max():.3f} rint mean "
                       f"{float(np.asarray(metrics['rint_mean']).mean()):.4f} "
                       f"elapsed {now - t0:.0f}s")
        jax.block_until_ready(state.params)
        return state, {"iterations": num_iterations,
                       "global_step": num_iterations * cfg.num_steps * cfg.n_copies * cfg.n_envs,
                       "seconds": time.time() - t0}


def main():
    """Smoke / bench entry."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--style", default="epoch_minibatch")
    args = ap.parse_args()
    trainer = JaxPPORND(PPOConfig(n_copies=args.n_copies, update_style=args.style))
    _, stats = trainer.train(args.iters, log_every_seconds=0)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
