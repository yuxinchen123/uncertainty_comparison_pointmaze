"""One training iteration, built for exactly one (environment, agent, bonus) combination.

Nothing here is chosen at run time. The update style, the rollout shape, the bonus and whether
coverage is tracked are all Python decisions taken while the function is being built, so the
compiled program holds one of each and no branch that picks between them.

The iteration: roll out T steps, score the states the rollout reached, filter and normalise that
intrinsic reward, compute both advantage streams backwards, assemble the update batch, and take
the update.
"""
import jax
import jax.numpy as jnp

from .. import F32, LOG2PI
from ..agents.ppo.losses import ppo_loss_per_copy, ppo_loss_per_copy_discrete
from ..agents.ppo.networks import actor_logits, actor_mean, critic_values, logprob_discrete
from ..agents.ppo.update import build_update
from ..statistics import rms_update
from .state import named_params, put_params, stored_params


def build_total_loss(cfg, bonus, layout, discrete: bool = False):
    """The scalar the optimizer minimises: the agent's per-copy loss plus the bonus's, summed.

    Adding the two [C] vectors before the sum, rather than summing each separately, is what keeps
    this identical to the single-module trainer down to the last bit. `discrete` picks the
    categorical actor's loss — a Python decision taken when the program is built.
    """
    agent_loss = ppo_loss_per_copy_discrete if discrete else ppo_loss_per_copy

    def total_loss(params, batch, style_a):
        """Sum over copies of (agent loss + bonus loss) on one (mini)batch."""
        # in the one-array form the gradient is taken with respect to the flat array, so the
        # named tensors the networks expect are cut out of it here
        named = layout.unpack(params) if layout.packed else params
        loss_c = agent_loss(cfg, named["agent"], batch, style_a)
        return (loss_c + bonus.loss(named["bonus"], batch)).sum()
    return total_loss


def build_iteration(cfg, env, bonus, sweep, layout, capture_batch: bool = False):
    """Return the unjitted iteration function for this configuration.

    capture_batch makes the returned function hand back the update batch as a third result. It
    exists for the test that checks the hoisted rollout against the in-scan one, which has to
    compare the batch the two forms build.
    """
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    discrete = getattr(env, "action_kind", "continuous") == "discrete"

    # how far to unroll the rollout scan. One rollout step does more work the more copies there
    # are, so the point where a longer program stops paying moves with the copy count: at 8
    # copies unrolling 32 steps beat 16 in 11 of 11 paired rounds, while at 32 and 128 copies 16
    # beat 32 in 11 of 11. Unrolling does not change the arithmetic, and the parameters after one
    # iteration are bit-identical at 4, 16 and 32 — EXCEPT with the rollout hoist off, where the
    # in-scan critic and log-probability fuse differently at 32 (4 against 32 deviates 3.9e-04,
    # 4 against 16 stays exact). So 32 is taken only where it is bit-neutral. A value given
    # explicitly is used as given.
    scan_unroll = cfg.scan_unroll or (32 if cfg.n_copies <= 8 and cfg.hoist_rollout else 16)

    if not cfg.hoist_rollout and bonus.rollout_step is None:
        raise ValueError(
            f"the {bonus.name} bonus cannot score one rollout step at a time, so it cannot be "
            "composed with hoist_rollout=False; leave the hoist on")

    update_fn = build_update(cfg, build_total_loss(cfg, bonus, layout, discrete),
                             sweep.lr_per_copy)

    def iteration(state, lr):
        """Roll out, score, form advantages, update; returns (state, metrics)."""
        # the iteration's key comes from the state, so the driver does not have to keep a
        # counter beside it: iteration k folds k into the run's key, exactly as before
        step = state.step + 1
        key = jax.random.fold_in(state.rng, step)

        # the rollout reads the networks by name, so the one-array form is cut apart once here
        params = named_params(stored_params(state, layout), layout)
        agent_params, bonus_params = params["agent"], params["bonus"]
        bonus_state_old = state.bonus_state
        # the per-step action noise: Gaussian for a continuous actor, Gumbel for a discrete
        # one (argmax of logits + Gumbel draws exactly the categorical distribution)
        if discrete:
            u = jax.random.uniform(jax.random.fold_in(key, 0), (T, C, N, env.act_dim), F32,
                                   minval=1e-7, maxval=1.0 - 1e-7)
            z_all = -jnp.log(-jnp.log(u))
            logstd = None
        else:
            z_all = jax.random.normal(jax.random.fold_in(key, 0), (T, C, N, env.act_dim), F32)
            logstd = agent_params["actor"]["logstd"]
        # before: a rollout buffer [T, C, N, k]; after: [C, T*N, k], one row per copy per step
        flat = lambda x: x.transpose(1, 0, 2, *range(3, x.ndim)).reshape(C, T * N, *x.shape[3:])
        # before: [C, T*N]; after: [T, C, N] — the inverse of flat for a scalar-per-step field
        unflat = lambda x: x.reshape(C, T, N).transpose(1, 0, 2)

        if cfg.hoist_rollout:
            # The scan carries only what is genuinely sequential: the action (which the
            # environment consumes) and the environment step. The critic values, the
            # log-probability and the intrinsic reward are pure functions of data the scan
            # already stores and of parameters that do not change during a rollout, so they are
            # computed once afterwards over the whole [C, T*N, ...] batch. A discrete actor's
            # log-probability needs its logits, so it is taken inside the scan where the
            # logits already exist, at the cost of one cheap gather per step.
            if discrete:
                def body(carry, z):
                    """One rollout step: sample from the logits, step, store what the rest needs."""
                    env_state, obs = carry
                    logits = actor_logits(agent_params["actor"], obs)
                    action = jnp.argmax(logits + z, axis=-1).astype(jnp.int32)
                    logp = logprob_discrete(logits, action)
                    env_state, next_obs, r_ext, term, trunc, final_obs = env.step(env_state,
                                                                                  action)
                    out = (obs, action, logp, final_obs, r_ext,
                           term.astype(F32), (term | trunc).astype(F32))
                    return (env_state, next_obs), out

                (env_state, obs_last), outs = jax.lax.scan(
                    body, (state.env_state, state.obs), z_all, unroll=scan_unroll)
                obs_buf, act_buf, logp_buf, nobs_buf, rext_buf, term_buf, done_buf = outs
            else:
                def body(carry, z):
                    """One rollout step: act, step the environment, store what the rest needs."""
                    env_state, obs = carry
                    mean = actor_mean(agent_params["actor"], obs)
                    action = mean + jnp.exp(logstd)[:, None, :] * z
                    env_state, next_obs, r_ext, term, trunc, final_obs = env.step(env_state,
                                                                                  action)
                    out = (obs, action, final_obs, r_ext,
                           term.astype(F32), (term | trunc).astype(F32))
                    return (env_state, next_obs), out

                (env_state, obs_last), outs = jax.lax.scan(
                    body, (state.env_state, state.obs), z_all, unroll=scan_unroll)
                obs_buf, act_buf, nobs_buf, rext_buf, term_buf, done_buf = outs

                # log-probability of the sampled actions depends only on the noise and logstd
                logp_buf = (-0.5 * z_all * z_all - logstd[None, :, None, :]
                            - 0.5 * LOG2PI).sum(-1)
            # one critic pass covers the on-step values and the bootstrap values together
            obs_flat, nobs_flat = flat(obs_buf), flat(nobs_buf)
            vall_e, vall_i = critic_values(
                agent_params["critic"], jnp.concatenate([obs_flat, nobs_flat], axis=1))
            vext_buf, vint_buf = unflat(vall_e[:, :T * N]), unflat(vall_i[:, :T * N])
            vext_next, vint_next = unflat(vall_e[:, T * N:]), unflat(vall_i[:, T * N:])
            scored = None
        else:
            # round-1 form, kept so the hoist can be measured and checked against it
            def body(carry, z):
                """One rollout step, with the critic, the log-probability and the bonus inside."""
                env_state, obs = carry
                vext, vint = critic_values(agent_params["critic"], obs)
                if discrete:
                    logits = actor_logits(agent_params["actor"], obs)
                    action = jnp.argmax(logits + z, axis=-1).astype(jnp.int32)
                    logp = logprob_discrete(logits, action)
                else:
                    mean = actor_mean(agent_params["actor"], obs)
                    action = mean + jnp.exp(logstd)[:, None, :] * z
                    logp = (-0.5 * z * z - logstd[:, None, :] - 0.5 * LOG2PI).sum(-1)
                env_state, next_obs, r_ext, term, trunc, final_obs = env.step(env_state, action)
                r_int = bonus.rollout_step(bonus_params, bonus_state_old, final_obs)
                out = (obs, action, logp, vext, vint, final_obs, r_ext,
                       term.astype(F32), (term | trunc).astype(F32), r_int)
                return (env_state, next_obs), out

            (env_state, obs_last), outs = jax.lax.scan(
                body, (state.env_state, state.obs), z_all, unroll=scan_unroll)
            (obs_buf, act_buf, logp_buf, vext_buf, vint_buf, nobs_buf, rext_buf,
             term_buf, done_buf, rint_scan) = outs
            nobs_flat = flat(nobs_buf)
            vext_next, vint_next = critic_values(agent_params["critic"], nobs_flat)
            vext_next, vint_next = unflat(vext_next), unflat(vint_next)
            scored = flat(rint_scan)

        # the bonus scores the rollout (or takes the scores the scan already made), moves its own
        # state on, and hands over whatever extra fields its loss will need in the update batch
        bonus_state, rint_flat, batch_extra = bonus.post_rollout(
            bonus_params, bonus_state_old, nobs_flat, scored)
        rint_buf = unflat(rint_flat)

        # intrinsic filter forward in time, then per-copy normalization (spec 5.2)
        def filt_body(f, r):
            """The discounted forward filter, one rollout step at a time."""
            f = cfg.gamma_int * f + r
            return f, f
        int_filter, filt = jax.lax.scan(filt_body, state.agent_state["int_filter"], rint_buf)
        int_rms = rms_update(state.agent_state["int_rms"],
                             filt.transpose(1, 0, 2).reshape(C, T * N, 1), cfg.batch_stats_f32)
        int_std = jnp.sqrt(int_rms.var + 1e-8).astype(F32).reshape(1, C, 1)
        rint_hat = rint_buf / int_std

        # two-stream generalized advantage estimation, backwards (spec 7)
        boot_mask = term_buf if cfg.bootstrap_on_truncation else done_buf

        def gae_body(carry, xs):
            """One backwards step of both advantage streams."""
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

        # the weight on the intrinsic advantage: one scalar for every copy, or — when a sweep
        # varies it — one value per copy broadcast along the copy axis of [T, C, N]
        if sweep.beta_per_copy is None:
            adv = cfg.int_coef * aint_buf + cfg.ext_coef * aext_buf
        else:
            adv = sweep.beta_per_copy.reshape(1, C, 1) * aint_buf + cfg.ext_coef * aext_buf
        ret_ext = aext_buf + vext_buf
        ret_int = aint_buf + vint_buf

        batch = {
            "obs": flat(obs_buf), "actions": flat(act_buf), "old_logprob": flat(logp_buf),
            "adv": flat(adv), "ret_ext": flat(ret_ext), "ret_int": flat(ret_int),
            "vext_old": flat(vext_buf), **batch_extra,
        }

        # cumulative visited-cell map, kept on the device so no iteration synchronises; the
        # driver reads it only on the iterations it records
        # before: nobs_flat [C, T*N, obs] world coordinates; after: visited [C, rows*cols] bool
        visited = state.visited
        if cfg.track_coverage:
            visited = visited.at[jnp.arange(C)[:, None], env.cell_index(nobs_flat)].set(True)

        state = state._replace(
            env_state=env_state, obs=obs_last, bonus_state=bonus_state, visited=visited,
            step=step,
            agent_state={"int_filter": int_filter, "int_rms": int_rms})
        new_params, opt, loss = update_fn(stored_params(state, layout), state.opt, batch, lr,
                                          jax.random.fold_in(key, 1))
        state = put_params(state, new_params, layout)._replace(opt=opt)

        metrics = {"loss": loss, "reward_ext_sum": rext_buf.sum(axis=(0, 2)),
                   "rint_mean": rint_buf.mean(axis=(0, 2)),
                   **bonus.metrics(bonus_params, bonus_state)}
        if capture_batch:
            return state, metrics, batch
        return state, metrics

    return iteration


def build_prime_step(cfg, env, bonus, layout):
    """Return the unjitted warm-up pass: T random-action steps that only move the bonus's state.

    A bonus with no warm-up hook gets no program at all — the runner skips priming entirely.
    """
    if bonus.prime is None:
        return None
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps

    def prime_step(state, key):
        """One warm-up rollout (spec 4.3): random actions, the bonus's statistics updated."""
        if getattr(env, "action_kind", "continuous") == "discrete":
            acts = jax.random.randint(key, (T, C, N), 0, env.n_actions, jnp.int32)
        else:
            acts = jax.random.uniform(key, (T, C, N, env.act_dim), F32, -1.0, 1.0)

        def body(carry, a):
            """One environment step with a random action; only the next observation is kept."""
            env_state = carry
            env_state, _, _, _, _, final_obs = env.step(env_state, a)
            return env_state, final_obs

        env_state, nobs = jax.lax.scan(body, state.env_state, acts)
        # before: nobs [T, C, N, obs]; after: [C, T*N, obs], the shape the statistics consume
        bonus_params = named_params(stored_params(state, layout), layout)["bonus"]
        bonus_state = bonus.prime(bonus_params, state.bonus_state,
                                  nobs.transpose(1, 0, 2, 3).reshape(C, T * N, env.obs_dim))
        return state._replace(env_state=env_state, bonus_state=bonus_state)

    return prime_step
