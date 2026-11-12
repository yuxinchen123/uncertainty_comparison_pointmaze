"""One training iteration, built for exactly one (environment, agent, bonus) combination.

Nothing here is chosen at run time. The update style, the rollout shape, the bonus and whether
coverage is tracked are all Python decisions taken while the function is being built, so the
compiled program holds one of each and no branch that picks between them.

The iteration: roll out T steps, score the states the rollout reached, filter and normalise that
intrinsic reward, compute both advantage streams backwards, assemble the update batch, and take
the update.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .. import F32, LOG2PI
from ..agents.ppo.losses import (loss_coefficients, ppo_loss_per_copy,
                                 ppo_loss_per_copy_discrete)
from ..agents.ppo.networks import actor_logits, actor_mean, critic_values, logprob_discrete
from ..bonuses.rnd.networks import whiten
from ..agents.ppo.update import build_update
from ..statistics import rms_update
from . import diagnostics as diag
from .episode_log import add_rollout, extend_running_return
from .state import named_params, put_params, stored_params


def build_total_loss(cfg, bonus, layout, discrete: bool = False, sweep=None):
    """The scalar the optimizer minimises: the agent's per-copy loss plus the bonus's, summed.

    Adding the two [C] vectors before the sum, rather than summing each separately, is what keeps
    this identical to the single-module trainer down to the last bit. `discrete` picks the
    categorical actor's loss — a Python decision taken when the program is built. `sweep` supplies
    the clip, entropy and value coefficients that differ between copies; without it, or for a
    coefficient it does not hold, the configuration's scalar is used.
    """
    agent_loss = ppo_loss_per_copy_discrete if discrete else ppo_loss_per_copy
    if cfg.actor_critic_joint_trunk:
        if discrete:
            raise ValueError("actor_critic_joint_trunk is built for the continuous actor only")
        agent_loss = partial(ppo_loss_per_copy, joint_trunk=True)
    coef = loss_coefficients(cfg, sweep)

    def total_loss(params, batch, style_a):
        """Sum over copies of (agent loss + bonus loss) on one (mini)batch."""
        # in the one-array form the gradient is taken with respect to the flat array, so the
        # named tensors the networks expect are cut out of it here
        named = layout.unpack(params) if layout.packed else params
        loss_c = agent_loss(coef, named["agent"], batch, style_a)
        return (loss_c + bonus.loss(named["bonus"], batch)).sum()
    return total_loss


def rollout_unroll(cfg) -> int:
    """How many rollout steps one body of the rollout scan holds.

    One rollout step does more work the more copies there are, so the point where a longer
    program stops paying moves with the copy count: at 8 copies unrolling 32 steps beat 16 in 11
    of 11 paired rounds, while at 32 and 128 copies 16 beat 32 in 11 of 11. Unrolling does not
    change the arithmetic, and the parameters after one iteration are bit-identical at 4, 16 and
    32 — EXCEPT with the rollout hoist off, where the in-scan critic and log-probability fuse
    differently at 32 (4 against 32 deviates 3.9e-04, 4 against 16 stays exact). So 32 is taken
    only where it is bit-neutral. A value given explicitly is used as given.

    before: scan_unroll 0, 8 copies, hoist on -> 32;  scan_unroll 0, 256 copies -> 16
    after:  scan_unroll 8, any copy count     -> 8
    """
    return cfg.scan_unroll or (32 if cfg.n_copies <= 8 and cfg.hoist_rollout else 16)


def action_noise(cfg, key, shape):
    """The standard normal action noise of one rollout, float32 of the given shape.

    "threefry" is JAX's default generator: the same numbers on every machine. "rbg" keys the
    compiler's own bit generator with four words drawn from the same key; it is another random
    scheme, and its numbers may differ between card models and library versions.
    """
    if cfg.action_noise_generator == "threefry":
        return jax.random.normal(key, shape, F32)
    if cfg.action_noise_generator != "rbg":
        raise ValueError(f"unknown action_noise_generator {cfg.action_noise_generator!r}; use "
                         "'threefry' or 'rbg'")
    words = jax.random.bits(key, (4,), jnp.uint32)
    return jax.random.normal(jax.random.wrap_key_data(words, impl="rbg"), shape, F32)


def intrinsic_sum_matrices(gamma_int: float, gae_lambda: float, steps: int):
    """The fixed matrices that turn the intrinsic filter and the intrinsic advantage into products.

    Filter: f[t] = gamma f[t-1] + r[t], so f[t] = sum over s <= t of gamma^(t-s) r[s]
    + gamma^(t+1) f[-1]. Advantage: a[t] = d[t] + gamma lambda a[t+1], so a[t] = sum over
    s >= t of (gamma lambda)^(s-t) d[s].

    before: gamma 0.5, lambda 1, 3 steps
    after:  filter [[1, 0, 0], [0.5, 1, 0], [0.25, 0.5, 1]], carry powers [0.5, 0.25, 0.125],
            advantage [[1, 0.5, 0.25], [0, 1, 0.5], [0, 0, 1]]
    Returns (filter [T, T], carry powers [T], advantage [T, T]), float32.
    """
    lag = np.arange(steps)[:, None] - np.arange(steps)[None, :]
    filter_matrix = np.where(lag >= 0, float(gamma_int) ** np.maximum(lag, 0), 0.0)
    carry_powers = float(gamma_int) ** (np.arange(steps) + 1.0)
    advantage_matrix = np.where(lag <= 0, float(gamma_int * gae_lambda) ** np.maximum(-lag, 0),
                                0.0)
    return (jnp.asarray(filter_matrix, F32), jnp.asarray(carry_powers, F32),
            jnp.asarray(advantage_matrix, F32))


REWARD_DIVISORS = ("running_std", "frozen_after_priming", "none")


def filtered_intrinsic_sum(gamma_int: float, carried, rint_buf, unroll: int = 1):
    """The discounted forward filter of the intrinsic reward, one rollout step at a time.

    f[t] = gamma f[t-1] + r[t], started from the value carried in from the rollout before.
    before: gamma 0.5, carried 4, rewards 1, 1, 1
    after:  (2.125, [3, 2.5, 2.125]): the value to carry on, and the filter at every step
    carried [C, N], rint_buf [T, C, N].
    """
    def filt_body(f, r):
        """The discounted forward filter, one rollout step at a time."""
        f = gamma_int * f + r
        return f, f
    return jax.lax.scan(filt_body, carried, rint_buf, unroll=unroll)


def intrinsic_divisor(reward_divisor: str, int_rms, n_copies: int):
    """What each copy's intrinsic reward is divided by, [C] float32.

    "running_std" and "frozen_after_priming" read the standard deviation of the reward's
    discounted sum from the statistics as they stand; "none" is 1.
    """
    if reward_divisor == "none":
        return jnp.ones((n_copies,), F32)
    return jnp.sqrt(int_rms.var + 1e-8).astype(F32).reshape(n_copies)


def build_iteration(cfg, env, bonus, sweep, layout, capture_batch: bool = False):
    """Return the unjitted iteration function for this configuration.

    capture_batch makes the returned function hand back the update batch as a third result. It
    exists for the test that checks the hoisted rollout against the in-scan one, which has to
    compare the batch the two forms build.
    """
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    discrete = getattr(env, "action_kind", "continuous") == "discrete"
    scan_unroll = rollout_unroll(cfg)

    if not cfg.hoist_rollout and bonus.rollout_step is None:
        raise ValueError(
            f"the {bonus.name} bonus cannot score one rollout step at a time, so it cannot be "
            "composed with hoist_rollout=False; leave the hoist on")

    update_fn = build_update(cfg, build_total_loss(cfg, bonus, layout, discrete, sweep),
                             sweep.lr_per_copy, bonus.own_learning_rate)

    # what the bonus's reward is divided by, from the family's own record
    reward_divisor = bonus.reward_divisor
    if reward_divisor not in REWARD_DIVISORS:
        raise ValueError(f"unknown reward divisor {reward_divisor!r} of the {bonus.name} bonus; "
                         f"choose from {list(REWARD_DIVISORS)}")

    # the reward-decline study's switches, all decided here while the program is built
    if cfg.int_coef_schedule not in ("constant", "linear_decay"):
        raise ValueError(f"unknown int_coef_schedule {cfg.int_coef_schedule!r}")
    if cfg.int_coef_schedule == "linear_decay" and cfg.total_iterations <= 0:
        raise ValueError("int_coef_schedule='linear_decay' needs total_iterations")
    track = cfg.diagnostics or cfg.success_gate_rate > 0
    if cfg.normalize_agent_obs and not cfg.hoist_rollout:
        raise ValueError("normalize_agent_obs is built only for the hoisted rollout")
    # a step paid more than the reward shift ended inside the goal radius, whatever the payment
    # rule: per_step pays shift + 1 there, remaining_steps pays shift + at least 1
    inside_threshold = env.cfg.reward_shift + 0.5

    # the PointMaze and AntMaze study's keys, also decided here. The reward the learner sees is
    # scale x the environment's reward + offset: Python numbers when every copy shares them, one
    # value per copy broadcast over the [T, C, N] rollout otherwise. Scale 1 and offset 0 as
    # Python numbers put no arithmetic into the program.
    reward_scale = sweep.setting(cfg, "ext_reward_scale", (1, C, 1))
    reward_offset = sweep.setting(cfg, "ext_reward_offset", (1, C, 1))
    maps_reward = not (isinstance(reward_scale, (int, float)) and reward_scale == 1
                       and isinstance(reward_offset, (int, float)) and reward_offset == 0)
    if cfg.episode_log and not cfg.hoist_rollout:
        raise ValueError("episode_log is built only for the hoisted rollout")
    # the speed keys of the reward filter and the advantage sums
    if cfg.advantage_scan_unroll < 1:
        raise ValueError(f"advantage_scan_unroll must be at least 1; it is "
                         f"{cfg.advantage_scan_unroll}")
    if cfg.action_noise_generator != "threefry" and discrete:
        raise ValueError("action_noise_generator other than 'threefry' is built for the "
                         "continuous actor only")
    if cfg.intrinsic_sums_by_matrix_product:
        filter_matrix, carry_powers, advantage_matrix = intrinsic_sum_matrices(
            cfg.gamma_int, cfg.gae_lambda, T)

    # next values read from the following step's values: possible only where no next value
    # other than a truncated step's has to come from a final observation, and where one
    # environment is truncated at most once in a rollout
    shifted_values = cfg.next_values_from_shifted_values
    if shifted_values:
        if not cfg.hoist_rollout or cfg.intrinsic_bootstrap_observation != "reset_observation":
            raise ValueError("next_values_from_shifted_values needs the hoisted rollout and "
                             "intrinsic_bootstrap_observation='reset_observation'")
        if (not getattr(env, "truncates_only_at_step_limit", False)
                or env.cfg.max_episode_steps < T):
            raise ValueError("next_values_from_shifted_values needs an environment whose only "
                             "truncation is its step limit, with a limit of at least one "
                             f"rollout ({T} steps), so that an environment is truncated at "
                             "most once in a rollout")

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
            z_all = action_noise(cfg, jax.random.fold_in(key, 0), (T, C, N, env.act_dim))
            logstd = agent_params["actor"]["logstd"]
        # The current PointMaze configuration may whiten its inputs; historical configurations do not.
        if cfg.normalize_agent_obs:
            agent_rms = state.agent_state["obs_rms"]
            seen = lambda x: whiten(x, agent_rms)
        else:
            seen = lambda x: x
        # before: a rollout buffer [T, C, N, k]; after: [C, T*N, k], one row per copy per step
        flat = lambda x: x.transpose(1, 0, 2, *range(3, x.ndim)).reshape(C, T * N, *x.shape[3:])
        # before: [C, T*N]; after: [T, C, N] — the inverse of flat for a scalar-per-step field
        unflat = lambda x: x.reshape(C, T, N).transpose(1, 0, 2)

        # With the episode log on, the hoisted rollout also carries each environment's running
        # episode return and stores, per step, that return and the step's number in its episode.
        # With it off the carry and the stored fields are exactly what they always were.
        rollout_start = (state.env_state, state.obs)
        if cfg.episode_log:
            rollout_start += (state.agent_state["episode_log"]["running_return"],)

        def logged_step(carry, env_state, stepped, next_obs, r_ext, term, trunc, out):
            """The scan step's result: the next carry and the stored fields, with the log's two.

            env_state is the environment state before the step and `stepped` the one after it.
            before (log off): carry (state, obs),         out (..., reward, terminated, done)
            after  (log on):  carry (state, obs, return), out (..., done, return so far, step
                              number in the episode)
            """
            if not cfg.episode_log:
                return (stepped, next_obs), out
            running, so_far = extend_running_return(carry[2], r_ext, term | trunc)
            return (stepped, next_obs, running), out + (so_far, env_state.step_count + 1)

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
                    env_state, obs = carry[:2]
                    logits = actor_logits(agent_params["actor"], seen(obs))
                    action = jnp.argmax(logits + z, axis=-1).astype(jnp.int32)
                    logp = logprob_discrete(logits, action)
                    stepped, next_obs, r_ext, term, trunc, final_obs = env.step(env_state,
                                                                                action)
                    out = (obs, action, logp, final_obs, r_ext,
                           term.astype(F32), (term | trunc).astype(F32))
                    return logged_step(carry, env_state, stepped, next_obs, r_ext,
                                       term, trunc, out)

                rollout_carry, outs = jax.lax.scan(
                    body, rollout_start, z_all, unroll=scan_unroll)
                obs_buf, act_buf, logp_buf, nobs_buf, rext_buf, term_buf, done_buf = outs[:7]
            else:
                def body(carry, z):
                    """One rollout step: act, step the environment, store what the rest needs."""
                    env_state, obs = carry[:2]
                    mean = actor_mean(agent_params["actor"], seen(obs))
                    action = mean + jnp.exp(logstd)[:, None, :] * z
                    stepped, next_obs, r_ext, term, trunc, final_obs = env.step(env_state,
                                                                                action)
                    out = (obs, action, final_obs, r_ext,
                           term.astype(F32), (term | trunc).astype(F32))
                    return logged_step(carry, env_state, stepped, next_obs, r_ext,
                                       term, trunc, out)

                rollout_carry, outs = jax.lax.scan(
                    body, rollout_start, z_all, unroll=scan_unroll)
                obs_buf, act_buf, nobs_buf, rext_buf, term_buf, done_buf = outs[:6]

                # log-probability of the sampled actions depends only on the noise and logstd
                logp_buf = (-0.5 * z_all * z_all - logstd[None, :, None, :]
                            - 0.5 * LOG2PI).sum(-1)
            env_state, obs_last = rollout_carry[:2]
            obs_flat, nobs_flat = flat(obs_buf), flat(nobs_buf)
            if shifted_values:
                # A step's final observation IS the next step's observation unless the step
                # ended an episode, so its value is already among the on-step values. A step that
                # ended an episode needs: nothing after a termination (the extrinsic return stops
                # there), the reset observation's intrinsic value (again the next step's value),
                # and, after a truncation only, the extrinsic value of the final observation.
                # An environment is truncated at most once in a rollout, so that is one
                # observation per environment.
                # before: critic rows per copy 2 x T*N (observations, final observations)
                # after:  T*N + 2 x N (observations, the observation after the last step, each
                #         environment's final observation at its truncated step)
                trunc_buf = (done_buf - term_buf) > 0
                trunc_step = jnp.argmax(trunc_buf, axis=0)
                final_at_trunc = jnp.take_along_axis(
                    nobs_buf, trunc_step[None, :, :, None], axis=0)[0]
                vall_e, vall_i = critic_values(agent_params["critic"], seen(
                    jnp.concatenate([obs_flat, obs_last, final_at_trunc], axis=1)))
                vext_buf, vint_buf = unflat(vall_e[:, :T * N]), unflat(vall_i[:, :T * N])
                vext_last, vint_last = vall_e[:, T * N:T * N + N], vall_i[:, T * N:T * N + N]
                # before: values [T, C, N] and the value after the last step [C, N]
                # after:  [T, C, N] where entry t is the value at step t + 1
                following = lambda v, last: jnp.concatenate([v[1:], last[None]], axis=0)
                vext_next = jnp.where(trunc_buf, vall_e[:, T * N + N:][None],
                                      following(vext_buf, vext_last))
                vint_next = following(vint_buf, vint_last)
            else:
                # one critic pass covers the on-step values and the bootstrap values together
                vall_e, vall_i = critic_values(
                    agent_params["critic"], seen(jnp.concatenate([obs_flat, nobs_flat], axis=1)))
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
                    logits = actor_logits(agent_params["actor"], seen(obs))
                    action = jnp.argmax(logits + z, axis=-1).astype(jnp.int32)
                    logp = logprob_discrete(logits, action)
                else:
                    mean = actor_mean(agent_params["actor"], seen(obs))
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

        # The current PointMaze configuration continues intrinsic returns from the reset state.
        # Historical configurations retain their final-observation calculation for reproduction.
        # Before: final observations at resets; after: reset observations only when selected.
        if cfg.intrinsic_bootstrap_observation == "reset_observation" and not shifted_values:
            _, vint_last = critic_values(agent_params["critic"], seen(obs_last))
            vint_after_reset = jnp.concatenate([vint_buf[1:], vint_last[None]], axis=0)
            vint_next = jnp.where(done_buf.astype(bool), vint_after_reset, vint_next)

        # the bonus scores the rollout (or takes the scores the scan already made), moves its own
        # state on, and hands over whatever extra fields its loss will need in the update batch
        bonus_state, rint_flat, batch_extra = bonus.post_rollout(
            bonus_params, bonus_state_old, nobs_flat, scored)
        rint_buf = unflat(rint_flat)

        # intrinsic filter forward in time, then per-copy normalization (spec 5.2)
        if reward_divisor == "running_std":
            if cfg.intrinsic_sums_by_matrix_product:
                # before: rewards [T, C, N] and the filter value carried in, [C, N]
                # after:  filt[t] = sum over s <= t of gamma^(t-s) r[s]
                #         + gamma^(t+1) x carried value
                filt = (jnp.matmul(filter_matrix, rint_buf.reshape(T, C * N)).reshape(T, C, N)
                        + carry_powers[:, None, None] * state.agent_state["int_filter"][None])
                int_filter = filt[-1]
            else:
                int_filter, filt = filtered_intrinsic_sum(
                    cfg.gamma_int, state.agent_state["int_filter"], rint_buf,
                    cfg.advantage_scan_unroll)
            int_rms = rms_update(state.agent_state["int_rms"],
                                 filt.transpose(1, 0, 2).reshape(C, T * N, 1),
                                 cfg.batch_stats_f32)
            int_std = jnp.sqrt(int_rms.var + 1e-8).astype(F32).reshape(1, C, 1)
            rint_hat = rint_buf / int_std
        else:
            # the statistics are not moved: a frozen divisor keeps the value the priming
            # rollouts left, and no divisor leaves the reward as the bonus gave it
            int_filter = state.agent_state["int_filter"]
            int_rms = state.agent_state["int_rms"]
            int_std = intrinsic_divisor(reward_divisor, int_rms, C).reshape(1, C, 1)
            rint_hat = rint_buf
            if reward_divisor == "frozen_after_priming":
                rint_hat = rint_buf / int_std

        # study additions: the episode tracker (and the running goal-reaching rate it feeds), and
        # the task reward scaled by its running discounted-sum std. Neither runs by default.
        agent_state = {"int_filter": int_filter, "int_rms": int_rms}
        if cfg.normalize_agent_obs:
            agent_state["obs_rms"] = rms_update(agent_rms, flat(obs_buf), cfg.batch_stats_f32)
        if track:
            tracker, ended = diag.track_episodes(state.agent_state["episode"],
                                                 rext_buf > inside_threshold, rext_buf, done_buf)
            agent_state["episode"] = tracker
        if cfg.success_gate_rate > 0:
            success_old = state.agent_state["success_rate"]
            agent_state["success_rate"] = diag.update_success_rate(
                success_old, ended["episodes"], ended["reached"], cfg.success_gate_rate)
        else:
            success_old = None
        if cfg.episode_log:
            agent_state["episode_log"] = add_rollout(
                state.agent_state["episode_log"], rollout_carry[2], term_buf, done_buf,
                outs[-2], outs[-1], step, N)
        if bonus.record_on_maze_grid:
            # one visit per state the rollout reached, counted in the cell it lies in; the
            # driver reads the counts once per logging window and sets them back to zero
            # before: positions of [C, T*N] states; after: counts [C, rows*cols] grown by T*N
            agent_state["maze_grid_visits"] = state.agent_state["maze_grid_visits"].at[
                jnp.arange(C)[:, None], env.cell_index(nobs_flat)].add(1)
        # before: the environment's reward, 1 on arrival and 0 elsewhere, [T, C, N]
        # after:  with scale 1/400 and offset -1/400 for a copy, 0 on arrival and -1/400 elsewhere
        rext_used = rext_buf * reward_scale + reward_offset if maps_reward else rext_buf
        if cfg.normalize_ext_reward:
            ext_filter, ext_rms, rext_used, ext_std = diag.normalize_task_reward(
                state.agent_state["ext_filter"], state.agent_state["ext_rms"], rext_used,
                done_buf, cfg.gamma_ext)
            agent_state["ext_filter"], agent_state["ext_rms"] = ext_filter, ext_rms

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
        if cfg.intrinsic_sums_by_matrix_product:
            def gae_ext_body(aext, xs):
                """One backwards step of the extrinsic advantage, cut where an episode ended."""
                r_e, v_e, vn_e, bm, dn = xs
                d_ext = r_e + cfg.gamma_ext * vn_e * (1 - bm) - v_e
                aext = d_ext + cfg.gamma_ext * cfg.gae_lambda * (1 - dn) * aext
                return aext, aext

            _, aext_buf = jax.lax.scan(
                gae_ext_body, zero, (rext_used, vext_buf, vext_next, boot_mask, done_buf),
                reverse=True, unroll=cfg.advantage_scan_unroll)
            # before: one-step intrinsic differences d [T, C, N]
            # after:  aint[t] = sum over s >= t of (gamma lambda)^(s-t) d[s]
            d_int = rint_hat + cfg.gamma_int * vint_next - vint_buf
            aint_buf = jnp.matmul(advantage_matrix, d_int.reshape(T, C * N)).reshape(T, C, N)
        else:
            _, (aext_buf, aint_buf) = jax.lax.scan(
                gae_body, (zero, zero),
                (rext_used, vext_buf, vext_next, rint_hat, vint_buf, vint_next,
                 boot_mask, done_buf), reverse=True, unroll=cfg.advantage_scan_unroll)

        # No-bonus policies use task advantages only. Other bonuses use either a scalar
        # intrinsic weight or per-copy weights broadcast over the [T, C, N] rollout; a schedule
        # or the goal-rate gate multiplies that weight by a [C] factor.
        factor = diag.intrinsic_weight_factor(cfg, step, success_old)
        beta_c = None
        if bonus.name == "none" and cfg.no_bonus_advantage_stream == "task_only":
            adv = cfg.ext_coef * aext_buf
        elif factor is not None:
            base = sweep.beta_per_copy if sweep.beta_per_copy is not None else cfg.int_coef
            beta_c = base * factor
            adv = beta_c.reshape(1, C, 1) * aint_buf + cfg.ext_coef * aext_buf
        elif sweep.beta_per_copy is None:
            adv = cfg.int_coef * aint_buf + cfg.ext_coef * aext_buf
        else:
            adv = sweep.beta_per_copy.reshape(1, C, 1) * aint_buf + cfg.ext_coef * aext_buf
        ret_ext = aext_buf + vext_buf
        ret_int = aint_buf + vint_buf

        batch = {
            "obs": seen(flat(obs_buf)), "actions": flat(act_buf), "old_logprob": flat(logp_buf),
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
            step=step, agent_state=agent_state)
        updated = update_fn(stored_params(state, layout), state.opt, batch, lr,
                            jax.random.fold_in(key, 1))
        new_params, opt, loss = updated[:3]
        state = put_params(state, new_params, layout)._replace(opt=opt)

        metrics = {"loss": loss, "reward_ext_sum": rext_buf.sum(axis=(0, 2)),
                   "rint_mean": rint_buf.mean(axis=(0, 2)),
                   **bonus.metrics(bonus_params, bonus_state)}
        if cfg.diagnostics:
            # per copy, [C] each: what the policy, the critic and the advantage looked like this
            # iteration, and the episodes that ended inside the rollout
            new_agent = named_params(new_params, layout)["agent"]
            beta_now = (beta_c if beta_c is not None else
                        sweep.beta_per_copy if sweep.beta_per_copy is not None else
                        jnp.full((C,), 0.0 if bonus.name == "none" and cfg.no_bonus_advantage_stream == "task_only"
                                 else cfg.int_coef, F32))
            speed = jnp.sqrt((nobs_flat[..., 2:] ** 2).sum(-1))
            d = {"logstd_mean": new_agent["actor"]["logstd"].mean(axis=1),
                 "approx_kl": diag.gaussian_kl(agent_params["actor"], new_agent["actor"],
                                               batch["obs"]),
                 "ext_explained_variance": diag.explained_variance(batch["vext_old"],
                                                                   batch["ret_ext"]),
                 "vext_mean": batch["vext_old"].mean(axis=1),
                 "ret_ext_mean": batch["ret_ext"].mean(axis=1),
                 "int_std": int_std.reshape(C),
                 "rint_hat_mean": rint_hat.mean(axis=(0, 2)),
                 "adv_int_rms": jnp.sqrt(((beta_now.reshape(1, C, 1) * aint_buf) ** 2
                                          ).mean(axis=(0, 2))),
                 "adv_ext_rms": jnp.sqrt(((cfg.ext_coef * aext_buf) ** 2).mean(axis=(0, 2))),
                 "beta_effective": beta_now,
                 "speed_mean": speed.mean(axis=1),
                 **{f"episode_{name}": value for name, value in ended.items()},
                 **updated[3]}
            if cfg.success_gate_rate > 0:
                d["success_rate"] = agent_state["success_rate"]
            if cfg.normalize_ext_reward:
                d["ext_std"] = ext_std
            metrics["diagnostics"] = d
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
    frozen_divisor = bonus.reward_divisor == "frozen_after_priming"
    if frozen_divisor and bonus.score_alone is None:
        raise ValueError(f"the {bonus.name} bonus cannot score the priming rollouts, so its "
                         "reward divisor cannot be frozen after priming")
    if frozen_divisor and cfg.prime_iterations < 1:
        raise ValueError("a reward divisor frozen after priming needs at least one priming "
                         "rollout; prime_iterations is 0")

    def prime_step(state, key):
        """One warm-up rollout (spec 4.3): random actions, the bonus's statistics updated.

        With a divisor frozen after priming, the rollout is also scored and the running
        statistics of the reward's discounted sum take it in, exactly as a training rollout's
        would; the divisor is then what these statistics hold when priming ends.
        """
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
        nobs_flat = nobs.transpose(1, 0, 2, 3).reshape(C, T * N, env.obs_dim)
        bonus_state = bonus.prime(bonus_params, state.bonus_state, nobs_flat)
        state = state._replace(env_state=env_state, bonus_state=bonus_state)
        if not frozen_divisor:
            return state
        # score the rollout with the statistics that now include it, then filter and fold in
        # before: rewards [C, T*N]; after: [T, C, N], the order the filter runs in
        rint_buf = bonus.score_alone(bonus_params, bonus_state, nobs_flat).reshape(
            C, T, N).transpose(1, 0, 2)
        int_filter, filt = filtered_intrinsic_sum(
            cfg.gamma_int, state.agent_state["int_filter"], rint_buf)
        int_rms = rms_update(state.agent_state["int_rms"],
                             filt.transpose(1, 0, 2).reshape(C, T * N, 1), cfg.batch_stats_f32)
        return state._replace(agent_state={**state.agent_state, "int_filter": int_filter,
                                           "int_rms": int_rms})

    return prime_step
