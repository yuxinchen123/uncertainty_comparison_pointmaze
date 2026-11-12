"""Where one training iteration of the PointMaze study's learner spends its time, piece by piece.

The learner is the study's (see `study_learner.py`). One process measures one rollout shape: a
number of policies and a number of environments per policy.

1. The whole iteration, as the study runs it (episode log and coverage map on), without the
   episode log, and without the coverage map. The three programs are built in this process and
   timed in alternating rounds, each round a block of chained calls with one wait at the end, as
   a training loop calls them. The difference between two programs is the cost of the piece that
   one of them leaves out.
2. Each piece alone: its own compiled function that returns everything it computes, timed with a
   wait for the card after every call. The pieces of the update: the whole update; the shuffle and
   the gather; the permutation draw; the sixteen gradients; the gradients with the clip. The
   pieces of the rollout and of what follows it: the rollout scan; the environment; the policy
   forward; the action noise; the critic pass; the random-network-distillation forward and its
   statistics; the reward filter and the advantage sums; putting the rows in per-policy order; the
   coverage map; the episode log's sums.

Timing depends on array shapes and not on values, so the pieces run on arrays of the right shapes
filled with random numbers. The gradient-only pieces compute a different function on purpose (they
do not change the parameters); they exist only to attribute time. A piece timed alone waits for
the card after every call, so the pieces add up to more than the chained iteration.

The record is rewritten after every piece, so a job that reaches its time limit keeps what it
measured. Pieces not reached before the time budget are recorded as not measured.

Usage (inside a Slurm job on one card):
  python profile_study_iteration_pieces.py --copies 256 --envs-per-copy 16 --output <file.json>
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_learner import (block_seconds, call_seconds, machine_record, median,  # noqa: E402
                           study_configuration, throughput_columns)
from profile_jax_phases import make_batch  # noqa: E402


def write_record(path: Path, record: dict) -> None:
    """Write the record through a temporary file, so a reader never sees half of it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1))
    temporary.replace(path)


def adaptive_call_seconds(fn, target_seconds: float = 2.0) -> list:
    """Time fn() over as many calls as fit in about `target_seconds`, between 5 and 30 calls."""
    first = call_seconds(fn, reps=1, warmup=3)[0]
    reps = min(30, max(5, int(target_seconds / max(first, 1e-6))))
    return call_seconds(fn, reps=reps, warmup=0)


def build_pieces(tr, state, key) -> dict:
    """The compiled pieces of one iteration, each a function of no arguments that runs it once.

    `state` is a training state that no other program consumes; the pieces read its parameters,
    statistics, environment state and coverage map and never donate them.
    """
    import jax
    import jax.numpy as jnp
    from exploration_platform import F32, LOG2PI
    from exploration_platform.agents.ppo.networks import actor_mean, critic_values
    from exploration_platform.agents.ppo.update import (build_update, clip_per_copy,
                                                        scan_minibatches)
    from exploration_platform.statistics import rms_update
    from exploration_platform.training.episode_log import add_rollout
    from exploration_platform.training.state import named_params, stored_params
    from exploration_platform.training.train_step import rollout_unroll

    cfg, env = tr.cfg, tr.env
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    rows = T * N
    obs_dim, act_dim = env.obs_dim, env.act_dim
    unroll = rollout_unroll(cfg)
    stored = stored_params(state, tr.layout)
    named = named_params(stored, tr.layout)
    agent, bonus_params = named["agent"], named["bonus"]
    lr = tr.lr_argument(1, 1)

    # arrays of the shapes the iteration handles, filled with random numbers
    keys = iter(jax.random.split(key, 32))
    normal = lambda *shape: jax.random.normal(next(keys), shape, F32)
    z_all, obs_buf = normal(T, C, N, act_dim), normal(T, C, N, obs_dim)
    actions = jax.random.uniform(next(keys), (T, C, N, act_dim), F32, -1.0, 1.0)
    obs_flat, nobs_flat, obs_last = normal(C, rows, obs_dim), normal(C, rows, obs_dim), \
        normal(C, N, obs_dim)
    rint_buf, rext_buf = jnp.abs(normal(T, C, N)), normal(T, C, N)
    vext_buf, vext_next, vint_buf, vint_next = (normal(T, C, N) for _ in range(4))
    logp_buf, adv_buf, ret_ext_buf, ret_int_buf = (normal(T, C, N) for _ in range(4))
    done_buf = (jax.random.uniform(next(keys), (T, C, N)) < 1.0 / 400).astype(F32)
    term_buf = done_buf * (jax.random.uniform(next(keys), (T, C, N)) < 0.5).astype(F32)
    length_buf = jnp.ones((T, C, N), jnp.int32)
    batch = make_batch(cfg, next(keys), obs_dim, act_dim)
    ukey, nkey = next(keys), next(keys)
    n_steps = cfg.update_epochs * cfg.num_minibatches
    mb_size = rows // cfg.num_minibatches

    # ---- the update and its parts ----
    update_fn = build_update(cfg, tr.total_loss, tr.sweep.lr_per_copy)
    update = jax.jit(lambda p, o, b: update_fn(p, o, b, lr, ukey))

    def shuffle_and_gather(b, k):
        """The permutations and every minibatch of every epoch, with each row summed once."""
        body = lambda carry, mb: (carry + sum(v.sum() for v in mb.values()), None)
        return scan_minibatches(cfg, b, k, jnp.zeros((), F32), body)[0]

    def permutation_draw(k):
        """One shuffled order of the rows per epoch and policy, as the update draws it."""
        return jnp.argsort(jax.random.uniform(k, (cfg.update_epochs, C, rows), F32), axis=-1)

    def gradients(params, b, with_clip):
        """Sixteen scanned gradients on one minibatch shape, optionally clipped, never applied."""
        # before: one batch field [C, rows, k]; after: [16, C, rows / 4, k], the same rows for
        # every step, because only the shapes affect the time
        mbs = {name: jnp.broadcast_to(v[None, :, :mb_size], (n_steps, C, mb_size) + v.shape[2:])
               for name, v in b.items()}

        def body(carry, mb):
            """One gradient; every leaf is consumed so none is removed as unused."""
            g = jax.grad(lambda p: tr.total_loss(p, mb, style_a=False))(params)
            if with_clip:
                g = clip_per_copy(g, C, cfg.max_grad_norm)
            return carry + sum(leaf.sum() for leaf in jax.tree.leaves(g)), None

        return jax.lax.scan(body, jnp.zeros((), F32), mbs, unroll=cfg.update_unroll)[0]

    # ---- the rollout and its parts ----
    def policy_step(actor, obs, z):
        """The action of one rollout step: the policy's mean plus scaled noise."""
        return actor_mean(actor, obs) + jnp.exp(actor["logstd"])[:, None, :] * z

    def rollout_scan(actor, env_state, obs, z):
        """The rollout as the iteration runs it: policy forward and environment step per step."""
        def body(carry, noise):
            """One rollout step, storing what the rest of the iteration reads."""
            env_state, obs = carry
            action = policy_step(actor, obs, noise)
            stepped, next_obs, r_ext, term, trunc, final_obs = env.step(env_state, action)
            return (stepped, next_obs), (obs, action, final_obs, r_ext, term.astype(F32),
                                         (term | trunc).astype(F32))
        return jax.lax.scan(body, (env_state, obs), z, unroll=unroll)

    def environment_scan(env_state, acts):
        """128 environment steps on given actions, storing what the rollout stores."""
        def body(env_state, action):
            """One environment step."""
            stepped, next_obs, r_ext, term, trunc, final_obs = env.step(env_state, action)
            return stepped, (next_obs, final_obs, r_ext, term.astype(F32),
                             (term | trunc).astype(F32))
        return jax.lax.scan(body, env_state, acts, unroll=unroll)

    def policy_scan(actor, observations, z):
        """128 policy forwards on given observations, one per rollout step, and the log-probability."""
        def body(carry, xs):
            """One policy forward."""
            return carry, policy_step(actor, xs[0], xs[1])
        _, acts = jax.lax.scan(body, 0, (observations, z), unroll=unroll)
        logp = (-0.5 * z * z - actor["logstd"][None, :, None, :] - 0.5 * LOG2PI).sum(-1)
        return acts, logp

    def action_noise(k):
        """The standard normal draws of one rollout."""
        return jax.random.normal(k, (T, C, N, act_dim), F32)

    def critic_pass(critic, obs_rows, nobs_rows, last, vint, done):
        """Values of the rollout's observations and next observations in one pass, and the
        intrinsic value after a reset."""
        vall_e, vall_i = critic_values(critic, jnp.concatenate([obs_rows, nobs_rows], axis=1))
        _, vint_last = critic_values(critic, last)
        after_reset = jnp.concatenate([vint[1:], vint_last[None]], axis=0)
        return vall_e, vall_i, jnp.where(done.astype(bool), after_reset, vint)

    def distillation(params, bonus_state, nobs_rows):
        """The intrinsic reward of the rollout, the input statistics, the whitened batch input."""
        return tr.bonus.post_rollout(params, bonus_state, nobs_rows, None)

    def filter_and_advantages(int_filter, int_rms, r_int, r_ext, v_e, vn_e, v_i, vn_i, term,
                                done):
        """The intrinsic reward filter and its statistics, then both advantage streams."""
        def filt_body(f, r):
            """The discounted forward filter, one rollout step at a time."""
            f = cfg.gamma_int * f + r
            return f, f
        int_filter, filt = jax.lax.scan(filt_body, int_filter, r_int)
        int_rms = rms_update(int_rms, filt.transpose(1, 0, 2).reshape(C, rows, 1),
                             cfg.batch_stats_f32)
        r_hat = r_int / jnp.sqrt(int_rms.var + 1e-8).astype(F32).reshape(1, C, 1)

        def gae_body(carry, xs):
            """One backwards step of both advantage streams."""
            aext, aint = carry
            r_e, ve, vne, r_i, vi, vni, bm, dn = xs
            aext = (r_e + cfg.gamma_ext * vne * (1 - bm) - ve
                    + cfg.gamma_ext * cfg.gae_lambda * (1 - dn) * aext)
            aint = r_i + cfg.gamma_int * vni - vi + cfg.gamma_int * cfg.gae_lambda * aint
            return (aext, aint), (aext, aint)

        zero = jnp.zeros((C, N), F32)
        _, (aext, aint) = jax.lax.scan(gae_body, (zero, zero),
                                       (r_ext, v_e, vn_e, r_hat, v_i, vn_i, term, done),
                                       reverse=True)
        return (int_filter, int_rms, cfg.int_coef * aint + cfg.ext_coef * aext, aext + v_e,
                aint + v_i)

    def rows_in_policy_order(*fields):
        """before: rollout buffers [T, C, N, k]; after: [C, T*N, k], one row per policy per step."""
        return tuple(x.transpose(1, 0, 2, *range(3, x.ndim)).reshape(C, rows, *x.shape[3:])
                     for x in fields)

    def coverage_map(visited, nobs_rows):
        """The visited-cell map with one rollout's cells added."""
        return visited.at[jnp.arange(C)[:, None], env.cell_index(nobs_rows)].set(True)

    def episode_log_sums(log, term, done, returns, lengths, step):
        """The episode log with one rollout's finished episodes added."""
        return add_rollout(log, log["running_return"], term, done, returns, lengths, step, N)

    jit = jax.jit
    update_j, shuffle_j, permutation_j = update, jit(shuffle_and_gather), jit(permutation_draw)
    grad_j = jit(lambda p, b: gradients(p, b, False))
    gradclip_j = jit(lambda p, b: gradients(p, b, True))
    rollout_j, env_j, policy_j = jit(rollout_scan), jit(environment_scan), jit(policy_scan)
    noise_j, critic_j, rnd_j = jit(action_noise), jit(critic_pass), jit(distillation)
    filter_j, order_j, coverage_j = (jit(filter_and_advantages), jit(rows_in_policy_order),
                                     jit(coverage_map))
    pieces = {
        "update (shuffle, gather, sixteen gradients, clip, Adam)":
            lambda: update_j(stored, state.opt, batch),
        "shuffle and gather alone": lambda: shuffle_j(batch, ukey),
        "permutation draw alone (uniform numbers and their sort)": lambda: permutation_j(ukey),
        "sixteen gradients alone": lambda: grad_j(stored, batch),
        "sixteen gradients with the clip": lambda: gradclip_j(stored, batch),
        "rollout scan (policy forward and environment, 128 steps)":
            lambda: rollout_j(agent["actor"], state.env_state, state.obs, z_all),
        "environment alone, 128 steps": lambda: env_j(state.env_state, actions),
        "policy forward alone, 128 steps": lambda: policy_j(agent["actor"], obs_buf, z_all),
        "action noise draw": lambda: noise_j(nkey),
        "critic pass after the rollout":
            lambda: critic_j(agent["critic"], obs_flat, nobs_flat, obs_last, vint_buf, done_buf),
        "distillation forward, input statistics and whitening":
            lambda: rnd_j(bonus_params, state.bonus_state, nobs_flat),
        "reward filter, its statistics and both advantage sums":
            lambda: filter_j(state.agent_state["int_filter"], state.agent_state["int_rms"],
                             rint_buf, rext_buf, vext_buf, vext_next, vint_buf, vint_next,
                             term_buf, done_buf),
        "rows put in per-policy order (seven fields)":
            lambda: order_j(obs_buf, z_all, logp_buf, adv_buf, ret_ext_buf, ret_int_buf, vext_buf),
    }
    if cfg.track_coverage:
        pieces["coverage map alone"] = lambda: coverage_j(state.visited, nobs_flat)
    if cfg.episode_log:
        log_j = jit(episode_log_sums)
        pieces["episode log sums alone"] = lambda: log_j(
            state.agent_state["episode_log"], term_buf, done_buf, rext_buf, length_buf, state.step)
    return pieces


def iteration_variants(tr, state) -> dict:
    """The compiled iteration as the study runs it, without the episode log, without coverage.

    Returns name -> (program, its own starting state). Every program gets its own copy of the
    state, because an iteration consumes the state it is given.

    before: a study state with agent_state["episode_log"] and visited [C, cells]
    after:  "without the episode log": the same state with no episode_log entry;
            "without the coverage map": the same state with visited [C, 0]
    """
    import jax
    import jax.numpy as jnp
    from exploration_platform.training.train_step import build_iteration

    copy = lambda tree: jax.tree.map(jnp.copy, tree)
    build = lambda cfg: jax.jit(build_iteration(cfg, tr.env, tr.bonus, tr.sweep, tr.layout),
                                donate_argnums=(0,))
    no_log = {k: v for k, v in state.agent_state.items() if k != "episode_log"}
    return {
        "as the study runs it": (tr.iterate, copy(state)),
        "without the episode log": (build(replace(tr.cfg, episode_log=False)),
                                    copy(state._replace(agent_state=no_log))),
        "without the coverage map": (build(replace(tr.cfg, track_coverage=False)),
                                     copy(state._replace(
                                         visited=jnp.zeros((tr.cfg.n_copies, 0), bool)))),
    }


def main():
    """Measure one rollout shape and write its record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, required=True)
    ap.add_argument("--envs-per-copy", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--budget-seconds", type=float, default=780.0,
                    help="no new piece starts after this many seconds of wall time")
    args = ap.parse_args()
    started = time.time()

    import jax
    from exploration_platform.training.runner import Runner

    cfg, env_cfg = study_configuration(args.copies, args.envs_per_copy)
    record = {"record": "time breakdown of one iteration", "environment": "pointmaze-large",
              "copies": args.copies, "envs_per_copy": args.envs_per_copy,
              "rollout_steps": cfg.num_steps, "update": "4 epochs x 4 minibatches, clipped",
              "gather_minibatches_per_epoch": cfg.gather_minibatches_per_epoch,
              "machine": machine_record(), "pieces": {}, "not_measured": []}
    say = lambda text: print(f"[{time.time() - started:6.1f} s] {text}", flush=True)

    # the study's program and a primed state, with two iterations run so the state is typical
    tr = Runner(cfg, env_cfg=env_cfg)
    lr = tr.lr_argument(1, 1)
    state = tr.prime(tr.init_state(run_seed=0))
    record["seconds_to_build_and_prime"] = time.time() - started
    say(f"built and primed: {args.copies} policies x {args.envs_per_copy} environments")
    variants = iteration_variants(tr, state)
    piece_state = jax.tree.map(jax.numpy.copy, state)

    # two warm-up calls per program: the second call, on donated buffers, compiles once more
    states = {}
    for name, (program, own_state) in variants.items():
        t0 = time.time()
        for _ in range(2):
            own_state = program(own_state, lr)[0]
            jax.block_until_ready(own_state)
        states[name] = own_state
        say(f"compiled and warmed up the iteration {name}: {time.time() - t0:.1f} s")
    record["seconds_to_first_timed_call"] = time.time() - started

    # one call at a time with a wait, for the study's program: the scale of the pieces below
    step = {name: (lambda s, program=program: program(s, lr)[0])
            for name, (program, _) in variants.items()}
    waited = []
    for _ in range(20):
        seconds, states["as the study runs it"] = block_seconds(
            step["as the study runs it"], states["as the study runs it"], 1)
        waited.append(seconds)
    record["whole_iteration_one_call_with_wait_seconds"] = median(waited)

    # alternating rounds of chained calls; the order is reversed on every second round
    block = max(5, int(1.5 / median(waited)))
    rounds = {name: [] for name in variants}
    for r in range(args.rounds):
        for name in (list(variants) if r % 2 == 0 else list(variants)[::-1]):
            seconds, states[name] = block_seconds(step[name], states[name], block)
            rounds[name].append(seconds)
    record["whole_iteration_chained_seconds_per_round"] = rounds
    record["chained_calls_per_block"] = block
    whole = median(rounds["as the study runs it"])
    record["throughput"] = throughput_columns(args.copies, args.envs_per_copy, cfg.num_steps,
                                              whole)
    say(f"whole iteration: {whole * 1e3:.2f} ms chained, {median(waited) * 1e3:.2f} ms with a "
        f"wait; {record['throughput']['total_steps_per_second_millions']:.2f} million steps "
        "per second")
    for name in list(variants)[1:]:
        say(f"  {name}: {median(rounds[name]) * 1e3:.2f} ms chained")
    write_record(args.output, record)

    # every piece alone
    for name, piece in build_pieces(tr, piece_state, jax.random.PRNGKey(7)).items():
        if time.time() - started > args.budget_seconds:
            record["not_measured"].append(name)
            continue
        times = adaptive_call_seconds(piece)
        record["pieces"][name] = {"median_seconds": median(times), "least_seconds": min(times),
                                  "calls": len(times)}
        say(f"{median(times) * 1e3:9.3f} ms  {name}")
        write_record(args.output, record)
    record["wall_seconds"] = time.time() - started
    write_record(args.output, record)
    say(f"wrote {args.output}; not measured: {record['not_measured'] or 'nothing'}")


if __name__ == "__main__":
    main()
