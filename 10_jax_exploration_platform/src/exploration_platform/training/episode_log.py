"""A per-copy log of the training episodes that end in each rollout, kept by plain reductions.

The rollout scan carries one number per environment, the return of its current episode, and
stores for every step that return and the step's number in its episode. Everything else is a
masked sum over the stored [T, C, N] buffers, so the log adds no loop to the compiled program.

The sums stay in the training state and grow inside the compiled iteration. The driver reads them
once per logging window and sets them back to zero; the running returns and each copy's first
arrival step are kept.

Returns are of the ENVIRONMENT's reward, before the per-copy scale and offset. The return of the
reward the learner sees follows exactly, because the mapping is the same for every step of a copy:
scale x (sum of returns) + offset x (sum of lengths).
"""
import jax.numpy as jnp
import numpy as np

from .. import F32

# the per-copy sums the driver reads and clears once per logging window
SUM_FIELDS = ("episodes", "arrivals", "return_sum", "length_sum", "steps_to_goal_sum")


def log_init(n_copies: int, n_envs: int) -> dict:
    """An empty log: zero running returns, zero sums, and no first arrival yet (-1).

    Every field is its own array: the iteration donates its input state, and one buffer passed
    as two donated fields is refused on the card.
    """
    count = lambda: jnp.zeros((n_copies,), jnp.int64)
    return {"running_return": jnp.zeros((n_copies, n_envs), F32),
            "episodes": count(), "arrivals": count(),
            "return_sum": jnp.zeros((n_copies,), jnp.float64),
            "length_sum": count(), "steps_to_goal_sum": count(),
            "first_goal_step": jnp.full((n_copies,), -1, jnp.int64)}


def extend_running_return(running_return, reward, done):
    """One rollout step of every environment's episode return.

    before: running return 0.5, this step's reward 1.0, the step ends the episode
    after:  (0.0, 1.5): the next step starts a new episode, and this episode's return is 1.5
    Returns (the carry for the next step, the return up to and including this step), [C, N] each.
    """
    so_far = running_return + reward
    return jnp.where(done, 0.0, so_far), so_far


def rollout_sums(term_buf, done_buf, return_buf, length_buf) -> dict:
    """Per-copy [C] sums over the episodes that ended in one rollout of [T, C, N] steps.

    term_buf, done_buf: nonzero where the step ended the episode at the goal / ended it at all.
    return_buf: the episode's return up to and including the step. length_buf: the step's number
    in its episode, counted from 1.

    before: one environment, four steps with (terminated, done) = (0,0) (1,1) (0,0) (0,1),
            returns so far 0 1 0 0, step numbers 1 2 1 2
    after:  episodes 2, arrivals 1, return_sum 1, length_sum 4, steps_to_goal_sum 2
    """
    done, term = done_buf > 0, term_buf > 0
    count = lambda mask: mask.sum(axis=(0, 2), dtype=jnp.int64)
    masked_lengths = lambda mask: jnp.where(mask, length_buf, 0).sum(axis=(0, 2),
                                                                     dtype=jnp.int64)
    return {"episodes": count(done), "arrivals": count(term),
            "return_sum": jnp.where(done, return_buf, 0.0).astype(jnp.float64).sum(axis=(0, 2)),
            "length_sum": masked_lengths(done), "steps_to_goal_sum": masked_lengths(term)}


def first_arrival_step(term_buf, iteration, n_envs: int):
    """Each copy's training step at the first rollout step of this rollout with an arrival.

    A copy's training step counts its own environment steps: iteration k (from 1) covers steps
    (k - 1) x T x N + 1 to k x T x N, and rollout index t (from 0) of it ends at
    (k - 1) x T x N + (t + 1) x N.

    before: T = 4, N = 2, iteration 3, the first arrival at rollout index 1
    after:  (True, 2 x 8 + 2 x 2 = 20)
    Returns (whether the copy had an arrival [C] bool, that step [C] int64; the step is
    meaningful only where the first is True).
    """
    steps = term_buf.shape[0]
    any_arrival = (term_buf > 0).any(axis=2)
    first_index = jnp.argmax(any_arrival, axis=0).astype(jnp.int64)
    step = ((iteration.astype(jnp.int64) - 1) * (steps * n_envs) + (first_index + 1) * n_envs)
    return any_arrival.any(axis=0), step


def add_rollout(log: dict, running_return, term_buf, done_buf, return_buf, length_buf,
                iteration, n_envs: int) -> dict:
    """The log after one more rollout: its sums added, and a copy's first arrival step set once."""
    sums = rollout_sums(term_buf, done_buf, return_buf, length_buf)
    arrived, step = first_arrival_step(term_buf, iteration, n_envs)
    first = jnp.where((log["first_goal_step"] < 0) & arrived, step, log["first_goal_step"])
    return {"running_return": running_return, "first_goal_step": first,
            **{name: log[name] + sums[name] for name in SUM_FIELDS}}


def read_window(log: dict) -> dict:
    """The log's per-copy numbers on the host: the window's sums and the first arrival steps."""
    return {name: np.asarray(log[name]) for name in (*SUM_FIELDS, "first_goal_step")}


def clear_window(log: dict) -> dict:
    """The log with its window sums back at zero; running returns and first arrivals are kept."""
    return {**log, **{name: jnp.zeros_like(log[name]) for name in SUM_FIELDS}}
