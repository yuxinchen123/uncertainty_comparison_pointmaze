"""Faster forms of the fast last layer's rollout bonus; in exact arithmetic each equals last_layer.fast_layer_scores.

Notation for one sharing group of g environments over the T steps of a rollout (F = 513 inputs, the
last one the bias input; K = 512 outputs): R_t [g, F] the unit rows of step t, L_t [g, 1] their
lengths, Y_t [g, K] the targets, W_0 [F, K] the predictor's output layer, eta the step size, and
F_t = (R_t R_t^T + ridge)^-1 R_t the fit rows of step t (last_layer.fit_rows). The reference keeps a
temporary layer W_t and runs, step after step,

    E_t = L_t (R_t W_t) - Y_t,    W_{t+1} = W_t - eta F_t^T (E_t / L_t),    bonus_t = |E_t|^2 / 2.

Define X_t = E_t / L_t and the predictor's plain error e_t = L_t R_t W_0 - Y_t. Writing W_t as W_0
plus its changes gives X_t = e_t / L_t - eta sum_{s<t} (R_t F_s^T) X_s: the X of all steps solve
one block lower triangular system with unit diagonal,

    (I + eta Q) X = e / L,    Q_ts = R_t F_s^T for s < t and 0 otherwise,

which the reference solves by forward substitution, one step at a time, through W_t. The chunked
form cuts the T steps into chunks of B steps. Inside a chunk it solves the system of the chunk's
B g rows with one triangular solve; between chunks it carries only the change U of the layer
(W = W_0 + U), so the right side of a chunk is e / L + R U. With B = T there is one chunk, no
change to carry and no loop over steps at all.

The options, each off in the reference form:
- chunk_steps B (a divisor of T): the chunked form; 0 keeps the reference.
- plain errors: the rollout stores the predictor's float32 plain error e instead of the target
  output, so the right side e / L needs no float64 product with W_0 (the predictor's own bonus is
  then |e|^2 / 2 as well).
- float32_products "changes": each chunk's change of the layer, F^T X, in float32, added to a
  float64 change U that is applied (R U) in float64; "changes_and_applications": both products in
  float32 and U kept in float32. The fits, Q and the solves stay float64 in every case.
- small_solve "unrolled": the fit rows of a small group by a Cholesky factorization written out
  entry by entry (elementwise operations over the whole rollout) instead of the batched library
  factorization.

Measured on pixel Montezuma (runs/2026-11-17-21-26_*/speed_ideas.md), chunk_steps_for chooses the
chunk by sharing group: a group whose whole rollout is small (4 environments, 512 rows) is one
chunk and never forms the layer's change; other groups of up to SMALL_GROUP_LIMIT environments use
chunks of about 128 rows; larger groups, whose single step already is a large system, one step.
Float32 products are faster on cards with slow float64 arithmetic, but on real rollouts they changed
small bonuses of groups of 128 by up to 5.6e-4 (changes only) and 0.32 (both products) relative, more
than the 1e-4 allowed, so the trainer's default is "none".
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..bonuses.rnd.last_layer import RIDGE, fit_rows


class FastLayerForm(NamedTuple):
    """How the fast last layer's bonus is computed; the defaults are the reference form."""
    chunk_steps: int = 0
    plain_errors: bool = False
    float32_products: str = "none"
    small_solve: str = "cholesky"


REFERENCE_FORM = FastLayerForm()
FLOAT32_PRODUCTS = ("none", "changes", "changes_and_applications")
# The largest sharing group that is solved in chunks of several steps, and whose factorization
# "unrolled_for_small_groups" writes out entry by entry (a written-out factorization of 128 rows
# would be a program of about 350,000 operations).
SMALL_GROUP_LIMIT = 16


def small_solve_for(setting, group):
    """The factorization of one sharing group: the configured one, with "unrolled_for_small_groups" resolved.

    before: "unrolled_for_small_groups", group 16 ; after: "unrolled"
    before: "unrolled_for_small_groups", group 128 ; after: "cholesky"
    """
    if setting == "unrolled_for_small_groups":
        return "unrolled" if group <= SMALL_GROUP_LIMIT else "cholesky"
    return setting


def chunk_steps_for(rows_per_chunk, whole_rollout_rows, steps, group):
    """Steps per chunk of one sharing group of `group` environments over `steps` rollout steps.

    The whole rollout when it has at most whole_rollout_rows rows; else one step for groups larger
    than SMALL_GROUP_LIMIT; else the largest divisor of the steps whose chunk holds at most
    rows_per_chunk rows.
    before: 128 rows, whole 512, 128 steps, group 4 ; after: 128 (the whole rollout, 512 rows)
    before: 128 rows, whole 512, 128 steps, group 16 ; after: 8 (chunks of 128 rows)
    before: 128 rows, whole 512, 128 steps, group 128 ; after: 1 (one step at a time)
    before: 512 rows, whole 0, 128 steps, group 16 ; after: 32 ; before: 8 rows, whole 0, 128 steps,
    group 16 ; after: 1 (a chunk is never shorter than one step)
    """
    if steps * group <= whole_rollout_rows:
        return steps
    if group > SMALL_GROUP_LIMIT:
        return 1
    fitting = [count for count in range(1, steps + 1) if steps % count == 0 and count * group <= rows_per_chunk]
    return max(fitting, default=1)


def unrolled_fit_rows(rows):
    """last_layer.fit_rows for a small group, with the g x g factorization written out entry by entry.

    rows [..., g, F] unit rows -> (rows rows^T + ridge)^-1 rows [..., g, F], float64.
    before: two states with the same unit row r ; after: [r / 2, r / 2] (as fit_rows)
    """
    envs = rows.shape[-2]
    if envs == 1:
        # one unit-length row: rows rows^T is 1, and the row itself is the answer
        return rows
    gram = jnp.matmul(rows, jnp.swapaxes(rows, -1, -2)) + RIDGE * jnp.eye(envs, dtype=rows.dtype)
    # Cholesky factor, row by row: factor[i][j] for j <= i, each an array over the leading axes.
    factor = [[None] * envs for _ in range(envs)]
    for i in range(envs):
        for j in range(i + 1):
            total = gram[..., i, j] - sum(factor[i][k] * factor[j][k] for k in range(j))
            factor[i][j] = jnp.sqrt(total) if i == j else total / factor[j][j]
    # Forward substitution factor y = rows, then back substitution factor^T x = y, one row at a time.
    # before: rows [..., g, F] ; after: g arrays [..., F], then stacked back to [..., g, F]
    columns = [rows[..., i, :] for i in range(envs)]
    forward = []
    for i in range(envs):
        value = columns[i] - sum(factor[i][k][..., None] * forward[k] for k in range(i))
        forward.append(value / factor[i][i][..., None])
    backward = [None] * envs
    for i in reversed(range(envs)):
        value = forward[i] - sum(factor[k][i][..., None] * backward[k] for k in range(i + 1, envs))
        backward[i] = value / factor[i][i][..., None]
    return jnp.stack(backward, axis=-2)


def chunked_fast_layer_scores(rows, length, right_side, step_size, form):
    """The chunked form of last_layer.fast_layer_scores, from the right side e / L of every step.

    rows [T, P, g, F] unit rows, length [T, P, g, 1], right_side [T, P, g, K] = e / L, all float64,
    where P counts policies times sharing groups; step_size broadcasts against [P, 1, 1]. Returns
    half the squared error of every state, [T, P, g] float64.

    before: one state seen at steps 1, 2 and 3, eta = 0.5   ; after: bonus b, b / 4, b / 16
    """
    steps, policies, envs, inputs = rows.shape
    outputs = right_side.shape[-1]
    chunk = form.chunk_steps
    if chunk < 1 or steps % chunk:
        raise ValueError(f"chunk_steps must be a positive divisor of the {steps} rollout steps, got {chunk}")
    chunks, width = steps // chunk, chunk * envs
    fits = unrolled_fit_rows(rows) if form.small_solve == "unrolled" else fit_rows(rows)

    def by_chunk(values):
        """Put each chunk's rows of one policy together, in time order then environment order."""
        # before: [T, P, g, k] with T = 4 steps, chunk 2 ; after: [2 chunks, P, 2 * g, k], where
        # row b * g + j of a chunk is environment j at the chunk's step b.
        shaped = values.reshape(chunks, chunk, policies, envs, values.shape[-1])
        return shaped.transpose(0, 2, 1, 3, 4).reshape(chunks, policies, width, values.shape[-1])

    rows_c, fits_c, length_c, right_c = by_chunk(rows), by_chunk(fits), by_chunk(length), by_chunk(right_side)
    eta = jnp.broadcast_to(jnp.asarray(step_size, jnp.float64), (policies, 1, 1))
    if chunk == 1:
        # A chunk of one step has no earlier step: no system is formed.
        systems = None
    else:
        # The unit lower triangular matrix of each chunk: identity plus eta times the products of
        # every row with the fit rows of earlier steps of the chunk (same-step and later blocks are 0).
        # before: chunk of 2 steps, g = 1: Q = [[r0.f0, r0.f1], [r1.f0, r1.f1]] ; after: [[1, 0], [eta r1.f0, 1]]
        block = jnp.arange(width) // envs
        earlier = block[:, None] > block[None, :]
        systems = (jnp.where(earlier, eta * jnp.matmul(rows_c, jnp.swapaxes(fits_c, -1, -2)), 0.0)
                   + jnp.eye(width, dtype=jnp.float64))
    # Precision of the two products with the layer's change and of the change itself.
    if form.float32_products not in FLOAT32_PRODUCTS:
        raise ValueError(f"float32_products must be one of {FLOAT32_PRODUCTS}, got {form.float32_products}")
    change_dtype = jnp.float64 if form.float32_products == "none" else jnp.float32
    applied_dtype = jnp.float32 if form.float32_products == "changes_and_applications" else jnp.float64

    def solve(system, right):
        """X of one chunk from its triangular system: [P, B g, B g] and [P, B g, K] -> [P, B g, K]."""
        if chunk == 1:
            # A chunk of one step has no earlier step: its system is the identity.
            return right
        return jax.scipy.linalg.solve_triangular(system, right, lower=True, unit_diagonal=True)

    if chunks == 1:
        # One chunk: the layer never needs to be formed.
        solved = solve(None if systems is None else systems[0], right_c[0])[None]
    else:
        def step(change, inputs):
            """Solve one chunk with the change of the layer so far, then add the chunk's own change."""
            rows_t, fits_t, right_t, system_t = inputs
            right_t = right_t + jnp.matmul(rows_t.astype(applied_dtype), change.astype(applied_dtype)).astype(jnp.float64)
            solved_t = solve(system_t, right_t)
            wanted = (-eta * solved_t).astype(change_dtype)
            increment = jnp.matmul(jnp.swapaxes(fits_t.astype(change_dtype), -1, -2), wanted)
            return change + increment.astype(applied_dtype), solved_t

        _, solved = jax.lax.scan(step, jnp.zeros((policies, inputs, outputs), applied_dtype),
                                 (rows_c, fits_c, right_c, systems))
    error = length_c * solved
    bonus = 0.5 * (error * error).sum(axis=-1)
    # before: [chunks, P, B * g] ; after: [T, P, g] in the steps' own order
    return bonus.reshape(chunks, policies, chunk, envs).transpose(0, 2, 1, 3).reshape(steps, policies, envs)
