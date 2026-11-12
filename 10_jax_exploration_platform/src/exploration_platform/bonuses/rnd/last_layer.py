"""Two fits of the predictor's output layer that run while a rollout is scored.

Both visit the rollout's steps in time order with one output layer per policy: a matrix of
[features + 1, outputs], whose last row is the layer's bias (the last-hidden features get a
constant 1 appended). After the N states of a step are scored, the layer takes the change with
the smallest sum of squared entries that moves the error vector of each of those states by a
wanted amount, all N states at once:

    change of the layer = rows^T (rows rows^T)^-1 wanted

where `rows` [N, features + 1] holds the unit-length feature vectors of the step's states and
`wanted` [N, outputs] the wanted change of each state's error. For one state this is the wanted
change times that state's unit-length features.

- `fast_layer_scores`: the layer is a temporary copy of the trained predictor's output layer; each
  step multiplies the error of its states by 1 - eta. The copy is dropped after the rollout.
- `scheduled_layer_scores`: the layer is the bonus's own state, kept between rollouts; each step
  moves the squared error e of its states to (1 + e^(-1/a))^(-a), one place further along the
  schedule (1 + m)^(-a).

Arithmetic precision: FLOAT64 for the features after the body, the targets, the layer and the
fit. The body's forward pass stays float32 like every other network. Two reasons:

1. The N states of a step are often nearly the same state (every environment of a policy starts
   at the same cell), which makes rows rows^T nearly singular. The off-diagonal entries then
   differ from 1 by less than float32 can hold, and the solve would return rounding noise.
2. The scheduled layer is the sum of millions of small changes, and the squared error it must
   reproduce falls to 1e-6 and below. Float32 rounds a unit-length target by about 1e-7 per
   output, which is no longer small against such an error.

Cost per policy and rollout step: two products of [N, features + 1] with [features + 1,
outputs]. The N x N systems of all steps are solved once, before the loop over the steps, because
they depend on the features only.
"""
import jax
import jax.numpy as jnp

# Added to the diagonal of rows rows^T (whose diagonal is 1). It only decides how the change is
# shared between states with the same features, where the system has no unique solution; the fit
# of states with different features is off by at most this relative amount.
RIDGE = 1e-9


def unit_rows_with_bias(hidden):
    """Float32 last-hidden features [..., F] -> (float64 unit-length [..., F + 1], lengths [..., 1]).

    before: features [3.0, 0.0] ; after: ([3, 0, 1] / sqrt(10), sqrt(10))
    The appended 1 is the input of the layer's bias. It also keeps the length at 1 or more, so a
    state whose features are all zero (every unit of the last hidden layer off) still has a row.
    """
    features = hidden.astype(jnp.float64)
    rows = jnp.concatenate([features, jnp.ones(features.shape[:-1] + (1,), jnp.float64)], axis=-1)
    length = jnp.sqrt((rows * rows).sum(axis=-1, keepdims=True))
    return rows / length, length


def by_step(flat, steps: int, envs: int):
    """A flattened rollout [C, T*N, k] -> [T, C, N, k], the order the loop over the steps reads.

    Row t*N + n of the flat form is environment n at step t.
    """
    copies, rows, width = flat.shape
    if rows != steps * envs:
        raise ValueError(f"the rollout holds {rows} rows per policy, not {steps} steps x {envs} "
                         "environments; the last-layer fit needs the whole rollout in time order")
    return flat.reshape(copies, steps, envs, width).transpose(1, 0, 2, 3)


def flat_rollout(per_step):
    """Per-step values [T, C, N] -> [C, T*N], the inverse of `by_step` for one number per state."""
    steps, copies, envs = per_step.shape
    return per_step.transpose(1, 0, 2).reshape(copies, steps * envs)


def fit_rows(rows):
    """(rows rows^T + ridge)^-1 rows, [..., N, F + 1]: its transpose times the wanted changes of
    the N states' errors is the smallest change of the layer that makes them.

    before: one state, rows [[0.6, 0.8]]            ; after: [[0.6, 0.8]]
    before: two states with orthogonal unit rows    ; after: the same two rows
    before: two states with the same unit row r     ; after: [r / 2, r / 2] (each takes half)
    """
    envs = rows.shape[-2]
    if envs == 1:
        # one unit-length row: rows rows^T is 1, and the row itself is the answer
        return rows
    gram = jnp.matmul(rows, jnp.swapaxes(rows, -1, -2)) + RIDGE * jnp.eye(envs, dtype=rows.dtype)
    return jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(gram), True), rows)


def fast_layer_scores(rows, length, target, layer, step_size: float, bonus_of_error):
    """Score a rollout with a temporary output layer that learns each step right after scoring it.

    rows [T, C, N, F + 1] unit-length features, length [T, C, N, 1] their lengths before scaling,
    target [T, C, N, K], layer [C, F + 1, K] the predictor's output layer (weights, then bias),
    all float64. step_size is eta: after a step is scored, the error of its states is multiplied
    by 1 - eta. Returns the bonus of every state, [T, C, N] float64.

    before: one state seen at steps 1, 2 and 3, eta = 1   ; after: bonus b, 0, 0
    before: the same with eta = 0.5, half squared error   ; after: bonus b, b / 4, b / 16
    """
    fits = fit_rows(rows)

    def step(layer, inputs):
        """Score one step's states with the temporary layer, then fit the layer to them."""
        rows_t, fits_t, length_t, target_t = inputs
        # the error of the full-length features: length x (unit row . layer) - target
        error = length_t * jnp.matmul(rows_t, layer) - target_t
        # wanted change of each state's error is -eta x error; a unit row moves the error by
        # length x (its change of the layer), hence the division
        wanted = (-step_size) * error / length_t
        return layer + jnp.matmul(jnp.swapaxes(fits_t, -1, -2), wanted), bonus_of_error(error)

    _, bonus = jax.lax.scan(step, layer, (rows, fits, length, target))
    return bonus


def scheduled_layer_scores(rows, target, layer, decay: float):
    """Score a rollout with the kept output layer, moving each visited state along the schedule.

    rows [T, C, N, F + 1] unit-length features, target [T, C, N, K] unit-length targets, layer
    [C, F + 1, K], all float64. decay is a. The bonus of a state is its squared error e as the
    layer stands when its step is reached. The layer then moves the error of each state of the
    step from e to e' = (1 + e^(-1/a))^(-a): the schedule (1 + m)^(-a) gives the error e the
    visit count m = e^(-1/a) - 1, and e' is the schedule's value at m + 1. The error vector is
    scaled by sqrt(e' / e) = (1 + e^(1/a))^(-a/2), which needs no division by e.
    Returns (the layer after the rollout, the squared errors [T, C, N] float64).

    before: a = 1, one state seen at steps 1, 2 and 3, layer zero ; after: bonus 1, 1/2, 1/3,
            and the layer leaves that state a squared error of 1/4
    """
    fits = fit_rows(rows)

    def step(layer, inputs):
        """Read one step's squared errors, then move each of its states one place further."""
        rows_t, fits_t, target_t = inputs
        error = jnp.matmul(rows_t, layer) - target_t
        squared = (error * error).sum(axis=-1)
        scale = (1.0 + squared ** (1.0 / decay)) ** (-0.5 * decay)
        wanted = (scale - 1.0)[..., None] * error
        return layer + jnp.matmul(jnp.swapaxes(fits_t, -1, -2), wanted), squared

    return jax.lax.scan(step, layer, (rows, fits, target))
