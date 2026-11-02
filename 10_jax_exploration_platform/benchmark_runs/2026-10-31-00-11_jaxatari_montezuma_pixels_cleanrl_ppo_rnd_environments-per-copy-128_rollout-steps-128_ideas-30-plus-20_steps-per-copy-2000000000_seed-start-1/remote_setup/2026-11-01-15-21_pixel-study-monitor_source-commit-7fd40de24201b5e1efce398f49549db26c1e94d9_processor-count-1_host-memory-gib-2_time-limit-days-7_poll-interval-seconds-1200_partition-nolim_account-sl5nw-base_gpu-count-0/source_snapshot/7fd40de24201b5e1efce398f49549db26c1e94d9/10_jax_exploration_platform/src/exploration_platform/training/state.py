"""Everything one training iteration reads and writes, and how the trainable tree is stored.

The state is one immutable tree. Nothing on the hot path is an object with methods: the compiled
program takes this tree in and gives a new one back, and the donated buffers make that an update
in place on the device.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class TrainState(NamedTuple):
    """The whole of a run's mutable state.

    `agent_params` / `bonus_params` are the two halves of the trainable tree, and `opt` is ONE
    Adam over both of them together — the baseline trained the bonus's predictor jointly with the
    policy, and keeping that means the gradient-norm clip also spans both halves.

    `obs` is here rather than inside `env_state` because the environment's state object holds
    positions and velocities, not the observation the next rollout will start from. `visited` is a
    measurement accumulator, not part of the algorithm: it is a fixed-shape array so the compiled
    program's shape never depends on it, and it is [C, 0] when coverage is not being tracked.
    """
    env_state: NamedTuple      # the environment's own state, [C, N, ...]
    obs: jnp.ndarray           # [C, N, 4] the observation the next rollout starts from
    agent_params: dict         # the agent's parameter tree, or the whole packed array
    bonus_params: dict         # the bonus's parameter tree ({} when packed, or when it has none)
    opt: NamedTuple            # Adam over {"agent": ..., "bonus": ...}
    agent_state: dict          # what the agent carries between iterations
    bonus_state: dict          # what the bonus carries between iterations
    visited: jnp.ndarray       # [C, rows*cols] bool, cumulative visited-cell map (or [C, 0])
    rng: jnp.ndarray           # the run's key; iteration k folds k into it
    step: jnp.ndarray          # int32 scalar, the number of iterations run so far


class ParamLayout:
    """Where each named tensor sits inside the one-array parameter form, and whether it is used.

    Holding every parameter in ONE array of shape [copies, parameters per copy] instead of
    twenty-one named arrays makes the gradient clip one reduction and Adam a handful of
    elementwise operations rather than twenty-one of each. The networks are unchanged: the array
    is cut back into the named tensors before every forward pass.
    """

    def __init__(self, tree, packed: bool):
        """Record the flatten order, each tensor's trailing shape, and its column range."""
        # before: 21 arrays, e.g. actor W0 [C, 4, 64], actor b0 [C, 64], ...
        # after:  one array [C, P] with P = 4*64 + 64 + ..., tensor i occupying columns
        #         offsets[i] : offsets[i] + sizes[i], reshaped back to its own trailing shape
        self.packed = packed
        leaves, self.treedef = jax.tree.flatten(tree)
        self.shapes = [leaf.shape for leaf in leaves]
        self.sizes = [int(np.prod(leaf.shape[1:])) for leaf in leaves]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.flat_size = int(self.offsets[-1])

    def pack(self, tree):
        """The named parameter tensors -> one array [C, P], copies still on axis 0."""
        return jnp.concatenate(
            [leaf.reshape(leaf.shape[0], -1) for leaf in jax.tree.flatten(tree)[0]], axis=1)

    def unpack(self, flat):
        """One array [C, P] -> the named parameter tensors, each with its own shape."""
        # before: flat [C, P]; after: e.g. actor W0 = flat[:, 0:256] viewed as [C, 4, 64]
        C = flat.shape[0]
        parts = [flat[:, o:o + s].reshape((C,) + shape[1:])
                 for o, s, shape in zip(self.offsets, self.sizes, self.shapes)]
        return jax.tree.unflatten(self.treedef, parts)


def stored_params(state: TrainState, layout: ParamLayout):
    """The trainable parameters in the form the state keeps them in: one array, or two trees."""
    if layout.packed:
        return state.agent_params
    return {"agent": state.agent_params, "bonus": state.bonus_params}


def named_params(stored, layout: ParamLayout):
    """The trainable parameters as {"agent": ..., "bonus": ...}, whatever form they were stored in."""
    return layout.unpack(stored) if layout.packed else stored


def put_params(state: TrainState, stored, layout: ParamLayout) -> TrainState:
    """Write a stored-form parameter value back into the state."""
    if layout.packed:
        return state._replace(agent_params=stored)
    return state._replace(agent_params=stored["agent"], bonus_params=stored["bonus"])
