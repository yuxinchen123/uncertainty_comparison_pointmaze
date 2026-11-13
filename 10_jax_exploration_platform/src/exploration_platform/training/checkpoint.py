"""Save and restore one training unit's whole state, so an interrupted unit continues exactly.

A checkpoint is one `.npz` file of plain arrays and one JSON record; no object is pickled:

  state/00000 ...     every array of the TrainState, in the tree's own flatten order (weights,
                      Adam moments, environment state, statistics, the run key, the iteration)
  bonus_constant/...  the bonus's fixed arrays by name (the distillation target network), so a
                      resume on another card model cannot rebuild them with different last digits
  driver/...          what the driver holds outside the state (its partial window sums)
  record              the identity of the unit (configuration, seeds) and each leaf's shape and type

Restoring puts the saved arrays into the tree of a fresh `init_state`, after checking that the
identity and every shape and type agree. The iteration number is part of the state, and each
iteration's random key is derived from it, so a restored unit draws what the uninterrupted unit
would have drawn.
"""
import json
import os
from pathlib import Path

import numpy as np

FORMAT = 1


def flatten_named(tree, prefix: str = "") -> dict:
    """Nested dictionaries of arrays as one flat dictionary keyed by path.

    before: {"target": {"W0": a, "b0": b}}
    after:  {"target/W0": a, "target/b0": b}
    """
    flat = {}
    for name, value in tree.items():
        if "/" in name:
            raise ValueError(f"the name {name!r} holds the path separator '/'")
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(flatten_named(value, path + "/"))
        else:
            flat[path] = np.asarray(value)
    return flat


def unflatten_named(flat: dict) -> dict:
    """The inverse of `flatten_named`: {"target/W0": a} -> {"target": {"W0": a}}."""
    tree = {}
    for path, value in flat.items():
        node = tree
        *parents, leaf = path.split("/")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = value
    return tree


def normalized(record):
    """A record as JSON would return it (tuples become lists), so saved and fresh ones compare."""
    return json.loads(json.dumps(record))


def save_checkpoint(path, state, identity: dict, bonus_constants=None, driver=None) -> None:
    """Write the whole state to `path`, replacing an older checkpoint only when the write is done.

    identity: what must agree before these arrays may be restored (configuration, seeds).
    bonus_constants: nested dictionaries of the bonus's fixed arrays, or None.
    driver: nested dictionaries of arrays the driver keeps outside the state, or None.
    """
    import jax

    # device arrays -> host arrays, in the tree's flatten order
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(jax.device_get(state))]
    arrays = {f"state/{index:05d}": leaf for index, leaf in enumerate(leaves)}
    arrays.update({f"bonus_constant/{name}": value
                   for name, value in flatten_named(bonus_constants or {}).items()})
    arrays.update({f"driver/{name}": value
                   for name, value in flatten_named(driver or {}).items()})
    record = {"format": FORMAT, "identity": normalized(identity),
              "leaf_shapes": [list(leaf.shape) for leaf in leaves],
              "leaf_dtypes": [str(leaf.dtype) for leaf in leaves]}
    arrays["record"] = np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)

    # write beside the target and rename, so a stop during the write leaves the old file whole
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_checkpoint(path) -> dict:
    """Read a checkpoint into host arrays: its identity, state leaves, constants and driver sums.

    Returns {"identity", "leaves", "bonus_constants" (None when the bonus has none), "driver"}.
    Nothing is put on a device here, so the bonus constants can be handed to the trainer before
    it is built.
    """
    with np.load(Path(path)) as data:
        record = json.loads(bytes(data["record"]).decode())
        if record["format"] != FORMAT:
            raise ValueError(f"checkpoint format {record['format']} is not format {FORMAT}")
        names = sorted(data.files)
        leaves = [data[name] for name in names if name.startswith("state/")]
        part = lambda prefix: unflatten_named({name[len(prefix):]: data[name] for name in names
                                               if name.startswith(prefix)})
        constants, driver = part("bonus_constant/"), part("driver/")
    if ([list(leaf.shape) for leaf in leaves] != record["leaf_shapes"]
            or [str(leaf.dtype) for leaf in leaves] != record["leaf_dtypes"]):
        raise ValueError("the checkpoint's arrays do not match its own record of them")
    return {"identity": record["identity"], "leaves": leaves,
            "bonus_constants": constants or None, "driver": driver}


def check_identity(saved: dict, current: dict, environment_defaults: dict = None,
                   bonus_option_defaults: dict = None) -> None:
    """Refuse a checkpoint written by a different unit, naming every entry that differs.

    A trainer, environment or bonus-option field added after the checkpoint was written is
    absent from its record. Such a field is added with a default that keeps the earlier program,
    so the saved record is read as holding that default: the checkpoint continues when the
    current unit holds the default too, and is refused when the current unit has changed the
    field.
    environment_defaults: the default fields of the current unit's environment configuration
    class; without them a later environment field is refused.
    bonus_option_defaults: every option of the current unit's bonus family at its default;
    without them a later bonus option is refused.

    before: saved trainer_config has no "permutation_draw"; current holds "argsort_uniform"
    after:  accepted; with current "sort_packed_keys" it is refused, naming trainer_config
    before: saved environment_config has no "physics"; current holds "jax"
    after:  accepted; with current "warp" it is refused, naming environment_config
    before: saved bonus_options has no "fast_last_layer_until_first_arrival"; current holds False
    after:  accepted; with current True it is refused, naming bonus_options
    """
    from dataclasses import asdict

    from ..agents.ppo.config import PPOConfig

    current = normalized(current)
    defaults = {"trainer_config": normalized(asdict(PPOConfig())),
                "environment_config": normalized(environment_defaults or {}),
                "bonus_options": normalized(bonus_option_defaults or {})}
    for part, part_defaults in defaults.items():
        saved_part, current_part = saved.get(part), current.get(part)
        if isinstance(saved_part, dict) and isinstance(current_part, dict):
            later = {name: part_defaults[name] for name in current_part
                     if name not in saved_part and name in part_defaults}
            saved = {**saved, part: {**later, **saved_part}}
    differing = sorted(name for name in set(saved) | set(current)
                       if saved.get(name) != current.get(name))
    if differing:
        raise ValueError("the checkpoint belongs to a different unit: these entries differ: "
                         f"{differing}")


def restore_state(template, leaves):
    """The saved arrays in the tree of a fresh state; every shape and type must agree.

    template: `Runner.init_state(...)` of a trainer built for the same configuration.
    leaves: the "leaves" of `read_checkpoint`.
    """
    import jax
    import jax.numpy as jnp

    fresh, treedef = jax.tree.flatten(template)
    if len(fresh) != len(leaves):
        raise ValueError(f"the checkpoint holds {len(leaves)} arrays and this unit's state "
                         f"holds {len(fresh)}")
    for index, (expected, saved) in enumerate(zip(fresh, leaves)):
        if tuple(expected.shape) != tuple(saved.shape) or str(expected.dtype) != str(saved.dtype):
            raise ValueError(f"state array {index} is {saved.shape}/{saved.dtype} in the "
                             f"checkpoint and {expected.shape}/{expected.dtype} in this unit")
    return jax.tree.unflatten(treedef, [jnp.asarray(leaf) for leaf in leaves])
