"""Restore the complete learner only when its policy seeds and environment settings agree."""

from dataclasses import asdict
from pathlib import Path
import pickle


def load_checkpoint(path, configuration, copy_seeds, environment_configuration=None):
    """Validate recorded run identity before restoring every array onto the selected device."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload["configuration"] != asdict(configuration):
        raise ValueError("checkpoint learner configuration does not match this unit")
    if tuple(payload["copy_seeds"]) != tuple(copy_seeds):
        raise ValueError("checkpoint policy seeds do not match this unit")
    if "environment_configuration" not in payload:
        raise ValueError("checkpoint has no recorded environment configuration")
    if payload["environment_configuration"] != dict(environment_configuration or {}):
        raise ValueError("checkpoint environment settings do not match this unit")

    # Random keys, environment state and optimizer moments resume together; no priming is repeated.
    import jax
    return jax.tree.map(jax.device_put, payload["state"])
