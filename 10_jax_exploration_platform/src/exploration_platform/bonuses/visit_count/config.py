"""Knobs of the oracle visit-count bonus."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VisitCountConfig:
    """The counted state's discretisation and how fast the bonus falls with the count.

    decay is negative so the bonus decreases as a state is revisited: -0.5 gives 1/sqrt(n) and
    -1 gives 1/n. The velocity settings reproduce 07_reconstruction's: the environment clips each
    velocity component to +-5 m/s, and that range is cut into 10 equal bins per axis.
    """
    decay: float = -0.5
    velocity_bins: int = 10
    velocity_clip: float = 5.0
