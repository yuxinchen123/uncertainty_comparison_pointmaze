"""Knobs of the random-network-distillation bonus."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RNDConfig:
    """The two network widths; everything else about this bonus is fixed by the algorithm."""
    feature_dim: int = 128   # width of the target's output, which the predictor tries to match
    hidden: int = 256        # width of both networks' hidden layers
