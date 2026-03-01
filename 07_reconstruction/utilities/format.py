import numpy as np
import torch
from typing import Union


def to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Convert numpy / tensor input to a float32 tensor on the given device."""
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    return torch.as_tensor(x, dtype=torch.float32, device=device)

