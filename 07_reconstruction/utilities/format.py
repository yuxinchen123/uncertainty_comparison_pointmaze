import numpy as np
import torch
from typing import Union


def to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Convert numpy / tensor input to a float32 tensor on the given device."""
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def to_numpy_flat(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert numpy or tensor input to a 1D float32 numpy array."""
    # .ravel() flattens to 1D in row-major order (no copy when layout allows).
    # e.g. shape (256, 1) -> (256,); (batch_size, 4) -> (batch_size * 4,)
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32).ravel()
    return np.asarray(x, dtype=np.float32).ravel()

