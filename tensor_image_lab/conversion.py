import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor from any device into a NumPy array."""
    return tensor.detach().cpu().numpy()