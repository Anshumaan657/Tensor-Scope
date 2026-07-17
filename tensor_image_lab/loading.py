import numpy as np
import torch
from PIL import Image

def create_synthetic_batch(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Create a synthetic NCHW image batch with values in [0, 1)."""
    return torch.rand(batch_size, channels, height, width)


def load_image(path: str) -> torch.Tensor:
    """Load an RGB image as an HWC uint8 tensor."""

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        image_array = np.array(rgb_image)

    return torch.from_numpy(image_array)