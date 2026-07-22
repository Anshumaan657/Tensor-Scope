import numpy as np
import torch
from PIL import Image


def _require_positive_dimensions(**dimensions: int) -> None:
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def create_synthetic_batch(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Create a synthetic NCHW image batch with values in [0, 1)."""
    _require_positive_dimensions(
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    return torch.rand(batch_size, channels, height, width)


def load_image(path: str) -> torch.Tensor:
    """Load an RGB image as an HWC uint8 tensor."""

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        image_array = np.array(rgb_image)

    return torch.from_numpy(image_array)


def stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    """Combine equally shaped CHW images into one NCHW batch."""
    if not images:
        raise ValueError("At least one image tensor is required.")

    expected_shape = images[0].shape
    for image in images:
        if image.ndim != 3:
            raise ValueError("Every image must have CHW shape [C, H, W].")
        if image.shape != expected_shape:
            raise ValueError("All images must have the same CHW shape.")

    return torch.stack(images, dim=0)


def load_image_batch(
    paths: list[str],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Load multiple images as an equally sized NCHW float32 batch."""

    if not paths:
        raise ValueError("At least one image path is required.")

    target_height, target_width = target_size
    _require_positive_dimensions(
        target_height=target_height,
        target_width=target_width,
    )
    images = []

    for path in paths:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            resized_image = rgb_image.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            )
            image_array = np.array(resized_image)

        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.to(torch.float32) / 255.0

        images.append(image_tensor)

    return torch.stack(images, dim=0)
