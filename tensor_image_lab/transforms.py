import torch


def crop_batch(
    images: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Crop the same rectangular region from every NCHW image."""

    bottom = top + height
    right = left + width

    return images[:, :, top:bottom, left:right]

def hwc_to_chw(image: torch.Tensor) -> torch.Tensor:
    """Rearrange one image from HWC layout to CHW layout."""
    return image.permute(2, 0, 1)

def uint8_to_float(image: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 image from [0, 255] to float32 in [0, 1]."""
    return image.to(torch.float32) / 255.0