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

def channel_statistics(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-channel mean and standard deviation for an NCHW batch."""

    means = batch.mean(dim=(0, 2, 3))
    standard_deviations = batch.std(
        dim=(0, 2, 3),
        unbiased=False,
    )

    return means, standard_deviations
def normalize_batch(
    batch: torch.Tensor,
    means: torch.Tensor,
    standard_deviations: torch.Tensor,
) -> torch.Tensor:
    """Normalize an NCHW batch using per-channel statistics."""

    broadcast_means = means.view(1, -1, 1, 1)
    broadcast_stds = standard_deviations.view(1, -1, 1, 1)

    return (batch - broadcast_means) / broadcast_stds

def rgb_to_grayscale(batch: torch.Tensor) -> torch.Tensor:
    """Convert an NCHW RGB batch into an NCHW grayscale batch."""

    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=batch.dtype,
        device=batch.device,
    ).view(1, 3, 1, 1)

    return (batch * weights).sum(dim=1, keepdim=True)

def nchw_to_nhwc(batch: torch.Tensor) -> torch.Tensor:
    """Rearrange a batch from NCHW to NHWC."""
    return batch.permute(0, 2, 3, 1)


def nhwc_to_nchw(batch: torch.Tensor) -> torch.Tensor:
    """Rearrange a batch from NHWC to NCHW."""
    return batch.permute(0, 3, 1, 2)
