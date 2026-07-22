import torch


def _validate_nchw_batch(
    batch: torch.Tensor,
    *,
    channels: int | None = None,
) -> None:
    if batch.ndim != 4:
        raise ValueError("Expected an NCHW batch with shape [N, C, H, W].")
    if 0 in batch.shape:
        raise ValueError("An image batch cannot contain empty dimensions.")
    if channels is not None and batch.shape[1] != channels:
        raise ValueError(f"Expected {channels} channels, got {batch.shape[1]}.")


def crop_batch(
    images: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Crop the same rectangular region from every NCHW image."""
    _validate_nchw_batch(images)

    if top < 0 or left < 0:
        raise ValueError("Crop coordinates must be non-negative.")
    if height <= 0 or width <= 0:
        raise ValueError("Crop height and width must be positive.")

    bottom = top + height
    right = left + width

    if bottom > images.shape[2] or right > images.shape[3]:
        raise ValueError("Crop extends beyond the image boundaries.")

    return images[:, :, top:bottom, left:right]


def hwc_to_chw(image: torch.Tensor) -> torch.Tensor:
    """Rearrange one image from HWC layout to CHW layout."""
    if image.ndim != 3:
        raise ValueError("Expected an HWC image with shape [H, W, C].")
    return image.permute(2, 0, 1)


def uint8_to_float(image: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 image from [0, 255] to float32 in [0, 1]."""
    if image.dtype != torch.uint8:
        raise TypeError("Expected a torch.uint8 image tensor.")
    return image.to(torch.float32) / 255.0


def channel_statistics(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-channel mean and standard deviation for an NCHW batch."""
    _validate_nchw_batch(batch)
    if not batch.is_floating_point():
        raise TypeError("Channel statistics require a floating-point batch.")

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
    _validate_nchw_batch(batch)

    channels = batch.shape[1]
    if means.ndim != 1 or means.numel() != channels:
        raise ValueError("Means must have shape [C].")
    if standard_deviations.ndim != 1 or standard_deviations.numel() != channels:
        raise ValueError("Standard deviations must have shape [C].")
    if means.device != batch.device or standard_deviations.device != batch.device:
        raise ValueError("Batch, means, and standard deviations must share a device.")
    if torch.any(standard_deviations <= 0):
        raise ValueError("Standard deviations must be greater than zero.")

    broadcast_means = means.view(1, -1, 1, 1)
    broadcast_stds = standard_deviations.view(1, -1, 1, 1)

    return (batch - broadcast_means) / broadcast_stds


def rgb_to_grayscale(batch: torch.Tensor) -> torch.Tensor:
    """Convert an NCHW RGB batch into an NCHW grayscale batch."""
    _validate_nchw_batch(batch, channels=3)
    if not batch.is_floating_point():
        raise TypeError("Grayscale conversion requires a floating-point batch.")

    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=batch.dtype,
        device=batch.device,
    ).view(1, 3, 1, 1)

    return (batch * weights).sum(dim=1, keepdim=True)


def nchw_to_nhwc(batch: torch.Tensor) -> torch.Tensor:
    """Rearrange a batch from NCHW to NHWC."""
    _validate_nchw_batch(batch)
    return batch.permute(0, 2, 3, 1)


def nhwc_to_nchw(batch: torch.Tensor) -> torch.Tensor:
    """Rearrange a batch from NHWC to NCHW."""
    if batch.ndim != 4:
        raise ValueError("Expected an NHWC batch with shape [N, H, W, C].")
    if 0 in batch.shape:
        raise ValueError("An image batch cannot contain empty dimensions.")
    return batch.permute(0, 3, 1, 2)
