import torch


def select_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Move an image batch to the selected device."""
    return batch.to(device)