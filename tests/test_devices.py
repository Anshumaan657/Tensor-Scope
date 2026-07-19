import torch

from tensor_image_lab.devices import (
    move_batch_to_device,
    select_device,
)


def test_select_device() -> None:
    device = select_device()

    assert device.type in {"cpu", "cuda", "mps"}


def test_move_batch_to_device() -> None:
    batch = torch.rand(2, 3, 8, 8)
    device = select_device()

    moved_batch = move_batch_to_device(batch, device)

    assert moved_batch.device.type == device.type
    assert moved_batch.shape == batch.shape
    assert moved_batch.dtype == batch.dtype