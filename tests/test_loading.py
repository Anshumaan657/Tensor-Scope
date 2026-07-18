import torch

from tensor_image_lab.loading import (
    create_synthetic_batch,
    stack_images,
                                      )


def test_create_synthetic_batch_shape() -> None:
    images = create_synthetic_batch(
        batch_size=4,
        channels=3,
        height=64,
        width=64,
    )

    assert images.shape == (4, 3, 64, 64)


def test_create_synthetic_batch_properties() -> None:
    images = create_synthetic_batch(
        batch_size=2,
        channels=3,
        height=16,
        width=16,
    )

    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() < 1.0

def test_stack_images_creates_nchw_batch() -> None:
    first_image = torch.zeros(3, 4, 5)
    second_image = torch.ones(3, 4, 5)

    batch = stack_images([
        first_image,
        second_image,
    ])

    assert batch.shape == (2, 3, 4, 5)
    assert batch.ndim == 4
    assert torch.equal(batch[0], first_image)
    assert torch.equal(batch[1], second_image)