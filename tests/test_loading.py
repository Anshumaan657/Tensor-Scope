import pytest
import torch
from PIL import Image

from tensor_image_lab.loading import (
    create_synthetic_batch,
    load_image_batch,
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


def test_load_image_batch_standardizes_sizes(tmp_path) -> None:
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"

    Image.new(
        "RGB",
        size=(20, 10),
        color=(255, 0, 0),
    ).save(first_path)

    Image.new(
        "RGB",
        size=(40, 30),
        color=(0, 255, 0),
    ).save(second_path)

    batch = load_image_batch(
        paths=[str(first_path), str(second_path)],
        target_size=(16, 24),
    )

    assert batch.shape == (2, 3, 16, 24)
    assert batch.dtype == torch.float32
    assert batch.min().item() >= 0.0
    assert batch.max().item() <= 1.0


def test_load_image_batch_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="At least one image path"):
        load_image_batch(paths=[], target_size=(16, 16))


def test_load_image_batch_rejects_invalid_target_size() -> None:
    with pytest.raises(ValueError, match="target_height"):
        load_image_batch(paths=["unused.jpg"], target_size=(0, 16))


def test_stack_images_rejects_mismatched_shapes() -> None:
    images = [
        torch.zeros(3, 10, 10),
        torch.zeros(3, 20, 10),
    ]

    with pytest.raises(ValueError, match="same CHW shape"):
        stack_images(images)
