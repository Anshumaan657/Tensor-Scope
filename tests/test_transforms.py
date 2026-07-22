import pytest
import torch

from tensor_image_lab.loading import create_synthetic_batch
from tensor_image_lab.transforms import (
    channel_statistics,
    crop_batch,
    nchw_to_nhwc,
    nhwc_to_nchw,
    normalize_batch,
    rgb_to_grayscale,
)


def test_crop_batch_shape() -> None:
    images = create_synthetic_batch(
        batch_size=4,
        channels=3,
        height=64,
        width=64,
    )

    cropped_images = crop_batch(
        images,
        top=8,
        left=12,
        height=24,
        width=40,
    )

    assert cropped_images.shape == (4, 3, 24, 40)


def test_channel_statistics() -> None:
    batch = torch.tensor(
        [
            [
                [[0.0, 2.0]],
                [[2.0, 4.0]],
                [[4.0, 6.0]],
            ]
        ]
    )

    means, standard_deviations = channel_statistics(batch)

    expected_means = torch.tensor([1.0, 3.0, 5.0])
    expected_stds = torch.tensor([1.0, 1.0, 1.0])

    assert means.shape == (3,)
    assert standard_deviations.shape == (3,)
    torch.testing.assert_close(means, expected_means)
    torch.testing.assert_close(standard_deviations, expected_stds)


def test_normalize_batch() -> None:
    batch = torch.tensor(
        [
            [
                [[0.0, 2.0]],
                [[2.0, 4.0]],
                [[4.0, 6.0]],
            ]
        ]
    )
    means = torch.tensor([1.0, 3.0, 5.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])

    normalized = normalize_batch(batch, means, standard_deviations)

    expected = torch.tensor(
        [
            [
                [[-1.0, 1.0]],
                [[-1.0, 1.0]],
                [[-1.0, 1.0]],
            ]
        ]
    )

    assert normalized.shape == batch.shape
    torch.testing.assert_close(normalized, expected)


def test_rgb_to_grayscale() -> None:
    red_image = torch.tensor(
        [
            [
                [[1.0]],
                [[0.0]],
                [[0.0]],
            ]
        ]
    )

    grayscale = rgb_to_grayscale(red_image)

    assert grayscale.shape == (1, 1, 1, 1)
    torch.testing.assert_close(grayscale, torch.tensor([[[[0.299]]]]))


def test_nchw_nhwc_round_trip() -> None:
    batch = torch.arange(
        2 * 3 * 4 * 5,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5)

    nhwc_batch = nchw_to_nhwc(batch)
    restored_batch = nhwc_to_nchw(nhwc_batch)

    assert nhwc_batch.shape == (2, 4, 5, 3)
    assert restored_batch.shape == batch.shape
    assert torch.equal(restored_batch, batch)


def test_crop_batch_rejects_out_of_bounds_crop() -> None:
    batch = torch.rand(2, 3, 16, 16)

    with pytest.raises(ValueError, match="beyond the image boundaries"):
        crop_batch(batch, top=8, left=8, height=16, width=16)


def test_channel_statistics_rejects_non_nchw_tensor() -> None:
    image = torch.rand(3, 16, 16)

    with pytest.raises(ValueError, match="Expected an NCHW batch"):
        channel_statistics(image)


def test_normalize_batch_rejects_wrong_channel_statistics() -> None:
    batch = torch.rand(2, 3, 8, 8)
    means = torch.zeros(2)
    standard_deviations = torch.ones(3)

    with pytest.raises(ValueError, match=r"Means must have shape \[C\]"):
        normalize_batch(batch, means, standard_deviations)


def test_normalize_batch_rejects_zero_standard_deviation() -> None:
    batch = torch.rand(2, 3, 8, 8)
    means = torch.zeros(3)
    standard_deviations = torch.tensor([1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="greater than zero"):
        normalize_batch(batch, means, standard_deviations)


def test_rgb_to_grayscale_requires_three_channels() -> None:
    grayscale_batch = torch.rand(2, 1, 8, 8)

    with pytest.raises(ValueError, match="Expected 3 channels"):
        rgb_to_grayscale(grayscale_batch)
