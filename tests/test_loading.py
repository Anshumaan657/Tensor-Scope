from tensor_image_lab.loading import create_synthetic_batch


def test_create_synthetic_batch_shape() -> None:
    images = create_synthetic_batch(
        batch_size=4,
        channels=3,
        height=64,
        width=64,
    )

    assert images.shape == (4, 3, 64, 64)