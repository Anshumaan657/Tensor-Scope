from tensor_image_lab.loading import create_synthetic_batch
from tensor_image_lab.transforms import crop_batch


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