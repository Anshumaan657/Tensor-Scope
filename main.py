from tensor_image_lab.loading import (
    create_synthetic_batch,
    load_image,
    stack_images,
)
from tensor_image_lab.transforms import (
    channel_statistics,
    crop_batch,
    hwc_to_chw,
    uint8_to_float,
)
def main() -> None:
    images = create_synthetic_batch(
        batch_size=4,
        channels=3,
        height=64,
        width=64,
    )

    cropped_images = crop_batch(
        images,
        top=16,
        left=16,
        height=32,
        width=32,
    )

    print("Synthetic batch shape:", images.shape)
    print("Cropped batch shape:", cropped_images.shape)

    real_image = load_image("images/sample.jpg")
    chw_image = hwc_to_chw(real_image)
    float_image = uint8_to_float(chw_image)

    real_batch = stack_images([
        float_image,
        float_image,
    ])
    channel_means, channel_stds = channel_statistics(real_batch)

    print("\nLoaded image (HWC):")
    print("Shape:", real_image.shape)
    print("Datatype:", real_image.dtype)
    print(
        "Value range:",
        real_image.min().item(),
        "to",
        real_image.max().item(),
    )

    print("\nPrepared image (CHW):")
    print("Shape:", float_image.shape)
    print("Datatype:", float_image.dtype)
    print(
        "Value range:",
        float_image.min().item(),
        "to",
        float_image.max().item(),
    )

    print("\nReal image batch:")
    print("Shape:", real_batch.shape)
    print("Dimensions:", real_batch.ndim)
    print("Datatype:", real_batch.dtype)
    print("Batch size:", real_batch.shape[0])

    print("\nChannel statistics:")
    print("Means:", channel_means.tolist())
    print("Standard deviations:", channel_stds.tolist())


if __name__ == "__main__":
    main()
