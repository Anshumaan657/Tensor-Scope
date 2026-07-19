import torch

from tensor_image_lab.devices import (
    move_batch_to_device,
    select_device,
)
from tensor_image_lab.loading import (
    create_synthetic_batch,
    load_image,
    stack_images,
)
from tensor_image_lab.transforms import (
    channel_statistics,
    crop_batch,
    hwc_to_chw,
    nchw_to_nhwc,
    nhwc_to_nchw,
    normalize_batch,
    rgb_to_grayscale,
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

    device = select_device()
    real_batch = move_batch_to_device(real_batch, device)

    channel_means, channel_stds = channel_statistics(real_batch)
    normalized_batch = normalize_batch(
        real_batch,
        channel_means,
        channel_stds,
    )
    normalized_means, normalized_stds = channel_statistics(
        normalized_batch
    )
    grayscale_batch = rgb_to_grayscale(normalized_batch)
    nhwc_batch = nchw_to_nhwc(real_batch)
    restored_nchw_batch = nhwc_to_nchw(nhwc_batch)

    print("\nDevice:")
    print("Selected device:", device)
    print("Batch device:", real_batch.device)

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
    print("Means:", channel_means.cpu().tolist())
    print("Standard deviations:", channel_stds.cpu().tolist())

    print("\nNormalized batch:")
    print("Shape:", normalized_batch.shape)
    print("Channel means:", normalized_means.cpu().tolist())
    print(
        "Channel standard deviations:",
        normalized_stds.cpu().tolist(),
    )

    print("\nGrayscale batch:")
    print("Shape:", grayscale_batch.shape)
    print("Datatype:", grayscale_batch.dtype)
    print(
        "Value range:",
        grayscale_batch.min().item(),
        "to",
        grayscale_batch.max().item(),
    )

    print("\nBatch layouts:")
    print("NCHW shape:", real_batch.shape)
    print("NHWC shape:", nhwc_batch.shape)
    print("Restored NCHW shape:", restored_nchw_batch.shape)
    print(
        "Round trip preserved values:",
        torch.equal(real_batch, restored_nchw_batch),
    )


if __name__ == "__main__":
    main()
