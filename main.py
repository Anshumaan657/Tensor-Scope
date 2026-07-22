from pathlib import Path

import torch

from tensor_image_lab.conversion import tensor_to_numpy
from tensor_image_lab.devices import (
    move_batch_to_device,
    select_device,
)
from tensor_image_lab.loading import (
    create_synthetic_batch,
    load_image_batch,
)
from tensor_image_lab.transforms import (
    channel_statistics,
    crop_batch,
    nchw_to_nhwc,
    nhwc_to_nchw,
    normalize_batch,
    rgb_to_grayscale,
)
from tensor_image_lab.visualization import visualize_transformations


def discover_image_paths(image_directory: Path) -> list[str]:
    """Return sorted paths for supported images in a directory."""
    supported_extensions = {".jpg", ".jpeg", ".png"}

    if not image_directory.is_dir():
        return []

    return sorted(
        str(path)
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )


def main(show_visualization: bool = True) -> None:
    synthetic_batch = create_synthetic_batch(
        batch_size=4,
        channels=3,
        height=64,
        width=64,
    )
    cropped_batch = crop_batch(
        synthetic_batch,
        top=16,
        left=16,
        height=32,
        width=32,
    )

    print("Synthetic batch shape:", synthetic_batch.shape)
    print("Cropped batch shape:", cropped_batch.shape)

    image_paths = discover_image_paths(Path("images"))
    if not image_paths:
        raise FileNotFoundError(
            "Add at least one JPG, JPEG, or PNG file to the images/ directory."
        )

    real_batch = load_image_batch(
        paths=image_paths,
        target_size=(256, 256),
    )

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

    grayscale_batch = rgb_to_grayscale(real_batch)
    nhwc_batch = nchw_to_nhwc(real_batch)
    restored_nchw_batch = nhwc_to_nchw(nhwc_batch)
    image_array = tensor_to_numpy(nhwc_batch[0])

    print("\nDevice:")
    print("Selected device:", device)
    print("Batch device:", real_batch.device)

    print("\nReal image batch:")
    print("Image paths:", image_paths)
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

    print("\nNumPy conversion:")
    print("Type:", type(image_array))
    print("Shape:", image_array.shape)
    print("Datatype:", image_array.dtype)

    visualize_transformations(
        original_batch=real_batch,
        normalized_batch=normalized_batch,
        grayscale_batch=grayscale_batch,
        show=show_visualization,
    )


if __name__ == "__main__":
    main()
