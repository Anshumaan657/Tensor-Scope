import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from tensor_image_lab.conversion import tensor_to_numpy


def _scale_for_display(array: np.ndarray) -> np.ndarray:
    minimum = array.min()
    maximum = array.max()

    if maximum == minimum:
        return np.zeros_like(array)

    return (array - minimum) / (maximum - minimum)


def visualize_transformations(
    original_batch: torch.Tensor,
    normalized_batch: torch.Tensor,
    grayscale_batch: torch.Tensor,
    show: bool = True,
) -> Figure:
    """Display original, normalized and grayscale images."""

    original = tensor_to_numpy(
        original_batch[0].permute(1, 2, 0)
    )
    normalized = tensor_to_numpy(
        normalized_batch[0].permute(1, 2, 0)
    )
    grayscale = tensor_to_numpy(
        grayscale_batch[0, 0]
    )

    normalized = _scale_for_display(normalized)
    grayscale = _scale_for_display(grayscale)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("Original")

    axes[1].imshow(normalized)
    axes[1].set_title("Normalized")

    axes[2].imshow(grayscale, cmap="gray")
    axes[2].set_title("Grayscale")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    if show:
        plt.show()

    return figure