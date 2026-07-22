import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from tensor_image_lab.transforms import rgb_to_grayscale
from tensor_image_lab.visualization import visualize_transformations


def test_visualize_transformations() -> None:
    original = torch.rand(2, 3, 8, 8)
    normalized = torch.randn(2, 3, 8, 8)
    grayscale = rgb_to_grayscale(original)

    figure = visualize_transformations(
        original_batch=original,
        normalized_batch=normalized,
        grayscale_batch=grayscale,
        show=False,
    )

    assert len(figure.axes) == 3

    plt.close(figure)