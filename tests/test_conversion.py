import numpy as np
import torch

from tensor_image_lab.conversion import tensor_to_numpy
from tensor_image_lab.devices import select_device


def test_tensor_to_numpy() -> None:
    device = select_device()

    tensor = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device,
    )

    array = tensor_to_numpy(tensor)

    expected = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )

    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 2)
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, expected)