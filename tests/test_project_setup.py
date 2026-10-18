import numpy as np


def test_numpy_array_operations() -> None:
    image = np.array([[0, 1], [2, 3]])

    assert image.shape == (2, 2)
    assert int(image.sum()) == 6
