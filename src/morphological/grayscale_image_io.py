"""Read and write grayscale images as two-dimensional NumPy arrays."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

"""读取灰度图像"""
def read_grayscale_image(path: str | Path) -> NDArray[np.uint8]:
    """Read an image as a two-dimensional uint8 grayscale array."""
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)

"""写入灰度图像"""
def write_grayscale_image(array: ArrayLike, path: str | Path) -> Path:
    """Write a two-dimensional array of 8-bit grayscale values to an image."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("array must be two-dimensional")
    if 0 in values.shape:
        raise ValueError("array must not be empty")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("array must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("array values must be finite")
    if np.any((values < 0) | (values > 255)):
        raise ValueError("array values must be between 0 and 255")
    if np.any(values != np.floor(values)):
        raise ValueError("array values must be whole numbers")

    destination = Path(path)
    Image.fromarray(values.astype(np.uint8)).save(destination)
    return destination
