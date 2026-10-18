"""Read and write two-dimensional numeric text arrays."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def read_text_array(
    path: str | Path, delimiter: str | None = ","
) -> NDArray[np.float64]:
    """Read a delimited numeric text file as a two-dimensional NumPy array."""
    return np.loadtxt(path, delimiter=delimiter, ndmin=2)


def write_text_array(
    array: ArrayLike,
    path: str | Path,
    delimiter: str = ",",
) -> Path:
    """Write a two-dimensional numeric array as a delimited text file."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("array must be two-dimensional")
    if 0 in values.shape:
        raise ValueError("array must not be empty")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("array must contain numeric values")

    destination = Path(path)
    np.savetxt(destination, values, delimiter=delimiter, fmt="%.18g")
    return destination
