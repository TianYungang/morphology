from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def _as_binary_array(array: NDArray, name: str) -> NDArray[np.int_]:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must contain only binary values (0 and 1)")
    return values.astype(np.int_, copy=False)


def _prepare_arrays(
    arrays: tuple[NDArray, ...],
) -> tuple[NDArray[np.int_], ...]:
    """Validate and convert any number of binary arrays."""
    if not arrays:
        raise ValueError("At least one array is required")
    return tuple(
        _as_binary_array(array, f"array {index}")
        for index, array in enumerate(arrays, start=1)
    )


def _validate_structuring_element(structuring_element: NDArray) -> None:
    if 0 in structuring_element.shape or not np.any(structuring_element):
        raise ValueError("B must be non-empty and contain at least one active element")

"""对二值二维数组计算补集"""
def complements(A: NDArray) -> NDArray[np.int_]:
    """Return the binary complement of a two-dimensional array."""
    image = _as_binary_array(A, "A")
    return 1 - image

def erosions(A: NDArray, B: NDArray) -> NDArray[np.int_]:
    """Return the binary erosion of image A by structuring element B.

    The structuring-element origin is its center (floor for even dimensions).
    Pixels outside A are treated as background.
    """
    image, structuring_element = _prepare_arrays((A, B))
    _validate_structuring_element(structuring_element)
    height, width = image.shape
    se_height, se_width = structuring_element.shape
    center_y, center_x = (size // 2 for size in structuring_element.shape)
    """将原图像进行扩展，扩展的大小为结构元素的大小减去1，扩展的方式为常数填充0"""
    padded = np.pad(
        image,
        (
            (center_y, se_height - center_y - 1),
            (center_x, se_width - center_x - 1),
        ),
        mode="constant",
        constant_values=0,
    )

    result = np.ones(image.shape, dtype=np.int_)

    for se_row, se_column in np.argwhere(structuring_element):
        result &= padded[
            se_row : se_row + height,
            se_column : se_column + width,
        ]

    return result


def dilations(A: NDArray, B: NDArray) -> NDArray[np.int_]:
    """Return the binary dilation of image A by structuring element B.

    The structuring-element origin is its center (floor for even dimensions).
    Pixels outside A are treated as background.
    """
    image, structuring_element = _prepare_arrays((A, B))
    _validate_structuring_element(structuring_element)
    height, width = image.shape
    center_y, center_x = (size // 2 for size in structuring_element.shape)
    padded = np.pad(
        image,
        (
            (structuring_element.shape[0] - 1, structuring_element.shape[0] - 1),
            (structuring_element.shape[1] - 1, structuring_element.shape[1] - 1),
        ),
        mode="constant",
        constant_values=0,
    )
    result = np.zeros(image.shape, dtype=np.int_)

    for row, column in np.argwhere(structuring_element):
        start_y = structuring_element.shape[0] - 1 + center_y - row
        start_x = structuring_element.shape[1] - 1 + center_x - column
        result |= padded[start_y : start_y + height, start_x : start_x + width]

    return result


def openings(A: NDArray, B: NDArray) -> NDArray[np.int_]:
    """Return the binary opening of image A by structuring element B."""
    return dilations(erosions(A, B), B)


def closings(A: NDArray, B: NDArray) -> NDArray[np.int_]:
    """Return the binary closing of image A by structuring element B."""
    return erosions(dilations(A, B), B) 



# 按步数返回marker的重建过程，例如返回第5次，应该这样调用：
# for iteration, result in reconstruction_steps(A, B, M):
#     if iteration == 5:
#         print(result)
#         break
def reconstruction_steps(
    A: NDArray, B: NDArray, M: NDArray
) -> Iterator[tuple[int, NDArray[np.int_]]]:
    """Yield each dilation step of marker reconstruction, including convergence.

    Iteration numbering starts at 1. Each yielded array is a copy, so later
    iterations cannot modify previously yielded results.
    """
    mask, structuring_element, marker = _prepare_arrays((A, B, M))
    _validate_structuring_element(structuring_element)
    if marker.shape != mask.shape:
        raise ValueError("M must have the same shape as A")
    if np.any(marker > mask):
        raise ValueError("M must be a subset of A")

    result = marker.copy()
    iteration = 0
    while True:
        next_result = np.logical_and(
            dilations(result, structuring_element),
            mask,
        ).astype(np.int_)
        iteration += 1
        yield iteration, next_result.copy()
        if np.array_equal(next_result, result):
            return
        result = next_result


# 返回marker重建的最终结果
def reconstructions(A: NDArray, B: NDArray, M: NDArray) -> NDArray[np.int_]:
    """Reconstruct marker M inside mask A and return the final result."""
    result: NDArray[np.int_] | None = None
    for _, result in reconstruction_steps(A, B, M):
        pass
    if result is None:
        raise RuntimeError("Reconstruction produced no iterations")
    return result


def convex_hull(A: NDArray, B1: NDArray, B2: NDArray) -> NDArray[np.int_]:
    """Expand a binary image to a fixed point using a hit-or-miss pair.

    B1 matches the foreground pattern and B2 matches the background pattern.
    """
    (image,) = _prepare_arrays((A,))
    foreground_element, background_element = _prepare_arrays((B1, B2))
    _validate_structuring_element(foreground_element)
    _validate_structuring_element(background_element)

    while True:
        hit_or_miss = np.minimum(
            erosions(image, foreground_element),
            erosions(complements(image), background_element),
        )
        expanded = np.maximum(image, hit_or_miss)
        if np.array_equal(expanded, image):
            return image
        image = expanded