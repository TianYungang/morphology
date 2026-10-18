"""Flat-structuring-element operations for grayscale images."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _prepare_grayscale_image(image: ArrayLike) -> NDArray:
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if 0 in values.shape:
        raise ValueError("image must not be empty")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("image must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("image values must be finite")
    return values


def _prepare_flat_structuring_element(element: ArrayLike) -> NDArray[np.bool_]:
    """校验平坦结构元并将其转换为布尔数组。

    输入必须可转换为非空二维数值数组；非零位置表示结构元参与运算，
    零值位置不参与。返回的布尔数组可直接用于灰度侵蚀和扩张。
    参数使用 ArrayLike，因此既可传入 NumPy 数组，也可传入嵌套列表等
    可转换为数组的数据；二维形状由函数在运行时校验。
    """
    values = np.asarray(element)
    if values.ndim != 2:
        raise ValueError("structuring element must be two-dimensional")
    if 0 in values.shape:
        raise ValueError("structuring element must not be empty")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        raise ValueError("structuring element must contain numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("structuring element values must be finite")

    footprint = values != 0
    if not np.any(footprint):
        raise ValueError("structuring element must contain an active element")
    return footprint


def _grayscale_extreme(image: NDArray, *, erosion: bool) -> int | float:
    """返回灰度形态学运算在图像边界外使用的中性极值。

    侵蚀取局部最小值，因此边界外填充最大值；扩张取局部最大值，
    因此边界外填充最小值。浮点数组使用正负无穷表示这两个极值。
    """
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        return limits.max if erosion else limits.min
    return np.inf if erosion else -np.inf


def grayscale_erosions(image: ArrayLike, structuring_element: ArrayLike) -> NDArray:
    """Apply grayscale erosion using the local minimum over a flat footprint.

    The structuring-element origin is its center (floor for even dimensions).
    Values outside the image use the neutral maximum for erosion.
    """
    values = _prepare_grayscale_image(image)
    footprint = _prepare_flat_structuring_element(structuring_element)
    height, width = values.shape
    se_height, se_width = footprint.shape
    center_y, center_x = (size // 2 for size in footprint.shape)
    padded = np.pad(
        values,
        (
            (center_y, se_height - center_y - 1),
            (center_x, se_width - center_x - 1),
        ),
        mode="constant",
        constant_values=_grayscale_extreme(values, erosion=True),
    )

    active_positions = np.argwhere(footprint)
    first_row, first_column = active_positions[0]
    result = padded[
        first_row : first_row + height,
        first_column : first_column + width,
    ].copy()
    for row, column in active_positions[1:]:
        np.minimum(
            result,
            padded[row : row + height, column : column + width],
            out=result,
        )
    return result


def grayscale_dilations(
    image: ArrayLike, structuring_element: ArrayLike
) -> NDArray:
    """Apply grayscale dilation using the local maximum over a flat footprint.

    The structuring-element origin is its center (floor for even dimensions).
    Values outside the image use the neutral minimum for dilation.
    """
    values = _prepare_grayscale_image(image)
    footprint = _prepare_flat_structuring_element(structuring_element)
    height, width = values.shape
    se_height, se_width = footprint.shape
    center_y, center_x = (size // 2 for size in footprint.shape)
    padded = np.pad(
        values,
        (
            (se_height - 1, se_height - 1),
            (se_width - 1, se_width - 1),
        ),
        mode="constant",
        constant_values=_grayscale_extreme(values, erosion=False),
    )

    active_positions = np.argwhere(footprint)
    first_row, first_column = active_positions[0]
    start_y = se_height - 1 + center_y - first_row
    start_x = se_width - 1 + center_x - first_column
    result = padded[
        start_y : start_y + height,
        start_x : start_x + width,
    ].copy()
    for row, column in active_positions[1:]:
        start_y = se_height - 1 + center_y - row
        start_x = se_width - 1 + center_x - column
        np.maximum(
            result,
            padded[
                start_y : start_y + height,
                start_x : start_x + width,
            ],
            out=result,
        )
    return result


"""对灰度图像进行阈值分割操作，大于阈值的置为1，小于等于的置为0，变成二值数组"""
def binarize_grayscale(image: ArrayLike, threshold: int) -> NDArray[np.int_]:
    """Return 1 where grayscale pixels exceed threshold and 0 elsewhere."""
    values = _prepare_grayscale_image(image)
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, np.integer)
    ):
        raise ValueError("threshold must be an integer")
    return (values > threshold).astype(np.int_)
