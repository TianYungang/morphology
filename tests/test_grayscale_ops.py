import numpy as np
import pytest
from collections.abc import Callable
from numpy.typing import ArrayLike

from morphological.grayscale_ops import (
    binarize_grayscale,
    grayscale_dilations,
    grayscale_erosions,
)


def test_grayscale_erosion_uses_local_minimum() -> None:
    image = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.uint8)
    flat_element = np.ones((3, 3), dtype=np.uint8)

    result = grayscale_erosions(image, flat_element)

    np.testing.assert_array_equal(
        result,
        [[5, 4, 4], [2, 1, 1], [2, 1, 1]],
    )
    assert result.dtype == image.dtype


def test_grayscale_dilation_uses_local_maximum() -> None:
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    flat_element = np.ones((3, 3), dtype=np.uint8)

    result = grayscale_dilations(image, flat_element)

    np.testing.assert_array_equal(
        result,
        [[5, 6, 6], [8, 9, 9], [8, 9, 9]],
    )
    assert result.dtype == image.dtype


def test_grayscale_morphology_respects_flat_footprint_and_origin() -> None:
    image = np.array([[0, 10, 0], [20, 30, 40], [0, 50, 0]])
    flat_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    np.testing.assert_array_equal(
        grayscale_erosions(image, flat_element),
        [[0, 0, 0], [0, 10, 0], [0, 0, 0]],
    )
    np.testing.assert_array_equal(
        grayscale_dilations(image, flat_element),
        [[20, 30, 40], [30, 50, 40], [50, 50, 50]],
    )


def test_grayscale_morphology_handles_single_active_element() -> None:
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    flat_element = np.array([[0, 0], [1, 0]])

    np.testing.assert_array_equal(
        grayscale_erosions(image, flat_element),
        [[255, 1], [255, 3]],
    )
    np.testing.assert_array_equal(
        grayscale_dilations(image, flat_element),
        [[2, 0], [4, 0]],
    )


def test_binarize_grayscale_uses_strict_threshold() -> None:
    image = np.array([[0, 127, 128, 255]], dtype=np.uint8)

    result = binarize_grayscale(image, threshold=127)

    np.testing.assert_array_equal(result, [[0, 0, 1, 1]])
    assert np.issubdtype(result.dtype, np.integer)


@pytest.mark.parametrize(
    ("function", "image", "element"),
    [
        (grayscale_erosions, np.array([1, 2]), np.ones((1, 1))),
        (grayscale_dilations, np.ones((2, 2)), np.zeros((1, 1))),
        (grayscale_erosions, np.array([[np.nan]]), np.ones((1, 1))),
        (grayscale_dilations, np.ones((2, 2)), np.ones((0, 1))),
    ],
)
def test_grayscale_morphology_rejects_invalid_inputs(
    function: Callable[[ArrayLike, ArrayLike], np.ndarray],
    image: np.ndarray,
    element: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        function(image, element)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, 1.5, "1", True])
def test_binarize_grayscale_rejects_invalid_threshold(
    threshold: object,
) -> None:
    with pytest.raises(ValueError):
        binarize_grayscale(np.array([[0, 1]]), threshold)
