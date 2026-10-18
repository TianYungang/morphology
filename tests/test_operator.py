import numpy as np
import pytest

from morphological.morphology_ops import (
    complements,
    convex_hull,
    dilations,
    erosions,
    reconstructions,
    reconstruction_steps,
)


def test_complements_inverts_binary_values_and_preserves_shape() -> None:
    image = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)

    result = complements(image)

    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 1, 1]])
    assert result.shape == image.shape
    assert np.issubdtype(result.dtype, np.integer)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1]),
        np.array([[0, 2]]),
    ],
)
def test_complements_rejects_non_binary_or_non_2d_data(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        complements(values)


def test_dilations_spreads_foreground_and_keeps_image_shape() -> None:
    image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    structuring_element = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    )

    result = dilations(image, structuring_element)

    np.testing.assert_array_equal(
        result,
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    )
    assert result.shape == image.shape
    assert np.issubdtype(result.dtype, np.integer)


def test_dilations_reflects_asymmetric_structuring_element() -> None:
    image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    structuring_element = np.array(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    )

    result = dilations(image, structuring_element)

    np.testing.assert_array_equal(
        result,
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    )


def test_erosions_keeps_pixels_where_structuring_element_fits() -> None:
    image = np.ones((3, 3), dtype=np.uint8)
    structuring_element = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    )

    result = erosions(image, structuring_element)

    np.testing.assert_array_equal(
        result,
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    )
    assert np.issubdtype(result.dtype, np.integer)


def test_erosion_result_is_indexed_at_structuring_element_origin() -> None:
    image = np.ones((3, 3), dtype=np.uint8)
    structuring_element = np.array([[1, 0], [0, 0]])

    result = erosions(image, structuring_element)

    np.testing.assert_array_equal(
        result,
        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
    )


def test_dilations_and_erosions_handle_image_boundaries() -> None:
    image = np.array([[1, 0], [0, 0]])
    structuring_element = np.ones((3, 3), dtype=np.uint8)

    np.testing.assert_array_equal(
        dilations(image, structuring_element),
        [[1, 1], [1, 1]],
    )
    np.testing.assert_array_equal(
        erosions(image, structuring_element),
        [[0, 0], [0, 0]],
    )


@pytest.mark.parametrize(
    ("image", "structuring_element"),
    [
        (np.array([0, 1]), np.ones((1, 1))),
        (np.ones((2, 2)), np.array([[1, 2]])),
        (np.ones((2, 2)), np.zeros((2, 2))),
        (np.ones((2, 2)), np.empty((0, 1))),
    ],
)
def test_morphology_rejects_invalid_inputs(
    image: np.ndarray, structuring_element: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        dilations(image, structuring_element)


def test_reconstructions_prepares_three_arrays_and_grows_marker_within_mask() -> None:
    mask = np.array([[0, 1, 1, 1, 0]])
    structuring_element = np.ones((1, 3), dtype=np.uint8)
    marker = np.array([[0, 0, 1, 0, 0]])

    result = reconstructions(mask, structuring_element, marker)

    np.testing.assert_array_equal(result, mask)


def test_convex_hull_expands_until_hit_or_miss_no_longer_adds_pixels() -> None:
    image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    foreground_element = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    )
    background_element = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    )

    result = convex_hull(image, foreground_element, background_element)

    np.testing.assert_array_equal(
        result,
        [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
    )
    np.testing.assert_array_equal(
        convex_hull(result, foreground_element, background_element),
        result,
    )


def test_reconstruction_steps_yields_each_iteration_through_convergence() -> None:
    mask = np.ones((1, 7), dtype=np.uint8)
    structuring_element = np.ones((1, 3), dtype=np.uint8)
    marker = np.array([[0, 0, 0, 1, 0, 0, 0]])

    steps = list(reconstruction_steps(mask, structuring_element, marker))

    assert [iteration for iteration, _ in steps] == [1, 2, 3, 4]
    np.testing.assert_array_equal(steps[0][1], [[0, 0, 1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(steps[1][1], [[0, 1, 1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(steps[2][1], mask)
    np.testing.assert_array_equal(steps[3][1], mask)
    np.testing.assert_array_equal(
        reconstructions(mask, structuring_element, marker),
        mask,
    )


def test_reconstruction_steps_yields_independent_array_copies() -> None:
    mask = np.ones((1, 5), dtype=np.uint8)
    structuring_element = np.ones((1, 3), dtype=np.uint8)
    marker = np.array([[0, 0, 1, 0, 0]])

    steps = reconstruction_steps(mask, structuring_element, marker)
    first_iteration, first_result = next(steps)
    second_iteration, second_result = next(steps)

    assert first_iteration == 1
    assert second_iteration == 2
    np.testing.assert_array_equal(first_result, [[0, 1, 1, 1, 0]])
    np.testing.assert_array_equal(second_result, mask)
    assert not np.shares_memory(first_result, second_result)


@pytest.mark.parametrize(
    ("mask", "structuring_element", "marker"),
    [
        (np.ones((2, 2)), np.ones((1, 1)), np.ones((1, 2))),
        (np.array([[1, 0]]), np.ones((1, 1)), np.array([[0, 1]])),
    ],
)
def test_reconstructions_rejects_invalid_marker(
    mask: np.ndarray,
    structuring_element: np.ndarray,
    marker: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        reconstructions(mask, structuring_element, marker)
