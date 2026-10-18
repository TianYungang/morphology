from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from morphological import read_grayscale_image, write_grayscale_image


DATA_DIR = Path(__file__).parents[1] / "data" / "GrayscaleImg"


def test_read_rice_image_as_two_dimensional_grayscale_array() -> None:
    image = read_grayscale_image(DATA_DIR / "rice.png")

    assert image.shape == (256, 256)
    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255


def test_read_grayscale_image_preserves_pixel_values(
    tmp_path: Path,
) -> None:
    source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_path = tmp_path / "grayscale.png"
    Image.fromarray(source).save(image_path)

    image = read_grayscale_image(image_path)

    assert image.shape == (2, 2)
    np.testing.assert_array_equal(image, source)


def test_write_grayscale_image_round_trips_pixel_values(
    tmp_path: Path,
) -> None:
    source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_path = tmp_path / "round-trip.png"

    result_path = write_grayscale_image(source, image_path)

    assert result_path == image_path
    np.testing.assert_array_equal(read_grayscale_image(image_path), source)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1]),
        np.empty((0, 2)),
        np.array([[-1]]),
        np.array([[256]]),
        np.array([[1.5]]),
        np.array([[np.nan]]),
        np.array([["0"]]),
    ],
)
def test_write_grayscale_image_rejects_invalid_arrays(
    values: np.ndarray,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        write_grayscale_image(values, tmp_path / "invalid.png")
