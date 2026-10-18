from pathlib import Path

import numpy as np
import pytest

from morphological import read_text_array, write_text_array


DATA_DIR = Path(__file__).parents[1] / "data" / "BinaryImg"


@pytest.mark.parametrize("filename", ["butterfly.txt", "marker.txt", "se_1.txt"])
def test_read_binary_image_text_files(filename: str) -> None:
    array = read_text_array(DATA_DIR / filename)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    assert array.size > 0


def test_read_text_array_keeps_single_row_two_dimensional(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "single-row.txt"
    file_path.write_text("1,2,3", encoding="utf-8")

    array = read_text_array(file_path)

    assert array.shape == (1, 3)
    np.testing.assert_array_equal(array, [[1, 2, 3]])


def test_read_text_array_supports_whitespace_delimiter(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("1 2\n3 4\n", encoding="utf-8")

    array = read_text_array(file_path, delimiter=None)

    assert array.shape == (2, 2)
    np.testing.assert_array_equal(array, [[1, 2], [3, 4]])


def test_write_text_array_matches_comma_separated_integer_format(
    tmp_path: Path,
) -> None:
    array = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int64)
    file_path = tmp_path / "array.txt"

    result_path = write_text_array(array, file_path)

    assert result_path == file_path
    assert file_path.read_text(encoding="utf-8") == "0,1,0\n1,1,1\n"
    np.testing.assert_array_equal(read_text_array(file_path), array)


def test_write_text_array_preserves_floating_point_values(
    tmp_path: Path,
) -> None:
    array = np.array([[0.125, 1.2345678901234567]])
    file_path = tmp_path / "float-array.txt"

    write_text_array(array, file_path)

    np.testing.assert_array_equal(read_text_array(file_path), array)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1]),
        np.empty((0, 2)),
        np.array([["zero", "one"]]),
    ],
)
def test_write_text_array_rejects_invalid_input(
    values: np.ndarray, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        write_text_array(values, tmp_path / "invalid.txt")
