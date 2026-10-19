from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from morphological import visualize_binary
from morphological.visualize_binary import (
    _as_binary_array,
    display_binary_array,
    draw_binary_array,
)


def test_draw_binary_array_creates_svg(tmp_path: Path) -> None:
    array = np.array([[0, 1], [1, 0]])
    output = draw_binary_array(array, tmp_path / "image.svg", "Example")

    root = ET.parse(output).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    title = root.find(".//svg:text", namespace)
    background = root.find(".//svg:rect", namespace)
    foreground = root.find(".//svg:path", namespace)

    assert output.is_file()
    assert title is not None
    assert title.text is not None and title.text.startswith("Example (2 x 2)")
    assert title.attrib["fill"] == "#fff"
    assert title.attrib["stroke"] == "#000"
    assert title.attrib["paint-order"] == "stroke"
    assert title.attrib["stroke-linejoin"] == "round"
    assert root.attrib["width"] == "34"
    assert root.attrib["height"] == "58"
    assert root.attrib["viewBox"] == "0 0 34 58"
    assert background is not None
    assert background.attrib["fill"] == "#000"
    assert foreground is not None
    assert foreground.attrib["fill"] == "#fff"
    assert foreground.attrib["shape-rendering"] == "crispEdges"
    assert foreground.attrib["d"].startswith("M17,40H18V41H17Z")
    assert foreground.attrib["d"].count("M") == 2


def test_draw_binary_array_accepts_morphology_result(tmp_path: Path) -> None:
    opening_result = np.array([[0, 1], [0, 0]], dtype=np.int64)

    output = draw_binary_array(
        opening_result,
        tmp_path / "opening.svg",
        "Opening result",
    )

    assert output.is_file()
    assert "Opening result (2 x 2)" in output.read_text(encoding="utf-8")


def test_display_binary_array_saves_its_svg(tmp_path: Path) -> None:
    output = tmp_path / "saved.svg"
    saved_path = display_binary_array(
        np.array([[0, 1], [1, 0]]),
        output,
        "Single array",
    )

    assert saved_path == output.resolve()
    assert saved_path.is_file()
    assert "Single array (2 x 2)" in saved_path.read_text(encoding="utf-8")


def test_as_binary_array_converts_values_to_integer() -> None:
    array = _as_binary_array(np.array([[0.0, 1.0]]), "array")

    assert np.issubdtype(array.dtype, np.integer)
    np.testing.assert_array_equal(array, [[0, 1]])


def test_main_saves_generated_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "images.svg"
    monkeypatch.setattr(
        "sys.argv", ["visualize_binary", "--output", str(output)]
    )

    visualize_binary.main()

    assert output.is_file()


def test_main_reads_input_text_file_before_drawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("0,1\n1,0\n", encoding="utf-8")
    output = tmp_path / "sample.svg"
    monkeypatch.setattr(
        "sys.argv",
        ["visualize_binary", "--input", str(source), "--output", str(output)],
    )

    visualize_binary.main()

    assert "sample (2 x 2)" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1]),
        np.array([[0, 2]]),
    ],
)
def test_as_binary_array_rejects_non_binary_or_non_2d_data(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _as_binary_array(values, "array")
