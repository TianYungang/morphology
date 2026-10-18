"""Render the butterfly image and its structuring element as an SVG."""

import os
import sys

_package_directory = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
_source_directory = os.path.dirname(_package_directory)
sys.path[:] = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or os.curdir)) != _package_directory
]
if os.path.normcase(_source_directory) not in {
    os.path.normcase(os.path.abspath(entry or os.curdir)) for entry in sys.path
}:
    sys.path.insert(0, _source_directory)

import argparse
from pathlib import Path
import webbrowser
from xml.sax.saxutils import escape

import numpy as np
from numpy.typing import ArrayLike, NDArray

from morphological.array_txt_io import read_text_array


def _as_binary_array(array: NDArray, name: str) -> NDArray[np.int_]:
    values = np.asarray(array)
    if values.ndim != 2 or not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must be a two-dimensional binary array")
    return values.astype(np.int_, copy=False)


def _draw_array(
    array: NDArray[np.int_],
    *,
    label: str,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
) -> str:
    rows, columns = array.shape
    cell_size = min(max_width / columns, max_height / rows)
    width = columns * cell_size
    height = rows * cell_size
    offset_x = x + (max_width - width) / 2
    title = (
        f'<text x="{x + max_width / 2:.2f}" y="{y - 18:.2f}" '
        f'text-anchor="middle" font-size="18" font-family="sans-serif">'
        f"{escape(label)} ({rows} x {columns})</text>"
    )
    cells = [
        f'<rect x="{offset_x + start * cell_size:.3f}" '
        f'y="{y + row * cell_size:.3f}" '
        f'width="{(end - start) * cell_size:.3f}" '
        f'height="{cell_size:.3f}" fill="#000"/>'
        for row in range(rows)
        for start, end in _foreground_runs(array[row])
    ]
    frame = (
        f'<rect x="{offset_x:.3f}" y="{y:.3f}" width="{width:.3f}" '
        f'height="{height:.3f}" fill="none" stroke="#777" stroke-width="1"/>'
    )
    return "\n".join([title, *cells, frame])


def _foreground_runs(row: NDArray[np.int_]) -> list[tuple[int, int]]:
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def draw_binary_array(
    array: ArrayLike,
    output_path: str | Path | None = None,
    label: str = "Binary image",
) -> Path:
    """Draw one two-dimensional binary array as an SVG image."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "BinaryImg"
    binary_array = _as_binary_array(array, label)
    if binary_array.size == 0:
        raise ValueError("Array must not be empty")

    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 700">',
            '<rect width="900" height="700" fill="#fff"/>',
            _draw_array(
                binary_array,
                label=label,
                x=50,
                y=110,
                max_width=800,
                max_height=550,
            ),
            "</svg>",
        ]
    )

    destination = (
        Path(output_path)
        if output_path is not None
        else data_dir / "binary_image.svg"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return destination


def display_binary_array(
    array: ArrayLike,
    output_path: str | Path | None = None,
    label: str = "Binary image",
) -> Path:
    """Draw one binary array and open the generated SVG in a browser."""
    image_path = draw_binary_array(array, output_path, label).resolve()
    if not webbrowser.open(image_path.as_uri()):
        raise RuntimeError(f"Could not open the generated image: {image_path}")
    print(f"Displayed binary image: {image_path}")
    return image_path


def main(array: ArrayLike | None = None, label: str | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "data" / "BinaryImg" / "butterfly.txt"
    parser = argparse.ArgumentParser(
        description="Draw a two-dimensional binary array as an SVG image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="input text file (defaults to data/BinaryImg/butterfly.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="SVG output path (defaults to data/BinaryImg/binary_image.svg)",
    )
    args = parser.parse_args()
    if array is None:
        array = read_text_array(args.input)
        label = label or args.input.stem
    label = label or "Binary image"
    display_binary_array(array, args.output, label)


if __name__ == "__main__":
    main()
