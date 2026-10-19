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
) -> str:
    rows, columns = array.shape
    title = (
        f'<text x="{x + columns / 2:.2f}" y="{y - 16:.2f}" '
        f'text-anchor="middle" font-size="14" font-family="sans-serif" '
        f'fill="#fff" stroke="#000" stroke-width="2" '
        f'paint-order="stroke" stroke-linejoin="round">'
        f"{escape(label)} ({rows} x {columns})</text>"
    )
    foreground_paths = []
    for row in range(rows):
        top = y + row
        bottom = y + row + 1
        for start, end in _foreground_runs(array[row]):
            left = x + start
            right = x + end
            foreground_paths.append(
                f"M{left},{top}H{right}V{bottom}H{left}Z"
            )

    if not foreground_paths:
        return title

    foreground = (
        f'<path d="{"".join(foreground_paths)}" fill="#fff" '
        'shape-rendering="crispEdges"/>'
    )
    return "\n".join([title, foreground])


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

    rows, columns = binary_array.shape
    margin = 16
    image_x = margin
    image_y = 40
    canvas_width = columns + 2 * margin
    canvas_height = rows + image_y + margin
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_width}" height="{canvas_height}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}">',
            f'<rect width="{canvas_width}" height="{canvas_height}" fill="#000"/>',
            _draw_array(
                binary_array,
                label=label,
                x=image_x,
                y=image_y,
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
    """Save one binary array as an SVG image without opening it."""
    return draw_binary_array(array, output_path, label).resolve()


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
