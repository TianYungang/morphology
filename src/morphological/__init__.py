"""Tools for morphological image analysis."""

if not __package__:
    import os
    import sys

    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "morphological"

from .binary_experiments import run_binary_experiments
from .array_txt_io import read_text_array, write_text_array
from .grayscale_image_io import read_grayscale_image, write_grayscale_image
from .grayscale_experiments import run_grayscale_experiments
from .grayscale_ops import (
    binarize_grayscale,
    grayscale_dilations,
    grayscale_erosions,
)
from .morphology_ops import (
    convex_hull,
    erosions,
    openings,
    complements,
    reconstruction_steps,
)
from .visualize_binary import display_binary_array


def main() -> None:
    """Run all binary and grayscale morphology experiments."""
    run_binary_experiments()
    run_grayscale_experiments()



__all__ = [
    "main",
    "run_binary_experiments",
    "run_grayscale_experiments",
    "convex_hull",
    "erosions",
    "openings",
    "complements",
    "reconstruction_steps",
    "binarize_grayscale",
    "grayscale_dilations",
    "grayscale_erosions",
    "read_grayscale_image",
    "read_text_array",
    "write_grayscale_image",
    "write_text_array",
    "display_binary_array",
]


if __name__ == "__main__":
    main()
