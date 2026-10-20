"""Run the binary morphology experiments."""

from pathlib import Path

import numpy as np

from morphological.array_txt_io import read_text_array, write_text_array
from morphological.morphology_ops import (
    closings,
    convex_hull,
    complements,
    openings,
    reconstruction_steps,
)
from morphological.visualize_binary import display_binary_array


def run_binary_experiments() -> None:
    """Run and save all binary morphology experiments."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "BinaryImg"

    image = read_text_array(data_dir / "butterfly.txt")
    se = read_text_array(data_dir / "se_1.txt")

    """原始图进行开操作"""
    opening_result = openings(image, se)
    write_text_array(opening_result, data_dir / "opening_result.txt")
    display_binary_array(image, data_dir / "butterfly.svg", "Butterfly")
    display_binary_array(se, data_dir / "se_1.svg", "Structuring element")
    display_binary_array(
        opening_result,
        data_dir / "opening_result.svg",
        "Opening result",
    )

    """补集图进行开操作"""
    complement_image = complements(image)
    opening_result_complement = openings(complement_image, se)
    write_text_array(
        opening_result_complement,
        data_dir / "opening_result_complement.txt",
    )
    display_binary_array(complement_image, data_dir / "butterfly_complement.svg", "Butterfly (complement)")
    display_binary_array(
        opening_result_complement,
        data_dir / "opening_result_complement.svg",
        "Opening result (complement)",
    )

    """原始图进行闭操作"""
    closing_result = closings(image, se)
    write_text_array(closing_result, data_dir / "closing_result.txt")
    display_binary_array(
        closing_result,
        data_dir / "closing_result.svg",
        "Closing result",
    )

    """原始图闭操作结果取补集"""
    closing_result_complement = complements(closing_result)
    write_text_array(
        closing_result_complement,
        data_dir / "closing_result_complement.txt",
    )
    display_binary_array(
        closing_result_complement,
        data_dir / "closing_result_complement.svg",
        "Closing result (complement)",
    )

    """补集图闭闭操作结果取补集"""
    closing_result_complement_complement = complements(
        closing_result_complement
    )
    write_text_array(
        closing_result_complement_complement,
        data_dir / "closing_result_complement_complement.txt",
    )
    display_binary_array(
        closing_result_complement_complement,
        data_dir / "closing_result_complement_complement.svg",
        "Closing result (complement) (complement)",
    )

    """重建操作"""
    marker = read_text_array(data_dir / "marker.txt")
    display_binary_array(marker, data_dir / "marker.svg", "Marker")
    reconstruction_result = reconstruction_steps(complement_image, se, marker)
    selected_results = {}
    last_iteration = 0
    last_result = None
    for iteration, result in reconstruction_result:
        if iteration in (1, 5):
            selected_results[iteration] = result
        last_iteration = iteration
        last_result = result

    if last_result is not None and last_iteration not in selected_results:
        selected_results[last_iteration] = last_result

    for iteration, result in selected_results.items():
        write_text_array(
            result,
            data_dir / f"reconstruction_result_{iteration}.txt",
        )
        display_binary_array(
            result,
            data_dir / f"reconstruction_result_{iteration}.svg",
            f"Reconstruction result ({iteration})",
        )


    """凸包操作"""
    convex_image = last_result.copy()
    for i in range(1, 5):
        se1 = read_text_array(data_dir / f"convex_hull_se/B{i}1.txt")
        se2 = read_text_array(data_dir / f"convex_hull_se/B{i}2.txt")

        directional_hull = convex_hull(last_result, se1, se2)
        write_text_array(
            directional_hull,
            data_dir / f"convex_directional_{i}.txt",
        )
        display_binary_array(
            directional_hull,
            data_dir / f"convex_directional_{i}.svg",
            f"Convex Directional Hull {i}",
        )
        convex_image = np.maximum(convex_image, directional_hull)

    write_text_array(convex_image, data_dir / "convexImg.txt")
    display_binary_array(convex_image, data_dir / "convexImg.svg", "ConvexImg")


if __name__ == "__main__":
    run_binary_experiments()
