"""Tools for morphological image analysis."""

if not __package__:
    import os
    import sys

    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __package__ = "morphological"

from pathlib import Path
import numpy as np

from .array_txt_io import read_text_array, write_text_array
from .morphology_ops import (
    convex_hull,
    erosions,
    openings,
    complements,
    reconstruction_steps,
)
from .visualize_binary import display_binary_array


def main() -> None:
    """Compute and display an opening result with its input arrays."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "BinaryImg"

    """对butterfly图像进行开运算，使用结构元素se_1.txt"""
    # 读取butterfly 和 structuring element.
    image = read_text_array(data_dir / "butterfly.txt")
    se = read_text_array(data_dir / "se_1.txt")

    # 执行opening操作.
    opening_result = openings(image, se)

    # 写入文件并显示到图片中.
    write_text_array(opening_result, data_dir / "opening_result.txt")
    display_binary_array(image, data_dir / "butterfly.svg", "Butterfly")
    display_binary_array(se, data_dir / "se_1.svg", "Structuring element")
    display_binary_array(
        opening_result,
        data_dir / "opening_result.svg",
        "Opening result",
    )


    """对butterfly的补集进行开运算"""
    # 计算补集.
    complement_image = complements(image)
    # 执行opening操作.
    opening_result_complement = openings(complement_image, se)
    # 写入文件并显示到图片中.
    write_text_array(opening_result_complement, data_dir / "opening_result_complement.txt")
    display_binary_array(
        opening_result_complement,
        data_dir / "opening_result_complement.svg",
        "Opening result (complement)",
    )


    """对butterfly进行closing操作，使用结构元素se_1.txt"""
    # 执行closing操作.
    closing_result = openings(complements(image), se)
    # 写入文件并显示到图片中.
    write_text_array(closing_result, data_dir / "closing_result.txt")
    display_binary_array(
        closing_result,
        data_dir / "closing_result.svg",
        "Closing result",
    )


    """对butterfly补集进行重建，marker为marker.txt，se_1.txt为结构元素"""
    # 读取marker.
    marker = read_text_array(data_dir / "marker.txt")
    # 绘制marker图像.
    display_binary_array(marker, data_dir / "marker.svg", "Marker")
    # 执行重建操作.
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

    # 写入并显示第1次、第5次以及最后一次重建结果.
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


    """对重建的集合计算凸包"""
    convex_image = last_result.copy()

    #读取4组共8个结构元素数组
    for i in range(1, 5):
        se1 = read_text_array(data_dir / f"convex_hull_se/B{i}1.txt")
        se2 = read_text_array(data_dir / f"convex_hull_se/B{i}2.txt")

        # 每组结构元均从重建结果出发，迭代至该方向不再增长.
        directional_hull = convex_hull(last_result, se1, se2)
        write_text_array(directional_hull, data_dir / f"convex_directional_{i}.txt")
        display_binary_array(directional_hull, data_dir / f"convex_directional_{i}.svg", f"Convex Directional Hull {i}")
        # 将当前方向的凸包并入总结果.
        convex_image = np.maximum(convex_image, directional_hull)

    write_text_array(convex_image, data_dir / "convexImg.txt")
    display_binary_array(convex_image, data_dir / "convexImg.svg", "ConvexImg")




    
        



__all__ = ["main", "read_text_array", "write_text_array"]


if __name__ == "__main__":
    main()
