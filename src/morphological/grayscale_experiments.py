"""the grayscale morphology experiments."""

from pathlib import Path

import numpy as np

from morphological.array_txt_io import read_text_array
from morphological.visualize_binary import display_binary_array

from morphological.grayscale_image_io import (
    read_grayscale_image,
    write_grayscale_image,
)
from morphological.grayscale_ops import (
    binarize_grayscale,
    grayscale_dilations,
    openings,
    top_hat,
)


def run_grayscale_experiments() -> None:
    """Run and save all grayscale morphology experiments."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "GrayscaleImg"

    """1、灰度图阈值分割操作"""
    # 读取灰度图
    grayscale_image = read_grayscale_image(data_dir / "rice.png")
    # 阈值分割操作
    binary_image = binarize_grayscale(grayscale_image, threshold=150)
    # 保存二值图像
    display_binary_array(
        binary_image,
        data_dir / "rice_binary.svg",
        "Rice binary image",
    )

    """Flat SE B1对grayscale_image开运算"""
    # 读取结构元
    se_b1 = read_text_array(data_dir / "se_1.txt")
    # 开运算操作
    opened_image = openings(grayscale_image, se_b1)
    # 保存开运算后的图像
    write_grayscale_image(opened_image, data_dir / "rice_opened.png")


    """Flat SE B1对grayscale_image进行Top-Hat变换"""
    top_hat_image = top_hat(grayscale_image, se_b1)
    # 保存Top-Hat变换后的图像
    write_grayscale_image(top_hat_image, data_dir / "rice_top_hat.png")


    """"对top-hat结果进行阈值分割操作，阈值为60"""
    top_hat_binary_image = binarize_grayscale(top_hat_image, threshold=60)
    # 保存二值图像
    display_binary_array(
        top_hat_binary_image,
        data_dir / "rice_top_hat_binary.svg",
        "Rice top-hat binary image",
    )

    """对top-hat的分割结果进行开运算，去除噪声"""
    # 读取结构元se_2.txt
    se_b2 = read_text_array(data_dir / "se_2.txt")
    top_hat_opened_binary_image = openings(top_hat_binary_image, se_b2)
    # 保存开运算后的图像
    display_binary_array(
        top_hat_opened_binary_image,
        data_dir / "rice_top_hat_opened_binary.svg",
        "Rice top-hat opened binary image",
    )


    """进行H-dome重建"""
    # 计算marker 原图像-45，小于等于0取0
    marker = np.maximum(grayscale_image.astype(np.int16) - 45, 0).astype(
        np.uint8
    )
    write_grayscale_image(marker, data_dir / "rice_marker.png")
    # 执行H-dome重建，grayscale_image作为source，marker作为marker，se_b2作为结构元
    # 重建过程应该是 marker的膨胀操作，每次膨胀后取与source的最小值，直到不再变化为止
    h_dome_reconstructed_image = marker.copy()
    selected_reconstructions = {}
    iteration = 0
    while True:
        iteration += 1
        # 膨胀操作
        dilated_image = grayscale_dilations(h_dome_reconstructed_image, se_b2)
        # 与source取最小值
        new_reconstructed_image = np.minimum(dilated_image, grayscale_image)
        if iteration == 1:
            selected_reconstructions["1"] = new_reconstructed_image.copy()
        elif iteration == 5:
            selected_reconstructions["5"] = new_reconstructed_image.copy()
        # 检查是否收敛
        if np.array_equal(new_reconstructed_image, h_dome_reconstructed_image):
            selected_reconstructions[
                f"last_{iteration}"
            ] = new_reconstructed_image.copy()
            break
        h_dome_reconstructed_image = new_reconstructed_image
    for suffix, reconstruction in selected_reconstructions.items():
        write_grayscale_image(
            reconstruction,
            data_dir / f"rice_h_dome_reconstruction_{suffix}.png",
        )
    # 原图减去重建结果得到H-dome结果
    h_dome_image = grayscale_image.astype(np.int16) - h_dome_reconstructed_image
    # 保存H-dome重建后的图像
    write_grayscale_image(h_dome_image, data_dir / "rice_h_dome.png")

if __name__ == "__main__":
    run_grayscale_experiments()
