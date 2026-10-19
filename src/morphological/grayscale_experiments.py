"""the grayscale morphology experiments."""

from pathlib import Path

from morphological.array_txt_io import read_text_array
from morphological.visualize_binary import display_binary_array

from morphological.grayscale_image_io import (
    read_grayscale_image,
    write_grayscale_image,
)
from morphological.grayscale_ops import binarize_grayscale, openings, top_hat


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


if __name__ == "__main__":
    run_grayscale_experiments()
