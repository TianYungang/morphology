# Morphological

用于图像形态学分析的 Python 工程，使用 NumPy 进行数组运算。

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 运行测试

```powershell
python -m pytest
```

## 读取数值文本

`read_text_array` 可以读取逗号分隔的数值文本文件，并返回二维 NumPy 数组。
传入 `delimiter=None` 可读取空白字符分隔的数据。

```python
from morphological import read_text_array

image = read_text_array("data/BinaryImg/butterfly.txt")
marker = read_text_array("data/BinaryImg/marker.txt")
structuring_element = read_text_array("data/BinaryImg/se_1.txt")
```

可以使用 `write_text_array` 将二维数值数组写成逗号分隔文本：

```python
from morphological import write_text_array

write_text_array(structuring_element, "output/se_1.txt")
```

## 读取灰度图片

`read_grayscale_image` 将图像转换为 8 位灰度值，并返回形状为
`(高度, 宽度)` 的二维 NumPy 数组。例如 `data/GrayscaleImg/rice.png`
应得到形状为 `(256, 256)` 的数组。`write_grayscale_image` 可将二维数组保存为灰度图片；
数组值必须是 0 到 255 之间的整数。

```python
from morphological import read_grayscale_image, write_grayscale_image

image = read_grayscale_image("data/GrayscaleImg/rice.png")
print(image.shape, image.dtype)
write_grayscale_image(image, "output/rice.png")
```

## 二值形态学运算

`morphological.morphology_ops` 提供二值侵蚀、扩张、重建和凸包函数。参与运算的数组应为
只包含 `0` 和 `1` 的二维数组；结构元的中心作为原点，图像边界之外按背景 `0`
处理。`complements(A)` 可计算二值数组的补集。重建运算将标记图 `M` 限制在掩膜
`A` 内，并使用结构元 `B` 迭代扩张至稳定。

```python
from morphological.morphology_ops import (
    complements,
    convex_hull,
    dilations,
    erosions,
    reconstructions,
    reconstruction_steps,
)

inverted = complements(image)
eroded = erosions(image, structuring_element)
dilated = dilations(image, structuring_element)
reconstructed = reconstructions(mask, structuring_element, marker)
directional_hull = convex_hull(image, foreground_element, background_element)

for iteration, intermediate_result in reconstruction_steps(
    mask, structuring_element, marker
):
    if iteration in (1, 5):
        print(f"Iteration {iteration}:", intermediate_result)
```

## 灰度形态学运算

`morphological.grayscale_ops` 提供基于平坦结构元（非零元素作为足迹）的灰度侵蚀、扩张和阈值二值化。
侵蚀在足迹范围内取局部最小值，扩张取局部最大值；结构元原点为中心（偶数尺寸向下取整）。
边界外使用中性极值，不影响窗口内像素。二值化将严格大于阈值的像素置为 `1`，其余置为 `0`。

```python
import numpy as np

from morphological.grayscale_ops import (
    binarize_grayscale,
    grayscale_dilations,
    grayscale_erosions,
)

flat_se = np.ones((3, 3), dtype=np.uint8)
eroded = grayscale_erosions(image, flat_se)
dilated = grayscale_dilations(image, flat_se)
binary = binarize_grayscale(image, threshold=128)
```

## 绘制二值图像

绘图函数接收一个二维二值数组并绘制为 SVG。文本文件可先用
`read_text_array` 读取成数组；运行项目主函数时会直接绘制 `opening_result`。
生成的 SVG 会尝试在默认浏览器中打开。

```python
from morphological import read_text_array
from morphological.visualize_binary import draw_binary_array

image = read_text_array("data/BinaryImg/butterfly.txt")
draw_binary_array(image, "output/butterfly.svg", label="Butterfly")
```

命令行默认读取 `data/BinaryImg/butterfly.txt`，也可通过 `--input` 和 `--output`
指定输入文本文件和 SVG 输出位置。若要在代码中直接显示数组，可调用
`display_binary_array(array, output_path, label)`。

```powershell
.\.venv\Scripts\python.exe src\morphological\__init__.py
.\.venv\Scripts\python.exe -m morphological.visualize_binary
.\.venv\Scripts\python.exe src\morphological\visualize_binary.py --output output\binary_images.svg
```
