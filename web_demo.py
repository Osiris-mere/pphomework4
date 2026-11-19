from __future__ import annotations

import gradio as gr

from UI import create_demo_hw1, create_demo_hw2, create_demo_hw3, create_demo_hw4, create_demo_hw5
from functions import function_hw1, function_hw2, function_hw3, function_hw4, function_hw5


HTML_DESCRIPTION = """
<div align="center">
  <h1 style="font-weight: 900; margin-bottom: 7px;">图像处理网页演示工具</h1>
  <p>使用方式：在浏览器中打开 http://127.0.0.1:8088/</p>
  <p>作者：图小像</p>
  <p>学号：2400000000</p>
</div>
"""

MD_DESCRIPTION = """
## 作业四：进阶去噪与锐化

- 手动双边滤波
- 手动导向滤波及双边/导向锐化对比
- 积分图盒式滤波优化的快速导向滤波
- 手动 BM3D（硬阈值与 Wiener 两阶段）
- 创意：快速导向去噪 + 可调细节增强

本版本只保留要求2与进阶功能，不使用要求1中的 OpenCV 快捷滤波函数。
"""


def main():
    with gr.Blocks() as demo:
        gr.Markdown(HTML_DESCRIPTION)
        gr.Markdown(MD_DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem("作业1：图像色相/饱和度/亮度/对比度调整工具"):
                create_demo_hw1(function_hw1)
            with gr.TabItem("作业2：基于 GAN 的图像生成编辑工具"):
                create_demo_hw2(function_hw2)
            with gr.TabItem("作业3：替换屏幕内容工具"):
                create_demo_hw3(function_hw3)
            with gr.TabItem("作业4：进阶去噪与锐化"):
                create_demo_hw4(function_hw4)
            with gr.TabItem("作业5：图像提亮工具"):
                create_demo_hw5(function_hw5)
    demo.launch(server_port=8088)


if __name__ == "__main__":
    main()
