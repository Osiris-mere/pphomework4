from __future__ import annotations

import gradio as gr


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业一：图像色相/饱和度/亮度/对比度调整工具")
        with gr.Row():
            input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
        gr.Button("运行").click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo


def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业二：基于 GAN 的图像生成编辑工具")
        with gr.Row():
            input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
        gr.Button("运行").click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业三：替换屏幕内容工具")
        with gr.Row():
            input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
        gr.Button("运行").click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo


def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业四：进阶功能（要求2 + 进阶要求）")
        gr.Markdown(
            "本界面不使用 OpenCV 双边滤波和 NLM 快捷函数：手动实现双边滤波、"
            "导向滤波、积分图盒式优化快速导向滤波、手动 BM3D，并包含创意细节增强。"
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["upload", "webcam", "clipboard"],
                                       type="numpy", label="输入图像")
                method = gr.Dropdown(
                    choices=[
                        "手动双边滤波",
                        "手动导向滤波",
                        "锐化效果对比（双边 vs 导向）",
                        "快速导向滤波（积分图盒式滤波）",
                        "手动 BM3D（硬阈值 + Wiener）",
                        "创意：快速导向去噪 + 细节增强",
                    ],
                    value="手动双边滤波",
                    label="处理方法",
                )
                strength = gr.Slider(1, 100, value=35, step=1,
                                     label="手动双边滤波：灰度/颜色域强度")
                spatial = gr.Slider(1, 15, value=3, step=1,
                                    label="手动双边滤波：空间邻域半径")
                radius = gr.Slider(1, 30, value=8, step=1,
                                   label="导向/快速导向滤波：窗口半径")
                epsilon = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001,
                                    label="导向/快速导向滤波：正则化 ε")
                bm3d_sigma = gr.Slider(1, 100, value=25, step=1,
                                       label="手动 BM3D：噪声标准差")
                sharpen_amount = gr.Slider(0, 3, value=1, step=0.1,
                                           label="创意和对比：细节增强强度")
                run_button = gr.Button("开始处理", variant="primary")
            with gr.Column():
                output_image = gr.Image(
                    type="numpy",
                    label="处理结果（对比模式：左为手动双边，右为手动导向）",
                    interactive=False,
                )

        run_button.click(
            fn=process,
            inputs=[input_image, method, strength, spatial, radius,
                    epsilon, sharpen_amount, bm3d_sigma],
            outputs=[output_image],
        )
    return demo


def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业五：图像提亮工具")
        with gr.Row():
            input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
        gr.Button("运行").click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo
