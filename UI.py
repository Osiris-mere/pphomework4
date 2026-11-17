from __future__ import annotations

import gradio as gr


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业一：图像色相/饱和度/亮度/对比度调整工具")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
                mode = gr.Radio(["色相/饱和度", "亮度/对比度"], label="模式", value="色相/饱和度")
                h_slider = gr.Slider(-1, 1, label="色相", value=0, step=0.01)
                s_slider = gr.Slider(-1, 1, label="饱和度", value=0, step=0.01)
                l_slider = gr.Slider(-1, 1, label="亮度", value=0, step=0.01)
                brightness = gr.Slider(-1, 1, label="亮度", value=0, step=0.01, visible=False)
                contrast = gr.Slider(-1, 1, label="对比度", value=0, step=0.01, visible=False)
            with gr.Column():
                output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
                with gr.Row():
                    r_channel = gr.Image(type="numpy", label="R 通道", interactive=False)
                    g_channel = gr.Image(type="numpy", label="G 通道", interactive=False)
                    b_channel = gr.Image(type="numpy", label="B 通道", interactive=False)
                with gr.Row():
                    h_channel = gr.Image(type="numpy", label="H 通道", interactive=False)
                    s_channel = gr.Image(type="numpy", label="S 通道", interactive=False)
                    l_channel = gr.Image(type="numpy", label="L 通道", interactive=False)
        inputs = [input_image, mode, h_slider, s_slider, l_slider, brightness, contrast]
        outputs = [output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel]
        input_image.change(fn=process, inputs=inputs, outputs=outputs)

        def update_mode(selected):
            hsv_visible = selected == "色相/饱和度"
            return tuple(gr.update(visible=hsv_visible) for _ in range(3)) + tuple(
                gr.update(visible=not hsv_visible) for _ in range(2))

        mode.change(update_mode, inputs=[mode], outputs=[h_slider, s_slider, l_slider, brightness, contrast])
        for slider in (h_slider, s_slider, l_slider, brightness, contrast):
            slider.release(fn=process, inputs=inputs, outputs=outputs)
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
            with gr.Column():
                bg_img = gr.ImageEditor(sources=["upload", "webcam", "clipboard"], type="numpy", label="背景图像",
                                        brush=gr.Brush(default_size=4, colors=["#FF0000"], color_mode="fixed"),
                                        eraser=False, layers=False)
                reset_btn = gr.Button("重置并选择屏幕四个角点")
                fg_img = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="前景图像")
                mode = gr.Radio(choices=["Nearest", "Bilinear", "Bicubic"], value="Nearest", label="插值方法")
            with gr.Column():
                output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
                run_btn = gr.Button("替换屏幕内容")
        reset_btn.click(fn=lambda image: image["background"] if image else None,
                        inputs=[bg_img], outputs=[bg_img])
        run_btn.click(fn=process, inputs=[bg_img, fg_img, mode], outputs=[output_image])
    return demo


def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业四：图像去噪与锐化效果比较")
        gr.Markdown(
            "支持 OpenCV 双边滤波、OpenCV 彩色 NLM（fastNlMeansDenoisingColored），"
            "以及手动实现的双边滤波和导向滤波。可选择并排比较两种滤波结果用于细节锐化的效果。"
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
                method = gr.Dropdown(
                    choices=["双边滤波（OpenCV）", "NLM 去噪（OpenCV）", "手动双边滤波",
                             "手动导向滤波", "锐化效果对比（双边 vs 导向）"],
                    value="双边滤波（OpenCV）", label="处理方法")
                strength = gr.Slider(1, 100, value=35, step=1, label="滤波强度 / NLM h")
                spatial = gr.Slider(1, 15, value=3, step=1, label="空间邻域半径（双边滤波）")
                radius = gr.Slider(1, 30, value=5, step=1, label="导向滤波半径")
                epsilon = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001, label="导向滤波正则化参数 ε")
                sharpen_amount = gr.Slider(0, 3, value=1, step=0.1, label="细节锐化强度")
                run_button = gr.Button("开始处理", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="numpy", label="处理结果（锐化对比模式下：左为双边滤波，右为导向滤波）",
                                        interactive=False)
        run_button.click(fn=process,
                         inputs=[input_image, method, strength, spatial, radius, epsilon, sharpen_amount],
                         outputs=[output_image])
    return demo


def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业五：图像提亮工具")
        with gr.Row():
            input_image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="输入图像")
            output_image = gr.Image(type="numpy", label="输出图像", interactive=False)
        gr.Button("运行").click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo
