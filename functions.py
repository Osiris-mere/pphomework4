import cv2
import gradio as gr
import numpy as np


def function_hw1(input_image, mode, h_slider, s_slider, l_slider, brightness, contrast):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    # 作业1的图像处理逻辑由学生补充。
    return [input_image] * 7


def function_hw2(input_image):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    # 作业2的图像处理逻辑由学生补充。
    return input_image


def function_hw3(bg_image, fg_image, mode):
    if not bg_image or bg_image.get("background") is None:
        raise gr.Error("请先上传一张背景图片。", duration=5)
    if fg_image is None:
        raise gr.Error("请先上传一张前景图片。", duration=5)
    background = bg_image["background"]
    layers = bg_image.get("layers") or []
    if not layers:
        raise gr.Error("请在背景图上标记屏幕的四个角点。")
    mask_layer = layers[0][:, :, 0]
    _, binary = cv2.threshold(mask_layer, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(binary)
    centers = [tuple(centroids[i]) for i in range(1, num_labels)]
    if len(centers) != 4:
        raise gr.Error("请在屏幕四角分别标记一个点！")
    # 作业3的透视变换逻辑由学生补充。
    return background


def _as_uint8(image):
    """Normalize a Gradio numpy image to uint8 while preserving grayscale/RGB shape."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise gr.Error("输入图片格式不受支持，请上传灰度图或 RGB 图片。")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def _manual_bilateral(image, diameter=5, sigma_color=35.0, sigma_space=3.0):
    """A straightforward, vectorized-per-neighborhood bilateral filter implementation."""
    src = image.astype(np.float32)
    gray = src.ndim == 2
    if gray:
        src = src[:, :, None]
    radius = max(1, int(diameter) // 2)
    h, w, channels = src.shape
    padded = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    result = np.zeros_like(src, dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    sigma_color = max(float(sigma_color), 1e-6)
    sigma_space = max(float(sigma_space), 1e-6)
    center = padded[radius:radius + h, radius:radius + w]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            spatial_weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space))
            color_distance = np.sum((neighbor - center) ** 2, axis=2, keepdims=True)
            range_weight = np.exp(-color_distance / (2.0 * sigma_color * sigma_color))
            weight = spatial_weight * range_weight
            result += neighbor * weight
            weight_sum += weight
    result /= np.maximum(weight_sum, 1e-12)
    if gray:
        result = result[:, :, 0]
    return np.clip(result, 0, 255).astype(np.uint8)


def _guided_filter_color(image, radius=8, eps=1e-3):
    """Manual grayscale-guided filter, applied independently to each color channel."""
    src = image.astype(np.float32) / 255.0
    gray = src.ndim == 2
    if gray:
        color = src[:, :, None]
        guide = src
    else:
        color = src
        guide = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    radius = max(1, int(radius))
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box_mean(values):
        return cv2.boxFilter(values, ddepth=-1, ksize=ksize, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mean_i = box_mean(guide)
    variance_i = box_mean(guide * guide) - mean_i * mean_i
    outputs = []
    for channel in range(color.shape[2]):
        p = color[:, :, channel]
        mean_p = box_mean(p)
        covariance_ip = box_mean(guide * p) - mean_i * mean_p
        a = covariance_ip / (variance_i + max(float(eps), 1e-8))
        b = mean_p - a * mean_i
        mean_a = box_mean(a)
        mean_b = box_mean(b)
        outputs.append(mean_a * guide + mean_b)
    filtered = np.stack(outputs, axis=2)
    if gray:
        filtered = filtered[:, :, 0]
    return np.clip(filtered * 255.0, 0, 255).astype(np.uint8)


def _sharpen(original, filtered, amount=1.0):
    src = original.astype(np.float32)
    detail = src - filtered.astype(np.float32)
    return np.clip(src + float(amount) * detail, 0, 255).astype(np.uint8)


def function_hw4(input_image, method="双边滤波（OpenCV）", strength=35, spatial=3,
                 radius=5, epsilon=0.01, sharpen_amount=1.0):
    """Denoise an image using OpenCV or hand-written traditional filters.

    Methods include OpenCV bilateral/NLM, manual bilateral/guided filters, and a
    side-by-side comparison of detail sharpening based on the two manual filters.
    """
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    image = _as_uint8(input_image)
    strength = max(1, int(strength))
    spatial = max(1, int(spatial))
    radius = max(1, int(radius))

    if method == "双边滤波（OpenCV）":
        return cv2.bilateralFilter(image, d=2 * spatial + 1,
                                   sigmaColor=float(strength), sigmaSpace=float(strength))
    if method == "NLM 去噪（OpenCV）":
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, None, h=float(strength),
                                            templateWindowSize=7, searchWindowSize=21)
        if image.shape[2] == 4:
            rgb, alpha = image[:, :, :3], image[:, :, 3]
            denoised = cv2.fastNlMeansDenoisingColored(rgb, None, h=float(strength),
                                                       hColor=float(strength),
                                                       templateWindowSize=7, searchWindowSize=21)
            return np.dstack((denoised, alpha))
        return cv2.fastNlMeansDenoisingColored(image, None, h=float(strength),
                                               hColor=float(strength),
                                               templateWindowSize=7, searchWindowSize=21)
    if method == "手动双边滤波":
        return _manual_bilateral(image, diameter=2 * spatial + 1,
                                 sigma_color=float(strength), sigma_space=float(spatial))
    if method == "手动导向滤波":
        return _guided_filter_color(image, radius=radius, eps=float(epsilon))
    if method == "锐化效果对比（双边 vs 导向）":
        bilateral = _manual_bilateral(image, diameter=2 * spatial + 1,
                                       sigma_color=float(strength), sigma_space=float(spatial))
        guided = _guided_filter_color(image, radius=radius, eps=float(epsilon))
        bilateral_sharp = _sharpen(image, bilateral, sharpen_amount)
        guided_sharp = _sharpen(image, guided, sharpen_amount)
        # 左右并排展示两种滤波器的高频细节锐化效果。
        return np.concatenate((bilateral_sharp, guided_sharp), axis=1)
    raise gr.Error("请选择有效的去噪方法。")


def function_hw5(input_image):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    # 作业5的图像处理逻辑由学生补充。
    return input_image
