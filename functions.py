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


def function_hw5(input_image):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    # 作业5的图像处理逻辑由学生补充。
    return input_image


def _as_uint8(image):
    """Normalize a Gradio numpy image to uint8 while preserving its channel layout."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise gr.Error("输入图片格式不受支持，请上传灰度图或 RGB 图片。")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def _manual_bilateral(image, diameter=5, sigma_color=35.0, sigma_space=3.0):
    """Bilateral filter implemented from spatial and range Gaussian weights."""
    src = image.astype(np.float32)
    gray = src.ndim == 2
    if gray:
        src = src[:, :, None]
    radius = max(1, int(diameter) // 2)
    h, w, _ = src.shape
    padded = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    result = np.zeros_like(src, dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    sigma_color = max(float(sigma_color), 1e-6)
    sigma_space = max(float(sigma_space), 1e-6)
    center = padded[radius:radius + h, radius:radius + w]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space))
            distance = np.sum((neighbor - center) ** 2, axis=2, keepdims=True)
            weight = spatial * np.exp(-distance / (2.0 * sigma_color * sigma_color))
            result += neighbor * weight
            weight_sum += weight
    result /= np.maximum(weight_sum, 1e-12)
    if gray:
        result = result[:, :, 0]
    return np.clip(result, 0, 255).astype(np.uint8)


def _box_mean_manual(values, radius):
    """Fast box mean using a padded integral image (constant-time per pixel)."""
    radius = max(0, int(radius))
    if radius == 0:
        return values.astype(np.float32, copy=True)
    size = 2 * radius + 1
    padded = cv2.copyMakeBorder(values.astype(np.float32), radius, radius, radius, radius,
                                cv2.BORDER_REFLECT)
    integral = cv2.integral(padded, sdepth=cv2.CV_64F)
    h, w = values.shape[:2]
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    sums = (integral[y0 + size, x0 + size] - integral[y0, x0 + size]
            - integral[y0 + size, x0] + integral[y0, x0])
    return (sums / float(size * size)).astype(np.float32)


def _guided_filter_color(image, radius=8, eps=1e-3):
    """Manual grayscale-guided filter, applied independently to each color channel."""
    src = image.astype(np.float32) / 255.0
    gray = src.ndim == 2
    color = src[:, :, None] if gray else src
    guide = src if gray else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    radius = max(1, int(radius))

    def box_mean(values):
        return _box_mean_manual(values, radius)

    mean_i = box_mean(guide)
    variance_i = np.maximum(box_mean(guide * guide) - mean_i * mean_i, 0.0)
    outputs = []
    for channel in range(color.shape[2]):
        p = color[:, :, channel]
        mean_p = box_mean(p)
        covariance = box_mean(guide * p) - mean_i * mean_p
        a = covariance / (variance_i + max(float(eps), 1e-8))
        b = mean_p - a * mean_i
        outputs.append(box_mean(a) * guide + box_mean(b))
    filtered = np.stack(outputs, axis=2)
    if gray:
        filtered = filtered[:, :, 0]
    return np.clip(filtered * 255.0, 0, 255).astype(np.uint8)


def _fast_guided_filter(image, radius=8, eps=1e-3, subsampling=4):
    """Fast guided filter: estimate coefficients on a reduced image, then upsample."""
    src = image.astype(np.float32) / 255.0
    gray = src.ndim == 2
    color = src[:, :, None] if gray else src
    guide = src if gray else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    h, w = guide.shape
    scale = max(1, int(subsampling))
    small_w, small_h = max(1, int(np.ceil(w / scale))), max(1, int(np.ceil(h / scale)))
    small_i = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_color = cv2.resize(color, (small_w, small_h), interpolation=cv2.INTER_AREA)
    if small_color.ndim == 2:
        small_color = small_color[:, :, None]
    small_radius = max(1, int(round(float(radius) / scale)))
    eps = max(float(eps), 1e-8)
    mean_i = _box_mean_manual(small_i, small_radius)
    var_i = np.maximum(_box_mean_manual(small_i * small_i, small_radius) - mean_i * mean_i, 0.0)
    outputs = []
    for channel in range(small_color.shape[2]):
        p = small_color[:, :, channel]
        mean_p = _box_mean_manual(p, small_radius)
        cov_ip = _box_mean_manual(small_i * p, small_radius) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        mean_a = cv2.resize(_box_mean_manual(a, small_radius), (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(_box_mean_manual(b, small_radius), (w, h), interpolation=cv2.INTER_LINEAR)
        outputs.append(mean_a * guide + mean_b)
    result = np.stack(outputs, axis=2)
    if gray:
        result = result[:, :, 0]
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _sharpen(original, filtered, amount=1.0):
    src = original.astype(np.float32)
    detail = src - filtered.astype(np.float32)
    return np.clip(src + float(amount) * detail, 0, 255).astype(np.uint8)


def _dct_matrix(size):
    """Orthonormal DCT-II matrix for collaborative BM3D group transforms."""
    x = np.arange(size, dtype=np.float32)
    matrix = np.empty((size, size), dtype=np.float32)
    for k in range(size):
        alpha = np.sqrt(1.0 / size) if k == 0 else np.sqrt(2.0 / size)
        matrix[k] = alpha * np.cos(np.pi * (2.0 * x + 1.0) * k / (2.0 * size))
    return matrix


def _bm3d_stage(noisy, guide, sigma, basic=None, block=8, search=12, step=4, max_group=8):
    """One collaborative-filtering stage; aggregation combines overlapping patches."""
    h, w, channels = noisy.shape
    margin = search + block
    padded_noisy = cv2.copyMakeBorder(noisy, margin, margin, margin, margin, cv2.BORDER_REFLECT)
    padded_guide = cv2.copyMakeBorder(guide, margin, margin, margin, margin, cv2.BORDER_REFLECT)
    padded_basic = None if basic is None else cv2.copyMakeBorder(
        basic, margin, margin, margin, margin, cv2.BORDER_REFLECT)
    accum = np.zeros_like(padded_noisy, dtype=np.float32)
    weights = np.zeros(padded_noisy.shape[:2], dtype=np.float32)
    dct_group = _dct_matrix(max_group)
    sigma = max(float(sigma), 1e-6)

    for y in range(0, h, step):
        for x in range(0, w, step):
            py, px = y + margin, x + margin
            reference = (padded_guide if basic is None else
                         cv2.cvtColor(padded_basic, cv2.COLOR_RGB2GRAY)
                         if padded_basic.shape[2] == 3 else padded_basic[:, :, 0])
            ref_patch = reference[py:py + block, px:px + block]
            matches = []
            for dy in range(-search, search + 1, 2):
                cy = py + dy
                if cy < 0 or cy + block > padded_guide.shape[0]:
                    continue
                for dx in range(-search, search + 1, 2):
                    cx = px + dx
                    if cx < 0 or cx + block > padded_guide.shape[1]:
                        continue
                    candidate = reference[cy:cy + block, cx:cx + block]
                    distance = float(np.mean((candidate - ref_patch) ** 2))
                    matches.append((distance, cy, cx))
            matches.sort(key=lambda item: item[0])
            group = matches[:max_group]
            locations = [(gy, gx) for _, gy, gx in group]
            count = len(locations)
            if not count:
                continue
            for channel in range(channels):
                noisy_coeff = np.empty((count, block, block), dtype=np.float32)
                basic_coeff = np.empty_like(noisy_coeff) if padded_basic is not None else None
                for index, (gy, gx) in enumerate(locations):
                    noisy_coeff[index] = cv2.dct(padded_noisy[gy:gy + block, gx:gx + block, channel])
                    if basic_coeff is not None:
                        basic_coeff[index] = cv2.dct(padded_basic[gy:gy + block, gx:gx + block, channel])
                transformed = dct_group[:count, :count] @ noisy_coeff.reshape(count, -1)
                transformed = transformed.reshape(count, block, block)
                if basic_coeff is None:
                    threshold = 2.7 * sigma
                    filtered_coeff = transformed * (np.abs(transformed) >= threshold)
                    nonzero = max(1, int(np.count_nonzero(filtered_coeff)))
                    aggregation_weight = 1.0 / (sigma * sigma * nonzero)
                else:
                    basic_transformed = (dct_group[:count, :count] @ basic_coeff.reshape(count, -1))
                    basic_transformed = basic_transformed.reshape(count, block, block)
                    gain = basic_transformed ** 2 / (basic_transformed ** 2 + sigma * sigma)
                    filtered_coeff = transformed * gain
                    aggregation_weight = 1.0 / max(sigma * sigma * float(np.sum(gain ** 2)), 1e-8)
                restored = (dct_group[:count, :count].T @ filtered_coeff.reshape(count, -1))
                restored = restored.reshape(count, block, block)
                for index, (gy, gx) in enumerate(locations):
                    patch = cv2.idct(restored[index])
                    accum[gy:gy + block, gx:gx + block, channel] += aggregation_weight * patch
            for gy, gx in locations:
                weights[gy:gy + block, gx:gx + block] += aggregation_weight

    safe_weights = np.maximum(weights, 1e-12)
    result = accum / safe_weights[:, :, None]
    return result[margin:margin + h, margin:margin + w]


def _bm3d_denoise(image, sigma=25):
    """Manual two-stage BM3D approximation (3-D DCT, hard threshold, Wiener stage)."""
    gray = image.ndim == 2
    rgb = image[:, :, None] if gray else image[:, :, :3]
    float_rgb = rgb.astype(np.float32)
    if gray:
        work = float_rgb
        guide = float_rgb[:, :, 0]
    else:
        ycc = cv2.cvtColor(float_rgb, cv2.COLOR_RGB2YCrCb)
        work = ycc
        guide = ycc[:, :, 0]
    basic = _bm3d_stage(work, guide, sigma, block=8, search=8, step=6, max_group=8)
    denoised = _bm3d_stage(work, guide, sigma, basic=basic,
                           block=8, search=8, step=6, max_group=8)
    if gray:
        result = denoised[:, :, 0]
    else:
        result = cv2.cvtColor(denoised, cv2.COLOR_YCrCb2RGB)
    return np.clip(result, 0, 255).astype(np.uint8)


def function_hw4(input_image, method="双边滤波（OpenCV）", strength=35, spatial=3,
                 radius=5, epsilon=0.01, sharpen_amount=1.0):
    """Homework 4 filters: OpenCV baselines, manual filters, fast guided, BM3D and detail enhancement."""
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    image = _as_uint8(input_image)
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    working = image[:, :, :3] if alpha is not None else image
    strength = max(1, int(strength))
    spatial = max(1, int(spatial))
    radius = max(1, int(radius))

    if method == "双边滤波（OpenCV）":
        result = cv2.bilateralFilter(working, d=2 * spatial + 1,
                                     sigmaColor=float(strength), sigmaSpace=float(strength))
    elif method == "NLM 去噪（OpenCV）":
        if working.ndim == 2:
            result = cv2.fastNlMeansDenoising(working, None, h=float(strength),
                                              templateWindowSize=7, searchWindowSize=21)
        else:
            result = cv2.fastNlMeansDenoisingColored(working, None, h=float(strength),
                                                     hColor=float(strength),
                                                     templateWindowSize=7, searchWindowSize=21)
    elif method == "手动双边滤波":
        result = _manual_bilateral(working, diameter=2 * spatial + 1,
                                   sigma_color=float(strength), sigma_space=float(spatial))
    elif method == "手动导向滤波":
        result = _guided_filter_color(working, radius=radius, eps=float(epsilon))
    elif method == "快速导向滤波（盒式滤波优化）":
        result = _fast_guided_filter(working, radius=radius, eps=float(epsilon))
    elif method == "BM3D（手动实现）":
        result = _bm3d_denoise(working, sigma=float(strength))
    elif method == "创意：快速导向去噪 + 细节增强":
        filtered = _fast_guided_filter(working, radius=radius, eps=float(epsilon))
        result = _sharpen(working, filtered, sharpen_amount)
    elif method == "锐化效果对比（双边 vs 导向）":
        bilateral = _manual_bilateral(working, diameter=2 * spatial + 1,
                                      sigma_color=float(strength), sigma_space=float(spatial))
        guided = _guided_filter_color(working, radius=radius, eps=float(epsilon))
        bilateral_sharp = _sharpen(working, bilateral, sharpen_amount)
        guided_sharp = _sharpen(working, guided, sharpen_amount)
        result = np.concatenate((bilateral_sharp, guided_sharp), axis=1)
        if alpha is not None:
            alpha = np.concatenate((alpha, alpha), axis=1)
    else:
        raise gr.Error("请选择有效的去噪方法。")
    return np.dstack((result, alpha)) if alpha is not None else result
