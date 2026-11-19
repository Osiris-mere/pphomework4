import cv2
import gradio as gr
import numpy as np


def function_hw1(input_image, mode, h_slider, s_slider, l_slider, brightness, contrast):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    return [input_image] * 7


def function_hw2(input_image):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    return input_image


def function_hw3(bg_image, fg_image, mode):
    if not bg_image or bg_image.get("background") is None:
        raise gr.Error("请先上传一张背景图片。", duration=5)
    if fg_image is None:
        raise gr.Error("请先上传一张前景图片。", duration=5)
    return bg_image["background"]


def function_hw5(input_image):
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)
    return input_image


def _as_uint8(image):
    """将 Gradio 图像转换为 uint8，并保留灰度/RGB/RGBA 通道结构。"""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise gr.Error("输入图片格式不受支持，请上传灰度图或 RGB 图片。")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def _manual_bilateral(image, diameter=5, sigma_color=35.0, sigma_space=3.0):
    """手动双边滤波：逐邻域计算空间高斯权重和灰度/颜色域高斯权重。"""
    src = image.astype(np.float32)
    gray = src.ndim == 2
    if gray:
        src = src[:, :, None]

    radius = max(1, int(diameter) // 2)
    h, w, channels = src.shape
    padded = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    center = padded[radius:radius + h, radius:radius + w]
    result = np.zeros_like(src, dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)

    sigma_color = max(float(sigma_color), 1e-6)
    sigma_space = max(float(sigma_space), 1e-6)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius + dy:radius + dy + h,
                              radius + dx:radius + dx + w]
            spatial_weight = np.exp(-(dx * dx + dy * dy) /
                                    (2.0 * sigma_space * sigma_space))
            color_distance = np.sum((neighbor - center) ** 2,
                                    axis=2, keepdims=True)
            range_weight = np.exp(-color_distance /
                                  (2.0 * sigma_color * sigma_color))
            weight = spatial_weight * range_weight
            result += neighbor * weight
            weight_sum += weight

    result /= np.maximum(weight_sum, 1e-12)
    if gray:
        result = result[:, :, 0]
    return np.clip(result, 0, 255).astype(np.uint8)


def _box_mean_integral(values, radius):
    """使用积分图和 reflect 填充实现盒式均值滤波。"""
    radius = max(0, int(radius))
    if radius == 0:
        return values.astype(np.float32, copy=True)
    size = 2 * radius + 1
    padded = cv2.copyMakeBorder(values.astype(np.float32), radius, radius,
                                radius, radius, cv2.BORDER_REFLECT)
    integral = cv2.integral(padded, sdepth=cv2.CV_64F)
    h, w = values.shape[:2]
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    sums = (integral[y0 + size, x0 + size]
            - integral[y0, x0 + size]
            - integral[y0 + size, x0]
            + integral[y0, x0])
    return (sums / float(size * size)).astype(np.float32)


def _guided_filter_manual(image, radius=8, eps=1e-3):
    """手动导向滤波：局部线性模型 q=aI+b，并用盒式滤波求解均值。"""
    src = image.astype(np.float32) / 255.0
    gray = src.ndim == 2
    color = src[:, :, None] if gray else src
    guide = src if gray else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    mean_i = _box_mean_integral(guide, radius)
    mean_ii = _box_mean_integral(guide * guide, radius)
    variance_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)

    outputs = []
    eps = max(float(eps), 1e-8)
    for channel in range(color.shape[2]):
        p = color[:, :, channel]
        mean_p = _box_mean_integral(p, radius)
        covariance_ip = _box_mean_integral(guide * p, radius) - mean_i * mean_p
        a = covariance_ip / (variance_i + eps)
        b = mean_p - a * mean_i
        mean_a = _box_mean_integral(a, radius)
        mean_b = _box_mean_integral(b, radius)
        outputs.append(mean_a * guide + mean_b)

    filtered = np.stack(outputs, axis=2)
    if gray:
        filtered = filtered[:, :, 0]
    return np.clip(filtered * 255.0, 0, 255).astype(np.uint8)


def _fast_guided_filter_manual(image, radius=8, eps=1e-3, subsampling=4):
    """快速导向滤波：在缩小图上估计系数，再双线性上采样系数。"""
    src = image.astype(np.float32) / 255.0
    gray = src.ndim == 2
    color = src[:, :, None] if gray else src
    guide = src if gray else cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    h, w = guide.shape

    scale = max(1, int(subsampling))
    small_w = max(1, int(np.ceil(w / scale)))
    small_h = max(1, int(np.ceil(h / scale)))
    small_i = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_color = cv2.resize(color, (small_w, small_h), interpolation=cv2.INTER_AREA)
    if small_color.ndim == 2:
        small_color = small_color[:, :, None]

    small_radius = max(1, int(round(float(radius) / scale)))
    eps = max(float(eps), 1e-8)
    mean_i = _box_mean_integral(small_i, small_radius)
    var_i = np.maximum(_box_mean_integral(small_i * small_i, small_radius)
                       - mean_i * mean_i, 0.0)

    outputs = []
    for channel in range(small_color.shape[2]):
        p = small_color[:, :, channel]
        mean_p = _box_mean_integral(p, small_radius)
        cov_ip = _box_mean_integral(small_i * p, small_radius) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * small_i
        mean_a = cv2.resize(_box_mean_integral(a, small_radius),
                            (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(_box_mean_integral(b, small_radius),
                            (w, h), interpolation=cv2.INTER_LINEAR)
        outputs.append(mean_a * guide + mean_b)

    result = np.stack(outputs, axis=2)
    if gray:
        result = result[:, :, 0]
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _detail_enhance(original, filtered, amount=1.0):
    """创意功能：利用滤波前后差值进行可调反遮罩锐化。"""
    src = original.astype(np.float32)
    detail = src - filtered.astype(np.float32)
    return np.clip(src + float(amount) * detail, 0, 255).astype(np.uint8)


def _dct_matrix(size):
    """生成正交 DCT-II 基矩阵，用于 BM3D 块组变换。"""
    x = np.arange(size, dtype=np.float32)
    matrix = np.empty((size, size), dtype=np.float32)
    for k in range(size):
        alpha = np.sqrt(1.0 / size) if k == 0 else np.sqrt(2.0 / size)
        matrix[k] = alpha * np.cos(np.pi * (2.0 * x + 1.0) * k / (2.0 * size))
    return matrix


def _bm3d_stage(noisy, guide, sigma, basic=None, block=8,
                search=8, step=6, max_group=8):
    """手动 BM3D 单阶段：块匹配、三维 DCT、系数滤波、加权聚合。"""
    h, w, channels = noisy.shape
    margin = search + block
    padded_noisy = cv2.copyMakeBorder(noisy, margin, margin, margin, margin,
                                      cv2.BORDER_REFLECT)
    padded_basic = None
    if basic is not None:
        padded_basic = cv2.copyMakeBorder(basic, margin, margin, margin, margin,
                                          cv2.BORDER_REFLECT)

    padded_guide = cv2.copyMakeBorder(guide, margin, margin, margin, margin,
                                      cv2.BORDER_REFLECT)
    accum = np.zeros_like(padded_noisy, dtype=np.float32)
    # Each color plane has its own Wiener/threshold aggregation weight.
    weights = np.zeros_like(padded_noisy, dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    step = max(1, int(step))
    block = max(1, int(block))
    search = max(0, int(search))
    max_group = max(1, int(max_group))

    # The second-stage estimate is already in YCrCb, so its Y plane is the
    # correct matching guide. Convert/select it once, not once per reference
    # block (which previously made processing large images needlessly slow).
    if padded_basic is not None:
        match_reference = (padded_basic[:, :, 0] if padded_basic.shape[2] == 3
                           else padded_basic[:, :, 0])
    else:
        match_reference = padded_guide

    dct_cache = {}
    for y in range(0, h, step):
        for x in range(0, w, step):
            py, px = y + margin, x + margin
            ref_patch = match_reference[py:py + block, px:px + block]
            candidates = []
            for dy in range(-search, search + 1, 2):
                cy = py + dy
                if cy < 0 or cy + block > match_reference.shape[0]:
                    continue
                for dx in range(-search, search + 1, 2):
                    cx = px + dx
                    if cx < 0 or cx + block > match_reference.shape[1]:
                        continue
                    patch = match_reference[cy:cy + block, cx:cx + block]
                    distance = float(np.mean((patch - ref_patch) ** 2))
                    candidates.append((distance, cy, cx))

            candidates.sort(key=lambda item: item[0])
            locations = [(gy, gx) for _, gy, gx in candidates[:max_group]]
            if not locations:
                continue

            count = len(locations)
            if count not in dct_cache:
                dct_cache[count] = _dct_matrix(count)
            dct = dct_cache[count]

            for channel in range(channels):
                group = np.empty((count, block, block), dtype=np.float32)
                basic_group = None
                if padded_basic is not None:
                    basic_group = np.empty_like(group)

                for index, (gy, gx) in enumerate(locations):
                    patch = padded_noisy[gy:gy + block, gx:gx + block, channel]
                    group[index] = cv2.dct(patch.astype(np.float32))
                    if basic_group is not None:
                        basic_patch = padded_basic[gy:gy + block,
                                                   gx:gx + block, channel]
                        basic_group[index] = cv2.dct(basic_patch.astype(np.float32))

                # The per-patch 2-D DCT has already been applied above; this
                # orthonormal DCT applies the third transform along the group.
                transformed = dct @ group.reshape(count, block * block)
                transformed = transformed.reshape(count, block, block)

                if basic_group is None:
                    threshold = 2.7 * sigma
                    filtered = transformed * (np.abs(transformed) >= threshold)
                    nonzero = max(1, int(np.count_nonzero(filtered)))
                    aggregation_weight = 1.0 / (sigma * sigma * nonzero)
                else:
                    basic_transformed = dct @ basic_group.reshape(count, block * block)
                    basic_transformed = basic_transformed.reshape(count, block, block)
                    power = basic_transformed * basic_transformed
                    gain = power / np.maximum(power + sigma * sigma, 1e-12)
                    filtered = transformed * gain
                    aggregation_weight = 1.0 / max(
                        sigma * sigma * float(np.sum(gain * gain)), 1e-12
                    )

                restored = dct.T @ filtered.reshape(count, block * block)
                restored = restored.reshape(count, block, block)
                for index, (gy, gx) in enumerate(locations):
                    patch = cv2.idct(restored[index])
                    accum[gy:gy + block, gx:gx + block, channel] += (
                        aggregation_weight * patch
                    )
                    weights[gy:gy + block, gx:gx + block, channel] += aggregation_weight

    result = accum / np.maximum(weights, 1e-12)
    result = np.nan_to_num(result, nan=0.0, posinf=255.0, neginf=0.0)
    return result[margin:margin + h, margin:margin + w]


def _bm3d_manual(image, sigma=25):
    """手动 BM3D：第一阶段硬阈值，第二阶段 Wiener 滤波。"""
    gray = image.ndim == 2
    rgb = image[:, :, None] if gray else image[:, :, :3]
    work_rgb = np.clip(rgb.astype(np.float32), 0, 255)

    if gray:
        work = work_rgb
        guide = work_rgb[:, :, 0]
    else:
        work = cv2.cvtColor(work_rgb.astype(np.uint8),
                            cv2.COLOR_RGB2YCrCb).astype(np.float32)
        guide = work[:, :, 0]

    basic = _bm3d_stage(work, guide, sigma, basic=None,
                        block=8, search=8, step=6, max_group=8)
    denoised = _bm3d_stage(work, guide, sigma, basic=basic,
                           block=8, search=8, step=6, max_group=8)

    denoised = np.nan_to_num(denoised, nan=0.0,
                             posinf=255.0, neginf=0.0)

    if gray:
        result = denoised[:, :, 0]
    else:
        result = cv2.cvtColor(
            np.clip(denoised, 0, 255).astype(np.uint8),
            cv2.COLOR_YCrCb2RGB
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def function_hw4(input_image, method="手动双边滤波", strength=35,
                 spatial=3, radius=8, epsilon=0.01,
                 sharpen_amount=1.0, bm3d_sigma=25):
    """进阶作业4：手动双边/导向/快速导向、手动 BM3D 和创意细节增强。"""
    if input_image is None:
        raise gr.Error("请先上传一张图片。", duration=5)

    image = _as_uint8(input_image)
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    working = image[:, :, :3] if alpha is not None else image

    strength = max(1, int(strength))
    spatial = max(1, int(spatial))
    radius = max(1, int(radius))
    bm3d_sigma = max(1, int(bm3d_sigma))

    if method == "手动双边滤波":
        result = _manual_bilateral(working, diameter=2 * spatial + 1,
                                   sigma_color=float(strength),
                                   sigma_space=float(spatial))
    elif method == "手动导向滤波":
        result = _guided_filter_manual(working, radius=radius,
                                       eps=float(epsilon))
    elif method == "锐化效果对比（双边 vs 导向）":
        bilateral = _manual_bilateral(working, diameter=2 * spatial + 1,
                                      sigma_color=float(strength),
                                      sigma_space=float(spatial))
        guided = _guided_filter_manual(working, radius=radius,
                                       eps=float(epsilon))
        bilateral_sharp = _detail_enhance(working, bilateral, sharpen_amount)
        guided_sharp = _detail_enhance(working, guided, sharpen_amount)
        result = np.concatenate((bilateral_sharp, guided_sharp), axis=1)
        if alpha is not None:
            alpha = np.concatenate((alpha, alpha), axis=1)
    elif method == "快速导向滤波（积分图盒式滤波）":
        result = _fast_guided_filter_manual(working, radius=radius,
                                            eps=float(epsilon))
    elif method == "手动 BM3D（硬阈值 + Wiener）":
        result = _bm3d_manual(working, sigma=float(bm3d_sigma))
    elif method == "创意：快速导向去噪 + 细节增强":
        filtered = _fast_guided_filter_manual(working, radius=radius,
                                              eps=float(epsilon))
        result = _detail_enhance(working, filtered, sharpen_amount)
    else:
        raise gr.Error("请选择有效的去噪方法。")

    return np.dstack((result, alpha)) if alpha is not None else result
