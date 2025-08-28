import cv2
import numpy as np
from PIL import Image


def read_image(file_bytes):
    """قراءة صورة من bytes وإرجاعها كـ BGR (OpenCV) و RGB و HSV."""
    img_pil = Image.open(file_bytes).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_bgr, img_rgb, img_hsv


def resize_keep_ratio(img_bgr, max_side=640):
    h, w = img_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr


def gray_world_white_balance(img_bgr):
    """توازن أبيض بسيط (Gray World) لتحسين ثبات اللون عبر إضاءات مختلفة."""
    result = img_bgr.copy().astype(np.float32)
    b, g, r = cv2.split(result)
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    # تجنب القسمة على صفر
    b = b * (avg_gray / (avg_b + 1e-6))
    g = g * (avg_gray / (avg_g + 1e-6))
    r = r * (avg_gray / (avg_r + 1e-6))
    result = cv2.merge([b, g, r])
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def get_dominant_colors_hsv(img_bgr, k=3, sample_fraction=0.25, seed=42):
    """استخراج k ألوان سائدة في HSV باستخدام KMeans داخل OpenCV.
    نعيد: [(h,s,v,ratio), ...] مرتبة تنازليًا حسب النسبة.
    """
    # تصغير لتسريع
    small = resize_keep_ratio(img_bgr, max_side=512)
    # خيار موازنة اللون
    balanced = gray_world_white_balance(small)
    img_hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)

    h, w = img_hsv.shape[:2]
    total = h * w

    # أخذ عينة عشوائية لتسريع الحساب
    rng = np.random.default_rng(seed)
    mask = rng.random(total) < sample_fraction
    pixels = img_hsv.reshape(-1, 3)[mask]

    # تحويل إلى float32
    Z = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, flags)
    centers = np.uint8(centers)

    # حساب نسب كل عنقود
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    ratios = counts / counts.sum()

    # ترتيب حسب النسبة
    order = np.argsort(-ratios)
    centers = centers[order]
    ratios = ratios[order]

    dom = []
    for i in range(k):
        h, s, v = centers[i]
        dom.append((int(h), int(s), int(v), float(ratios[i])))
    return dom


def hsv_to_name(h, s, v):
    """تحويل HSV مبسط إلى اسم لون بشري."""
    # النطاق: H:0-179 في OpenCV، نحوله إلى درجات 0-360 للمقارنة البديهية
    hue = (h / 179.0) * 360.0
    sat = s / 255.0
    val = v / 255.0

    if val > 0.9 and sat < 0.15:
        return "White"
    if val < 0.15:
        return "Black"
    if sat < 0.2:
        return "Gray"

    if hue < 15 or hue >= 345:
        return "Red"
    if 15 <= hue < 45:
        return "Orange"
    if 45 <= hue < 70:
        return "Yellow"
    if 70 <= hue < 170:
        return "Green"
    if 170 <= hue < 200:
        return "Cyan"
    if 200 <= hue < 255:
        return "Blue"
    if 255 <= hue < 290:
        return "Purple"
    return "Magenta"


def compute_stats_from_hsv_dominants(dominants):
    """حساب إحصائيات مبسطة من قائمة الألوان السائدة.
    نعيد قاموسًا يحتوي متوسط S و V ونسبة الأحمر/الأزرق، إلخ.
    """
    if not dominants:
        return {}
    # أوزان حسب النسبة
    weights = np.array([r for (_, _, _, r) in dominants])
    hs = np.array([h for (h, _, _, _) in dominants], dtype=np.float32)
    ss = np.array([s for (_, s, _, _) in dominants], dtype=np.float32) / 255.0
    vs = np.array([v for (_, _, v, _) in dominants], dtype=np.float32) / 255.0

    mean_s = float(np.sum(ss * weights))
    mean_v = float(np.sum(vs * weights))

    # حساب نسبة الألوان الحمراء/الزرقاء عالية الإشباع (قد تنزف)
    def is_red_like(h):
        hue = (h / 179.0) * 360.0
        return (hue < 20) or (hue >= 340)

    def is_blue_like(h):
        hue = (h / 179.0) * 360.0
        return 200 <= hue <= 260

    reds = weights[[is_red_like(h) for h in hs]].sum() if len(weights) else 0.0
    blues = weights[[is_blue_like(h) for h in hs]].sum() if len(weights) else 0.0

    # درجة خطر نزف اللون (تقديرية): أعلى عندما يكون S مرتفع و V متوسط/منخفض
    bleed_score = float(mean_s * (1.0 - abs(mean_v - 0.5) * 2.0))

    return {
        "mean_s": mean_s,
        "mean_v": mean_v,
        "red_ratio": float(reds),
        "blue_ratio": float(blues),
        "bleed_score": bleed_score,
    }


def render_swatch(h, s, v, size=80):
    """يُنشئ صورة صغيرة (RGB) تمثل اللون (H,S,V)."""
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    swatch = np.ones((size, size, 3), dtype=np.uint8) * rgb
    return swatch