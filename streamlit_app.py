# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import hashlib

st.set_page_config(page_title="مصنّف غسيل الملابس", layout="centered")
st.title(" مصنف غسيل الملابس")
st.markdown("صور أو ارفع صورة سلة الملابس ")
st.caption(""" حدد عدد القطع التقريبية في الصورة, استبعد الألوان الغير مهمة ان وجد""")


# --- أدوات مساعدة ---
def ensure_rgb(np_img):
    """يضمن الصورة RGB بغض النظر عن القناة/التدرج."""
    if np_img.ndim == 2:  # رمادي
        np_img = np.stack([np_img, np_img, np_img], axis=2)
    if np_img.shape[2] == 4:  # RGBA
        np_img = np_img[:, :, :3]
    return np_img.astype(np.uint8)

def sample_pixels(img_rgb, max_pixels=50000):
    pixels = img_rgb.reshape(-1, 3)
    if pixels.shape[0] > max_pixels:
        idx = np.random.choice(pixels.shape[0], max_pixels, replace=False)
        pixels = pixels[idx]
    return pixels

def kmeans_colors(img_rgb, k):
    """يرجع مراكز الألوان (RGB) مرتبة حسب النسبة الأكبر."""
    X = sample_pixels(img_rgb)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.astype(np.uint8)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(counts)[::-1]
    centers = centers[order]
    props = (counts[order] / counts.sum()).tolist()
    return centers, props

def rgb_to_hsv01(rgb):
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h = float(hsv[0]) * 2.0           # 0..360
    s = float(hsv[1]) / 255.0         # 0..1
    v = float(hsv[2]) / 255.0         # 0..1
    return h, s, v

def arabic_color_name(rgb):
    h, s, v = rgb_to_hsv01(rgb)
    if v > 0.92 and s < 0.15: return "أبيض"
    if v < 0.15: return "أسود"
    if s < 0.15: return "رمادي"
    # بني تقريبي
    if 10 <= h < 35 and v < 0.75 and s > 0.35: return "بني"
    if (h < 15) or (h >= 345): return "أحمر"
    if 15 <= h < 35: return "برتقالي"
    if 35 <= h < 65: return "أصفر"
    if 65 <= h < 170: return "أخضر"
    if 170 <= h < 200: return "سماوي"
    if 200 <= h < 255: return "أزرق"
    if 255 <= h < 290: return "بنفسجي"
    if 290 <= h < 345: return "وردي"
    return "لون"

def wash_group(rgb):
    """تصنيف تقريبي لمجموعة الغسيل."""
    h, s, v = rgb_to_hsv01(rgb)
    if v > 0.92 and s < 0.15: return "أبيض"
    if s < 0.18 and v >= 0.25: return "رمادي فاتح" if v > 0.6 else "رمادي داكن"
    if v < 0.25: return "غامق"
    if s > 0.6 and v > 0.5: return "ألوان قوية"
    if v >= 0.8: return "فاتح"
    return "ألوان"

def final_decision(groups):
    has_white = "أبيض" in groups
    has_dark = any(g in ["غامق", "رمادي داكن"] for g in groups)
    has_light = any(g in ["فاتح", "رمادي فاتح"] for g in groups)
    has_bright = "ألوان قوية" in groups

    msg = []
    if has_white and (has_dark or has_bright or len(groups - {"أبيض"}) > 0):
        msg.append("افصل الأبيض وحده.")
    if has_light and has_dark:
        msg.append("افصل الفاتح/الرمادي الفاتح عن الداكن.")
    if has_bright and (has_light or has_white):
        msg.append("الألوان القوية تُغسل أول مرة لوحدها أو مع ألوان مشابهه ، وبماء بارد.")
    if not msg:
        return " يمكن غسل القطع معًا.", ["استخدم ماء بارد وبرنامج عادي."]
    return " يُفضّل الفصل.", msg

def color_block(rgb, label):
    r, g, b = map(int, rgb)
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin:6px 0;">
      <div style="width:28px;height:28px;border-radius:6px;border:1px solid #ddd;background: rgb({r},{g},{b});"></div>
      <div style="font-weight:600">{label}</div>
    </div>
    """

# --- الإدخال ---
mode = st.radio("المصدر:", [" رفع صورة", " الكاميرا"], horizontal=True)
uploaded = None
if mode == " رفع صورة":
    uploaded = st.file_uploader("ارفع صورة (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded = st.camera_input("التقاط صورة")

num_items = st.slider("عدد القطع التقريبي في الصورة", 1, 10 , 5)
k = num_items + 2

if uploaded:
    # قراءة الصورة كـ RGB بشكل مضمون
    img_bytes = uploaded.getvalue()
    image = Image.open(uploaded).convert("RGB")
    img_rgb = ensure_rgb(np.array(image))

    st.image(img_rgb, caption="الصورة المدخلة", use_column_width=True)

    # إعادة تهيئة الاستبعاد عند تغيير الصورة
    img_hash = hashlib.sha1(img_bytes).hexdigest()
    if st.session_state.get("last_img_hash") != img_hash:
        st.session_state.last_img_hash = img_hash
        st.session_state.excluded_idx = set()

    # استخراج الألوان
    centers, props = kmeans_colors(img_rgb, k)

    st.subheader(" الألوان المكتشفة (حدد ما تريد استبعاده):")

    keep_indices = []
    for i, (rgb, p) in enumerate(zip(centers, props)):
        name_ar = arabic_color_name(rgb)
        col1, col2 = st.columns([6, 2])
        with col1:
            st.markdown(color_block(rgb, f"{name_ar}"), unsafe_allow_html=True)
        with col2:
            exclude = st.checkbox("استبعاد", key=f"ex_{i}", value=(i in st.session_state.excluded_idx))
            if exclude:
                st.session_state.excluded_idx.add(i)
            else:
                if i in st.session_state.excluded_idx:
                    st.session_state.excluded_idx.remove(i)

    keep_indices = [i for i in range(len(centers)) if i not in st.session_state.excluded_idx]
    kept_rgbs = [centers[i] for i in keep_indices]
    kept_groups = {wash_group(rgb) for rgb in kept_rgbs}

    st.markdown("---")
    st.subheader(" الحكم النهائي")
    if kept_rgbs:
        verdict, tips = final_decision(kept_groups)
        st.write(verdict)
        for t in tips:
            st.write("• " + t)

        # عرض المجموعات المقترحة
        st.write("**المجموعات  بعد الاستبعاد:** " + "، ".join(sorted(kept_groups)))
    else:
        st.warning("اختار على الأقل لونًا واحدًا (لا تستبعد كل الألوان).")

st.caption("ملاحظة: التصنيف تقريبي. عند الشك خاصة مع الألوان الحمراء/الداكنة الجديدة، اغسلها منفصلة أول مرة.")
