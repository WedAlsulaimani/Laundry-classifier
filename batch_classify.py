import os
import io
import pandas as pd
from color_utils import read_image, get_dominant_colors_hsv, compute_stats_from_hsv_dominants
from rules import recommendation

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def classify_folder(folder: str, k: int = 3) -> pd.DataFrame:
    rows = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if not os.path.isfile(path) or ext not in SUPPORTED:
            continue
        with open(path, "rb") as f:
            img_bgr, img_rgb, img_hsv = read_image(io.BytesIO(f.read()))
        dom = get_dominant_colors_hsv(img_bgr, k=k)
        stats = compute_stats_from_hsv_dominants(dom)
        rec = recommendation(dom, stats)
        rows.append({
            "filename": name,
            "group": rec["group"],
            "bleed_score": rec["bleed_score"],
            "mean_s": stats.get("mean_s"),
            "mean_v": stats.get("mean_v"),
            "red_ratio": stats.get("red_ratio"),
            "blue_ratio": stats.get("blue_ratio")
        })