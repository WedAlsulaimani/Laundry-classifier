from typing import List, Tuple, Dict

# نوع العنصر في dominant: (h, s, v, ratio)
Dominant = Tuple[int, int, int, float]


def _bucket(mean_s: float, mean_v: float, red_ratio: float) -> str:
    """قواعد مبسطة لتحديد مجموعة الغسيل."""
    # الأبيض الصريح
    if mean_v > 0.85 and mean_s < 0.18:
        return "WHITE"
    # ألوان فاتحة/باستل
    if mean_v > 0.70 and mean_s < 0.35:
        return "LIGHT"
    # ألوان قوية/حمراء (غالبًا تنزف)
    if mean_s > 0.55 and red_ratio > 0.25:
        return "BRIGHT (RED/STRONG)"
    # داكن
    if mean_v < 0.45:
        return "DARK"
    # ألوان عامة
    return "COLORS"


def recommendation(dominants: List[Dominant], stats: Dict) -> Dict:
    mean_s = stats.get("mean_s", 0.0)
    mean_v = stats.get("mean_v", 0.0)
    red_ratio = stats.get("red_ratio", 0.0)
    bleed_score = stats.get("bleed_score", 0.0)

    group = _bucket(mean_s, mean_v, red_ratio)

    advice = []
    if group == "WHITE":
        advice.append("غسل بمفرده أو مع الأبيض فقط، ماء دافئ/ساخن.")
    elif group == "LIGHT":
        advice.append("غسل مع الفواتح فقط، ماء بارد/فاتر.")
    elif group == "DARK":
        advice.append("غسل مع الغوامق فقط، ماء بارد، قلب الملابس من الداخل.")
    elif group.startswith("BRIGHT"):
        advice.append("غسل لوحده أول مرتين أو مع ألوان قوية مشابهة، ماء بارد ومعكوس.")
    else:
        advice.append("غسل مع الألوان المتوسطة المشابهة، ماء بارد.")

    # تحذير نزف اللون
    if bleed_score >= 0.35:
        advice.append(" احتمال نزف لون مرتفع. جرّب مناديل تثبيت الألوان أو نقع سريع بالخل المخفف.")

    return {
        "group": group,
        "advice": " ".join(advice),
        "bleed_score": round(float(bleed_score), 3),
    }