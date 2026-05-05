from __future__ import annotations

from typing import Dict, List


STRENGTH_TEXT_TO_LABEL = {
    "weak": 0,
    "medium": 1,
    "strong": 2,
}


_WEAK_PHRASES = [
    "weak",
    "low strength",
    "mild",
    "slight",
    "subtle",
    "gentle",
    "弱档",
    "轻微档",
    "弱一些",
    "偏弱",
    "轻微",
    "稍微增强一点",
    "轻微增强",
    "略微增强",
    "小幅增强",
    "温和增强",
    "增强一点",
    "轻微调整",
    "略微调整",
    "稍微调整",
    "弱一点",
    "轻一点",
]

_MEDIUM_PHRASES = [
    "medium",
    "moderate",
    "medium strength",
    "normal strength",
    "中等档",
    "中档",
    "中等",
    "适中档",
    "适当增强",
    "适度增强",
    "中等增强",
    "适中增强",
    "一般增强",
    "正常增强",
    "中度增强",
]

_STRONG_PHRASES = [
    "strong",
    "high strength",
    "intense",
    "clear",
    "obvious",
    "significant",
    "强档",
    "明显档",
    "强烈",
    "偏强",
    "明显",
    "尽量强",
    "尽可能强",
    "非常强",
    "明显增强",
    "显著增强",
    "大幅增强",
    "更强一些",
    "强一些",
]

_CONSTRAINT_PHRASES = {
    "avoid_overshoot": [
        "avoid overshoot",
        "do not overshoot",
        "not too much",
        "别过冲",
        "不要过冲",
        "别太夸张",
        "不要太夸张",
        "别太猛",
        "不要太猛",
    ],
    "conservative": [
        "conservative",
        "keep it conservative",
        "not too strong",
        "保守一点",
        "别太强",
        "不要太强",
    ],
}


def _collect_matches(text: str, phrases: List[str]) -> List[str]:
    return [phrase for phrase in phrases if phrase in text]


def parse_strength_text(text: str, default_strength: str = "medium") -> Dict[str, object]:
    normalized = str(text or "").strip().lower()
    matches = {
        "weak": _collect_matches(normalized, _WEAK_PHRASES),
        "medium": _collect_matches(normalized, _MEDIUM_PHRASES),
        "strong": _collect_matches(normalized, _STRONG_PHRASES),
    }

    constraint_type = "none"
    constraint_hits: List[str] = []
    for name, phrases in _CONSTRAINT_PHRASES.items():
        hits = _collect_matches(normalized, phrases)
        if hits:
            constraint_type = name
            constraint_hits.extend(hits)
            break

    if matches["strong"]:
        strength_text = "strong"
    elif matches["medium"]:
        strength_text = "medium"
    elif matches["weak"]:
        strength_text = "weak"
    elif constraint_type != "none":
        strength_text = "weak"
    else:
        strength_text = default_strength if default_strength in STRENGTH_TEXT_TO_LABEL else "medium"

    matched_tokens = matches[strength_text] if matches.get(strength_text) else []
    if constraint_hits:
        matched_tokens = matched_tokens + constraint_hits

    confidence = 0.95
    non_empty_strengths = sum(1 for value in matches.values() if value)
    if constraint_type != "none":
        confidence = min(confidence, 0.8)
    if non_empty_strengths > 1:
        confidence = min(confidence, 0.75)
    if not matched_tokens:
        confidence = 0.55

    parse_reason = "default_strength_fallback"
    if matched_tokens:
        parse_reason = "matched_tokens=" + "|".join(matched_tokens)
    if constraint_type != "none":
        parse_reason += f";constraint={constraint_type}"

    return {
        "strength_text": strength_text,
        "strength_label": STRENGTH_TEXT_TO_LABEL[strength_text],
        "constraint_type": constraint_type,
        "parse_reason": parse_reason,
        "confidence": float(confidence),
        "matched_tokens": matched_tokens,
    }
