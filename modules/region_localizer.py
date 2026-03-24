"""Lightweight temporal region localizer for BetterTSE.

This module implements a simple plan-then-execute localization stage inspired by
video temporal grounding:

1. Parse coarse temporal anchor from the prompt / LLM localization output
2. Generate candidate temporal windows around the anchor bucket
3. Rank windows using weak-signal and duration heuristics

The goal is not to replace the LLM, but to constrain it into a more stable
"temporal bounding box" workflow before editing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


POSITION_BUCKET_ALIASES = {
    "early": "early",
    "mid": "mid",
    "middle": "mid",
    "late": "late",
    "full": "full",
    "none": "none",
    "early_horizon": "early",
    "mid_horizon": "mid",
    "middle_horizon": "mid",
    "late_horizon": "late",
    "full_horizon": "full",
    "mid_early": "mid",
    "mid_late": "mid",
}


def normalize_position_bucket(bucket: str, fallback: str = "mid") -> str:
    return POSITION_BUCKET_ALIASES.get(str(bucket or "").strip().lower(), fallback)


EARLY_HINTS = ("清晨", "早班", "大清早", "早上")
MID_EARLY_HINTS = ("上午", "前半段", "较早时段")
MIDDLE_HINTS = ("中午", "运行中段", "刚才", "午后")
MID_LATE_HINTS = ("傍晚", "晚些时候", "恢复阶段")
LATE_HINTS = ("深夜", "夜间", "半夜", "夜里", "低谷期前")

SHORT_HINTS = ("短时", "瞬时", "很快", "短暂", "一度")
LONG_HINTS = ("持续", "维持", "一段时间", "逐步", "随后")
RECOVERY_HINTS = ("恢复", "回落", "回到常态", "逐步平稳", "恢复阶段")
SWITCH_HINTS = ("切换", "转移", "重置", "阶跃", "切到新的状态")
NOISE_HINTS = ("杂乱跳变", "失真", "无规律波动", "信号异常", "跳变")
HUMP_HINTS = ("短时冲高后回落", "先升后降", "先偏离后恢复", "阶段性抬升后逐步回落")
SHUTDOWN_HINTS = ("停摆", "停机", "极低水平", "接近停滞", "几乎中断")
ONSET_HINTS = ("开始", "交接后", "之后", "自", "从", "预计在")
VOLATILITY_CANONICAL_TOOLS = {
    "volatility_increase",
    "volatility_global_scale",
    "volatility_local_burst",
    "volatility_envelope_monotonic",
}


@dataclass
class CandidateWindow:
    start: int
    end: int
    score: float
    reason: str


def infer_position_bucket(text: str, fallback: str = "mid") -> str:
    if any(token in text for token in EARLY_HINTS):
        return "early"
    if any(token in text for token in MID_EARLY_HINTS):
        return "mid"
    if any(token in text for token in LATE_HINTS):
        return "late"
    if any(token in text for token in MID_LATE_HINTS):
        return "mid"
    if any(token in text for token in MIDDLE_HINTS):
        return "mid"
    return normalize_position_bucket(fallback)


def anchor_center(text: str, ts_length: int, bucket: str) -> int:
    normalized = text or ""
    bucket = normalize_position_bucket(bucket)
    if bucket == "early":
        ratio = 0.22
    elif bucket == "late":
        ratio = 0.78
    elif bucket == "full":
        ratio = 0.5
    else:
        ratio = 0.5

    if "刚才" in normalized:
        ratio = 0.52
    if "从今天中午开始" in normalized:
        ratio = 0.42
    if "预计在今晚深夜" in normalized or "今晚深夜" in normalized:
        ratio = 0.68
    if "就快到半夜的时候" in normalized:
        ratio = 0.66
    if "夜间低谷期前" in normalized:
        ratio = 0.7
    return max(0, min(ts_length - 1, int(round(ts_length * ratio))))


def infer_shape_hint(text: str) -> str:
    normalized = text or ""
    if any(token in normalized for token in SHUTDOWN_HINTS):
        return "flatline"
    if any(token in normalized for token in SWITCH_HINTS):
        return "step"
    if any(token in normalized for token in NOISE_HINTS):
        return "irregular_noise"
    if any(token in normalized for token in HUMP_HINTS):
        return "hump"
    if any(token in normalized for token in SHORT_HINTS):
        return "transient"
    return ""


def infer_onset_phrase(text: str) -> bool:
    normalized = text or ""
    return any(token in normalized for token in ONSET_HINTS)


def infer_duration_steps(
    text: str,
    ts_length: int,
    effect_family: str = "",
    canonical_tool: str = "",
    shape: str = "",
) -> int:
    if shape in {"step"} or any(token in text for token in SWITCH_HINTS):
        return max(10, min(24, ts_length // 8))
    if shape in {"hump"} or any(token in text for token in HUMP_HINTS):
        return max(12, min(28, ts_length // 8))
    if shape in {"irregular_noise"} or any(token in text for token in NOISE_HINTS):
        return max(14, min(30, ts_length // 7))
    if shape in {"flatline"} or any(token in text for token in SHUTDOWN_HINTS):
        return max(12, min(26, ts_length // 8))
    if effect_family in {"impulse"} or any(token in text for token in SHORT_HINTS):
        return max(8, min(20, ts_length // 10))
    if canonical_tool in VOLATILITY_CANONICAL_TOOLS or any(token in text for token in NOISE_HINTS):
        return max(14, min(28, ts_length // 7))
    if any(token in text for token in SWITCH_HINTS):
        return max(10, min(24, ts_length // 8))
    if effect_family in {"trend", "seasonality", "shutdown"} or any(token in text for token in LONG_HINTS):
        return max(16, min(48, ts_length // 4))
    if effect_family == "volatility":
        return max(12, min(36, ts_length // 6))
    return max(12, min(32, ts_length // 6))

def generate_candidate_windows(
    ts_length: int,
    center: int,
    duration_steps: int,
    shape: str = "",
    onset_anchored: bool = False,
) -> List[CandidateWindow]:
    half = max(2, duration_steps // 2)

    offsets = (-duration_steps, -half, 0, half, duration_steps)
    scales = (0.6, 0.75, 1.0, 1.25)
    candidates: List[CandidateWindow] = []
    for offset in offsets:
        for scale in scales:
            win = max(4, int(round(duration_steps * scale)))
            start = max(0, center + offset - win // 2)
            end = min(ts_length, start + win)
            start = max(0, end - win)
            candidates.append(CandidateWindow(start=start, end=end, score=0.0, reason=""))

    if shape == "step" and onset_anchored:
        start_offsets = (-duration_steps // 4, 0, duration_steps // 6)
        onset_scales = (0.75, 1.0, 1.25)
        for start_offset in start_offsets:
            for scale in onset_scales:
                win = max(8, int(round(duration_steps * scale)))
                start = max(0, min(ts_length - 1, center + start_offset))
                end = min(ts_length, start + win)
                start = max(0, end - win)
                candidates.append(CandidateWindow(start=start, end=end, score=0.0, reason=""))
    return candidates


def score_candidate_window(
    candidate: CandidateWindow,
    ts_length: int,
    bucket: str,
    center: int,
    duration_steps: int,
    effect_family: str,
    prompt_text: str,
    canonical_tool: str,
    shape: str,
    onset_anchored: bool = False,
) -> CandidateWindow:
    cand_center = (candidate.start + candidate.end) / 2
    cand_len = candidate.end - candidate.start

    center_penalty = abs(cand_center - center) / max(ts_length, 1)
    len_penalty = abs(cand_len - duration_steps) / max(duration_steps, 1)

    score = 1.0 - 0.7 * center_penalty - 0.3 * len_penalty
    if shape == "step":
        start_penalty = abs(candidate.start - center) / max(ts_length, 1)
        if onset_anchored:
            # For onset-anchored step events, the start boundary is more
            # informative than the window center.
            score = 1.0 - 0.2 * center_penalty - 0.25 * len_penalty - 0.55 * start_penalty
        else:
            score -= 0.35 * start_penalty
        if cand_len < max(12, ts_length // 12):
            score -= 0.08
        if cand_center > center:
            score -= 0.04
    if shape == "hump":
        if cand_len < max(18, ts_length // 8):
            score -= 0.08
        if cand_len > max(48, ts_length // 4):
            score -= 0.05
    if shape == "irregular_noise":
        if cand_len < max(16, ts_length // 9):
            score -= 0.08
        if cand_len > max(54, ts_length // 3):
            score -= 0.04
    if shape == "flatline" and cand_len < max(18, ts_length // 8):
        score -= 0.1
    if effect_family == "impulse" and cand_len > max(20, ts_length // 8):
        score -= 0.15
    if effect_family in {"trend", "seasonality", "shutdown"} and cand_len < max(12, ts_length // 10):
        score -= 0.15
    if canonical_tool in VOLATILITY_CANONICAL_TOOLS and cand_len > max(28, ts_length // 6):
        score -= 0.08
    if any(token in prompt_text for token in RECOVERY_HINTS) and effect_family in {"impulse", "trend"}:
        score += 0.04
    if any(token in prompt_text for token in SWITCH_HINTS) and cand_len > max(24, ts_length // 7):
        score -= 0.08
    if any(token in prompt_text for token in NOISE_HINTS) and canonical_tool in VOLATILITY_CANONICAL_TOOLS:
        score += 0.06
    if any(token in prompt_text for token in HUMP_HINTS) and shape == "hump":
        score += 0.05
    if any(token in prompt_text for token in SHUTDOWN_HINTS) and shape == "flatline":
        score += 0.05

    candidate.score = score
    candidate.reason = (
        f"bucket={bucket}, shape={shape or 'unknown'}, center≈{center}, duration_prior={duration_steps}, "
        f"candidate_len={cand_len}"
    )
    return candidate


def localize_region(
    prompt_text: str,
    ts_length: int,
    llm_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a refined region proposal from prompt text and coarse LLM plan."""
    localization = {}
    if llm_plan and isinstance(llm_plan.get("localization"), dict):
        localization = dict(llm_plan["localization"])

    anchor_phrase = localization.get("time_anchor_phrase") or prompt_text
    effect_family = ""
    canonical_tool = ""
    shape = ""
    if llm_plan and isinstance(llm_plan.get("intent"), dict):
        effect_family = llm_plan["intent"].get("effect_family", "")
        shape = llm_plan["intent"].get("shape", "")
    if llm_plan:
        canonical_tool = llm_plan.get("canonical_tool") or ""

    hinted_shape = infer_shape_hint(prompt_text)
    active_shape = hinted_shape or shape
    bucket = localization.get("position_bucket") or infer_position_bucket(anchor_phrase)
    center = anchor_center(anchor_phrase, ts_length=ts_length, bucket=bucket)
    heuristic_duration = infer_duration_steps(
        prompt_text,
        ts_length,
        effect_family,
        canonical_tool,
        active_shape,
    )
    llm_duration = localization.get("duration_steps")
    if llm_duration:
        llm_duration = int(llm_duration)
        if abs(llm_duration - heuristic_duration) <= max(6, heuristic_duration // 2):
            duration_steps = llm_duration
        else:
            duration_steps = int(round((llm_duration + heuristic_duration) / 2))
    else:
        duration_steps = heuristic_duration

    if active_shape == "step":
        if infer_onset_phrase(anchor_phrase):
            duration_steps = min(duration_steps, max(18, ts_length // 10))
        if normalize_position_bucket(bucket) == "mid":
            center = max(0, center - max(6, ts_length // 16))
        elif bucket == "late":
            center = max(0, center - max(8, ts_length // 14))
    if active_shape == "hump" and any(token in prompt_text for token in RECOVERY_HINTS):
        center = max(0, center - max(4, ts_length // 20))
    if active_shape == "irregular_noise":
        duration_steps = max(duration_steps, max(18, ts_length // 6))
    if infer_onset_phrase(anchor_phrase):
        if active_shape == "step":
            center = min(ts_length - 1, center + max(1, duration_steps // 6))
        elif active_shape in {"hump", "flatline", "irregular_noise", "transient"}:
            onset_shift = duration_steps // 2
            if "从今天中午开始" in anchor_phrase:
                onset_shift = max(2, duration_steps // 4)
            center = min(ts_length - 1, center + onset_shift)

    onset_anchored = bool(infer_onset_phrase(anchor_phrase))
    candidates = generate_candidate_windows(
        ts_length=ts_length,
        center=center,
        duration_steps=duration_steps,
        shape=active_shape,
        onset_anchored=onset_anchored,
    )
    scored = [
        score_candidate_window(
            candidate=c,
            ts_length=ts_length,
            bucket=bucket,
            center=center,
            duration_steps=duration_steps,
            effect_family=effect_family,
            prompt_text=prompt_text,
            canonical_tool=canonical_tool,
            shape=active_shape,
            onset_anchored=onset_anchored,
        )
        for c in candidates
    ]
    best = max(scored, key=lambda c: c.score)

    return {
        "time_anchor_phrase": localization.get("time_anchor_phrase") or anchor_phrase[:32],
        "position_bucket": bucket,
        "duration_steps": duration_steps,
        "evidence": localization.get("evidence") or best.reason,
        "region": [best.start, best.end],
        "confidence": float(max(0.0, min(1.0, best.score))),
        "proposals": [
            {
                "region": [c.start, c.end],
                "score": round(c.score, 4),
                "reason": c.reason,
            }
            for c in sorted(scored, key=lambda x: x.score, reverse=True)[:5]
        ],
    }
