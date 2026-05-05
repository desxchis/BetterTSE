from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK_FONT).exists():
    fm.fontManager.addfont(_CJK_FONT)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


def _pil_font(size: int = 11) -> ImageFont.ImageFont:
    if Path(_CJK_FONT).exists():
        return ImageFont.truetype(_CJK_FONT, size=size)
    return ImageFont.load_default()


def build_visualization_dir(output_path: str, explicit_dir: Optional[str] = None) -> Path:
    if explicit_dir:
        vis_dir = Path(explicit_dir)
    else:
        vis_dir = Path(output_path).resolve().parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


def build_visualization_path(
    vis_dir: Path,
    sample_id: str,
    target_feature: str,
    task_type: str,
    timestamp: Optional[str] = None,
) -> Path:
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_feature = _slugify(target_feature or "unknown_feature")
    safe_task = _slugify(task_type or "unknown_task")
    safe_sample = _slugify(sample_id or "sample")
    return vis_dir / f"{stamp}_{safe_sample}_{safe_feature}_{safe_task}.png"


def save_pipeline_visualization(
    sample_id: str,
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    generated_ts: np.ndarray,
    gt_start: int,
    gt_end: int,
    llm_start: int,
    llm_end: int,
    prompt: str,
    tool_name: str,
    metrics: Dict[str, Any],
    save_path: Path,
    bg_fidelity: Optional[Dict[str, Any]] = None,
) -> None:
    base = np.asarray(base_ts, dtype=np.float32).reshape(-1)
    target = np.asarray(target_ts, dtype=np.float32).reshape(-1)
    generated = np.asarray(generated_ts, dtype=np.float32).reshape(-1)
    delta = generated - base

    width, height = 1400, 1100
    margin_x = 80
    panel_h = 260
    panel_gap = 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _pil_font(11)
    small_font = _pil_font(10)

    title = " | ".join(
        [
            f"Sample {sample_id}",
            f"Tool {tool_name}",
            f"Pred [{llm_start},{llm_end}]",
            f"GT [{gt_start},{gt_end}]",
        ]
    )
    draw.text((20, 10), title, fill="black", font=font)
    prompt_text = f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}"
    draw.text((20, 30), prompt_text, fill="black", font=small_font)

    def _panel_bounds(panel_idx: int) -> tuple[int, int, int, int]:
        top = 70 + panel_idx * (panel_h + panel_gap)
        return margin_x, top, width - 30, top + panel_h

    def _normalize(vals: np.ndarray, top: int, bottom: int) -> list[float]:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        usable = bottom - top
        return [bottom - ((float(v) - vmin) / (vmax - vmin)) * usable for v in vals]

    def _draw_series_panel(panel_idx: int, title_text: str, series: list[tuple[np.ndarray, str, str]], shade_gt: bool, shade_pred: bool) -> None:
        left, top, right, bottom = _panel_bounds(panel_idx)
        draw.rectangle((left, top, right, bottom), outline="black", width=1)
        draw.text((left, top - 18), title_text, fill="black", font=small_font)
        n = max(len(arr) for arr, _, _ in series)
        x_positions = [left + (i / max(1, n - 1)) * (right - left) for i in range(n)]
        if shade_gt:
            xs = left + (gt_start / max(1, n - 1)) * (right - left)
            xe = left + (gt_end / max(1, n - 1)) * (right - left)
            draw.rectangle((xs, top, xe, bottom), fill=(255, 242, 170))
        if shade_pred:
            xs = left + (llm_start / max(1, n - 1)) * (right - left)
            xe = left + (llm_end / max(1, n - 1)) * (right - left)
            draw.rectangle((xs, top, xe, bottom), fill=(232, 220, 255))
        all_vals = np.concatenate([arr.astype(np.float32) for arr, _, _ in series], axis=0)
        ys_by_series = {}
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        for arr, label, color in series:
            ys = [bottom - ((float(v) - vmin) / (vmax - vmin)) * (bottom - top) for v in arr]
            xs = x_positions[: len(arr)]
            pts = list(zip(xs, ys))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
            for px, py in pts[:: max(1, len(pts) // 24)]:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color, outline=color)
            ys_by_series[label] = color
        for idx, (label, color) in enumerate(ys_by_series.items()):
            draw.text((right - 180, top + 8 + idx * 16), label, fill=color, font=small_font)

    _draw_series_panel(
        0,
        "Base vs Target",
        [(base, "Base", "#888888"), (target, "Target", "#C62828")],
        shade_gt=True,
        shade_pred=False,
    )
    _draw_series_panel(
        1,
        "Target vs Generated",
        [(target, "Target", "#C62828"), (generated, "Generated", "#1565C0")],
        shade_gt=True,
        shade_pred=True,
    )
    _draw_series_panel(
        2,
        (
            f"Edit Delta | t-IoU={metrics.get('t_iou', 0.0):.3f} | "
            f"Editability={metrics.get('editability_score', 0.0):.3f} | "
            f"Preservability={metrics.get('preservability_score', 0.0):.3f}"
            + (f" | BG max err={bg_fidelity.get('max_err', 0.0):.2e}" if bg_fidelity is not None else "")
        ),
        [(delta, "Delta", "#EF6C00")],
        shade_gt=True,
        shade_pred=True,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def save_forecast_revision_visualization(
    sample_id: str,
    history_ts: np.ndarray,
    future_gt: np.ndarray,
    base_forecast: np.ndarray,
    revision_target: np.ndarray,
    edited_forecast: np.ndarray,
    gt_region: tuple[int, int],
    pred_region: tuple[int, int],
    context_text: str,
    metrics: Dict[str, Any],
    save_path: Path,
    display_text: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), sharex=False)
    fig.suptitle(
        f"Forecast Revision | Sample {sample_id}",
        fontsize=10,
    )

    hist_x = np.arange(len(history_ts))
    future_x = np.arange(len(history_ts), len(history_ts) + len(base_forecast))

    def _connected_future(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        series = np.asarray(values, dtype=np.float64)
        x = np.concatenate([[len(history_ts) - 1], future_x])
        y = np.concatenate([[float(history_ts[-1])], series])
        return x, y

    ax = axes[0]
    ax.plot(hist_x, history_ts, color="0.45", lw=1.4, label="History")
    gt_x, gt_y = _connected_future(future_gt)
    base_x, base_y = _connected_future(base_forecast)
    edit_x, edit_y = _connected_future(edited_forecast)
    target_x, target_y = _connected_future(revision_target)
    ax.plot(gt_x, gt_y, color="black", lw=1.4, ls=":", label="Future GT")
    ax.plot(base_x, base_y, color="royalblue", lw=1.6, label="Base forecast")
    ax.plot(edit_x, edit_y, color="seagreen", lw=1.6, label="Edited forecast")
    ax.plot(target_x, target_y, color="crimson", lw=1.6, ls="--", label="Revision target")
    ax.axvspan(len(history_ts) + gt_region[0], len(history_ts) + gt_region[1], color="gold", alpha=0.18, label="GT edit")
    ax.set_title("History and forecast horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    raw_display = display_text if display_text is not None else context_text
    wrapped_context = "\n".join(textwrap.wrap(str(raw_display or ""), width=115, break_long_words=False, break_on_hyphens=False))
    ax.axis("off")
    ax.text(0.01, 0.98, "Prompt Text", va="top", ha="left", fontsize=11, fontweight="bold")
    ax.text(0.01, 0.90, wrapped_context, va="top", ha="left", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _slugify(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    text = "".join(chars).strip("_")
    return text or "item"
