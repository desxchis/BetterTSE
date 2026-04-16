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

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK_FONT).exists():
    fm.fontManager.addfont(_CJK_FONT)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


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
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    fig.suptitle(
        " | ".join(
            [
                f"Sample {sample_id}",
                f"Tool {tool_name}",
                f"Pred [{llm_start},{llm_end}]",
                f"GT [{gt_start},{gt_end}]",
            ]
        )
        + "\n"
        + f"Prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}",
        fontsize=9,
    )

    x = np.arange(len(base_ts))

    def shade(ax: plt.Axes, start: int, end: int, color: str, alpha: float, label: str) -> None:
        ax.axvspan(start, end, color=color, alpha=alpha, label=label)

    ax = axes[0]
    shade(ax, gt_start, gt_end, "gold", 0.28, f"GT [{gt_start},{gt_end}]")
    ax.plot(x, base_ts, color="0.55", lw=1.5, ls="--", label="Base")
    ax.plot(x, target_ts, color="crimson", lw=1.6, label="Target")
    ax.set_ylabel("Value")
    ax.set_title("Base vs Target")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    shade(ax, gt_start, gt_end, "gold", 0.28, f"GT [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "mediumpurple", 0.12, f"Pred [{llm_start},{llm_end}]")
    ax.plot(x, target_ts, color="crimson", lw=1.6, ls="--", label="Target")
    ax.plot(x, generated_ts, color="steelblue", lw=1.6, label="Generated")
    ax.set_ylabel("Value")
    ax.set_title("Target vs Generated")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    delta = generated_ts - base_ts
    colors = np.where(delta >= 0, "tomato", "steelblue")
    ax.bar(x, delta, color=colors, width=1.0, alpha=0.75)
    shade(ax, gt_start, gt_end, "orange", 0.16, f"GT [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "mediumpurple", 0.10, f"Pred [{llm_start},{llm_end}]")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Delta")
    title = (
        f"Edit Delta | t-IoU={metrics.get('t_iou', 0.0):.3f} | "
        f"Editability={metrics.get('editability_score', 0.0):.3f} | "
        f"Preservability={metrics.get('preservability_score', 0.0):.3f}"
    )
    if bg_fidelity is not None:
        title += f" | BG max err={bg_fidelity.get('max_err', 0.0):.2e}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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
