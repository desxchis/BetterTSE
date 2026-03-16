"""BetterTSE — 5-sample 快速验证脚本
====================================
从 results/testsets/event_driven/etth1_100_ultimate 中取前 5 个样本，
真实调用 TEdit 扩散模型 + DeepSeek LLM 完整 pipeline，
并保存每个样本的可视化结果图到 results/validation/etth1_5sample/。

运行方法:
    python run_5sample_validation.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------- project root on path ----------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_api_config, get_model_config
from modules.llm import CustomLLMClient, get_event_driven_plan
from tool.ts_editors import execute_llm_tool
from tool.tedit_wrapper import get_tedit_instance
from test_scripts.bettertse_cik_official import TSEditEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Validation")

# ---------- paths ----------
TESTSET_PATH = _ROOT / "results" / "testsets" / "event_driven" / "etth1_100_ultimate" / "event_driven_testset_ETTh1_100.json"
TEDIT_MODEL   = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"
TEDIT_CONFIG  = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"
VIS_DIR       = _ROOT / "results" / "validation" / "etth1_5sample"
RESULT_JSON   = VIS_DIR / "results.json"
N_SAMPLES     = 5


def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


DEVICE = "cuda:0" if _check_cuda() else "cpu"


# ---------- visualisation ----------

def save_vis(
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
    metrics: dict,
    save_path: Path,
    bg_fidelity: dict | None = None,
) -> None:
    """Save a 3-panel comparison figure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    fig.suptitle(
        f"Sample {sample_id}  |  Tool: {tool_name}  |  LLM region [{llm_start},{llm_end}]  |  GT region [{gt_start},{gt_end}]\n"
        f"Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}",
        fontsize=9,
    )

    x = np.arange(len(base_ts))

    def shade(ax, s, e, color, alpha, label):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label)

    # ── panel 0: base vs target ──
    # base == target everywhere except gt_start:gt_end (the physical injection region)
    ax = axes[0]
    shade(ax, gt_start, gt_end, "gold", 0.30, f"GT injection [{gt_start},{gt_end}]")
    ax.plot(x, base_ts,   color="steelblue", lw=1.8, label="Base TS")
    ax.plot(x, target_ts, color="crimson",   lw=1.8, ls="--", label="Target TS")
    ax.set_ylabel("Value")
    ax.set_title("Base TS  vs  Target TS  (differ only in GT injection region)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── panel 1: target vs generated ──
    # target = ideal edited result; generated = actual model output
    ax = axes[1]
    shade(ax, gt_start,  gt_end,  "gold",   0.30, f"GT injection [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "purple", 0.12, f"LLM edit [{llm_start},{llm_end}]")
    ax.plot(x, target_ts,    color="crimson",   lw=1.8, ls="--", label="Target TS (ideal)")
    ax.plot(x, generated_ts, color="steelblue", lw=1.8, label="Generated TS")
    ax.set_ylabel("Value")
    ax.set_title("Target TS (ideal)  vs  Generated TS")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── panel 2: edit delta (generated − base) ──
    ax = axes[2]
    diff = generated_ts - base_ts
    inside_mask = np.zeros(len(diff), dtype=bool)
    inside_mask[llm_start:llm_end] = True
    bar_colors = np.where(diff >= 0, "tomato", "steelblue")
    ax.bar(x, diff, color=bar_colors, width=1.0, alpha=0.75)
    shade(ax, gt_start,  gt_end,  "orange", 0.18, f"GT [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "purple", 0.10, f"LLM [{llm_start},{llm_end}]")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Δ (generated − base)")
    ax.set_title(
        f"Edit Delta  |  t-IoU={metrics.get('t_iou', 0):.3f}  "
        f"Editability={metrics.get('editability_score', 0):.3f}  "
        f"Preservability={metrics.get('preservability_score', 0):.3f}"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  图像已保存: {save_path}")


# ---------- main ----------

def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load testset ──
    log.info(f"加载测试集: {TESTSET_PATH}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    samples = testset["samples"][:N_SAMPLES]
    log.info(f"取前 {len(samples)} 个样本")

    # ── LLM client ──
    api_cfg = get_api_config()
    llm = CustomLLMClient(
        model_name=api_cfg["model_name"],
        base_url=api_cfg["base_url"],
        api_key=api_cfg["api_key"],
        temperature=0.3,
    )
    log.info(f"LLM: {api_cfg['model_name']}  base_url={api_cfg['base_url']}")

    # ── TEdit ──
    log.info(f"加载 TEdit 模型 ({DEVICE}): {TEDIT_MODEL}")
    tedit = get_tedit_instance(
        model_path=str(TEDIT_MODEL),
        config_path=str(TEDIT_CONFIG),
        device=DEVICE,
        force_reload=True,
    )
    log.info("TEdit 加载完成")

    # ── evaluator ──
    evaluator = TSEditEvaluator()

    all_results = []

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id", f"{i:03d}")
        log.info(f"\n{'='*60}")
        log.info(f"[{i+1}/{len(samples)}] 样本 {sid}  feature={sample['target_feature']}  task={sample['task_type']}")

        base_ts     = np.array(sample["base_ts"],   dtype=np.float32)
        target_ts   = np.array(sample["target_ts"],  dtype=np.float32)
        mask_gt     = np.array(sample["mask_gt"],    dtype=np.float32)
        gt_start    = int(sample["gt_start"])
        gt_end      = int(sample["gt_end"])

        # pick level-2 prompt (新闻主播)
        event_prompts = sample.get("event_prompts", [])
        sorted_ep = sorted(event_prompts, key=lambda x: abs(x.get("level", 0) - 2))
        prompt = sorted_ep[0]["prompt"] if sorted_ep else ""
        log.info(f"  Prompt (level={sorted_ep[0]['level']}): {prompt}")

        # ── Stage A: LLM plan ──
        try:
            plan = get_event_driven_plan(
                news_text="",
                instruction_text=prompt,
                ts_length=len(base_ts),
                llm=llm,
            )
            region = plan.get("parameters", {}).get("region", [0, len(base_ts)])
            llm_start, llm_end = int(region[0]), int(region[1])
            log.info(
                f"  LLM plan: tool={plan.get('tool_name')}  "
                f"region=[{llm_start},{llm_end}]  "
                f"math_shift={plan.get('parameters',{}).get('math_shift','N/A')}"
            )
        except Exception as e:
            log.error(f"  LLM 失败: {e}")
            continue

        # ── Stage B: TEdit edit ──
        try:
            generated_ts, edit_log = execute_llm_tool(
                plan, base_ts, tedit, use_soft_boundary=True
            )
            generated_ts = np.asarray(generated_ts, dtype=np.float32).flatten()
            log.info(f"  编辑完成: {edit_log}")
        except Exception as e:
            log.error(f"  编辑失败: {e}")
            continue

        # ── Stage B+: strict background fidelity check ──
        left_err  = float(np.max(np.abs(generated_ts[:llm_start] - base_ts[:llm_start]))) if llm_start > 0 else 0.0
        right_err = float(np.max(np.abs(generated_ts[llm_end:]   - base_ts[llm_end:])))   if llm_end < len(base_ts) else 0.0
        max_bg_err = max(left_err, right_err)
        bg_fidelity = {"pass": max_bg_err == 0.0, "max_err": max_bg_err,
                       "left_err": left_err, "right_err": right_err}
        status = "PASS ✓" if bg_fidelity["pass"] else f"FAIL ✗  max_err={max_bg_err:.2e}"
        log.info(
            f"  背景保真度: {status}  "
            f"left_max={left_err:.2e}  right_max={right_err:.2e}"
        )

        # ── Stage C: metrics ──
        gt_config = {
            "start_step":     gt_start,
            "end_step":       gt_end,
            "target_feature": sample.get("target_feature", ""),
        }
        llm_pred = {
            "target_feature": gt_config["target_feature"],
            "start_step":     llm_start,
            "end_step":       llm_end,
        }
        try:
            metrics_obj = evaluator.evaluate(
                base_ts=base_ts,
                target_ts=target_ts,
                generated_ts=generated_ts,
                gt_mask=mask_gt,
                gt_config=gt_config,
                llm_prediction=llm_pred,
            )
            metrics = {
                "t_iou":               metrics_obj.t_iou,
                "feature_accuracy":    metrics_obj.feature_accuracy,
                "mse_edit_region":     metrics_obj.mse_edit_region,
                "mae_edit_region":     metrics_obj.mae_edit_region,
                "mse_preserve_region": metrics_obj.mse_preserve_region,
                "mae_preserve_region": metrics_obj.mae_preserve_region,
                "editability_score":   metrics_obj.editability_score,
                "preservability_score":metrics_obj.preservability_score,
            }
            log.info(
                f"  t-IoU={metrics['t_iou']:.4f}  "
                f"Editability={metrics['editability_score']:.4f}  "
                f"Preservability={metrics['preservability_score']:.4f}"
            )
        except Exception as e:
            log.error(f"  评估失败: {e}")
            metrics = {}

        # ── Stage D: visualisation ──
        vis_path = VIS_DIR / f"sample_{sid}_{sample['target_feature']}_{sample['task_type']}.png"
        save_vis(
            sample_id=sid,
            base_ts=base_ts,
            target_ts=target_ts,
            generated_ts=generated_ts,
            gt_start=gt_start,
            gt_end=gt_end,
            llm_start=llm_start,
            llm_end=llm_end,
            prompt=prompt,
            tool_name=plan.get("tool_name", "?"),
            metrics=metrics,
            save_path=vis_path,
            bg_fidelity=bg_fidelity,
        )

        all_results.append({
            "sample_id":    sid,
            "target_feature": sample["target_feature"],
            "task_type":    sample["task_type"],
            "prompt":       prompt,
            "gt_region":    [gt_start, gt_end],
            "llm_region":   [llm_start, llm_end],
            "tool_name":    plan.get("tool_name", "?"),
            "edit_log":     edit_log,
            "metrics":      metrics,
            "bg_fidelity":  bg_fidelity,
        })

    # ── summary ──
    log.info(f"\n{'='*60}")
    log.info(f"完成 {len(all_results)}/{len(samples)} 个样本")
    if all_results:
        def avg(k):
            vals = [r["metrics"].get(k, 0) for r in all_results if r.get("metrics")]
            return float(np.mean(vals)) if vals else 0.0
        log.info(f"  avg t-IoU:         {avg('t_iou'):.4f}")
        log.info(f"  avg Editability:   {avg('editability_score'):.4f}")
        log.info(f"  avg Preservability:{avg('preservability_score'):.4f}")
        log.info(f"  avg Edit-MSE:      {avg('mse_edit_region'):.4f}")
        log.info(f"  avg Preserve-MSE:  {avg('mse_preserve_region'):.4f}")

        # Background fidelity summary
        bg_pass = sum(1 for r in all_results if r.get("bg_fidelity", {}).get("pass", False))
        bg_total = len(all_results)
        max_errs = [r["bg_fidelity"]["max_err"] for r in all_results if r.get("bg_fidelity")]
        log.info(f"\n  ━━━ 背景保真度汇总 ━━━")
        log.info(f"  PASS: {bg_pass}/{bg_total}")
        for r in all_results:
            bf = r.get("bg_fidelity", {})
            ok_str = "PASS ✓" if bf.get("pass") else "FAIL ✗"
            log.info(
                f"    sample {r['sample_id']} [{r['target_feature']}] "
                f"llm_region={r['llm_region']}  {ok_str}  "
                f"max_err={bf.get('max_err', -1):.2e}"
            )
        if max_errs:
            log.info(f"  overall max background error: {max(max_errs):.2e}")

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
    log.info(f"\n结果 JSON 已保存: {RESULT_JSON}")
    log.info(f"可视化图像保存在:  {VIS_DIR}/")


if __name__ == "__main__":
    main()
