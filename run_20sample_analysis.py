"""BetterTSE — 20-sample 扩展分析脚本
========================================
从 results/testsets/event_driven/etth1_100_ultimate 中取样本 6-25（跳过已跑的前5个），
完整跑 LLM + TEdit pipeline，将结果存到 results/validation/etth1_20sample/，
同时写详细 log 文件供事后分析。

运行方法:
    python run_20sample_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_api_config, get_model_config
from modules.llm import CustomLLMClient, get_event_driven_plan
from tool.ts_editors import execute_llm_tool
from tool.tedit_wrapper import get_tedit_instance
from test_scripts.bettertse_cik_official import TSEditEvaluator

# ---------- paths ----------
TESTSET_PATH = _ROOT / "results" / "testsets" / "event_driven" / "etth1_100_ultimate" / "event_driven_testset_ETTh1_100.json"
TEDIT_MODEL  = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"
TEDIT_CONFIG = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"
VIS_DIR      = _ROOT / "results" / "validation" / "etth1_20sample"
LOG_FILE     = VIS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RESULT_JSON  = VIS_DIR / "results.json"

SAMPLE_START = 5    # skip first 5 (already in results/validation/etth1_5sample)
SAMPLE_END   = 25   # exclusive → 20 samples total

# ---------- logging ----------
VIS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("Analysis")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(fmt)
log.addHandler(_sh)

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(fmt)
log.addHandler(_fh)

log.info(f"Log file: {LOG_FILE}")

# ---------- device ----------
def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

DEVICE = "cuda:0" if _check_cuda() else "cpu"


# ---------- visualisation ----------
def save_vis(
    sample_id, base_ts, target_ts, generated_ts,
    gt_start, gt_end, llm_start, llm_end,
    prompt, tool_name, metrics, save_path,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Sample {sample_id}  |  Tool: {tool_name}  |  LLM [{llm_start},{llm_end}]  |  GT [{gt_start},{gt_end}]\n"
        f"Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}",
        fontsize=9,
    )
    x = np.arange(len(base_ts))

    def shade(ax, s, e, color, alpha, label):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label)

    # panel 0: base vs target
    ax = axes[0]
    shade(ax, gt_start, gt_end, "gold", 0.30, f"GT injection [{gt_start},{gt_end}]")
    ax.plot(x, base_ts,   color="steelblue", lw=1.8, label="Base TS")
    ax.plot(x, target_ts, color="crimson",   lw=1.8, ls="--", label="Target TS")
    ax.set_ylabel("Value")
    ax.set_title("Base TS  vs  Target TS  (differ only in GT injection region)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # panel 1: target vs generated
    ax = axes[1]
    shade(ax, gt_start,  gt_end,  "gold",   0.30, f"GT injection [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "purple", 0.12, f"LLM edit [{llm_start},{llm_end}]")
    ax.plot(x, target_ts,    color="crimson",   lw=1.8, ls="--", label="Target TS (ideal)")
    ax.plot(x, generated_ts, color="steelblue", lw=1.8, label="Generated TS")
    ax.set_ylabel("Value")
    ax.set_title("Target TS (ideal)  vs  Generated TS")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # panel 2: edit delta
    ax = axes[2]
    diff = generated_ts - base_ts
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


# ---------- main ----------
def main():
    log.info("=" * 70)
    log.info("BetterTSE 20-sample 扩展分析")
    log.info(f"样本范围: [{SAMPLE_START}, {SAMPLE_END})")
    log.info("=" * 70)

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    samples = testset["samples"][SAMPLE_START:SAMPLE_END]
    log.info(f"共 {len(samples)} 个样本")

    api_cfg = get_api_config()
    llm = CustomLLMClient(
        model_name=api_cfg["model_name"],
        base_url=api_cfg["base_url"],
        api_key=api_cfg["api_key"],
        temperature=0.3,
    )
    log.info(f"LLM: {api_cfg['model_name']}")

    tedit = get_tedit_instance(
        model_path=str(TEDIT_MODEL),
        config_path=str(TEDIT_CONFIG),
        device=DEVICE,
        force_reload=True,
    )
    log.info(f"TEdit loaded on {DEVICE}")

    evaluator = TSEditEvaluator()
    all_results = []

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id", f"{SAMPLE_START+i:03d}")
        task = sample["task_type"]
        feat = sample["target_feature"]
        log.info(f"\n{'='*60}")
        log.info(f"[{i+1}/{len(samples)}] sample={sid}  feature={feat}  task={task}")

        base_ts   = np.array(sample["base_ts"],   dtype=np.float32)
        target_ts = np.array(sample["target_ts"],  dtype=np.float32)
        mask_gt   = np.array(sample["mask_gt"],    dtype=np.float32)
        gt_start  = int(sample["gt_start"])
        gt_end    = int(sample["gt_end"])
        gt_dur    = gt_end - gt_start
        gt_mag    = float(np.mean(np.abs(target_ts[gt_start:gt_end] - base_ts[gt_start:gt_end])))
        log.info(f"  GT region=[{gt_start},{gt_end}]  duration={gt_dur}  mean_injection_mag={gt_mag:.4f}")

        event_prompts = sample.get("event_prompts", [])
        sorted_ep = sorted(event_prompts, key=lambda x: abs(x.get("level", 0) - 2))
        prompt = sorted_ep[0]["prompt"] if sorted_ep else ""
        log.info(f"  Prompt: {prompt}")

        # Stage A: LLM
        try:
            plan = get_event_driven_plan(news_text="", instruction_text=prompt, ts_length=len(base_ts), llm=llm)
            region = plan.get("parameters", {}).get("region", [0, len(base_ts)])
            llm_start, llm_end = int(region[0]), int(region[1])
            llm_dur = llm_end - llm_start
            math_shift = plan.get("parameters", {}).get("math_shift", "N/A")
            tool_name = plan.get("tool_name", "?")
            log.info(f"  LLM → tool={tool_name}  region=[{llm_start},{llm_end}]  duration={llm_dur}  math_shift={math_shift}")

            # region overlap analysis
            overlap_start = max(gt_start, llm_start)
            overlap_end   = min(gt_end,   llm_end)
            overlap = max(0, overlap_end - overlap_start)
            union   = max(1, (max(gt_end, llm_end) - min(gt_start, llm_start)))
            t_iou_pre = overlap / union
            log.info(f"  Region analysis: GT_dur={gt_dur}  LLM_dur={llm_dur}  overlap={overlap}  pre-t-IoU={t_iou_pre:.3f}")
        except Exception as e:
            log.error(f"  LLM 失败: {e}")
            continue

        # Stage B: TEdit
        try:
            generated_ts, edit_log = execute_llm_tool(plan, base_ts, tedit, use_soft_boundary=True)
            generated_ts = np.asarray(generated_ts, dtype=np.float32).flatten()
            edit_mag = float(np.mean(np.abs(generated_ts[llm_start:llm_end] - base_ts[llm_start:llm_end])))
            log.info(f"  TEdit → {edit_log}  mean_edit_mag={edit_mag:.4f}")
        except Exception as e:
            log.error(f"  编辑失败: {e}")
            continue

        # Stage B+: background fidelity
        left_err  = float(np.max(np.abs(generated_ts[:llm_start] - base_ts[:llm_start]))) if llm_start > 0 else 0.0
        right_err = float(np.max(np.abs(generated_ts[llm_end:]   - base_ts[llm_end:])))   if llm_end < len(base_ts) else 0.0
        max_bg_err = max(left_err, right_err)
        bg_pass = max_bg_err == 0.0
        log.info(f"  BG fidelity: {'PASS' if bg_pass else 'FAIL'}  left={left_err:.2e}  right={right_err:.2e}")

        # Stage C: metrics
        try:
            metrics_obj = evaluator.evaluate(
                base_ts=base_ts, target_ts=target_ts, generated_ts=generated_ts,
                gt_mask=mask_gt,
                gt_config={"start_step": gt_start, "end_step": gt_end, "target_feature": feat},
                llm_prediction={"target_feature": feat, "start_step": llm_start, "end_step": llm_end},
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
                f"  Metrics: t-IoU={metrics['t_iou']:.4f}  "
                f"Editability={metrics['editability_score']:.4f}  "
                f"Preservability={metrics['preservability_score']:.4f}  "
                f"MSE_edit={metrics['mse_edit_region']:.4f}  "
                f"MSE_preserve={metrics['mse_preserve_region']:.4f}"
            )
        except Exception as e:
            log.error(f"  评估失败: {e}")
            metrics = {}

        # Stage D: vis
        vis_path = VIS_DIR / f"sample_{sid}_{feat}_{task}.png"
        save_vis(
            sample_id=sid, base_ts=base_ts, target_ts=target_ts,
            generated_ts=generated_ts, gt_start=gt_start, gt_end=gt_end,
            llm_start=llm_start, llm_end=llm_end, prompt=prompt,
            tool_name=tool_name, metrics=metrics, save_path=vis_path,
        )
        log.info(f"  图像已保存: {vis_path.name}")

        all_results.append({
            "sample_id":      sid,
            "target_feature": feat,
            "task_type":      task,
            "prompt":         prompt,
            "gt_region":      [gt_start, gt_end],
            "gt_duration":    gt_dur,
            "gt_mag":         gt_mag,
            "llm_region":     [llm_start, llm_end],
            "llm_duration":   llm_dur,
            "tool_name":      tool_name,
            "math_shift":     str(math_shift),
            "edit_mag":       edit_mag,
            "pre_t_iou":      t_iou_pre,
            "bg_fidelity":    {"pass": bg_pass, "max_err": max_bg_err},
            "metrics":        metrics,
        })

    # ── summary ──
    log.info(f"\n{'='*70}")
    log.info(f"完成 {len(all_results)}/{len(samples)} 个样本")

    if not all_results:
        return

    def avg(k):
        vals = [r["metrics"].get(k, 0) for r in all_results if r.get("metrics")]
        return float(np.mean(vals)) if vals else 0.0

    log.info("\n━━━ 整体指标均值 ━━━")
    log.info(f"  avg t-IoU:          {avg('t_iou'):.4f}")
    log.info(f"  avg Editability:    {avg('editability_score'):.4f}")
    log.info(f"  avg Preservability: {avg('preservability_score'):.4f}")
    log.info(f"  avg MSE_edit:       {avg('mse_edit_region'):.4f}")
    log.info(f"  avg MSE_preserve:   {avg('mse_preserve_region'):.4f}")

    log.info("\n━━━ 背景保真度 ━━━")
    bg_pass_n = sum(1 for r in all_results if r["bg_fidelity"]["pass"])
    log.info(f"  PASS: {bg_pass_n}/{len(all_results)}")

    log.info("\n━━━ 按 task_type 分组 ━━━")
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r["task_type"]].append(r)
    for task, rs in sorted(by_task.items()):
        t_ious = [r["metrics"].get("t_iou", 0) for r in rs if r.get("metrics")]
        edits  = [r["metrics"].get("editability_score", 0) for r in rs if r.get("metrics")]
        pres   = [r["metrics"].get("preservability_score", 0) for r in rs if r.get("metrics")]
        log.info(
            f"  {task:<20}  n={len(rs)}  "
            f"t-IoU={np.mean(t_ious):.3f}  "
            f"Edit={np.mean(edits):.3f}  "
            f"Pres={np.mean(pres):.3f}"
        )

    log.info("\n━━━ 按 tool_name 分组 ━━━")
    by_tool = defaultdict(list)
    for r in all_results:
        by_tool[r["tool_name"]].append(r)
    for tool, rs in sorted(by_tool.items()):
        t_ious = [r["metrics"].get("t_iou", 0) for r in rs if r.get("metrics")]
        edits  = [r["metrics"].get("editability_score", 0) for r in rs if r.get("metrics")]
        log.info(
            f"  {tool:<20}  n={len(rs)}  "
            f"t-IoU={np.mean(t_ious):.3f}  "
            f"Edit={np.mean(edits):.3f}"
        )

    log.info("\n━━━ Region 定位分析（LLM vs GT） ━━━")
    pre_ious = [r["pre_t_iou"] for r in all_results]
    gt_durs  = [r["gt_duration"] for r in all_results]
    llm_durs = [r["llm_duration"] for r in all_results]
    log.info(f"  avg pre-t-IoU (region overlap before edit): {np.mean(pre_ious):.4f}")
    log.info(f"  avg GT duration:  {np.mean(gt_durs):.1f}")
    log.info(f"  avg LLM duration: {np.mean(llm_durs):.1f}")
    dur_ratios = [r["llm_duration"] / max(r["gt_duration"], 1) for r in all_results]
    log.info(f"  avg LLM/GT duration ratio: {np.mean(dur_ratios):.2f}  (>1=too wide, <1=too narrow)")

    log.info("\n━━━ Edit magnitude 分析 ━━━")
    edit_mags = [r["edit_mag"] for r in all_results]
    gt_mags   = [r["gt_mag"]   for r in all_results]
    log.info(f"  avg GT injection magnitude:  {np.mean(gt_mags):.4f}")
    log.info(f"  avg Model edit magnitude:    {np.mean(edit_mags):.4f}")
    log.info(f"  avg edit/GT mag ratio:       {np.mean([e/max(g,1e-6) for e,g in zip(edit_mags,gt_mags)]):.3f}")

    log.info("\n━━━ 逐样本明细 ━━━")
    log.info(f"  {'sid':>6}  {'feat':>6}  {'task':<20}  {'tool':<16}  "
             f"{'GT_reg':>12}  {'LLM_reg':>12}  {'t-IoU':>6}  {'Edit':>6}  {'Pres':>6}")
    for r in all_results:
        log.info(
            f"  {r['sample_id']:>6}  {r['target_feature']:>6}  {r['task_type']:<20}  "
            f"{r['tool_name']:<16}  "
            f"[{r['gt_region'][0]:>3},{r['gt_region'][1]:>3}]  "
            f"[{r['llm_region'][0]:>3},{r['llm_region'][1]:>3}]  "
            f"{r['metrics'].get('t_iou',0):>6.3f}  "
            f"{r['metrics'].get('editability_score',0):>6.3f}  "
            f"{r['metrics'].get('preservability_score',0):>6.3f}"
        )

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
    log.info(f"\n结果 JSON: {RESULT_JSON}")
    log.info(f"Log 文件:  {LOG_FILE}")
    log.info(f"图像目录:  {VIS_DIR}/")


if __name__ == "__main__":
    main()
