"""BetterTSE — Traffic 数据集验证脚本
=====================================
从 results/testsets/event_driven/traffic_20 中取全部 20 个样本，
完整跑 LLM + TEdit pipeline，输出可视化到 results/validation/traffic_20sample/。

运行方法:
    python run_traffic_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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

from config import get_api_config
from modules.llm import CustomLLMClient, get_event_driven_plan
from tool.ts_editors import execute_llm_tool
from tool.tedit_wrapper import get_tedit_instance
from test_scripts.bettertse_cik_official import TSEditEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("TrafficValidation")

TESTSET_PATH = _ROOT / "results" / "testsets" / "event_driven" / "traffic_20" / "event_driven_testset_Traffic_20.json"
TEDIT_MODEL  = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"
TEDIT_CONFIG = _ROOT / "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"
VIS_DIR      = _ROOT / "results" / "validation" / "traffic_20sample"
RESULT_JSON  = VIS_DIR / "results.json"
N_SAMPLES    = 20


def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


DEVICE = "cuda:0" if _check_cuda() else "cpu"


def save_vis(
    sample_id, base_ts, target_ts, generated_ts,
    gt_start, gt_end, llm_start, llm_end,
    prompt, tool_name, metrics, save_path, bg_fidelity,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Traffic Sample {sample_id}  |  Tool: {tool_name}  |  LLM [{llm_start},{llm_end}]  |  GT [{gt_start},{gt_end}]\n"
        f"Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}",
        fontsize=9,
    )
    x = np.arange(len(base_ts))

    def shade(ax, s, e, color, alpha, label):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label)

    ax = axes[0]
    shade(ax, gt_start, gt_end, "gold", 0.30, f"GT [{gt_start},{gt_end}]")
    ax.plot(x, base_ts,   color="steelblue", lw=1.5, label="Base TS")
    ax.plot(x, target_ts, color="crimson",   lw=1.5, ls="--", label="Target TS")
    ax.set_ylabel("Occupancy")
    ax.set_title("Base TS  vs  Target TS")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    shade(ax, gt_start,  gt_end,  "gold",   0.30, f"GT [{gt_start},{gt_end}]")
    shade(ax, llm_start, llm_end, "purple", 0.12, f"LLM [{llm_start},{llm_end}]")
    ax.plot(x, target_ts,    color="crimson",   lw=1.5, ls="--", label="Target (ideal)")
    ax.plot(x, generated_ts, color="steelblue", lw=1.5, label="Generated")
    ax.set_ylabel("Occupancy")
    ax.set_title("Target TS (ideal)  vs  Generated TS")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

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
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  图像已保存: {save_path}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"加载测试集: {TESTSET_PATH}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    samples = testset["samples"][:N_SAMPLES]
    log.info(f"共 {len(samples)} 个样本")

    api_cfg = get_api_config()
    llm = CustomLLMClient(
        model_name=api_cfg["model_name"],
        base_url=api_cfg["base_url"],
        api_key=api_cfg["api_key"],
        temperature=0.3,
    )
    log.info(f"LLM: {api_cfg['model_name']}")

    log.info(f"加载 TEdit ({DEVICE}): {TEDIT_MODEL}")
    tedit = get_tedit_instance(
        model_path=str(TEDIT_MODEL),
        config_path=str(TEDIT_CONFIG),
        device=DEVICE,
        force_reload=True,
    )
    log.info("TEdit 加载完成")

    evaluator = TSEditEvaluator()
    all_results = []

    for i, sample in enumerate(samples):
        sid  = sample.get("sample_id", f"{i:03d}")
        feat = sample.get("target_feature", "?")
        task = sample.get("task_type", "?")
        log.info(f"\n{'='*60}")
        log.info(f"[{i+1}/{len(samples)}] sample={sid}  feature={feat}  task={task}")

        base_ts   = np.array(sample["base_ts"],  dtype=np.float32)
        target_ts = np.array(sample["target_ts"], dtype=np.float32)
        mask_gt   = np.array(sample["mask_gt"],   dtype=np.float32)
        gt_start  = int(sample["gt_start"])
        gt_end    = int(sample["gt_end"])

        event_prompts = sample.get("event_prompts", [])
        sorted_ep = sorted(event_prompts, key=lambda x: abs(x.get("level", 0) - 2))
        prompt = sorted_ep[0]["prompt"] if sorted_ep else ""
        log.info(f"  Prompt (level={sorted_ep[0]['level'] if sorted_ep else '?'}): {prompt}")

        # Stage A: LLM plan
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

        # Stage B: TEdit edit
        try:
            generated_ts, edit_log = execute_llm_tool(plan, base_ts, tedit, use_soft_boundary=True)
            generated_ts = np.asarray(generated_ts, dtype=np.float32).flatten()
            log.info(f"  编辑完成: {edit_log}")
        except Exception as e:
            log.error(f"  编辑失败: {e}")
            continue

        # Stage B+: background fidelity
        left_err  = float(np.max(np.abs(generated_ts[:llm_start] - base_ts[:llm_start]))) if llm_start > 0 else 0.0
        right_err = float(np.max(np.abs(generated_ts[llm_end:]   - base_ts[llm_end:])))   if llm_end < len(base_ts) else 0.0
        bg_fidelity = {"pass": max(left_err, right_err) == 0.0, "max_err": max(left_err, right_err),
                       "left_err": left_err, "right_err": right_err}
        status = "PASS ✓" if bg_fidelity["pass"] else f"FAIL ✗  max_err={bg_fidelity['max_err']:.2e}"
        log.info(f"  背景保真度: {status}  left={left_err:.2e}  right={right_err:.2e}")

        # Stage C: metrics
        gt_config = {"start_step": gt_start, "end_step": gt_end, "target_feature": feat}
        llm_pred  = {"target_feature": feat, "start_step": llm_start, "end_step": llm_end}
        try:
            m = evaluator.evaluate(
                base_ts=base_ts, target_ts=target_ts, generated_ts=generated_ts,
                gt_mask=mask_gt, gt_config=gt_config, llm_prediction=llm_pred,
            )
            metrics = {
                "t_iou": m.t_iou, "feature_accuracy": m.feature_accuracy,
                "mse_edit_region": m.mse_edit_region, "mae_edit_region": m.mae_edit_region,
                "mse_preserve_region": m.mse_preserve_region, "mae_preserve_region": m.mae_preserve_region,
                "editability_score": m.editability_score, "preservability_score": m.preservability_score,
            }
            log.info(
                f"  t-IoU={m.t_iou:.4f}  "
                f"Editability={m.editability_score:.4f}  "
                f"Preservability={m.preservability_score:.4f}"
            )
        except Exception as e:
            log.error(f"  评估失败: {e}")
            metrics = {}

        # Stage D: visualisation
        vis_path = VIS_DIR / f"sample_{sid}_{feat}_{task}.png"
        save_vis(
            sample_id=sid, base_ts=base_ts, target_ts=target_ts, generated_ts=generated_ts,
            gt_start=gt_start, gt_end=gt_end, llm_start=llm_start, llm_end=llm_end,
            prompt=prompt, tool_name=plan.get("tool_name", "?"),
            metrics=metrics, save_path=vis_path, bg_fidelity=bg_fidelity,
        )

        all_results.append({
            "sample_id": sid, "target_feature": feat, "task_type": task,
            "prompt": prompt, "gt_region": [gt_start, gt_end],
            "llm_region": [llm_start, llm_end],
            "tool_name": plan.get("tool_name", "?"),
            "edit_log": edit_log, "metrics": metrics, "bg_fidelity": bg_fidelity,
        })

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"完成 {len(all_results)}/{len(samples)} 个样本")
    if all_results:
        def avg(k):
            vals = [r["metrics"].get(k, 0) for r in all_results if r.get("metrics")]
            return float(np.mean(vals)) if vals else 0.0

        log.info(f"  avg t-IoU:          {avg('t_iou'):.4f}")
        log.info(f"  avg Editability:    {avg('editability_score'):.4f}")
        log.info(f"  avg Preservability: {avg('preservability_score'):.4f}")
        log.info(f"  avg Edit-MSE:       {avg('mse_edit_region'):.6f}")
        log.info(f"  avg Preserve-MSE:   {avg('mse_preserve_region'):.6f}")

        bg_pass = sum(1 for r in all_results if r.get("bg_fidelity", {}).get("pass", False))
        log.info(f"\n  ━━━ 背景保真度 ━━━")
        log.info(f"  PASS: {bg_pass}/{len(all_results)}")
        for r in all_results:
            bf = r.get("bg_fidelity", {})
            ok = "PASS ✓" if bf.get("pass") else "FAIL ✗"
            log.info(
                f"    sample {r['sample_id']} [feat={r['target_feature']}] "
                f"llm_region={r['llm_region']}  {ok}  max_err={bf.get('max_err',-1):.2e}"
            )

        # Tool distribution
        from collections import Counter
        tool_counts = Counter(r["tool_name"] for r in all_results)
        log.info(f"\n  ━━━ Tool 分布 ━━━")
        for tool, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {tool}: {cnt}")

        # t-IoU by task type
        from collections import defaultdict
        task_ious = defaultdict(list)
        for r in all_results:
            if r.get("metrics"):
                task_ious[r["task_type"]].append(r["metrics"].get("t_iou", 0))
        log.info(f"\n  ━━━ t-IoU 按任务类型 ━━━")
        for task, ious in sorted(task_ious.items()):
            log.info(f"    {task}: avg={np.mean(ious):.4f}  n={len(ious)}")

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"dataset": "Traffic", "results": all_results}, f, ensure_ascii=False, indent=2)
    log.info(f"\n结果 JSON: {RESULT_JSON}")
    log.info(f"可视化图像: {VIS_DIR}/")


if __name__ == "__main__":
    main()
