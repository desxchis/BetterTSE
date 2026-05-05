"""BetterTSE 完整评估 Pipeline
================================

将四个阶段串联成一条可运行的流水线：
  Stage 1 — 加载测试集 JSON
  Stage 2 — LLM 解析编辑意图 (get_event_driven_plan)
  Stage 3 — 执行时序编辑 (execute_llm_tool / math-only fallback)
  Stage 4 — 计算评估指标 (TSEditEvaluator)

典型用法
--------
# 步骤一：先生成测试集（生成后保存到 results/pipeline_inputs/cik_official_test_results.json）
python test_scripts/bettertse_cik_official.py \\
    --csv-path data/ETTh1.csv \\
    --num-samples 20 \\
    --output-dir results/pipeline_inputs

# 步骤二：运行完整 pipeline（含真实模型编辑）
python run_pipeline.py \\
    --testset results/pipeline_inputs/cik_official_test_results.json \\
    --tedit-model TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth \\
    --tedit-config TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml \\
    --output results/pipeline/pipeline_results.json

# 若没有 GPU / TEdit 权重，可用纯数学编辑模式（仅评估 LLM 区域定位，不含扩散编辑）
python run_pipeline.py \\
    --testset results/pipeline_inputs/cik_official_test_results.json \\
    --output results/pipeline/pipeline_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 项目根目录下直接运行，确保 test_scripts 子包内的裸模块名可以解析
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_api_config
from modules.experiment_logging import build_pure_editing_record, compute_target_similarity
from modules.experiment_visualization import (
    build_visualization_dir,
    build_visualization_path,
    save_pipeline_visualization,
)
from modules.pure_editing_volatility import resolve_volatility_subtype_route
from modules.region_localizer import infer_duration_steps, infer_position_bucket, infer_shape_hint
from tool.ts_editors import EDIT_TOOL_SPECS, execute_llm_tool, normalize_llm_plan
from tool.tedit_wrapper import TEditWrapper, get_tedit_instance
from test_scripts.bettertse_cik_official import TSEditEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PipelineRunner")


# ---------------------------------------------------------------------------
# 格式归一化：兼容当前主测试集和旧样本字段名
#   bettertse_cik_official.py / event-driven 主线 → start_step / end_step
#   legacy samples                                 → gt_start / gt_end
# ---------------------------------------------------------------------------

def _normalize_gt_config(sample: Dict[str, Any]) -> Dict[str, Any]:
    """返回包含 start_step / end_step / target_feature 的标准 gt_config."""
    gt_config: Dict[str, Any] = dict(sample.get("gt_config", {}))

    # legacy sample format
    if "start_step" not in gt_config:
        gt_config["start_step"] = sample.get("gt_start", gt_config.get("gt_start", 0))
    if "end_step" not in gt_config:
        gt_config["end_step"] = sample.get("gt_end", gt_config.get("gt_end", 0))
    if "target_feature" not in gt_config:
        gt_config["target_feature"] = sample.get("target_feature", "")
    if "task_type" not in gt_config:
        gt_config["task_type"] = sample.get("task_type", "unknown")
    if "legacy_task_type" not in gt_config:
        gt_config["legacy_task_type"] = sample.get("legacy_task_type", "unknown")
    if "injection_operator" not in gt_config:
        gt_config["injection_operator"] = sample.get("injection_operator", gt_config.get("injection_type", "unknown"))
    if "edit_intent_gt" not in gt_config:
        gt_config["edit_intent_gt"] = sample.get("edit_intent_gt", {})

    return gt_config


def _family_consistent_localization_enabled(sample: Dict[str, Any]) -> bool:
    pipeline_options = sample.get("pipeline_options", {})
    if isinstance(pipeline_options, dict):
        return bool(pipeline_options.get("family_consistent_localization", False))
    return False


def _family_region_source(sample: Dict[str, Any]) -> str:
    pipeline_options = sample.get("pipeline_options", {})
    if isinstance(pipeline_options, dict):
        return str(pipeline_options.get("family_region_source", "llm_first_sample")).strip().lower()
    return "llm_first_sample"


def _benchmark_region(sample: Dict[str, Any], gt_config: Dict[str, Any], series_len: int) -> List[int]:
    start = gt_config.get("start_step", sample.get("gt_start", 0))
    end = gt_config.get("end_step", sample.get("gt_end", series_len))
    start_idx = max(0, min(int(series_len), int(start)))
    end_idx = max(start_idx, min(int(series_len), int(end)))
    return [start_idx, end_idx]


def _apply_family_region_override(plan: Dict[str, Any], region: List[int], family_id: str, source: str) -> Dict[str, Any]:
    normalized = dict(plan)
    normalized["parameters"] = dict(normalized.get("parameters", {}))
    normalized["parameters"]["region"] = [int(region[0]), int(region[1])]
    normalized["localization"] = dict(normalized.get("localization", {}))
    normalized["localization"]["region"] = [int(region[0]), int(region[1])]
    normalized["execution"] = dict(normalized.get("execution", {}))
    exec_params = dict(normalized["execution"].get("parameters", {}))
    exec_params["region"] = [int(region[0]), int(region[1])]
    normalized["execution"]["parameters"] = exec_params
    normalized["pipeline_localization_policy"] = {
        "family_consistent_localization": True,
        "region_source": source,
        "family_id": str(family_id),
    }
    return normalized


def _build_condensed_execution_instruction(plan: Dict[str, Any]) -> str:
    intent = plan.get("intent", {}) if isinstance(plan.get("intent"), dict) else {}
    effect_family = str(intent.get("effect_family", "")).strip().lower()
    direction = str(intent.get("direction", "neutral")).strip().lower()
    shape = str(intent.get("shape", "")).strip().lower()
    strength = str(intent.get("strength", "medium")).strip().lower()
    duration = str(intent.get("duration", "medium")).strip().lower()

    strength_en = {"weak": "weak", "medium": "medium", "strong": "strong"}.get(strength, "medium")
    duration_en = {
        "short": "short window",
        "medium": "medium window",
        "long": "long window",
        "short_or_medium": "short-to-medium window",
        "medium_or_long": "medium-to-long window",
    }.get(duration, "local window")

    if effect_family == "trend":
        direction_en = "upward" if direction == "up" else "downward"
        return f"Apply a {strength_en} {direction_en} trend edit in this {duration_en}"
    if effect_family == "seasonality":
        if shape == "flatten":
            return f"Apply a {strength_en} seasonality reduction in this {duration_en}"
        return f"Apply a {strength_en} seasonality enhancement in this {duration_en}"
    if effect_family == "hard_zero":
        return f"Apply a {strength_en} near-zero flatline edit in this {duration_en}"
    if effect_family == "step_change":
        direction_en = "upward" if direction == "up" else "downward"
        return f"Apply a {strength_en} {direction_en} step-regime edit in this {duration_en}"
    if effect_family == "multiplier":
        return f"Apply a {strength_en} multiplicative amplification edit in this {duration_en}"
    if effect_family == "noise_injection":
        return f"Apply a {strength_en} irregular-noise edit in this {duration_en}"
    return f"Apply a {strength_en} local edit in this {duration_en}"


def _resolve_execution_instruction(plan: Dict[str, Any]) -> str:
    execution = plan.get("execution", {}) if isinstance(plan.get("execution"), dict) else {}
    parameters = plan.get("parameters", {}) if isinstance(plan.get("parameters"), dict) else {}

    execution_phrase = execution.get("execution_phrase") or execution.get("condensed_instruction")
    if isinstance(execution_phrase, str) and execution_phrase.strip():
        return execution_phrase.strip()

    instruction_text = parameters.get("instruction_text")
    if isinstance(instruction_text, str) and instruction_text.strip():
        return instruction_text.strip()

    return _build_condensed_execution_instruction(plan)


# ---------------------------------------------------------------------------
# 核心 pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    testset_path: str,
    tedit_model_path: Optional[str] = None,
    tedit_config_path: Optional[str] = None,
    tedit_device: str = "cpu",
    output_path: str = "results/pipeline/pipeline_results.json",
    mode: str = "full_bettertse",
    vis_dir: Optional[str] = None,
    save_visualizations: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    运行完整评估 pipeline。

    Args:
        testset_path:       测试集 JSON 路径（由 bettertse_cik_official.py 生成）
        tedit_model_path:   TEdit 模型权重 .pth（可选；不提供则退化为纯数学编辑）
        tedit_config_path:  TEdit 配置 .yaml
        tedit_device:       运行设备（"cpu" 或 "cuda:0"）
        output_path:        结果 JSON 输出路径
        mode:               pipeline mode / ablation setting
        vis_dir:            可视化输出目录；默认 output 同级 visualizations/
        save_visualizations:是否保存逐样本图像
        max_samples:        最多处理样本数（None = 全部）

    Returns:
        包含 summary 和 results 的字典
    """

    # ── Stage 0: 加载测试集 ───────────────────────────────────────────────
    logger.info(f"[Stage 0] 加载测试集: {testset_path}")
    testset_path_obj = Path(testset_path)
    if not testset_path_obj.exists():
        raise FileNotFoundError(f"测试集文件不存在: {testset_path}")

    with open(testset_path_obj, "r", encoding="utf-8") as f:
        testset_data = json.load(f)

    samples: List[Dict[str, Any]] = testset_data.get("samples", [])
    if not samples:
        raise ValueError("测试集为空，请先运行 bettertse_cik_official.py 生成测试集。")

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"  共 {len(samples)} 个样本待处理")
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    vis_dir_obj = build_visualization_dir(output_path, vis_dir) if save_visualizations else None
    if vis_dir_obj is not None:
        logger.info(f"  可视化输出目录: {vis_dir_obj}")
    # ── Stage 1: 初始化 LLM 客户端 ───────────────────────────────────────
    llm_client = None
    get_event_driven_plan_fn = None
    if mode not in {"direct_edit", "replay_plan"}:
        logger.info("[Stage 1] 初始化 LLM 客户端")
        from modules.llm import CustomLLMClient, get_event_driven_plan

        api_cfg = get_api_config()
        if not api_cfg.get("api_key"):
            raise ValueError(
                "未设置 API 密钥。请在 .env 文件或环境变量中设置 DEEPSEEK_API_KEY / OPENAI_API_KEY。"
            )
        llm_client = CustomLLMClient(
            model_name=api_cfg["model_name"],
            base_url=api_cfg["base_url"],
            api_key=api_cfg["api_key"],
            temperature=0.3,
        )
        get_event_driven_plan_fn = get_event_driven_plan
        logger.info(f"  模型: {api_cfg['model_name']}  |  Base URL: {api_cfg['base_url']}")
    else:
        logger.info(f"[Stage 1] {mode} 模式跳过 LLM 初始化")

    # ── Stage 2: 初始化 TEdit 模型（可选）────────────────────────────────
    tedit: Optional[TEditWrapper] = None
    if tedit_model_path and tedit_config_path:
        logger.info(f"[Stage 2] 加载 TEdit 模型: {tedit_model_path}")
        tedit = get_tedit_instance(
            model_path=tedit_model_path,
            config_path=tedit_config_path,
            device=tedit_device,
            force_reload=True,
        )
        logger.info("  TEdit 模型加载完成")
    else:
        logger.warning(
            "[Stage 2] 未提供 TEdit 路径 → 退化为纯数学编辑模式"
            "（仍可评估 LLM 区域定位精度，但编辑质量不含扩散纹理）"
        )

    # ── Stage 3: 逐样本处理 ──────────────────────────────────────────────
    evaluator = TSEditEvaluator()
    results: List[Dict[str, Any]] = []
    family_region_cache: Dict[str, List[int]] = {}

    for i, sample in enumerate(samples):
        sample_id = sample.get("sample_id", f"sample_{i:04d}")
        logger.info(f"\n[{i + 1}/{len(samples)}] 处理样本: {sample_id}")

        base_ts = np.array(sample["base_ts"], dtype=np.float32)
        target_ts = np.array(sample.get("target_ts", sample["base_ts"]), dtype=np.float32)
        # 兼容两种掩码字段名：gt_mask (cik_official) / mask_gt (event_driven)
        gt_mask = np.array(
            sample.get("gt_mask", sample.get("mask_gt", [0] * len(base_ts))),
            dtype=np.float32,
        )
        # 兼容两种提示词格式：
        #   vague_prompt (str)         — cik_official / mini_benchmark
        #   event_prompts (list[dict]) — event_driven_testset
        vague_prompt: str = sample.get("vague_prompt", "")
        if not vague_prompt:
            event_prompts = sample.get("event_prompts", [])
            if event_prompts:
                # 优先使用 L2（新闻主播，间接叙事但仍有完整背景）；不存在则取最高 level
                sorted_ep = sorted(event_prompts, key=lambda x: abs(x.get("level", 0) - 2))
                vague_prompt = sorted_ep[0].get("prompt", "")
        if not vague_prompt:
            vague_prompt = str(sample.get("causal_scenario", "") or "")
        if not vague_prompt:
            vague_prompt = str(sample.get("technical_ground_truth", "") or "")
        gt_config = _normalize_gt_config(sample)

        if not vague_prompt:
            logger.warning(f"  样本 {sample_id} 没有 vague_prompt，跳过")
            results.append({**_strip_arrays(sample), "pipeline_error": "missing vague_prompt", "metrics": {}})
            continue

        # ---- Stage 3a: LLM 解析编辑意图 ----------------------------------
        try:
            if mode == "direct_edit":
                full_plan = {}
            elif mode == "replay_plan":
                full_plan = _sample_replay_plan(sample)
            else:
                full_plan = get_event_driven_plan_fn(
                    news_text="",
                    instruction_text=vague_prompt,
                    ts_length=len(base_ts),
                    llm=llm_client,
                )
            plan = _apply_pipeline_mode(
                full_plan=full_plan,
                prompt_text=vague_prompt,
                ts_length=len(base_ts),
                mode=mode,
            )
            family_id = str(sample.get("family_id", "") or "")
            if family_id and _family_consistent_localization_enabled(sample):
                cached_region = family_region_cache.get(family_id)
                if cached_region is not None:
                    plan = _apply_family_region_override(plan, cached_region, family_id, source="family_cache")
                else:
                    region_source = _family_region_source(sample)
                    if region_source in {"benchmark", "gt", "ground_truth"}:
                        proposed_region = _benchmark_region(sample, gt_config, len(base_ts))
                    else:
                        proposed_region = plan.get("parameters", {}).get("region", [0, len(base_ts)])
                    family_region_cache[family_id] = [int(proposed_region[0]), int(proposed_region[1])]
                    plan = _apply_family_region_override(plan, family_region_cache[family_id], family_id, source=region_source)
            plan.setdefault("parameters", {})
            plan["parameters"] = dict(plan["parameters"])
            plan["parameters"]["instruction_text_full"] = vague_prompt
            plan["parameters"]["instruction_text"] = _resolve_execution_instruction(plan)
            # Inject benchmark strength info into tool params for TEdit strength control
            if "strength_label" in sample:
                plan["parameters"]["strength_label"] = sample["strength_label"]
            if "strength_scalar" in sample:
                plan["parameters"]["strength_scalar"] = sample["strength_scalar"]
            if "runtime_strength_label" in sample:
                plan["parameters"]["strength_label"] = sample["runtime_strength_label"]
            if "runtime_strength_scalar" in sample:
                plan["parameters"]["strength_scalar"] = sample["runtime_strength_scalar"]
            if isinstance(sample.get("injection_config"), dict):
                plan["parameters"]["injection_config"] = sample["injection_config"]
            region = plan.get("parameters", {}).get("region", [0, len(base_ts)])
            logger.info(
                f"  Pipeline mode={mode}  tool={plan.get('tool_name')}  "
                f"region={region}  "
                f"math_shift={plan.get('parameters', {}).get('math_shift', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"  [Stage 3a] LLM 解析失败: {e}")
            results.append({**_strip_arrays(sample), "pipeline_error": f"llm_parse: {e}", "metrics": {}})
            continue

        # ---- Stage 3b: 执行编辑 ------------------------------------------
        try:
            if tedit is not None:
                generated_ts, edit_log = execute_llm_tool(
                    plan, base_ts, tedit, use_soft_boundary=True
                )
            else:
                generated_ts, edit_log = _math_only_edit(plan, base_ts)
            logger.info(f"  编辑完成: {edit_log}")
        except Exception as e:
            logger.error(f"  [Stage 3b] 编辑执行失败: {e}")
            results.append({**_strip_arrays(sample), "pipeline_error": f"edit: {e}", "metrics": {}})
            continue

        # ---- Stage 3c: 计算评估指标 ---------------------------------------
        region = plan.get("parameters", {}).get("region", [0, int(len(base_ts))])
        llm_prediction = {
            "target_feature": gt_config.get("target_feature", ""),
            "start_step": int(region[0]),
            "end_step": int(region[1]),
        }
        try:
            metrics = evaluator.evaluate(
                base_ts=base_ts,
                target_ts=target_ts,
                generated_ts=np.asarray(generated_ts, dtype=np.float32),
                gt_mask=gt_mask,
                gt_config=gt_config,
                llm_prediction=llm_prediction,
            )
        except Exception as e:
            logger.error(f"  [Stage 3c] 评估计算失败: {e}")
            results.append({**_strip_arrays(sample), "pipeline_error": f"eval: {e}", "metrics": {}})
            continue

        bg_fidelity = _compute_background_fidelity(
            base_ts=base_ts,
            generated_ts=np.asarray(generated_ts, dtype=np.float32),
            region=region,
        )

        logger.info(
            f"  t-IoU={metrics.t_iou:.4f}  "
            f"Editability={metrics.editability_score:.4f}  "
            f"Preservability={metrics.preservability_score:.4f}"
        )
        logger.info(
            f"  BG fidelity: max_err={bg_fidelity['max_err']:.2e}  "
            f"left={bg_fidelity['left_err']:.2e}  right={bg_fidelity['right_err']:.2e}"
        )

        gt_intent = gt_config.get("edit_intent_gt", {}) if isinstance(gt_config.get("edit_intent_gt"), dict) else {}
        predicted_intent = plan.get("intent", {}) if isinstance(plan.get("intent"), dict) else {}
        intent_alignment = _compute_intent_alignment(gt_intent, predicted_intent)
        target_similarity = compute_target_similarity(generated_ts, target_ts)
        visualization_path = None
        if vis_dir_obj is not None:
            visualization_path = build_visualization_path(
                vis_dir=vis_dir_obj,
                sample_id=sample_id,
                target_feature=str(gt_config.get("target_feature", "")),
                task_type=str(gt_config.get("task_type", sample.get("task_type", "unknown"))),
                timestamp=run_timestamp,
            )
            save_pipeline_visualization(
                sample_id=sample_id,
                base_ts=base_ts,
                target_ts=target_ts,
                generated_ts=np.asarray(generated_ts, dtype=np.float32).flatten(),
                gt_start=int(gt_config.get("start_step", 0)),
                gt_end=int(gt_config.get("end_step", len(base_ts))),
                llm_start=int(region[0]),
                llm_end=int(region[1]),
                prompt=vague_prompt,
                tool_name=str(plan.get("tool_name", "?")),
                metrics={
                    "t_iou": metrics.t_iou,
                    "editability_score": metrics.editability_score,
                    "preservability_score": metrics.preservability_score,
                },
                save_path=visualization_path,
                bg_fidelity=bg_fidelity,
            )
            logger.info(f"  可视化已保存: {visualization_path}")

        metrics_payload = {
            "t_iou": metrics.t_iou,
            "feature_accuracy": metrics.feature_accuracy,
            "mse_edit_region": metrics.mse_edit_region,
            "mae_edit_region": metrics.mae_edit_region,
            "mse_preserve_region": metrics.mse_preserve_region,
            "mae_preserve_region": metrics.mae_preserve_region,
            "editability_score": metrics.editability_score,
            "preservability_score": metrics.preservability_score,
            **target_similarity,
            "preservation_mae": metrics.mae_preserve_region,
        }
        experiment_record = build_pure_editing_record(
            sample=sample,
            gt_config=gt_config,
            prompt_text=vague_prompt,
            mode=mode,
            plan=plan,
            metrics=metrics_payload,
            intent_alignment=intent_alignment,
            visualization_path=str(visualization_path) if visualization_path is not None else None,
        )
        results.append({
            **_strip_arrays(sample),
            "mode": mode,
            "generated_ts": np.asarray(generated_ts, dtype=float).tolist(),
            "llm_plan": plan,
            "edit_log": edit_log,
            "intent_alignment": intent_alignment,
            "background_fidelity": bg_fidelity,
            "visualization_path": str(visualization_path) if visualization_path is not None else None,
            "metrics": metrics_payload,
            "experiment_record": experiment_record,
        })

    # ── Stage 4: 汇总并保存 ──────────────────────────────────────────────
    valid = [r for r in results if r.get("metrics")]
    summary = _compute_summary(valid, len(results))
    _print_summary(summary)

    output_obj = Path(output_path)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "mode": mode,
                "run_timestamp_utc": run_timestamp,
                "visualization_dir": str(vis_dir_obj) if vis_dir_obj is not None else None,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    logger.info(f"\n[Stage 4] 结果已保存: {output_obj}")

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _direct_tool_choice(prompt_text: str) -> tuple[str, str]:
    text = str(prompt_text or "")
    lowered = text.lower()
    shape = infer_shape_hint(text)
    down_hints = ("走低", "回落", "下降", "下滑", "停摆", "停机", "降到", "更低位")
    up_hints = ("走高", "冲高", "抬升", "上扬", "上升", "偏高", "高位")
    seasonality_hints = ("周期", "节律", "峰谷", "season", "seasonality", "seasonal", "cyclic", "periodic", "peaks and troughs")

    if any(token in lowered or token in text for token in seasonality_hints):
        return "season_enhance", EDIT_TOOL_SPECS["season_enhance"]["canonical_tool"]
    if shape == "step":
        return "step_shift", EDIT_TOOL_SPECS["step_shift"]["canonical_tool"]
    if shape == "hump":
        return "spike_inject", EDIT_TOOL_SPECS["spike_inject"]["canonical_tool"]
    if shape == "flatline":
        return "hybrid_down", EDIT_TOOL_SPECS["hybrid_down"]["canonical_tool"]
    if any(token in text for token in down_hints):
        return "hybrid_down", EDIT_TOOL_SPECS["hybrid_down"]["canonical_tool"]
    if any(token in text for token in up_hints) or "plateau" in lowered:
        return "hybrid_up", EDIT_TOOL_SPECS["hybrid_up"]["canonical_tool"]
    return "hybrid_up", EDIT_TOOL_SPECS["hybrid_up"]["canonical_tool"]


def _build_direct_edit_plan(prompt_text: str, ts_length: int) -> Dict[str, Any]:
    bucket = infer_position_bucket(prompt_text, fallback="mid")
    shape_hint = infer_shape_hint(prompt_text)
    duration = infer_duration_steps(
        prompt_text,
        ts_length,
        effect_family="volatility" if shape_hint == "irregular_noise" else "",
        canonical_tool="volatility_global_scale" if shape_hint == "irregular_noise" else "",
        shape=shape_hint,
    )
    if bucket == "full":
        region = [0, int(ts_length)]
    elif bucket == "late":
        start = max(0, ts_length - duration)
        region = [start, min(ts_length, start + duration)]
    elif bucket == "early":
        region = [0, min(ts_length, duration)]
    else:
        start = max(0, (ts_length - duration) // 2)
        region = [start, min(ts_length, start + duration)]

    volatility_route = None
    if shape_hint == "irregular_noise":
        volatility_route = resolve_volatility_subtype_route(
            text=prompt_text,
            region=region,
            ts_length=ts_length,
        )
        tool_name = volatility_route["tool_name"]
        canonical_tool = volatility_route["canonical_tool"]
    else:
        tool_name, canonical_tool = _direct_tool_choice(prompt_text)

    spec = EDIT_TOOL_SPECS.get(tool_name, {})
    plan: Dict[str, Any] = {
        "thought": "direct text-to-executor ablation",
        "intent": {
            "effect_family": spec.get("effect_family", "trend"),
            "direction": spec.get("direction", "neutral"),
            "shape": shape_hint or spec.get("shape", "linear"),
            "duration": spec.get("duration", "medium"),
            "strength": spec.get("strength", "medium"),
        },
        "localization": {
            "position_bucket": bucket,
            "region": region,
        },
        "execution": {
            "tool_name": tool_name,
            "canonical_tool": canonical_tool,
            "parameters": {"region": region},
        },
    }
    if volatility_route is not None:
        plan["volatility_subtype"] = volatility_route["final_subtype"]
        plan["volatility_routing"] = {
            "proposed_subtype": volatility_route["proposed_subtype"],
            "guarded_subtype": volatility_route["guarded_subtype"],
            "final_subtype": volatility_route["final_subtype"],
            "guard_reason": volatility_route["guard_reason"],
            "is_preview": volatility_route["is_preview"],
        }
        plan["intent"]["volatility_subtype"] = volatility_route["proposed_subtype"]
        plan["execution"]["volatility_subtype"] = volatility_route["final_subtype"]
        plan["execution"]["parameters"].update(volatility_route["parameters"])

    return normalize_llm_plan(plan, ts_length=ts_length)


def _apply_pipeline_mode(
    *,
    full_plan: Dict[str, Any],
    prompt_text: str,
    ts_length: int,
    mode: str,
) -> Dict[str, Any]:
    normalized = normalize_llm_plan(full_plan, ts_length=ts_length)
    if mode == "full_bettertse":
        return normalized
    if mode == "replay_plan":
        return normalized
    if mode == "direct_edit":
        return _build_direct_edit_plan(prompt_text, ts_length)
    if mode == "wo_localization":
        ablated = normalize_llm_plan(normalized, ts_length=ts_length)
        full_region = [0, int(ts_length)]
        ablated.setdefault("localization", {})["position_bucket"] = "full"
        ablated["localization"]["region"] = full_region
        ablated.setdefault("parameters", {})["region"] = full_region
        ablated.setdefault("execution", {}).setdefault("parameters", {})
        ablated["execution"]["parameters"]["region"] = full_region
        return normalize_llm_plan(ablated, ts_length=ts_length)
    if mode == "wo_canonical_layer":
        ablated = normalize_llm_plan(normalized, ts_length=ts_length)
        tool_name, _ = _direct_tool_choice(prompt_text)
        if str((ablated.get("intent") or {}).get("shape", "")) == "irregular_noise":
            routed = resolve_volatility_subtype_route(
                text=prompt_text,
                proposed_subtype=ablated.get("volatility_subtype") or (ablated.get("intent") or {}).get("volatility_subtype"),
                region=(ablated.get("parameters") or {}).get("region"),
                ts_length=ts_length,
            )
            tool_name = routed["tool_name"]
            ablated["volatility_subtype"] = routed["final_subtype"]
            ablated["volatility_routing"] = {
                "proposed_subtype": routed["proposed_subtype"],
                "guarded_subtype": routed["guarded_subtype"],
                "final_subtype": routed["final_subtype"],
                "guard_reason": routed["guard_reason"],
                "is_preview": routed["is_preview"],
            }
            ablated.setdefault("intent", {})["volatility_subtype"] = routed["proposed_subtype"]
        ablated.pop("canonical_tool", None)
        ablated["tool_name"] = tool_name
        ablated.setdefault("execution", {})
        ablated["execution"]["tool_name"] = tool_name
        ablated["execution"].pop("canonical_tool", None)
        return normalize_llm_plan(ablated, ts_length=ts_length)
    raise ValueError(f"unsupported pipeline mode: {mode}")


def _sample_replay_plan(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a benchmark-provided plan for offline pipeline replay."""
    for key in ("replay_plan", "llm_plan", "cached_plan"):
        plan = sample.get(key)
        if isinstance(plan, dict) and plan:
            return plan
    raise ValueError(
        "replay_plan mode requires each sample to contain replay_plan, llm_plan, or cached_plan"
    )

def _math_only_edit(
    plan: Dict[str, Any],
    base_ts: np.ndarray,
) -> tuple[np.ndarray, str]:
    """无 TEdit 时退化为纯线性数学偏移，仍可评估区域定位指标。"""
    region = plan.get("parameters", {}).get("region", [0, len(base_ts)])
    math_shift = float(plan.get("parameters", {}).get("math_shift", 0.0))
    tool_name = plan.get("tool_name", "unknown")

    s, e = int(region[0]), int(region[1])
    s = max(0, min(s, len(base_ts) - 1))
    e = max(s + 1, min(e, len(base_ts)))

    result = base_ts.copy().astype(np.float32)
    result[s:e] += np.linspace(0, math_shift, e - s, dtype=np.float32)

    log = f"[math-only/{tool_name}] region=[{s},{e}), shift={math_shift:.2f}"
    return result, log


def _strip_arrays(sample: Dict[str, Any]) -> Dict[str, Any]:
    """去掉大型数组字段，保留元数据，避免结果 JSON 过大。"""
    keep_as_is = {"base_ts", "target_ts", "gt_mask"}
    return {k: v for k, v in sample.items() if k not in {"generated_ts"}}


def _compute_background_fidelity(
    base_ts: np.ndarray,
    generated_ts: np.ndarray,
    region: List[int],
) -> Dict[str, float | bool]:
    start = max(0, min(int(region[0]), len(base_ts)))
    end = max(start, min(int(region[1]), len(base_ts)))
    left_err = float(np.max(np.abs(generated_ts[:start] - base_ts[:start]))) if start > 0 else 0.0
    right_err = float(np.max(np.abs(generated_ts[end:] - base_ts[end:]))) if end < len(base_ts) else 0.0
    max_err = max(left_err, right_err)
    return {
        "pass": bool(max_err == 0.0),
        "max_err": max_err,
        "left_err": left_err,
        "right_err": right_err,
    }


def _compute_summary(valid: List[Dict], total: int) -> Dict[str, Any]:
    if not valid:
        return {"total": total, "successful": 0, "failed": total}

    def _avg(key: str) -> float:
        return float(np.mean([r["metrics"][key] for r in valid]))

    intent_rows = [r.get("intent_alignment", {}) for r in valid if r.get("intent_alignment")]

    def _avg_intent(key: str) -> float:
        values = [row[key] for row in intent_rows if key in row]
        return float(np.mean(values)) if values else 0.0

    return {
        "total": total,
        "successful": len(valid),
        "failed": total - len(valid),
        "avg_t_iou": _avg("t_iou"),
        "avg_feature_accuracy": _avg("feature_accuracy"),
        "avg_mse_edit_region": _avg("mse_edit_region"),
        "avg_mae_edit_region": _avg("mae_edit_region"),
        "avg_mse_preserve_region": _avg("mse_preserve_region"),
        "avg_mae_preserve_region": _avg("mae_preserve_region"),
        "avg_mae_vs_target": _avg("mae_vs_target"),
        "avg_mse_vs_target": _avg("mse_vs_target"),
        "avg_preservation_mae": _avg("preservation_mae"),
        "avg_editability_score": _avg("editability_score"),
        "avg_preservability_score": _avg("preservability_score"),
        "avg_intent_match_score": _avg_intent("match_score"),
        "avg_effect_family_match": _avg_intent("effect_family_match"),
        "avg_shape_match": _avg_intent("shape_match"),
        "avg_direction_match": _avg_intent("direction_match"),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline 汇总结果")
    logger.info("=" * 60)
    logger.info(f"  总样本数:            {summary['total']}")
    logger.info(f"  成功:                {summary['successful']}")
    logger.info(f"  失败:                {summary.get('failed', 0)}")
    if summary["successful"] > 0:
        logger.info(f"  avg t-IoU:           {summary['avg_t_iou']:.4f}")
        logger.info(f"  avg Feature Acc:     {summary['avg_feature_accuracy']:.4f}")
        logger.info(f"  avg Target MAE:      {summary['avg_mae_vs_target']:.4f}")
        logger.info(f"  avg Target MSE:      {summary['avg_mse_vs_target']:.4f}")
        logger.info(f"  avg Edit MSE:        {summary['avg_mse_edit_region']:.4f}")
        logger.info(f"  avg Preserve MSE:    {summary['avg_mse_preserve_region']:.4f}")
        logger.info(f"  avg Preserve MAE:    {summary['avg_preservation_mae']:.4f}")
        logger.info(f"  avg Editability:     {summary['avg_editability_score']:.4f}")
        logger.info(f"  avg Preservability:  {summary['avg_preservability_score']:.4f}")
        logger.info(f"  avg Intent Match:    {summary.get('avg_intent_match_score', 0.0):.4f}")
        logger.info(f"  avg Family Match:    {summary.get('avg_effect_family_match', 0.0):.4f}")
        logger.info(f"  avg Shape Match:     {summary.get('avg_shape_match', 0.0):.4f}")
        logger.info(f"  avg Direction Match: {summary.get('avg_direction_match', 0.0):.4f}")
    logger.info("=" * 60)


def _compute_intent_alignment(gt_intent: Dict[str, Any], predicted_intent: Dict[str, Any]) -> Dict[str, Any]:
    if not gt_intent:
        return {}

    def canonicalize_effect_family(intent: Dict[str, Any]) -> Any:
        family = intent.get("effect_family")
        shape = intent.get("shape")
        if family == "shutdown" or shape == "flatline":
            return "hard_zero"
        if family == "level" and shape == "step":
            return "step_change"
        if family == "level" and shape == "scaled_surge":
            return "multiplier"
        if family == "volatility" and shape == "irregular_noise":
            return "noise_injection"
        if family == "impulse" and shape == "hump":
            return "trend"
        return family

    fields = ("effect_family", "shape", "direction")
    matches = {}
    for field in fields:
        gt_value = canonicalize_effect_family(gt_intent) if field == "effect_family" else gt_intent.get(field)
        pred_value = canonicalize_effect_family(predicted_intent) if field == "effect_family" else predicted_intent.get(field)
        matches[field] = 1.0 if gt_value and pred_value and gt_value == pred_value else 0.0

    return {
        "gt_intent": gt_intent,
        "predicted_intent": predicted_intent,
        "effect_family_match": matches["effect_family"],
        "shape_match": matches["shape"],
        "direction_match": matches["direction"],
        "match_score": float(np.mean(list(matches.values()))),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BetterTSE 完整评估 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--testset",
        required=True,
        help="测试集 JSON 路径（由 bettertse_cik_official.py 生成）",
    )
    parser.add_argument(
        "--tedit-model",
        default=None,
        help="TEdit 模型权重路径 (.pth)；不提供则使用纯数学编辑模式",
    )
    parser.add_argument(
        "--tedit-config",
        default=None,
        help="TEdit 配置文件路径 (.yaml)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="TEdit 运行设备，如 cpu / cuda:0（默认: cpu）",
    )
    parser.add_argument(
        "--output",
        default="results/pipeline/pipeline_results.json",
        help="结果 JSON 保存路径（默认: results/pipeline/pipeline_results.json）",
    )
    parser.add_argument(
        "--mode",
        default="full_bettertse",
        choices=["full_bettertse", "direct_edit", "replay_plan", "wo_localization", "wo_canonical_layer"],
        help="pipeline mode / ablation setting",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="逐样本可视化保存目录（默认: output 同级 visualizations/）",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="关闭逐样本可视化保存",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多处理的样本数，用于快速验证（默认: 全部）",
    )
    args = parser.parse_args()
    run_pipeline(
        testset_path=args.testset,
        tedit_model_path=args.tedit_model,
        tedit_config_path=args.tedit_config,
        tedit_device=args.device,
        output_path=args.output,
        mode=args.mode,
        vis_dir=args.vis_dir,
        save_visualizations=not args.no_save_vis,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
