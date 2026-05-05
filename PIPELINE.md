# Pipeline Execution Order - 实验执行流程

> 最后更新：2026-04-20
>
> 本文档只覆盖 pure-editing strength mainline，不覆盖 forecast-revision。

## 当前推荐的执行顺序

### 先统一当前三层口径

- benchmark / builder 支持范围：6 类
  - `seasonality`
  - `trend`
  - `hard_zero`
  - `step_change`
  - `multiplier`
  - `noise_injection`
- 当前真正完成 strength-control 训练闭环并进入 v1 主实验的：2 类
  - `seasonality`
  - `trend`
- 其余 4 类当前定位：
  - `hard_zero / step_change / multiplier / noise_injection`
  - 属于 `pipeline-coverage / partial-validation` 层
  - 不算当前 v1 主实验类，也不算 formal strength mainline

### Step 1: 锁定协议与主线边界

- 入口文档：`docs/pure_editing_how_much_protocol.md`
- 目的：确认当前只跑 pure-editing `how much / strength injection` 主线
- 重点边界：
  - 第一阶段主线是 `strength_label + pooled text_context`
  - 机制是 `Numeric Projector / residual modulation injection`
  - Phase 1 不把 `task_id` 当作 active mainline 必选项
  - retired path 不应重新写回 runtime：
    - teacher search 作为主方法故事
    - pure-editing student training / runtime override
    - `clip / clip_guard / clip_softguard`
- 输出：统一实验口径，避免跑偏

### Step 2: 构建 family-based 训练数据

- 脚本：`python test_scripts/build_tedit_strength_trend_family_dataset.py --csv-path <csv> --dataset-name <name> --output-dir <out> --train-families <n> --valid-families <n> --test-families <n>`
- 目的：生成 family-based strength 数据
- 当前分层：
  - formal mainline 训练集：`trend_injection` 与 `seasonality_injection`
  - 局部 family / partial-validation：`hard_zero / step_change / multiplier / noise_injection`
- 固定原则：
  - 同一个 family 固定 `source/region/template`
  - family 内只改变 `weak / medium / strong`
  - 训练显式使用 `src_x / tgt_x / mask_gt / strength_label / instruction_text`
- hard_zero 补充约束：
  - builder 会重采样退化 hard_zero family，避免 source 区域贴近零、target edit gain 间隔过小、或 target floor distance 随 strength 反向增大的样本进入 dedicated leaf
  - 构建 hard_zero dedicated leaf 时优先使用 collection root 形态：`--collection-root <root> --selector hard_zero --injection-types hard_zero`，保证实际叶子路径为 `<root>/hard_zero`
- 输出：
  - `<out>/train.json`
  - `<out>/valid.json`
  - `<out>/test.json`
  - `<out>/meta.json`

### Step 3: 构建或补强旧 synthetic strength 标签（仅历史兼容）

- 脚本：`python test_scripts/build_tedit_strength_dataset.py --dataset-folder <dataset_folder>`
- 目的：为历史 synthetic finetune 数据补齐 `strength_label`
- 依赖：
  - `<dataset_folder>/meta.json`
  - `<dataset_folder>/{train,valid,test}_attrs_idx.npy`
- 输出：
  - `<dataset_folder>/{train,valid,test}_strength.npy`
  - `<dataset_folder>/{train,valid,test}_task_id.npy`
  - `meta.json` 中的 `strength_control` 元信息

### Step 4: 准备 strength benchmark

- 脚本：`python test_scripts/build_tedit_strength_discrete_benchmark.py --csv-path <csv> --dataset-name <name> --output-dir <out> --num-families <n> --seq-len <len> --random-seed <seed>`
- 目的：建立离散 `weak / medium / strong` 的 pure-editing strength family benchmark
- 当前 benchmark / builder 支持 6 类：
  - `trend_injection`
  - `seasonality_injection`
  - `hard_zero`
  - `step_change`
  - `multiplier`
  - `noise_injection`
- 推荐参考：
  - 同一个 family 固定 `source/tool/region/shape template`
  - family 内只改变 `strength_label`
- 输出：
  - `tedit_discrete_strength_benchmark_<dataset>_<n>families.json`
  - 对应 `README.md`
  - 可选 health 检查：`python test_scripts/check_tedit_strength_discrete_benchmark.py --benchmark <json> --output <health_json>`

### Step 5: 进行 family-based strength-conditioned TEdit finetune

- 入口脚本：`python TEdit-main/run_finetune.py ...`
- 相关配置：
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`
  - `TEdit-main/configs/synthetic/finetune_strength_seasonality_family.yaml`
  - `TEdit-main/configs/synthetic/model_multi_weaver.yaml`
- 目的：训练只服务当前 formal strength mainline 的 TEdit
- 当前训练关注点：
  - `strength_control.enabled = true`
  - `use_text_context = true`
  - `use_task_id = false`
  - 局部编辑 family 可显式设置 `train.strength_control.final_output_strength_mapping.scope = edit_region`；默认仍是 `global`
  - 当前已完成训练闭环并进入 v1 主实验的：`trend / seasonality`
  - 已做小样本正向验证的局部 family：`hard_zero / step_change / multiplier / noise_injection`
  - 新输出层监督：
    - `edit_region_loss_weight`
    - `background_loss_weight`
    - `monotonic_loss_weight`
    - `monotonic_margin`
- 注意：
  - `run_finetune.py` 的 CLI 默认 `--epochs=50` 会覆盖 yaml 里的 `train.epochs`；跑小样本对照时应显式传 `--epochs 10` 或目标 epoch 数。
  - synthetic pretrain 基线通常需要显式 `--pretrained_dir save/synthetic/pretrain_multi_weaver`，避免落到不存在的默认 `save/synthetic/pretrain`。
- 关键输出：
  - `save/.../ckpts/model_best.pth`
  - `model_configs.yaml`
  - `finetune_configs.yaml`
  - `results_finetune.csv`

### Step 6: 运行 internal probe

- 脚本：`python test_scripts/probe_tedit_strength_internal.py --model-path <pth> --config-path <yaml> --dataset-folder <dataset_folder> --output <json> [--task-id <id>]`
- 目的：验证仅改变 `strength_label` 时，内部 diffusion 输出是否发生变化
- mask-local 注意：
  - dedicated leaf / collection-root probe 会把 `mask_gt` 传入 `TEditWrapper.edit_time_series(edit_mask=...)`。
  - 如果绕过该脚本直接调用 wrapper，`final_output_strength_mapping.scope=edit_region` 需要显式传 `edit_mask`；否则会按兼容逻辑退回 global。
- 固定原则：
  - 同一个样本
  - 同一个 seed
  - 只改 `strength_label`
  - 第一阶段使用 `--task-id -1`
- 核心输出：
  - `rows[].peak_abs_delta`
  - `rows[].mean_abs_delta`
  - `diff_0_1_linf`
  - `diff_1_2_linf`
  - `diff_0_2_linf`

### Step 7: 做 monotonicity / preservation / ablation 评估

- 简化入口脚本：`python test_scripts/evaluate_tedit_strength_effect.py --model-path <pth> --config-path <yaml> --dataset-folder <dataset_folder> --output <json>`
- discrete benchmark 主实验：`python test_scripts/run_tedit_trend_monotonic_eval.py --benchmark <json> --model-path <pth> --config-path <yaml> --output <json>`
- mask-local 注意：
  - `evaluate_tedit_strength_effect.py` 会把 benchmark record 中的 `edit_mask` 路由到 wrapper；这是验证 `scope=edit_region` 的必要条件。
- 评估目标：
  - monotonicity：`weak < medium < strong`
  - preservation：背景区误差不可失控
  - ablation：
    - original TEdit
    - prompt-only strength
    - internal residual modulation
- 当前通用/兼容指标：
  - `edit_gain_weak / medium / strong`
  - `monotonic_hit_rate`
  - `strong_minus_weak_edit_gain`
  - `bg_mae_strong_minus_weak`
  - `edit_gap / abs(bg_gap)`
- 当前 family-specific 主指标：
  - `primary_strength_metric`
  - `weak_primary_strength_value / medium_primary_strength_value / strong_primary_strength_value`
  - `primary_min_adjacent_gap_mean`
  - `primary_monotonic_hit`
  - trend 使用 `edit_gain`
  - seasonality 使用编辑增量 `edited - base` 上的 `fixed_period_fourier_amplitude`，并要求 benchmark 样本保留 `injection_config.cycles / expected_period`
  - hard_zero 使用 `zero_suppression_delta`
  - step_change 使用 `step_level_shift`
  - multiplier 使用 `multiplicative_abs_ratio`
  - noise_injection 使用 `local_noise_roughness_delta`
- seasonality 防作弊诊断：
  - `dominant_period_error_max`：在编辑增量 `edited - base` 上做 dominant period 诊断，避免源序列自身低频结构污染判定。
  - `level_drift_max`
  - `trend_drift_max`
  - 旧 run 若缺少 `injection_config`，summary 会回退到兼容 `edit_gain`，不能作为 season amplitude 主结论。
- strength pipeline benchmark 会写入 `pipeline_options.family_region_source = "benchmark"`：
  - full pipeline 仍可调用 LLM 做 intent/tool parsing
  - 但 family strength benchmark 的执行窗口锚到 benchmark GT region，避免 LLM 首样本窗口偏移破坏 fixed-period / same-family strength calibration
- `run_pipeline.py --mode replay_plan` 用于离线复现实验：
  - 每个 sample 必须包含 `replay_plan`、`llm_plan` 或 `cached_plan`
  - Stage 1 跳过 LLM，后续仍走同一 TEdit executor / evaluator / output schema
  - 用途是 API 不可用时验证 benchmark target、region policy、runtime strength 注入和 family-specific summary 是否闭合
  - 不能替代 fresh `full_bettertse` 的 planner 质量结论
- `test_scripts/build_strength_pipeline_replay_benchmark.py` 会把普通 strength benchmark 转成 replay benchmark：
  - 为每个 sample 写入 `replay_plan`
  - 对 frozen trend legacy checkpoint 可写入 `runtime_strength_scalar = strength_label`
  - 推荐作为 `run_pipeline.py --mode replay_plan` 的输入生成步骤
- `test_scripts/run_strength_pipeline_replay_validation.py` 是一键 replay validation 入口：
  - 构建 replay benchmark
  - 调用 `test_scripts/run_strength_pipeline_main_experiment.py --mode replay_plan`
  - 生成 summary/report
  - 调用 `verify_strength_pipeline_summary.py`
  - 输出 `validation_manifest.json`
- `test_scripts/verify_strength_pipeline_summary.py` 是 strength gate verifier：
  - 检查 required families 是否齐全
  - 检查每类 `primary_strength_metric` 是否匹配 family-specific 定义
  - 检查 `weak < medium < strong` 与 `primary_monotonic_hit`
  - 检查背景漂移阈值
  - 对 seasonality 额外检查 `dominant_period_error_max`
- seasonality amplitude benchmark 会通过 `injection_config` 向执行器传递固定 `cycles / phase / amplitude`：
  - 执行器优先使用 fixed-period scaffold
  - 本阶段不允许通过 source residual 重新估计 frequency
  - benchmark/runtime 的 fixed-period wave 必须对常数项和线性项正交化，并按 fixed-period Fourier amplitude 归一化；否则会把 level/trend drift 混入 season amplitude 主指标
- local-family benchmark 会通过 `injection_config` 向执行器传递对应物理 target：
  - `hard_zero`：`floor_value / ramp`
  - `step_change`：`magnitude / ramp_out`
  - `multiplier`：`multiplier / ramp_out`
  - `noise_injection`：`noise_std_ratio / baseline_offset / region_noise_std / noise_template`
  - 执行器优先使用 target-defined scaffold，避免通过 trend/level/frequency/随机纹理伪造 strength。
- volatility/noise 辅助指标：
  - `local_std_strong_minus_weak_mean`
  - `local_energy_strong_minus_weak_mean`
  - `local_roughness_strong_minus_weak_mean`
  - `bg_mae_strong_minus_weak`
  - `mae_vs_target`
  - `mse_vs_target`
  - `preservation_mae`
- volatility 家族额外关注：
  - `local_std_error`
  - `roughness_error`
  - `windowed_energy_profile_error`

### Step 8: 更新研究记录

- 更新 `PROGRESS.md`
  - 记录这次实验目的、关键改动、结论
- 更新 `EXPERIMENTS.csv`
  - 无论成功还是失败都落一条记录
- 如主线边界或执行顺序变化，更新本文档

## 当前推荐的主线 artifact

- `tmp/strength_phase1/probe/pretrained_probe.json`
- `tmp/strength_phase1/probe/post_finetune_probe.json`
- `tmp/strength_phase1/eval/phase1b_strength_effect.json`

## 注意事项

- 当前账本只服务 pure-editing strength 主线，不要把 forecast-revision 结果混进来
- `run_pipeline.py` 不应重新引入 retired pure-editing student runtime override
- 旧的 teacher / student 历史可作为背景参考，但不是当前 mainline 的结果展示结构
- 当前第一阶段口径是 `strength + text`，不要把 `task_id` 重新写成 Phase 1 必选条件
- 当前正式 strength mainline 是 family-based `trend_injection + seasonality_injection`
- `hard_zero / step_change / multiplier / noise_injection` 当前属于 pipeline-coverage / partial-validation，不要在文案上与 formal mainline 混层
- 如果实验参数无法完整恢复，允许在 `EXPERIMENTS.csv` 中用摘要式结论回填，但必须明确标注为历史回填
