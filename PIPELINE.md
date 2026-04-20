# Pipeline Execution Order - 实验执行流程

> 最后更新：2026-04-20
>
> 本文档只覆盖 pure-editing strength mainline，不覆盖 forecast-revision。

## 当前推荐的执行顺序

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

### Step 2: 构建新线 family-based 训练数据

- 脚本：`python test_scripts/build_tedit_strength_trend_family_dataset.py --csv-path <csv> --dataset-name <name> --output-dir <out> --train-families <n> --valid-families <n> --test-families <n>`
- 目的：生成只服务 `trend_injection` 强度注入主线的 family 训练集
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

### Step 4: 准备主线 trend benchmark

- 脚本：`python test_scripts/build_tedit_strength_discrete_benchmark.py --csv-path <csv> --dataset-name <name> --output-dir <out> --num-families <n> --seq-len <len> --random-seed <seed>`
- 目的：建立离散 `weak / medium / strong` 的 pure-editing strength family benchmark
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
  - `TEdit-main/configs/synthetic/model_multi_weaver.yaml`
- 目的：训练只服务当前强度注入主线的 TEdit
- 当前训练关注点：
  - `strength_control.enabled = true`
  - `use_text_context = true`
  - `use_task_id = false`
  - 局部编辑 family 可显式设置 `train.strength_control.final_output_strength_mapping.scope = edit_region`；默认仍是 `global`
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
- 当前主指标：
  - `edit_gain_weak / medium / strong`
  - `monotonic_hit_rate`
  - `strong_minus_weak_edit_gain`
  - `bg_mae_strong_minus_weak`
  - `edit_gap / abs(bg_gap)`
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
- 当前第一阶段训练主线是 family-based `trend_injection` 数据，不再用 `trend_types` 属性差异监督充当强度训练主线
- 如果实验参数无法完整恢复，允许在 `EXPERIMENTS.csv` 中用摘要式结论回填，但必须明确标注为历史回填
