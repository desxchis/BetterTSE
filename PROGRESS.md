# Research Progress Log - 科研代码进度记录

按时间倒序记录 pure-editing strength mainline 的关键推进，不混入 forecast-revision 主线。

## [2026-04-28] - trend legacy scalar restore 最小闭环完成；R0/R1 都未穿透真实 local-path spacing blocker

**本轮目标**：
- 不改 route，不回 old `standard-route` 救援线。
- 只验证旧主线 `scalar contract = 0 / 1 / 2` 能否单独把 `trend` 的 raw/local spacing 拉起来。
- 若仅恢复 legacy scalar 不够，再加一组旧主线风格的中等 spacing pressure。

**固定验收路径**：
- canonical local path 仍锁定为：
  - `edit_time_series(..., edit_mask=mask_gt)`
  - `final_output_strength_mapping.scope = edit_region`
- benchmark/effect/probe 全部统一到：
  - `acceptance_route = local_path_mask_routed`
  - `scalar_scheme = legacy_0_1_2`

**代码与数据改动**：
- 新增 trend-only legacy scalar family 构造：
  - `test_scripts/build_tedit_strength_trend_family_dataset.py`
  - 支持 `--scalar-scheme legacy_0_1_2`
  - strength scalar 改为 `weak=0.0 / medium=1.0 / strong=2.0`
- probe / effect / monotonic benchmark 全部识别并写出：
  - `scalar_scheme = legacy_0_1_2`
  - legacy sweep:
    - `0.0, 0.5, 1.0, 1.5, 2.0`
- 并行新配置：
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family_restore_legacy_scalar_r0.yaml`
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family_restore_legacy_scalar_r1.yaml`
- 并行新数据集：
  - `TEdit-main/datasets/discrete_strength_trend_family_restore_legacy_scalar`

**数据构造与健康检查**：
- 构造命令：
  - `python test_scripts/build_tedit_strength_trend_family_dataset.py --csv-path data/ETTh1.csv --dataset-name ETTh1 --output-dir TEdit-main/datasets/discrete_strength_trend_family_restore_legacy_scalar --seq-len 192 --random-seed 17 --train-families 96 --valid-families 24 --test-families 24 --injection-types trend_injection --selector trend_injection --scalar-scheme legacy_0_1_2`
- health：
  - `python test_scripts/check_tedit_strength_discrete_benchmark.py --benchmark TEdit-main/datasets/discrete_strength_trend_family_restore_legacy_scalar/test.json --output TEdit-main/datasets/discrete_strength_trend_family_restore_legacy_scalar/test_benchmark_health.json`
- 结果：
  - `complete_strength_rate = 1.0`
  - `target_monotonic_rate = 1.0`
  - `zero_background_leak_rate = 1.0`
  - `health_pass = true`

**R0：legacy scalar restore only**
- 训练配置：
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family_restore_legacy_scalar_r0.yaml`
- 保存目录：
  - `TEdit-main/save/synthetic/R0_legacy_scalar_restore`
- 训练结论：
  - best checkpoint 停在 `epoch 1`
  - `best_model_selection.json`:
    - `selection_score = 1.0648`
    - `pred_min_adjacent_gap_mean = 0.9814`
  - 后续 epoch 回落，所以中途截断，直接用当前 best 做验证。

**R0 验证**
- probe：
  - `tmp/R0_legacy_scalar_restore_probe.json`
  - projector separation 存在：
    - `0.0000_1.0000 = 0.0126`
    - `0.0000_2.0000 = 0.0418`
  - final mapping 有序：
    - `0.99286 -> 0.99994 -> 1.00703`
  - 但单样本 raw edit gain 基本不分：
    - `1.49984 -> 1.49981 -> 1.49919`
- effect audit：
  - `tmp/R0_legacy_scalar_restore_effect.json`
  - `monotonic_hit_rate = 0.4167`
  - `weak_le_medium_pass_rate = 0.4167`
  - `medium_le_strong_pass_rate = 0.4583`
  - `raw_min_adjacent_gap_mean = -0.00206`
  - `strong_minus_weak_edit_gain_mean = -0.00332`
  - `preservation_pass_rate = 1.0`
- monotonic benchmark：
  - `tmp/R0_legacy_scalar_restore_monotonic.json`
  - `adjacent_monotonic_pass_rate = 0.3333`
  - `weak_le_medium_pass_rate = 0.4167`
  - `medium_le_strong_pass_rate = 0.5`
  - `min_adjacent_gap_mean = -0.000913`
  - `adjacent_gap_collapse_rate = 0.625`
  - `preservation_pass_rate = 1.0`
- bucket 结论：
  - `short` 有一点正 spacing：
    - `min_adjacent_gap_mean = 0.000481`
  - `medium`、`long` 仍系统性负 gap

**R1：legacy scalar + moderate spacing pressure**
- 训练配置：
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family_restore_legacy_scalar_r1.yaml`
- 参数只加旧主线风格中等 pressure：
  - `strength_lr_scale = 5.0`
  - `gain_match_loss_weight = 8.0`
  - `family_relative_gain_loss_weight = 6.0`
  - `constant_gain_penalty_weight = 3.0`
  - `minimum_family_gain_std = 0.08`
  - `monotonic_loss_weight = 0.30`
- 保存目录：
  - `TEdit-main/save/synthetic/R1_legacy_scalar_restore_plus_spacing_pressure`
- 训练结论：
  - best checkpoint 仍停在 `epoch 1`
  - `best_model_selection.json`:
    - `selection_score = 1.3227`
    - `pred_min_adjacent_gap_mean = 1.2393`
  - `epoch 2` 已明显回落，所以直接截断，用 best 做验证。

**R1 验证**
- probe：
  - `tmp/R1_legacy_scalar_restore_plus_spacing_pressure_probe.json`
  - projector separation 仍在，但更小：
    - `0.0000_1.0000 = 0.00666`
    - `0.0000_2.0000 = 0.00735`
  - final mapping 仍有序：
    - `0.99289 -> 0.99997 -> 1.00706`
  - 单样本 raw edit gain 仍几乎不分：
    - `1.50080 -> 1.50078 -> 1.49998`
- effect audit：
  - `tmp/R1_legacy_scalar_restore_plus_spacing_pressure_effect.json`
  - `monotonic_hit_rate = 0.4583`
  - `weak_le_medium_pass_rate = 0.5`
  - `medium_le_strong_pass_rate = 0.4583`
  - `raw_min_adjacent_gap_mean = -0.00200`
  - `strong_minus_weak_edit_gain_mean = -0.00334`
  - `preservation_pass_rate = 1.0`
- monotonic benchmark：
  - `tmp/R1_legacy_scalar_restore_plus_spacing_pressure_monotonic.json`
  - `adjacent_monotonic_pass_rate = 0.4167`
  - `weak_le_medium_pass_rate = 0.5`
  - `medium_le_strong_pass_rate = 0.5`
  - `min_adjacent_gap_mean = -0.000888`
  - `adjacent_gap_collapse_rate = 0.5417`
  - `preservation_pass_rate = 1.0`
- bucket 结论：
  - `short` 比 R0 更好：
    - `adjacent_monotonic_pass_rate = 0.625`
    - `weak_le_medium_pass_rate = 0.875`
    - `medium_le_strong_pass_rate = 0.75`
    - `min_adjacent_gap_mean = 0.000502`
  - `medium`、`long` 仍是负 gap
  - `long` 仍然最差：
    - `min_adjacent_gap_mean = -0.001903`
    - `adjacent_gap_collapse_rate = 0.75`

**本轮判断**：
- 恢复旧主线 `0 / 1 / 2` scalar contract 是有信息量的：
  - valid-side predicted spacing 会一度明显转正
  - `short` bucket 在真实 local-path test 上也确实比旧窄标尺更稳
- 但这还不足以穿透当前真实 blocker：
  - canonical local-path test 上，`edit_gain_mean` 仍几乎恒定
  - R0:
    - `4.7607 -> 4.7591 -> 4.7573`
  - R1:
    - `4.7620 -> 4.7605 -> 4.7587`
- 所以结论不是 route 问题，也不是 background drift 问题，而是：
  - raw/local final gain 仍然被一个几乎恒定的大 base edit 主导
  - legacy scalar restore 和中等 spacing pressure 只改善了 `short`
  - 还没有把 `medium/long` 的真实 spacing 拉到可用水平

**下一步边界**：
- 不再回 old `standard-route`
- 不再继续磨 runtime-only final mapping
- 也不能再把“legacy scalar restore”当作单独解法
- 后续若继续修 `trend`，必须继续留在 training-side raw/local spacing-first 主线，但需要比这轮更有穿透力的训练修复

## [2026-04-28] - trend spacing 第一轮重训 + A0/A1/A2 顺序预验证完成；runtime-only mapping 线收口

**本轮目标**：
- 不回旧 `standard-route` 救援线。
- 锁定 `trend` 的 active acceptance path：
  - `edit_time_series(..., edit_mask=mask_gt)`
  - `final_output_strength_mapping.scope=edit_region`
- 先看 training-side 的最小 spacing-first 重训是否能把 canonical local-path 下的相邻 spacing 立起来。
- 再顺序跑 `A0 -> A1 -> A2`，验证 runtime-only final mapping 拉伸能不能单独解决 benchmark-facing blocker。

**训练执行**：
- 环境：
  - `source /root/miniconda3/bin/activate tedit`
  - `CUDA_VISIBLE_DEVICES=0`
- 已完整跑完：
  - `TEdit-main/configs/synthetic/finetune_strength_trend_family_spacing_t1.yaml`
- 产物：
  - best checkpoint:
    - `TEdit-main/save/synthetic/finetune_strength_trend_family_spacing_t1/0/trend_injection/ckpts/model_best.pth`
  - selection:
    - `TEdit-main/save/synthetic/finetune_strength_trend_family_spacing_t1/0/trend_injection/best_model_selection.json`
- 结果：
  - best 落在 `epoch 9`
  - `avg_loss_valid = 104.9662`
  - predicted-spacing `selection_score = 0.6867`
- `T3` 已启动过一轮，但前几个 epoch 明显落后 `T1`，中途停止，不继续烧 GPU：
  - `TEdit-main/save/synthetic/finetune_strength_trend_family_spacing_t3/0/trend_injection/best_model_selection.json`
- `T4` 本轮未启动。

**T1 验证结果**：
- `probe`：
  - `tmp/trend_spacing_t1_probe.json`
  - projector separation 仍在：
    - `0.0000_0.5000 = 0.0163`
    - `0.0000_1.0000 = 0.0256`
  - final output mapping 仍有序：
    - `0.99298 -> 1.00006 -> 1.00714`
- canonical-route `effect audit`：
  - `tmp/trend_spacing_t1_effect.json`
  - `monotonic_hit_rate = 1.0`
  - `weak_le_medium_pass_rate = 1.0`
  - `medium_le_strong_pass_rate = 1.0`
  - `strong_minus_weak_edit_gain_mean = 0.0205`
  - `preservation_pass = true`
- canonical-route monotonic benchmark：
  - `tmp/trend_spacing_t1_monotonic.json`
  - `adjacent_monotonic_pass_rate = 0.625`
  - `weak_le_medium_pass_rate = 0.625`
  - `medium_le_strong_pass_rate = 0.75`
  - `min_adjacent_gap_mean = -0.00242`
  - `adjacent_gap_collapse_rate = 0.375`
  - `gain_range_mean = -0.00903`

**A0/A1/A2 顺序预验证**：
- 目标：
  - 只看 runtime-only `scalar_prior_scale` 能不能把 canonical local-path 的 spacing blocker 推过线。
- 路线统一：
  - `generation_route = standard`
  - `eval_mask_routed = true`
  - `final_mapping_scope = edit_region`
- 产物：
  - `A0`: `tmp/trend_spacing_t1_monotonic.json`
  - `A1`: `tmp/preval_a1/trend_spacing_t1_A1_monotonic.json`
  - `A2`: `tmp/preval_a2/trend_spacing_t1_A2_monotonic.json`

**A0/A1/A2 关键对比**：
- `A0` baseline：
  - `adjacent_monotonic_pass_rate = 0.625`
  - `min_adjacent_gap_mean = -0.00242`
  - `adjacent_gap_collapse_rate = 0.375`
- `A1` (`scalar_prior_scale = 0.16`)：
  - `adjacent_monotonic_pass_rate = 0.5`
  - `min_adjacent_gap_mean = -0.00499`
  - `adjacent_gap_collapse_rate = 0.375`
- `A2` (`scalar_prior_scale = 0.24`)：
  - `adjacent_monotonic_pass_rate = 0.5`
  - `min_adjacent_gap_mean = -0.00871`
  - `adjacent_gap_collapse_rate = 0.375`

**bucket-level 结论**：
- `long` bucket 在 `A0/A1/A2` 都是健康的：
  - monotonic pass `= 1.0`
  - `min_adjacent_gap_mean > 0`
- 真正塌的是 `short / medium`：
  - `A0` 已经是负 gap
  - `A1/A2` 继续把负 gap 拉大
  - `A2` 比 `A1` 更差，尤其 `medium`

**本轮结论**：
- `T1` 确实比旧 trend checkpoint 更好：
  - internal separation 在
  - canonical-route effect 单调
  - locality / preservation 正常
- 但 active benchmark gate 仍未通过：
  - raw/local adjacent spacing 还是太小
  - `short / medium` 仍会 collapse 或反向
- `A0 -> A1 -> A2` 已经给出足够证据：
  - runtime-only final mapping 拉伸不能解决当前 blocker
  - 而且更高的 `scalar_prior_scale` 会把 benchmark-facing spacing 推得更差

**当前决定**：
- 收口 runtime-only mapping 这条线，不再继续往 `A3/A5` 扩。
- 后续如果继续修 `trend`，重点应回到 training-side 的 raw/local spacing-first 修复，而不是继续做 runtime-only 拉伸。

## [2026-04-28] - seasonality 正式版数据集 + full training 已跑通；local-route test 全量通过

**本轮目标**：
- 把 seasonality 从 smoke 规模推进到正式训练规模。
- 沿用当前 pure-editing strength 主线，不另造新训练流程：
  - family dataset
  - `strength_scalar = 0 / 0.5 / 1.0`
  - `strength_lr_scale = 10`
  - output branch carrier + final output strength mapping
  - local-route effect eval 统一走 `edit_region_soft`
- 在 `cuda:0` 上跑完整 finetune，并看全量 test split 的真实编辑效果。

**数据集升级**：
- 原 `TEdit-main/datasets/discrete_strength_seasonality_family` 之前只是 smoke：
  - train/valid/test families = `6 / 3 / 3`
- 已重建为正式版：
  - train/valid/test families = `96 / 24 / 24`
  - num samples = `288 / 72 / 72`
- 构建命令：
  - `python test_scripts/build_tedit_strength_seasonality_family_dataset.py --csv-path data/ETTh1.csv --dataset-name ETTh1 --output-dir TEdit-main/datasets/discrete_strength_seasonality_family --seq-len 192 --random-seed 17 --train-families 96 --valid-families 24 --test-families 24`
- health check：
  - `python test_scripts/check_tedit_strength_discrete_benchmark.py --benchmark TEdit-main/datasets/discrete_strength_seasonality_family/test.json --output TEdit-main/datasets/discrete_strength_seasonality_family/test_benchmark_health.json`
- health 结果：
  - `num_families = 24`
  - `complete_strength_rate = 1.0`
  - `target_monotonic_rate = 1.0`
  - `zero_background_leak_rate = 1.0`
  - `health_pass = true`

**训练设置**：
- 环境：
  - `source /root/miniconda3/bin/activate tedit`
  - `CUDA_VISIBLE_DEVICES=0`
- checkpoint / config：
  - pretrained: `save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth`
  - finetune config: `TEdit-main/configs/synthetic/finetune_strength_seasonality_family.yaml`
  - save folder: `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full`
- 训练命令：
  - `cd TEdit-main && python run_finetune.py --pretrained_dir save/synthetic/pretrain_multi_weaver --model_config_path model_configs.yaml --pretrained_model_path ckpts/model_best.pth --finetune_config_path configs/synthetic/finetune_strength_seasonality_family.yaml --evaluate_config_path configs/synthetic/evaluate.yaml --data_folder ./datasets/discrete_strength_seasonality_family --save_folder ./save/synthetic/finetune_strength_seasonality_family_full --n_runs 1 --epochs 10 --include_self 0 --strength-diagnostics 1 --strength-diagnostics-interval 1`

**训练产物**：
- best checkpoint:
  - `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/ckpts/model_best.pth`
- runtime config:
  - `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/resolved_runtime_config.json`
- diagnostics:
  - `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/strength_diagnostics.jsonl`
  - `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/valid_epoch_summary.jsonl`
  - `TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/best_model_selection.json`

**训练结果**：
- best model selection 落在 `epoch 9`
- `avg_loss_valid = 27.1934`
- 训练过程中 best checkpoint 更新轨迹：
  - `epoch 0 -> 1 -> 5 -> 7 -> 8 -> 9`
- 说明：
  - 这轮没有发散；
  - valid loss 随训练总体下降；
  - seasonality 线在正式规模数据上能稳定完成 full training。

**重要限制（需要留痕）**：
- 当前 `valid_epoch_summary.jsonl` / `best_model_selection.json` 里的 spacing summary 仍然是 target-side summary：
  - `target_medium_minus_weak_mean`
  - `target_strong_minus_medium_mean`
  - `target_min_adjacent_gap_mean`
- 这些值在 epoch 间基本不变，因此本轮 best checkpoint 的选择，本质上是：
  - spacing score 相同
  - 再由 `avg_loss_valid` 做 tie-break
- 这意味着：
  - 当前训练期 selection payload 还不能直接代表“模型真实输出的 spacing quality”；
  - 真实效果必须靠独立的 local-route effect eval 来看。

**全量 local-route effect eval**：
- 命令：
  - `python test_scripts/evaluate_tedit_strength_effect.py --model-path TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/ckpts/model_best.pth --config-path TEdit-main/save/synthetic/finetune_strength_seasonality_family_full/0/seasonality_injection/resolved_runtime_config.json --dataset-folder TEdit-main/datasets/discrete_strength_seasonality_family --split test --max-samples 24 --device cuda:0 --edit-steps 10 --condition-mode both --selector seasonality_injection --final-mapping-scope edit_region --output tmp/seasonality_tedit_strength_eval_softlocal_fulltrain.json`
- 产物：
  - `tmp/seasonality_tedit_strength_eval_softlocal_fulltrain.json`

**全量 effect 结果**：
- `generation_route = edit_region_soft`
- `monotonic_hit_rate = 1.0`
- `weak_le_medium_pass_rate = 1.0`
- `medium_le_strong_pass_rate = 1.0`
- `strength_visible_in_final = true`
- `preservation_pass = true`
- `final_output_strength_mapping_by_scalar`：
  - `0.0000 -> 0.9296`
  - `0.5000 -> 1.0021`
  - `1.0000 -> 1.0798`
- `final_output_strength_mapping_transformed_scalar_by_scalar`：
  - `0 -> 0`
  - `0.5 -> 1`
  - `1.0 -> 2`

**全量 effect 核心数值**：

| metric | value |
|---|---:|
| target_edit_gain_mean weak / medium / strong | `0.3601 / 0.7202 / 1.2603` |
| final edit_gain_mean weak / medium / strong | `0.3505 / 0.4261 / 0.5094` |
| final strong-minus-weak | `0.1589` |
| final medium-minus-weak | `0.0756` |
| final strong-minus-medium | `0.0832` |
| final min adjacent gap mean | `0.0727` |
| bg_mae_mean weak / medium / strong | `6.38e-4 / 6.63e-4 / 6.83e-4` |
| bg_mae_strong_minus_weak | `4.52e-5` |
| raw_to_final_monotonic_drop | `0.0` |
| gain_calibration_mae_mean | `0.3924` |

**duration bucket 结果**：
- short:
  - `monotonic_hit_rate = 1.0`
  - `strong_minus_weak = 0.1194`
  - `preservation_pass_rate = 1.0`
- medium:
  - `monotonic_hit_rate = 1.0`
  - `strong_minus_weak = 0.2578`
  - `preservation_pass_rate = 1.0`
- long:
  - `monotonic_hit_rate = 1.0`
  - `strong_minus_weak = 0.0994`
  - `preservation_pass_rate = 1.0`

**阶段结论**：
- seasonality 现在已经从“接入 + smoke 可跑”推进到：
  - `formal-dataset trained`
  - `full local-route test passed`
- 它在正式版数据和 full local-route effect eval 上已经满足当前主线期望：
  - 强度单调可见
  - bucket 全通过
  - 非编辑区保真稳定
- 但还需要保留一个工程性注意点：
  - 训练期 best-model selection 仍偏 target-side，尚不能替代独立 effect eval。

## [2026-04-27] - pure-editing strength eval 正式切到 `edit_region_soft`；seasonality local-mask 路线通过，trend 旧 checkpoint 仍弱

**本轮目标**：
- 复核“当前 acceptance/local-path 定义是否真的走了 local mask 方法”。
- 修正 pure-editing strength effect eval 的真实生成路由，避免把 `edit_time_series(..., edit_mask=...)` 和 `edit_region_soft(...)` 混为一谈。
- 在不重训 seasonality 的前提下，确认：
  - local-mask 路线是否真的生效；
  - seasonality checkpoint 在真实 local route 下是否还能保持 `weak < medium < strong`；
  - trend 旧 checkpoint 在同一路线上是否只是“没走 local mask”，还是模型本身就弱。

**关键判断更新**：
- 之前的 `evaluate_tedit_strength_effect.py` 实际走的是：
  - `edit_time_series(..., edit_mask=...)`
  - 这本质上仍是 standard global edit，不是 `edit_region_soft` 的 latent blending local path。
- 因此：
  - seasonality 那轮“背景漂移很大”的根因之一，确实是评估主线没走真正的 local mask；
  - 先前为压住背景漂移加的 wrapper output blend 只能算临时兜底，不是这条评估主线的正确结构。

**代码改动**：
- `test_scripts/evaluate_tedit_strength_effect.py`
  - effect eval 真实生成路由从 `edit_time_series(..., edit_mask=...)` 切到 `edit_region_soft(...)`；
  - 使用 family 里的 `region_start / region_end` 作为局部编辑区域；
  - 缺失 region 时从 `edit_mask` 反推；
  - route metadata 统一改成：
    - `generation_route = edit_region_soft`
    - `generation_route_label = soft_local_edit_region`
    - `acceptance_route = local_path_soft_mask`
    - `route_statement = edit_region_soft + soft local mask + latent blending`
- `test_scripts/test_output_branch_carrier_contract.py`
  - 新增 contract：`evaluate_tedit_strength_effect.py` 必须调用 `edit_region_soft`，不能再回退到 `edit_time_series`；
  - 保留原有 `edit_time_series + edit_mask` wrapper contract，不破坏其他调用链。

**静态 / contract 验证**：
- `python -m py_compile test_scripts/evaluate_tedit_strength_effect.py test_scripts/test_output_branch_carrier_contract.py`
- `python -m unittest -v test_scripts.test_output_branch_carrier_contract`

**seasonality local-route 复核**：
- 运行：
  - `source /root/miniconda3/bin/activate tedit`
  - `CUDA_VISIBLE_DEVICES=0 python test_scripts/evaluate_tedit_strength_effect.py --model-path TEdit-main/save/synthetic/finetune_strength_seasonality_family/0/seasonality_injection/ckpts/model_best.pth --config-path TEdit-main/save/synthetic/finetune_strength_seasonality_family/0/seasonality_injection/resolved_runtime_config.json --dataset-folder TEdit-main/datasets/discrete_strength_seasonality_family --split test --max-samples 3 --device cuda:0 --edit-steps 10 --condition-mode both --selector seasonality_injection --final-mapping-scope edit_region --output tmp/seasonality_tedit_strength_eval_softlocal.json`
- 产物：
  - `tmp/seasonality_tedit_strength_eval_softlocal.json`
- 结果：
  - `generation_route = edit_region_soft`
  - `monotonic_hit_rate = 1.0`
  - `weak_le_medium_pass_rate = 1.0`
  - `medium_le_strong_pass_rate = 1.0`
  - `strength_visible_in_final = true`
  - `preservation_pass = true`
  - `final_background_mean_abs_delta` 约 `5e-4`
  - `final_output_strength_mapping_by_scalar` 仍按 `0.0000 < 0.5000 < 1.0000` 有序
- 当前结论：
  - seasonality 这条线现在已经通过“真正 local mask 路由”验证；
  - 它不再依赖 global route 或 wrapper 后融合来保背景；
  - 当前阶段可以把 seasonality 视为已进入 `local-path verified` 状态。

**trend 对照复核**：
- 运行：
  - `source /root/miniconda3/bin/activate tedit`
  - `CUDA_VISIBLE_DEVICES=0 python test_scripts/evaluate_tedit_strength_effect.py --model-path TEdit-main/save/synthetic/finetune_strength_trend_family_semantic_split/0/trend_injection/ckpts/model_best.pth --config-path TEdit-main/save/synthetic/finetune_strength_trend_family_semantic_split/0/trend_injection/model_configs.yaml --dataset-folder TEdit-main/datasets/discrete_strength_trend_family --split test --max-samples 3 --device cuda:0 --edit-steps 10 --condition-mode both --selector trend_injection --final-mapping-scope edit_region --output tmp/trend_tedit_strength_eval_softlocal.json`
- 产物：
  - `tmp/trend_tedit_strength_eval_softlocal.json`
- 结果：
  - `generation_route = edit_region_soft`
  - `preservation_pass = true`
  - `monotonic_hit_rate = 0.3333`
  - `medium_minus_weak_edit_gain_mean < 0`
  - `final_output_strength_mapping_by_scalar` 为空
- 当前结论：
  - trend 旧 checkpoint 的弱表现，不是因为“没走 local mask”；
  - 即使切到正确 local route，它的强度 spacing 仍然弱且不稳；
  - 因而 trend 的活跃 blocker 继续保留为模型/训练质量，而不是 effect eval 路由定义。

**阶段结论**：
- acceptance/local-path 的正确定义现在更新为：
  - `edit_region_soft + soft local mask + latent blending`
- 旧的：
  - `edit_time_series(..., edit_mask=mask_gt) + final_output_strength_mapping.scope=edit_region`
  - 不再应被当成 pure-editing strength effect eval 的 canonical local path。
- seasonality 当前状态：
  - `local-path verified`
  - 已能在真实 local route 下保持单调强度与背景保真。
- trend 当前状态：
  - `still weak under true local route`
  - 后续若继续推进，应回到 checkpoint / finetune 设计，而不是再改 eval 路由。

## [2026-04-22] - final-mapping-only prevalidation 完成：runtime mapping 生效，但 raw spacing 仍过小

**当前所处阶段**：
- 已按新口径完成 `final-mapping-only prevalidation`（A0–A5），基线为 `stage2_freeze_true_lr10_trend_injection`。
- acceptance wording 保持不变：`edit_time_series(..., edit_mask=mask_gt)` + `final_output_strength_mapping.scope=edit_region`。
- 预验证前的唯一阻塞不是模型本身，而是共享 wrapper config 损坏：`results/stage4_narrow_repair_20260422/_wrapper_model_config.yaml` 尾部多出裸字符串 `01`，导致首轮 A0–A5 全部 YAML parse fail；现已修复并成功重跑。

**A0–A5 预验证设置**：

| run | scope | prior | transform |
|---|---|---:|---|
| A0 | `edit_region` | default | none |
| A1 | `edit_region` | 0.16 | none |
| A2 | `edit_region` | 0.24 | none |
| A3 | `edit_region` | 0.24 | `affine(scale=1.6, offset=0.0)` |
| A4 | `edit_region` | 0.32 | `affine(scale=1.6, offset=0.0)` |
| A5 | `edit_region` | 0.32 | `affine(scale=2.0, offset=0.0)` |

**共同结果**：
- A0–A5 全部 effect eval 成功跑通。
- 结论在所有 A* 上一致：
  - `raw_final_gap_edit_region_mean = 0.0`
  - `preservation_attenuation_ratio_mean = 1.0`
  - `monotonic_hit_rate = 1.0`
  - `weak<=medium = 1.0`，`medium<=strong = 1.0`
- 这进一步确认：当前问题不是 final stage 额外压平，而是 `raw spacing` 本身就非常小；runtime-only mapping 只能在极小 gap 上做轻微拉伸。

**代表性结果（A4 / A5）**：

| metric | A4 | A5 |
|---|---:|---:|
| prior | 0.32 | 0.32 |
| transform | affine 1.6 | affine 2.0 |
| raw medium-minus-weak | 0.002729 | 0.002729 |
| raw strong-minus-medium | 0.001534 | 0.001534 |
| raw strong-minus-weak | 0.004263 | 0.004263 |
| final min adjacent gap | 0.001534 | 0.001534 |
| raw→final gap | 0.0 | 0.0 |
| attenuation ratio | 1.0 | 1.0 |
| bg drift strong-minus-weak | 0.002204 | 0.002204 |
| mapping transformed scalar | `0/0.8/1.6` | `0/1/2` |
| mapping gain mean | `0.9716/0.9921/1.0120` | `0.9716/0.9987/1.0284` |

**阶段结论**：
- `final_output_strength_mapping` 的 runtime override 确实生效，尤其 A5 已把 transformed scalar 拉到 `0/1/2`，mapping gain 也按预期增大。
- 但 benchmark-facing spacing 几乎没被实质放大：相邻 gap 仍只有 `1e-3` 量级，`strong_minus_weak` 也仅约 `4.26e-3`。
- 因而最新判断进一步收敛为：
  - 不是 `final mapping flattening`；
  - 不是 acceptance route 定义错误；
  - 当前活跃 blocker 仍是 `local-path raw spacing` 太小、太不稳。
- 这也意味着：`更高 strength_lr_scale` 不是主解的判断继续成立；`lr10` 可保留为 baseline，`lr20` 继续只作为 negative evidence。

**主线状态更新**：

| 项 | 当前结论 |
|---|---|
| acceptance path | 保持 `local_path_mask_routed` + `scope=edit_region` |
| archived line | `standard-route loss-only repair` 继续封存 |
| final-stage flattening 假设 | 排除 |
| runtime-only mapping 预验证 | 已完成，结论偏负 |
| 当前 blocker | `raw spacing collapse / instability` |
| 下一阶段重点 | 转向 training-side `spacing-first` 修复，而不是继续扫 freeze/lr |

## [2026-04-21] - trend acceptance path 固定为 local-path spacing-stability；当前 checkpoint 仍未过线

**本轮决策**：
- 正式归档旧的 standard-route loss-only repair line，不再作为 trend promotion 的主验收路径。
- 这里的 `local-path` 默认特指：`TEditWrapper.edit_time_series(..., edit_mask=mask_gt)` + `final_output_strength_mapping.scope=edit_region` 的 mask-routed local final-mapping 路由；它**不包含** `edit_region_soft` latent blending，也不等同于未传 `edit_mask` 的 global standard route。
- 后续决策主指标改为 spacing metrics，而不是把 spacing 只当作附加报告项。
- acceptance path 已明确切换到 `local-path spacing-stability`，且这个切换是正确的；但当前 checkpoint 仍未通过这条新 acceptance path。

**local-path effect audit（通过）**：
- `weak<=medium` = 1.0，`medium<=strong` = 1.0；
- `final_min_adjacent_gap_mean` 为正，`final_adjacent_gap_collapse_rate = 0.0`；
- `bg_mae_strong_minus_weak` 近零；
- 说明 local-path 下强度差异可以在 edit region 内被干净兑现，不需要靠背景漂移实现。

**local-path monotonic benchmark（未通过）**：
- `weak<=medium_pass_rate = 0.5`，`medium<=strong_pass_rate = 0.5`；
- `min_adjacent_gap_mean < 0`，`adjacent_gap_collapse_rate = 0.5`；
- `preservation_pass_rate = 0.0`；
- `long` bucket 直接塌掉，`short/medium` 也只到 0.6667。

**当前诊断**：
- 问题已经不再是 local-path 定义是否正确，也不再是 raw-to-final attenuation 或旧 standard-route global drift；
- trend 线当前唯一活跃 blocker 是：`local-path spacing benchmark` 还没有稳定过线，尤其是 adjacent gap 稳定性、duration-bucket collapse 和 benchmark 条件下的 preservation。

**归档结论**：
- conditioning 已存在；
- effect 已存在；
- raw-to-final attenuation 不是当前 trend blocker 的主因；
- archived standard route 主要把 strength 实现成 global drift；
- 因此 `standard-route loss-only repair` 线正式关闭为 comparison-only evidence，不再作为未来 trend promotion 的 acceptance path。

**新的主线 framing**：
- 主线改为 `local-path spacing-stability`：
  - 不扩 benchmark；
  - 不引入新机制；
  - 复用现有 probe / effect / monotonic eval；
  - 把 weak/medium/strong 的 adjacent spacing、adjacent monotonicity、short/medium bucket stability 作为首要判据；
  - locality 继续保底，不允许用背景漂移换 spacing。
- `spacing metrics` 现在是后续训练/筛选配置的**一等主指标**：
  - `medium_minus_weak`、`strong_minus_medium`、`strong_minus_weak`；
  - `weak<=medium` 与 `medium<=strong` 分开统计；
  - `min_adjacent_gap`、collapse/gap-balance 指标；
  - locality safety：`bg_mae_strong_minus_weak`、preservation、edit-vs-background delta；
  - raw vs final 对照；
  - short / medium / long bucket split，重点看 short / medium，并显式盯住 long collapse。
- benchmark / effect-audit / finetune diagnostics 必须统一使用同一 acceptance wording：`edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region`。
- 后续 promotion 只看是否改善 active `local-path spacing-stability` blocker；旧 standard-route 结果只保留为 comparison-only evidence。

**实现约束**：
- repair 继续留在现有 loss surface 内：`monotonic_loss_weight`、`monotonic_margin`、`gain_match_loss_weight`、现有 background/edit-region 项。
- 保持 trend-only、direction-aligned、`enable_strength_control: true`、family grouping 与 `duration_bucket` metadata。
- 在新 spacing 指标落地前，不再把 standard-route rescue 作为主目标。

**代码整理与训练前准备**：
- `test_scripts/run_tedit_trend_monotonic_eval.py` 已补齐 final mapping scalar transform override parity（`scale / offset / name`），并与 effect audit 统一输出 route / acceptance / final-mapping metadata。
- `test_scripts/evaluate_tedit_strength_effect.py` 已统一 local-path wording 与 `final_mapping_overrides` 记录，避免 benchmark / effect 口径漂移。
- `TEdit-main/train/finetuner.py` 已把 spacing 指标接入 validation 记录与 best-checkpoint selection payload，输出 `valid_epoch_summary.jsonl` 与 `best_model_selection.json`，避免只按 `avg_loss_valid` 选点。
- 已补训练前路径清单：`TEdit-main/configs/synthetic/finetune_strength_trend_family_stage4_preflight_paths.json`，固定推荐 finetune config、pretrain run=0、`model_configs.yaml` 与 `model_best.pth` 路径，减少 GPU 开跑前的配置/权重报错。

**论文叙事决策（当前版本）**：
- 论文正文只保留主损失，不展开工程性修补项。
- 主损失聚合为四类目标：
  - 局部编辑一致性（edit-region fidelity）；
  - 非编辑区保持（out-of-region preservation）；
  - 强度到编辑幅度的对齐（strength-to-amplitude calibration）；
  - 跨强度级别的单调有序约束（monotonic ordering across strength levels）。
- `family_gap_match`、`family_relative_gain`、`constant_gain_penalty`、`numeric_only`、branch / final-mapping regularizers 等项在写作上统一降级为 implementation details / training stabilizers，不作为方法主公式主体。
- 这样写的主因不是删内容，而是把方法主张收束到一条清晰主线：统一 `strength_scalar` 作为共享控制轴，在保证局部性与保真的前提下，让强度值对应可校准、且单调可控的编辑幅度。

## [2026-04-20] - T3c trend mask-local final mapping：局部性复核通过，但仍不是 promotable

**本轮目标**：
- 接 T2b 结论继续验证 trend：
  - 不再做 scalar 合约对齐；
  - 不扩大 benchmark；
  - 不重训；
  - 只验证 direction/sign 已对齐后，trend 的 final mapping 正信号能否像 non-trend family 一样局部化。
- 严格隔离机制：
  - `eval_mask_routed` 走 `TEditWrapper.edit_time_series(..., edit_mask=mask_gt)`；
  - 不使用 `edit_region_soft`；
  - 不混入 soft-boundary latent blending / state-space mixing。

**代码改动**：
- `test_scripts/run_tedit_trend_monotonic_eval.py`
  - 增加 `--generation-route {soft_region,standard}`；
  - 增加 `--eval-mask-routed`；
  - 增加 final mapping runtime override：`--final-mapping-scalar-prior-scale`、`--final-mapping-gain-order-direction`、`--final-mapping-scope`；
  - `standard` route 调用普通 `edit_time_series`，并在 mask-routed 时传入 `mask_gt`。
- `test_scripts/probe_tedit_strength_internal.py`
  - 增加 `--final-mapping-scope`，输出记录 `final_mapping_overrides`。
- `test_scripts/evaluate_tedit_strength_effect.py`
  - 增加 `--final-mapping-scope`，输出 config 记录 final mapping overrides。
- `test_scripts/test_output_branch_carrier_contract.py`
  - 新增 contract：trend standard mask-routed eval 必须调用 `edit_time_series(..., edit_mask=...)`，不能调用 `edit_region_soft`。

**验证命令**：
- 新增 contract 先红后绿：
  - `python -m unittest test_scripts.test_output_branch_carrier_contract.TestOutputBranchCarrierContract.test_trend_standard_mask_routed_eval_uses_edit_time_series_not_soft_region`
- 相关 contract 通过：
  - `python -m unittest test_scripts.test_output_branch_carrier_contract`
- 语法检查通过：
  - `python -m py_compile TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py test_scripts/probe_tedit_strength_internal.py test_scripts/evaluate_tedit_strength_effect.py test_scripts/run_tedit_trend_monotonic_eval.py test_scripts/test_output_branch_carrier_contract.py`

**T3c 运行设置**：
- checkpoint:
  - `tmp/output_final_mapping_inverse004_3epoch/0/trend_injection/ckpts/model_best.pth`
- config:
  - `tmp/output_final_mapping_inverse004_3epoch/0/trend_injection/resolved_runtime_config.json`
- benchmark:
  - `TEdit-main/datasets/discrete_strength_trend_family/test.json`
- runtime overrides:
  - `final_mapping_scalar_prior_scale = 0.04`
  - `final_mapping_gain_order_direction = increasing`
  - `final_mapping_scope = edit_region`
  - monotonic eval: `generation_route = standard`
  - monotonic eval: `eval_mask_routed = true`

**T3c 产物**：
- `results/strength_t3c_trend_direction_masklocal_20260420/probe_sample0.json`
- `results/strength_t3c_trend_direction_masklocal_20260420/trend_monotonic_eval.json`
- `results/strength_t3c_trend_direction_masklocal_20260420/effect_eval.json`
- `results/strength_t3c_trend_direction_masklocal_20260420/_wrapper_model_config.yaml`

**T3c effect 结果**：

| check | T2b-B global | T3c mask-local |
|---|---:|---:|
| final_monotonic_hit_rate | 1.0 | 1.0 |
| strong_minus_weak_edit_gain_mean | 5.8907e-03 | 6.4203e-03 |
| bg_mae_strong_minus_weak | 5.7236e-03 | -1.0324e-04 |
| abs(bg/edit) ratio | 0.9716 | 0.0161 |
| family_spearman_rho_strength_gain_mean | 1.0 | 1.0 |
| preservation_pass | true | true |

**T3c effect duration bucket 结果**：

| bucket | n | final monotonic | strong-minus-weak | bg strength leak |
|---|---:|---:|---:|---:|
| short | 3 | 1.0 | 6.3785e-03 | 9.8546e-06 |
| medium | 3 | 1.0 | 6.6816e-03 | 1.1126e-05 |
| long | 2 | 1.0 | 6.0910e-03 | -4.4441e-04 |

**T3c standard-route monotonic sweep 结果**：

| check | T2b-B global / soft-region | T3c standard mask-routed |
|---|---:|---:|
| adjacent_monotonic_pass_rate | 0.25 | 0.75 |
| off_anchor_monotonic_pass_rate | 0.875 | 0.75 |
| gain_range_mean | 2.6940e-02 | -1.8622e-03 |
| family_spearman_rho_mean | 0.5125 | 0.5 |
| preservation_pass_rate | 1.0 | 0.0 |

**T3c standard-route bucket 结果**：

| bucket | n | adjacent | gain_range_mean | preservation |
|---|---:|---:|---:|---:|
| short | 3 | 0.6667 | -2.3473e-03 | 0.0 |
| medium | 3 | 0.6667 | -3.2795e-03 | 0.0 |
| long | 2 | 1.0 | 9.9136e-04 | 0.0 |

**阶段判断**：
- T3c 成功回答了本轮核心问题：
  - trend 的 T2b-B 正信号不必依赖 global final mapping 的背景同步放大；
  - 在普通 `edit_time_series(..., edit_mask=...)` 路径下，`scope=edit_region` 保住了 effect monotonic，并把背景 strength leak 从 `5.72e-03` 压到约 `1.03e-04`。
- 因此 trend 从 `direction-aligned promising, not promotable` 更新为：
  - `local-trend-promising, not promotable`。
- 仍不能 promotable：
  - standard-route monotonic sweep 的 `gain_range_mean` 仍略负；
  - short/medium bucket 在 sweep 里仍负；
  - standard route 的绝对背景漂移很大，`preservation_pass_rate = 0.0`，说明旧 checkpoint 的 raw full-sequence generation 本身仍不干净；
  - 本轮只能证明 mask-local final mapping 能解耦 strength leakage，不能证明 trend 已是稳定可发布 local family。

**下一步建议**：
- 不在旧 inverse004 checkpoint 上继续做更多 config-only localization 修补。
- 若继续推进 trend，应进入小规模 direction-aligned + mask-local 训练/再训练，验证 raw full-sequence drift 是否能被训练修正。
- promotable 门槛应绑定：
  - effect 8/8 monotonic；
  - background strength leak 低；
  - standard-route sweep bucket 整体正向；
  - absolute preservation 不再失败。

## [2026-04-20] - T2b final-only scalar transform / direction flip：合约对齐不是修复，方向翻转有正信号但未达 promotable

**本轮目标**：
- 按 T2b stop rule 做两步最小实验：
  - A：只在 final output strength mapping 内做 final-only scalar transform，把当前 `0 / 0.5 / 1.0` 映射到旧 smoke 的 `0 / 1 / 2`，不改变 projector/runtime 原始 scalar；
  - B：仅当 A 证明 transform 生效但 held-out 仍失败时，再做一次 final mapping direction / prior 翻转对照。
- 同时补可诊断字段，确保能区分：
  - projector 仍看到原始 runtime scalar；
  - final mapping 内部实际使用 transformed scalar；
  - final mapping gain 是否按预期改变。

**代码与诊断改动**：
- `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`
  - `final_output_strength_mapping.scalar_transform` 默认关闭；
  - transform 只在 final output strength mapping 内使用；
  - diagnostics 记录 `final_output_strength_mapping_transformed_scalar_by_scalar` 和 `final_output_strength_mapping_scalar_transform`。
- `test_scripts/probe_tedit_strength_internal.py`
  - 增加 final mapping scalar transform / prior / direction CLI override；
  - probe summary 输出 transformed scalar 与 mapping gain。
- `test_scripts/evaluate_tedit_strength_effect.py`
  - 增加同样 CLI override；
  - effect summary 输出 final mapping gain / transformed scalar diagnostics。
- `test_scripts/test_output_branch_carrier_contract.py`
  - 新增 contract：final-only transform 应改变 final mapping gain，但不改原始 scalar。

**静态 / contract 验证**：
- 新增 contract 先红后绿：
  - `python -m unittest test_scripts.test_output_branch_carrier_contract.TestOutputBranchCarrierContract.test_final_output_strength_mapping_scalar_transform_is_final_only`
- 相关 contract 通过：
  - `python -m unittest test_scripts.test_output_branch_carrier_contract`
- 语法检查通过：
  - `python -m py_compile TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py test_scripts/probe_tedit_strength_internal.py test_scripts/evaluate_tedit_strength_effect.py test_scripts/test_output_branch_carrier_contract.py`

**T2b-A：final-only scalar transform 对齐旧 `0 / 1 / 2` 合约**：
- 产物：
  - `results/strength_t2b_inverse004_scalar_aligned_20260420/probe_sample0.json`
  - `results/strength_t2b_inverse004_scalar_aligned_20260420/trend_monotonic_eval.json`
  - `results/strength_t2b_inverse004_scalar_aligned_20260420/effect_eval.json`
- probe 确认 transform 生效：
  - 原始 scalar: `0.0 / 0.5 / 1.0`
  - final mapping transformed scalar: `0.0 / 1.0 / 2.0`
  - final mapping gain: `1.039996 / 0.999996 / 0.959996`
- 但 sample0 raw/final 仍非 monotonic。

**T2b-A held-out 结果**：

| check | value |
|---|---:|
| monotonic eval adjacent pass rate | 0.0 |
| monotonic eval off-anchor pass rate | 0.375 |
| monotonic eval gain_range_mean | -5.8398e-02 |
| monotonic eval family_spearman_rho_mean | -0.400 |
| monotonic eval preservation_pass_rate | 1.0 |
| effect raw_monotonic_hit_rate | 0.0 |
| effect final_monotonic_hit_rate | 0.0 |
| effect strong_minus_weak_edit_gain_mean | -1.1667e-02 |
| effect bg_mae_strong_minus_weak | -1.1328e-02 |
| effect family_spearman_rho_strength_gain_mean | -1.0 |
| effect preservation_pass | true |

**T2b-A duration bucket 结果**：

| source | bucket | n | monotonic / adjacent | strong-minus-weak / gain range | preservation |
|---|---|---:|---:|---:|---:|
| monotonic eval | short | 3 | 0.0 | -4.7883e-02 | 1.0 |
| monotonic eval | medium | 3 | 0.0 | -1.0118e-01 | 1.0 |
| monotonic eval | long | 2 | 0.0 | -9.9989e-03 | 1.0 |
| effect eval | short | 3 | 0.0 | -1.1877e-02 | 1.0 |
| effect eval | medium | 3 | 0.0 | -1.3497e-02 | 1.0 |
| effect eval | long | 2 | 0.0 | -8.6057e-03 | 1.0 |

**T2b-B：一次 direction / prior 翻转对照**：
- 产物：
  - `results/strength_t2b_inverse004_direction_flip_20260420/probe_sample0.json`
  - `results/strength_t2b_inverse004_direction_flip_20260420/trend_monotonic_eval.json`
  - `results/strength_t2b_inverse004_direction_flip_20260420/effect_eval.json`
- probe sample0 转正：
  - final mapping gain: `0.959996 / 0.979997 / 0.999996`
  - raw/final monotonic: `true / true`
- effect eval held-out 转正：
  - `raw_monotonic_hit_rate = 1.0`
  - `final_monotonic_hit_rate = 1.0`
  - `strong_minus_weak_edit_gain_mean = 5.8907e-03`
  - `family_spearman_rho_strength_gain_mean = 1.0`
- 但 trend monotonic sweep 仍不全稳：
  - adjacent pass rate 只有 `0.25`
  - long bucket `gain_range_mean = -4.8505e-03`
  - 背景 gap 与 edit gap 同步上升，`bg_mae_strong_minus_weak = 5.7236e-03`

**T2b-B held-out 结果**：

| check | value |
|---|---:|
| monotonic eval adjacent pass rate | 0.25 |
| monotonic eval off-anchor pass rate | 0.875 |
| monotonic eval gain_range_mean | 2.6940e-02 |
| monotonic eval family_spearman_rho_mean | 0.5125 |
| monotonic eval preservation_pass_rate | 1.0 |
| effect raw_monotonic_hit_rate | 1.0 |
| effect final_monotonic_hit_rate | 1.0 |
| effect strong_minus_weak_edit_gain_mean | 5.8907e-03 |
| effect bg_mae_strong_minus_weak | 5.7236e-03 |
| effect family_spearman_rho_strength_gain_mean | 1.0 |
| effect preservation_pass | true |

**T2b-B duration bucket 结果**：

| source | bucket | n | monotonic / adjacent | strong-minus-weak / gain range | preservation |
|---|---|---:|---:|---:|---:|
| monotonic eval | short | 3 | 0.3333 | 2.5892e-02 | 1.0 |
| monotonic eval | medium | 3 | 0.3333 | 4.9182e-02 | 1.0 |
| monotonic eval | long | 2 | 0.0 | -4.8505e-03 | 1.0 |
| effect eval | short | 3 | 1.0 | 5.9350e-03 | 1.0 |
| effect eval | medium | 3 | 1.0 | 6.6050e-03 | 1.0 |
| effect eval | long | 2 | 1.0 | 4.7526e-03 | 1.0 |

**阶段判断**：
- T2b-A 证明：把当前 scalar final-only 变换回旧 `0 / 1 / 2` 合约不是正确修复；transform 路径正确生效，但 held-out trend 仍整体负向。
- T2b-B 证明：当前 held-out effect 口径下，final mapping 的 direction/sign 翻转可以恢复 8/8 effect monotonic。
- 但 B 不能直接升级为 promotable trend：
  - monotonic sweep adjacent 只有 `0.25`，long bucket 仍负；
  - 背景 gap 基本随 edit gap 同步上升，说明这还是 global final mapping 信号，不是合格的局部 trend 控制；
  - `probe_gate_pass` 仍为 false，说明旧内部路径本身没有重新获得 strength separability。
- 当前 trend verdict 从 `regression-only / no-regression-but-not-ready` 上调为：
  - `direction-aligned promising, not promotable`。

**下一步建议**：
- 不继续扩大 T2b benchmark，不把 B 当作最终 trend 修复。
- 如果继续 trend，应先做 region-local / mask-routed trend final mapping 或重新训练方向一致的 trend local family，再跑同一套 `probe -> monotonic eval -> effect eval`。
- 若保留 B，只能作为 sign/direction diagnostic baseline，不应并入默认配置。

## [2026-04-20] - trend inverse final mapping 旧成功线复核：smoke 成功依赖旧 strength scalar 合约，T2 held-out 方向反转

**本轮目标**：
- 回答“之前 trend 表现好的时候”到底是哪一层证据。
- 把 2026-04-19 的 `tmp/output_final_mapping_inverse004_3epoch` inverse final mapping 成功线，用同一套 T2 contract 复跑：
  - 同一个 healthy held-out trend benchmark；
  - 同样的 `probe -> monotonic eval -> effect eval`；
  - 同样的 duration bucket 汇总。
- 不扩大 benchmark，不改 Trainer / loss，不把 trend 提升为 promotable local family。

**复核对象**：
- checkpoint:
  - `tmp/output_final_mapping_inverse004_3epoch/0/trend_injection/ckpts/model_best.pth`
- training/runtime config:
  - `tmp/output_final_mapping_inverse004_3epoch/0/trend_injection/resolved_runtime_config.json`
- final mapping 关键设置：
  - `final_output_strength_mapping.enabled = true`
  - `gain_order_direction = decreasing`
  - `scalar_prior_scale = -0.04`
  - `learned_max_delta = 0.04`
  - `scalar_center = 1.0`

**历史正证据边界**：
- 2026-04-19 旧 smoke 结果来自：
  - `tmp/final_mapping_inverse004_eval_both.json`
- 当时 3-sample smoke 显示：
  - `final_monotonic_hit_rate = 1.0`
  - `strong_minus_weak_edit_gain_mean = 5.1547e-02`
  - `bg_mae_strong_minus_weak = 4.4010e-02`
  - `preservation_pass = true`
- 旧 probe 的 final mapping scalar/gain 关系近似为：
  - `0.0000 -> 1.039996`
  - `1.0000 -> 0.999996`
  - `2.0000 -> 0.959996`
- 因此旧结论应限定为：inverse final mapping 机制能在当时的 3-sample smoke 合约下显化 strength 差异。

**T2 复跑产物**：
- `results/strength_t2_inverse004_20260420/probe_sample0.json`
- `results/strength_t2_inverse004_20260420/trend_monotonic_eval.json`
- `results/strength_t2_inverse004_20260420/effect_eval.json`

**T2 关键发现**：
- 当前 held-out `discrete_strength_trend_family` 的 strength scalar 合约是：
  - `weak = 0.0`
  - `medium = 0.5`
  - `strong = 1.0`
- 同一个 inverse mapping 在当前合约下给出的 gain 顺序变成：
  - `0.0000 -> 1.039996`
  - `0.5000 -> 1.019996`
  - `1.0000 -> 0.999996`
- 这等价于 weak gain 最大、strong gain 最小，和当前 benchmark 的 `weak < medium < strong` 目标方向相反。

**T2 结果**：

| check | value |
|---|---:|
| probe gate `diff_0_2_linf` | 0.0 |
| monotonic eval adjacent pass rate | 0.0 |
| monotonic eval off-anchor pass rate | 0.625 |
| monotonic eval gain_range_mean | -2.9919e-02 |
| monotonic eval family_spearman_rho_mean | -0.325 |
| monotonic eval preservation_pass_rate | 1.0 |
| effect raw_monotonic_hit_rate | 0.0 |
| effect final_monotonic_hit_rate | 0.0 |
| effect strong_minus_weak_edit_gain_mean | -5.7758e-03 |
| effect bg_mae_strong_minus_weak | -5.6047e-03 |
| effect family_spearman_rho_strength_gain_mean | -1.0 |
| effect preservation_pass | true |

**duration bucket 结果**:

| source | bucket | n | monotonic | strong-minus-weak / gain range | preservation |
|---|---|---:|---:|---:|---:|
| monotonic eval | short | 3 | 0.0 | -2.2847e-02 | 1.0 |
| monotonic eval | medium | 3 | 0.0 | -5.0694e-02 | 1.0 |
| monotonic eval | long | 2 | 0.0 | -9.3655e-03 | 1.0 |
| effect eval | short | 3 | 0.0 | -5.9420e-03 | 1.0 |
| effect eval | medium | 3 | 0.0 | -6.8916e-03 | 1.0 |
| effect eval | long | 2 | 0.0 | -3.8528e-03 | 1.0 |

**阶段判断**：
- 旧 trend 正证据不应被抹掉，但必须分层：
  - 机制层：inverse final mapping 在旧 3-sample smoke 合约下确实能显化 strength 差异。
  - T2 held-out 层：同一个 inverse004 checkpoint/config 在当前 healthy held-out regression 上 raw/final 都不可分，且方向为负。
  - 合约层：失败不只是“小样本不泛化”，还暴露出旧 smoke 使用的 strength scalar 合约与当前 `0 / 0.5 / 1.0` family 合约不一致。
- 因此当前 trend 结论应更新为：
  - `trend` 仍停留在 `regression-only / no-regression-but-not-ready`；
  - 不能把 inverse004 旧 smoke 直接升级为 promotable local trend 证据；
  - 下一步如果继续 trend，不应扩大当前 benchmark，而应先做 scalar-contract-aligned 的最小配置实验。

**下一步建议**：
- 不再扩大当前 semantic-split checkpoint 或 inverse004 checkpoint 的 T2 benchmark。
- 若继续验证 trend 机制，优先做 config-only 合约对齐：
  - 明确当前 `0 / 0.5 / 1.0` scalar 合约下 final mapping 的期望方向；
  - 或把 final mapping 内部 scalar 规范到旧 `0 / 1 / 2` 合约后再比较；
  - 然后再跑同一套 T2 `probe -> monotonic eval -> effect eval`。

## [2026-04-20] - T2 trend regression rerun：配置 blocker 解除，但旧 checkpoint 行为不达标

**本轮目标**：
- 接手 Claude 中断点，修复 `finetune_strength_trend_family_semantic_split` checkpoint 在 T2 probe/eval 中的 architecture mismatch。
- 按当前主线计划执行 T2：先 probe，再 trend monotonic eval，再 effect eval。
- 不改 Trainer 主逻辑，不改监督公式，不把 trend 默认切到 mask-local。

**关键 blocker 与修复**：
- 目标 checkpoint：
  - `TEdit-main/save/synthetic/finetune_strength_trend_family_semantic_split/0/trend_injection/ckpts/model_best.pth`
- 旧失败不再是 CUDA 或 JSON/YAML resolver 本身，而是 checkpoint/config architecture mismatch：
  - checkpoint: `diff_model.strength_projector.mlp.0.weight = [64, 96]`
  - 当前运行时新结构默认会构造 `emb_dim * 2 + text_dim = 128` 或在错误推断后退到 64。
- 修复方式：
  - `StrengthProjector` 增加 `include_strength_scalar`，默认 `true`，不改变新训练行为。
  - `TEditWrapper.load_model()` 从 checkpoint 权重反推 legacy projector 输入结构。
  - 对该 checkpoint 自动推断为：
    - `emb_dim = 32`
    - `text_dim = 64`
    - `use_text_context = true`
    - `use_task_id = false`
    - `include_strength_scalar = false`
  - 构造后 `mlp.0.weight` 为 `[64, 96]`，和 checkpoint 对齐。
- 同时补齐：
  - dedicated family loader 透传 `duration_bucket`
  - trend monotonic/effect eval 的 duration bucket 汇总不再显示为 `unknown`
  - effect eval bucket preservation 汇总不再读取不存在的 row 字段

**T2 产物**：
- `results/strength_t2_regression_20260420/probe_sample0_after_legacy_projector_fix.json`
- `results/strength_t2_regression_20260420/trend_monotonic_eval_after_legacy_projector_fix.json`
- `results/strength_t2_regression_20260420/effect_eval_after_legacy_projector_fix.json`

**T2 结果**：

| check | value |
|---|---:|
| probe gate `diff_0_2_linf` | 0.0 |
| monotonic eval adjacent pass rate | 0.0 |
| monotonic eval gain_range_mean | -2.5144e-03 |
| monotonic eval preservation_pass_rate | 1.0 |
| effect raw_monotonic_hit_rate | 0.0 |
| effect final_monotonic_hit_rate | 0.0 |
| effect strong_minus_weak_edit_gain_mean | 3.3580e-05 |
| effect bg_mae_strong_minus_weak | 3.2900e-05 |
| effect preservation_pass | true |

**duration bucket 结果**：

| source | bucket | n | monotonic | strong-minus-weak / gain range | preservation |
|---|---|---:|---:|---:|---:|
| monotonic eval | short | 3 | 0.0 | 1.4576e-03 | 1.0 |
| monotonic eval | medium | 3 | 0.0 | 2.8090e-04 | 1.0 |
| monotonic eval | long | 2 | 0.0 | -1.2665e-02 | 1.0 |
| effect eval | short | 3 | 0.0 | 3.3696e-05 | 1.0 |
| effect eval | medium | 3 | 0.0 | 3.7511e-05 | 1.0 |
| effect eval | long | 2 | 0.0 | 2.7508e-05 | 1.0 |

**阶段判断**：
- T2 已从“跑不起来”推进为“能跑，但旧 checkpoint 行为不达标”。
- 当前旧 trend checkpoint 有 projector/modulation 信号，但输出层强度排序基本不可见：
  - raw/final 都不可分；
  - strong-minus-weak 量级只有 `3e-05`；
  - preservation 稳定通过，说明问题不是 final preservation flattening，而是 strength conditioning/训练动态没有转成有效输出排序。
- 按 Claude plan 的 stop rule：当前 healthy benchmark 上 raw 都不可分，因此不进入 T3 larger benchmark。
- trend verdict 应写为 `no-regression-but-not-ready`：
  - 没有证据说明 mask-local 会破坏 trend；
  - 但也没有证据支持把该旧 checkpoint 的局部 trend 纳入 promotable local family。

## [2026-04-20] - trend regression / noise calibration 计划收口到主线资产

**本轮目标**：
- 把 `mask-local` 从 hard_zero 专项推进为局部 family 可选机制后的结论，正式收口到主线资产。
- 明确 trend 下一步先做 regression check，不默认切到 local 主线。
- 明确 noise 下一步从“修 final mapping 泄漏”转向“做局部波动语义/统计校准”。
- 不改 Trainer 主逻辑，不扩展高层工具链，只补 benchmark/eval/reporting 支撑。

**代码改动**：
- `test_scripts/run_tedit_trend_monotonic_eval.py`
  - summary 新增 `duration_bucket_summary`，按 `short / medium / long` 聚合：
    - `adjacent_monotonic_pass_rate`
    - `off_anchor_monotonic_pass_rate`
    - `gain_range_mean`
    - `family_spearman_rho_mean`
    - `bg_mae_mean`
    - `preservation_pass_rate`
- `test_scripts/evaluate_tedit_strength_effect.py`
  - summary 新增 `duration_bucket_summary`，按 `short / medium / long` 聚合：
    - `monotonic_hit_rate`
    - `raw_monotonic_hit_rate`
    - `final_monotonic_hit_rate`
    - `strong_minus_weak_edit_gain_mean`
    - `local_std/local_energy/local_roughness strong-minus-weak`
    - `bg_mae_strong_minus_weak`
    - `preservation_pass_rate`
- `test_scripts/build_tedit_strength_discrete_benchmark.py`
  - `noise_injection` family 现默认先采样两类 pilot subtype：
    - `uniform_variance`
    - `local_burst`
  - family/sample payload 新增 `noise_subtype`，便于 health check 与 calibration baseline 按 subtype 切片。
- `test_scripts/check_tedit_strength_discrete_benchmark.py`
  - summary 新增 `noise_subtype_counts`，markdown health 报告同步输出 subtype 计数。
- `TEdit-main/data/discrete_strength_family.py`
  - family invariant 校验把 `noise_subtype` 纳入 metadata signature，保证同一 family 三个 strength 不发生 subtype 泄漏。
- `docs/pure_editing_how_much_protocol.md`
  - 追加主线 staging update：trend verdict/noise verdict 标签、duration bucket 报告口径、noise 2-subtype pilot 决策边界。

**当前结论锁定**：
- trend：先做 regression-only 检查；若 raw 可分但 final 塌缩，结论应记为 `no-regression-but-not-ready`。
- noise：默认先做 2-subtype pilot；若 benchmark 不健康或两类结构指标无稳定分离，不扩到第三类。
- `monotonic_envelope` 只在 pilot 已成立后再恢复为扩展阶段。

**验证状态**：
- 当前仅完成代码与文档收口。
- 按用户要求，涉及 GPU / CUDA 的训练与评估暂不执行，等待后续排队窗口。

## [2026-04-20] - mask-local 扩展到 step_change / multiplier / noise_injection 小样本对照

**本轮目标**：
- 回答“mask-local final strength mapping 是否只对 hard_zero 有效，还是对其他局部编辑 family 也有普遍收益”。
- 不改监督公式、不改 Trainer 主逻辑、不动 trend 既有成功主线。
- 对 `step_change / multiplier / noise_injection` 做同口径小样本训练与验证。
- 使用 `tedit` 环境、`CUDA_VISIBLE_DEVICES=0`、dedicated family leaf、`edit_mask` routed eval/probe。

**单变量可比性核对**：
- 本轮 mask-local 候选与已有 global baseline 保持一致：
  - `pretrained_dir = TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth`
  - `epochs = 10`
  - `freeze_backbone_for_strength = false`
  - `strength_lr_scale = 5.0`
  - `scalar_prior_scale = 0.12`
  - `learned_max_delta = 0.08`
  - `data.folder = TEdit-main/datasets/discrete_strength_family/<family>`
- 唯一核心变量：
  - global baseline: `final_output_strength_mapping.scope = global / None`
  - 本轮候选: `final_output_strength_mapping.scope = edit_region`

**代码改动**：
- `test_scripts/evaluate_tedit_strength_effect.py`
  - 新增 noise/volatility 辅助指标：
    - `local_std_delta`
    - `local_energy_delta`
    - `local_roughness`
  - per-sample 增加对应 by-strength 序列和 strong-minus-weak gap。
  - summary 增加：
    - `local_std_strong_minus_weak_mean`
    - `local_energy_strong_minus_weak_mean`
    - `local_roughness_strong_minus_weak_mean`
- 目的：`noise_injection` 不再只按平均 edit gain 判定，也能看局部波动性、能量和粗糙度是否随 strength 增强。

**训练与评估产物**：
- `results/strength_masklocal_step_change_s012_d008/`
  - `quick_eval_maskrouted.json`
  - `quick_probe_worst_sample0_maskrouted.json`
- `results/strength_masklocal_multiplier_s012_d008/`
  - `quick_eval_maskrouted.json`
  - `quick_probe_worst_sample0_maskrouted.json`
- `results/strength_masklocal_noise_injection_s012_d008/`
  - `quick_eval_maskrouted.json`
  - `quick_probe_worst_sample2_maskrouted.json`

**结果对照**：

| family | scope | raw mono | final mono | edit strong-weak | bg strong-weak | abs edit/bg | preservation |
|---|---|---:|---:|---:|---:|---:|---:|
| step_change | global baseline | 1.0 | 1.0 | 1.0928e-02 | 9.1185e-03 | 1.20 | true |
| step_change | edit_region | 1.0 | 1.0 | 1.7341e-02 | -3.3965e-04 | 51.06 | true |
| multiplier | global baseline | 1.0 | 1.0 | 1.0884e-02 | 1.0133e-02 | 1.07 | true |
| multiplier | edit_region | 1.0 | 1.0 | 1.7479e-02 | -4.0682e-04 | 42.96 | true |
| noise_injection | global baseline | 1.0 | 1.0 | 9.8289e-03 | 1.1738e-02 | 0.84 | true |
| noise_injection | edit_region | 1.0 | 1.0 | 1.4211e-02 | 6.7941e-04 | 20.92 | true |

**noise_injection 辅助指标**：

| metric | strong-minus-weak mean |
|---|---:|
| `local_std` | 6.7552e-04 |
| `local_energy` | 1.3990e-01 |
| `local_roughness` | 5.8001e-04 |

**worst-sample probe 策略**：
- 不再写死 `sample_idx=1`。
- 每个 family 先跑 3-sample quick eval，再选 `strong_minus_weak_edit_gain` 最小的样本做 probe：
  - `step_change`: worst `sample_idx=0`, gap `1.4496e-02`
  - `multiplier`: worst `sample_idx=0`, gap `1.4217e-02`
  - `noise_injection`: worst `sample_idx=2`, gap `7.8125e-03`

**阶段判断**：
- 这轮支持把 mask-local final mapping 从 hard_zero 专项提升为“局部编辑 family 的标准可选机制”。
- 三个 family 都保持 3/3 raw/final monotonic，edit gap 没有塌，background strong-minus-weak 被显著压低。
- `step_change` 和 `multiplier` 的收益非常干净：edit gap 上升且 background gap 接近 0。
- `noise_injection` 也有正信号，但问题更特殊：
  - 平均 edit gap 增加；
  - background gap 显著压低；
  - local energy / roughness 随 strength 增强；
  - 但绝对输出幅度仍远大于 target edit gain，说明它的主要瓶颈不只是 final mapping 泄漏，还包括 noise 语义/幅度校准。

**当前问题与可能原因**：
- `mask-local` 解决的是 final mapping 层的全局 gain 泄漏，不等于解决所有背景污染来源。
- 当前三类的 `bg_mae_mean` 绝对值仍大，说明编辑生成本身仍有基础漂移；本轮观察的是 strong-minus-weak 背景差是否随强度泄漏。
- `noise_injection` 的 target 是局部波动性而不是单纯均值幅度，后续需要继续把 local std / energy / roughness 纳入主判定。
- 如果某个 family 后续在 `scope=edit_region` 下仍出现 high bg strong-minus-weak，就应怀疑污染发生在更早的 residual / modulation / carrier 层，而不是继续加 global scalar。

**下一步建议**：
- 保持代码默认 `scope=global`，只在局部编辑 family 新实验中显式打开 `scope=edit_region`。
- 后续可按 `step_change -> multiplier -> noise_injection -> trend regression` 做更大样本验证。
- trend 暂时作为 regression check，不应无条件切换默认行为；局部 trend 适合 local，全序列 trend 不一定适合。

## [2026-04-20] - hard_zero mask-local final mapping：局部性显著改善，背景 gap 被压低

**本轮目标**：
- 接上上一轮结论：`scalar_prior_scale 0.08 -> 0.12` 能放大 hard_zero strength gap，但会把 background strong-minus-weak 也同步推高。
- 做最小改动实验：不改 Trainer 主逻辑、不改监督公式，只把 `final_output_strength_mapping` 的 gain 作用域从 global 改成可选 `edit_region`。
- 继续使用 `tedit` 环境和 `cuda:0`。

**代码改动**：
- `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`
  - `final_output_strength_mapping` 新增 `scope`，默认 `global`。
  - 新增 `edit_region` scope：`effective_gain = 1 + (gain - 1) * mask`。
  - mask 在 final mapping 内统一转 float、广播到 output 形状；无 mask 时自动保持旧 global 行为。
- `TEdit-main/models/conditional_generator.py`
  - 从 batch 中读取 `mask_gt`，规范为 `[B,K,L]`，只作为 `final_strength_mask` 传给 diffusion model。
  - `_noise_estimation_loss` / `_edit` / `predict_noise` 增加可选 mask 传递，不改变 loss 公式。
- `tool/tedit_wrapper.py`
  - `edit_time_series(..., edit_mask=...)` 新增可选参数，传入后写入 batch 的 `mask_gt`。
- `test_scripts/evaluate_tedit_strength_effect.py`
  - benchmark quick eval 现在把记录里的 `edit_mask` 传给 wrapper。
- `test_scripts/probe_tedit_strength_internal.py`
  - probe 同样传递 `edit_mask`，避免 mask-local 配置在评估侧退回 global。
- `test_scripts/test_output_branch_carrier_contract.py`
  - 新增 contract：`scope=edit_region` 只 gate `(gain - 1)`，背景位置保持 gain=1。

**重要中间问题**：
- 第一次 masklocal eval 得到 `edit_gap=0.01175 / bg_gap=0.01177`，看起来没有局部性收益。
- 复查后发现不是机制失败，而是 quick eval/probe 通过 `TEditWrapper.edit_time_series()` 推理时没有传 `edit_mask`，导致 `scope=edit_region` 无 mask 可用，按兼容逻辑退回 global。
- 修复 wrapper/eval/probe 的 mask 路由后，重新评估才是有效结果。
- 训练启动也有两个参数坑：
  - `run_finetune.py` 不接受 `--device`，本轮改用 `CUDA_VISIBLE_DEVICES=0`。
  - 脚本默认 `pretrained_dir=save/synthetic/pretrain` 不存在，本轮显式使用 `save/synthetic/pretrain_multi_weaver`。
  - 未显式传 `--epochs` 时 CLI 默认把配置里的 `epochs: 10` 覆盖成 `50`，所以本轮 masklocal 训练实际是 50 epoch；后续小样本变体需显式加 `--epochs 10`。

**实验结果**：

| run | raw mono | final mono | edit strong-weak | bg strong-weak | edit/bg ratio | preservation |
|---|---:|---:|---:|---:|---:|---:|
| strict m020 | 1.0 | 1.0 | 9.7496e-03 | 9.7671e-03 | 1.00 | true |
| strict s012_d008 global | 1.0 | 1.0 | 1.5578e-02 | 1.5606e-02 | 1.00 | true |
| masklocal without eval mask route | 1.0 | 1.0 | 1.1748e-02 | 1.1768e-02 | 1.00 | true |
| masklocal with eval mask route | 1.0 | 1.0 | 1.6285e-02 | 8.1547e-04 | 19.97 | true |

**关键产物**：
- 训练目录：
  - `results/strength_hard_zero_strict_leaf_s012_d008_masklocal/`
- 有效 quick eval：
  - `results/strength_hard_zero_strict_leaf_s012_d008_masklocal/quick_eval_maskrouted.json`
- 有效 probe：
  - `results/strength_hard_zero_strict_leaf_s012_d008_masklocal/quick_probe_sample1_maskrouted.json`
- 临时配置：
  - `tmp/strength_family_configs_strict/finetune_strength_hard_zero_strict_s012_d008_masklocal.yaml`

**probe 观察**：
- sample1 有效 probe 中：
  - `final_output_strength_mapping_mean = 0.9962`
  - `final_output_strength_mapping_by_scalar = {0.0000: 0.9884, 0.5000: 0.9974, 1.0000: 1.0028}`
- 因为 effective gain 已按 edit mask 稀释到全序列均值接近 1，但 edit region 内仍保留强度差，这和预期一致。

**阶段判断**：
- 这轮证明了当前最小 mask-aware final mapping 是有效的：
  - 不牺牲 3/3 monotonic。
  - edit gap 相对 global s012_d008 没有塌，反而略高。
  - background strong-minus-weak 从 `1.5606e-02` 压到 `8.1547e-04`，约降低 95%。
- 当前 hard_zero 主线可以更新为：
  - strict healthy leaf 上排序已经稳定；
  - global scalar gain 的背景泄漏可以通过 mask-local final mapping 显著抑制；
  - 剩余主要问题仍是幅度校准绝对值偏大/target gap mismatch，而不是排序或局部性。

## [2026-04-20] - hard_zero strict leaf 修复 raw inversion，排序小样本过线但幅度仍偏弱

**本轮目标**：
- 从 Claude 的 hard_zero 中断点继续，不回到 trend，不扩展全 family sweep。
- 继续保持：
  - 不改监督公式
  - 不改 Trainer 主逻辑
  - 不改 eval 指标定义
  - 使用 `tedit` 环境和 `cuda:0`
- 重点处理 `hard_zero sample_idx=1` 的 raw-stage inversion。

**关键发现**：
- 读取最新 `PROGRESS.md` 后确认，Claude 的主线已经从“non-trend final gap 太小”收敛到：
  - `hard_zero` 单 family
  - 固定难例 `sample_idx=1`
  - raw stage 已反向，不是 final attenuation
- 复查 artifact 后发现 `results/strength_nontrend_sweep_hard_zero_fbfalse` 实际已经训练和评估完成，虽然进度文字还写着“等待训练”。
- `fbfalse` 结果仍失败：

| metric | value |
|---|---:|
| raw_monotonic_hit_rate | 0.6667 |
| final_monotonic_hit_rate | 0.6667 |
| strong_minus_weak_edit_gain_mean | 8.7988e-04 |
| preservation_pass | true |
| sample_idx=1 strong_minus_weak_edit_gain | -2.5237e-02 |

**hard_zero 难例诊断**：
- 新增诊断产物：
  - `tmp/hard_zero_sample1_raw_inversion_diagnostic.json`
- 对比发现，旧 dedicated leaf 的 `test sample_idx=1` 不是单纯模型学不会，而是目标 profile 本身很弱且语义可疑：

| sample | source_abs | target gain seq | target floor seq | 模型结果 |
|---|---:|---:|---:|---|
| old test sample 0 | 4.3905 | `[2.8335, 2.8335, 2.8335]` | `[1.8232, 1.8232, 1.8232]` | 正向 |
| old test sample 1 | 1.1895 | `[0.0931, 0.1207, 0.1231]` | `[1.2651, 1.2983, 1.3011]` | 反向 |
| old test sample 2 | 12.0145 | `[2.8166, 3.7577, 4.2635]` | `[10.3335, 8.3927, 7.7701]` | 正向 |

- `sample_idx=1` 的 target edit gain 虽然按旧 health checker 是非降序，但：
  - target gap 极小，尤其 medium->strong 只有 `0.0024`
  - target floor distance 随 strength 反而略增
  - source region 已接近 floor
- 因此继续调 `strength_lr_scale / gain_match / monotonic / freeze_backbone` 很难稳定翻正这个样本。

**代码改动**：
- 只改 dataset builder：
  - `test_scripts/build_tedit_strength_discrete_benchmark.py`
- 新增 hard_zero family 采样健康约束：
  - `source_abs_mean >= 2.0`
  - weak->medium、medium->strong 的 target edit gain gap 均至少 `0.05`
  - target floor distance 随 strength 不增
  - hard_zero 最多 resample `256` 次
- 未改：
  - `TEdit-main/train/finetuner.py`
  - `TEdit-main/models/*`
  - `TEdit-main/data/discrete_strength_family.py`
  - eval 指标定义

**strict dedicated leaf**：
- 新建 scratch 数据，不覆盖现有 `TEdit-main/datasets`：
  - `tmp/hard_zero_strict_collection/hard_zero`
- 构建命令使用：
  - `conda run -n tedit python test_scripts/build_tedit_strength_trend_family_dataset.py ... --collection-root tmp/hard_zero_strict_collection --injection-types hard_zero --selector hard_zero`
- dedicated leaf 路径合同确认：
  - `meta.selector = hard_zero`
  - leaf basename = `hard_zero`
  - `run_finetune` runtime 指向 `tmp/hard_zero_strict_collection/hard_zero`
- strict profile 检查：

| split | strict_ok | min target gain gap |
|---|---:|---:|
| train | true | 0.1530 |
| valid | true | 0.1451 |
| test | true | 0.1749 |

**训练配置**：
- 输出目录：
  - `results/strength_hard_zero_strict_leaf_m020`
- 初始化：
  - `TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth`
- 关键参数：
  - `ctrl_attrs: [[hard_zero]]`
  - `freeze_backbone_for_strength: true`
  - `strength_lr_scale: 15.0`
  - `gain_match_loss_weight: 6.0`
  - `monotonic_loss_weight: 2.0`
  - `final_output_strength_mapping.gain_order_direction: increasing`
  - `scalar_prior_scale: 0.08`
  - `learned_max_delta: 0.08`
  - `min_gain/max_gain: 0.85 / 1.15`
- 训练结果：
  - epoch loss `21.0663 -> 20.5317`
  - best valid loss `43.6992 -> 42.8975`
  - `strength_diagnostics.jsonl` 正常落盘

**quick eval（test, max-samples=3, edit-steps=10）**：

| metric | old `fbfalse` | strict leaf `m020` |
|---|---:|---:|
| raw_monotonic_hit_rate | 0.6667 | 1.0000 |
| final_monotonic_hit_rate | 0.6667 | 1.0000 |
| strong_minus_weak_edit_gain_mean | 8.7988e-04 | 9.7496e-03 |
| family_spearman_rho_strength_gain_mean | 0.3333 | 1.0000 |
| gain_calibration_mae_mean | 3.6814 | 3.1546 |
| bg_mae_strong_minus_weak | N/A | 9.7671e-03 |
| preservation_pass | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 |
| strength_visible_in_final | true | true |

**strict leaf per-sample eval**：

| sample_idx | final_monotonic_hit | strong_minus_weak | pred_gap_seq | target_gap_seq |
|---|---:|---:|---:|---:|
| 0 | true | +1.0200e-02 | `[0.00539, 0.00481]` | `[0.39186, 0.32684]` |
| 1 | true | +9.2316e-03 | `[0.00503, 0.00420]` | `[0.17487, 0.50998]` |
| 2 | true | +9.8176e-03 | `[0.00523, 0.00459]` | `[0.53980, 0.36703]` |

**quick probe（test sample_idx=1）**：
- `raw_edit_region_mean_abs_delta`：
  - weak `3.781674`
  - medium `3.786707`
  - strong `3.790906`
- `final_edit_region_mean_abs_delta` 与 raw 完全一致。
- `blend_gap_edit_region_mean_abs = 0.0`
- `final_output_strength_mapping_by_scalar`：
  - weak `0.920723`
  - medium `0.960724`
  - strong `1.000712`
- 结论：旧固定反向样本对应的问题在 strict leaf 上已消失，raw/final 排序均正确。

**补充候选：strict leaf `s012_d008`**：
- 继续按单因素方式推进 gap 放大。
- 与 strict leaf `m020` 相比，只改：
  - `final_output_strength_mapping.scalar_prior_scale: 0.08 -> 0.12`
- 保持不变：
  - strict dedicated leaf
  - `freeze_backbone_for_strength: true`
  - `strength_lr_scale: 15.0`
  - `gain_match_loss_weight: 6.0`
  - `monotonic_loss_weight: 2.0`
  - `learned_max_delta: 0.08`
  - `min_gain/max_gain: 0.85 / 1.15`
- 输出目录：
  - `results/strength_hard_zero_strict_leaf_s012_d008`
- 训练结果：
  - epoch loss `21.1545 -> 20.6171`
  - best valid loss `43.7983 -> 42.9982`

**strict leaf `s012_d008` quick eval（test, max-samples=3, edit-steps=10）**：

| metric | strict `m020` | strict `s012_d008` |
|---|---:|---:|
| raw_monotonic_hit_rate | 1.0000 | 1.0000 |
| final_monotonic_hit_rate | 1.0000 | 1.0000 |
| strong_minus_weak_edit_gain_mean | 9.7496e-03 | 1.5578e-02 |
| bg_mae_strong_minus_weak | 9.7671e-03 | 1.5606e-02 |
| gain_calibration_mae_mean | 3.1546 | 3.1519 |
| preservation_pass | true | true |

**strict `s012_d008` per-sample eval**：

| sample_idx | strong_minus_weak | pred_gap_seq |
|---|---:|---:|
| 0 | +1.6159e-02 | `[0.00873, 0.00743]` |
| 1 | +1.4946e-02 | `[0.00834, 0.00661]` |
| 2 | +1.5631e-02 | `[0.00849, 0.00714]` |

**补充候选结论**：
- `scalar_prior_scale=0.12` 可以把 final gap 从约 `0.00975` 放大到 `0.01558`，约 1.6x。
- 排序仍保持 `3/3`，`preservation_pass` 仍为 true。
- 但背景强弱差几乎同比增长：
  - edit gap `+0.00583`
  - bg gap `+0.00584`
- 这说明继续加 global scalar gain 会同步推高背景漂移；后续更应该转向 region-local output control 或 mask-aware gain，而不是无脑继续放大 global mapping。

**当前结论**：
- Claude 当前 hard_zero 主线的关键断点已进一步明确：
  - 旧 dedicated leaf 的 `sample_idx=1` 失败主要由 hard_zero target profile 退化触发，而不是 output path 或训练压力不足。
  - 加入 hard_zero 采样健康约束后，小样本排序从 `2/3` 提升到 `3/3`。
- 但不能宣称 hard_zero 幅度校准已完成：
  - 目标 gap 是 `0.17~0.54` 量级
  - 模型 pred gap 仍只有 `0.004~0.005`
  - `gain_calibration_mae_mean` 仍有 `3.1546`
- 因此当前状态是：
  - **hard_zero raw inversion 已修通**
  - **final path 仍健康**
  - **排序小样本过线**
  - **幅度校准仍偏弱，后续应继续围绕 gap 放大与背景漂移控制推进**

**验证命令**：
- `conda run -n tedit python -m py_compile test_scripts/build_tedit_strength_discrete_benchmark.py test_scripts/build_tedit_strength_trend_family_dataset.py test_scripts/evaluate_tedit_strength_effect.py test_scripts/probe_tedit_strength_internal.py TEdit-main/run_finetune.py TEdit-main/data/discrete_strength_family.py`
- `conda run -n tedit python test_scripts/check_tedit_strength_discrete_benchmark.py --benchmark tmp/hard_zero_strict_collection/hard_zero/test.json --output tmp/hard_zero_strict_collection/hard_zero/test_health.json`
- `OMP_NUM_THREADS=1 conda run -n tedit python TEdit-main/run_finetune.py ... --data_folder tmp/hard_zero_strict_collection --save_folder results/strength_hard_zero_strict_leaf_m020 ...`
- `OMP_NUM_THREADS=1 conda run -n tedit python test_scripts/evaluate_tedit_strength_effect.py ... --dataset-folder tmp/hard_zero_strict_collection --selector hard_zero --output results/strength_hard_zero_strict_leaf_m020/quick_eval.json`
- `OMP_NUM_THREADS=1 conda run -n tedit python test_scripts/probe_tedit_strength_internal.py ... --sample-idx 1 --output results/strength_hard_zero_strict_leaf_m020/quick_probe_sample1.json`

**下一步建议**：
- 不回退旧 hard_zero leaf 的 sample_idx=1 作为模型失败证据；它现在应被记录为 data-profile 退化样本。
- 后续若继续 hard_zero，应在 strict leaf 上做下一轮：
  - 先保留当前 strict data
  - `s012_d008` 已证明 global scalar gain 能放大 gap，但背景差同步变大
  - 下一步不宜继续无脑扩大 global mapping，应转向 region-local / mask-aware output control，并继续检查 `bg_mae_strong_minus_weak`

## [2026-04-20] - hard_zero sample_idx=1 raw-stage inversion repair loop

**本轮目标**：
- 只修 `discrete_strength_family/hard_zero` 的固定难例 `sample_idx=1`。
- 不改 dataset contract、不改 family routing、不改 eval 指标定义、不改 model heads。
- 问题已收敛为 **raw stage weak/medium/strong inversion**，不再是 final attenuation。
- 严格按 `训练 -> sample_idx=1 quick probe -> hard_zero quick eval -> 记录结果` 执行。

**当前基线**：`results/strength_nontrend_sweep_hard_zero_s012_d008`

**基线配置**：
- `ctrl_attrs: [[hard_zero]]`
- `freeze_backbone_for_strength: true`
- `strength_lr_scale: 15.0`
- `instruction_text_dropout_prob: 0.0`
- `monotonic_loss_weight: 1.0`
- 其余 gain-matching / family-relative / output-path 保持当前 mainline

**基线快照**：

| check | value |
|---|---:|
| raw_monotonic_hit_rate | 0.6667 |
| final_monotonic_hit_rate | 0.6667 |
| strong_minus_weak_edit_gain_mean | 4.9108e-04 |
| preservation_pass | true |
| sample_idx=1 strong_minus_weak_edit_gain | -3.2149e-02 |
| sample_idx=1 raw_monotonic_hit | false |
| sample_idx=1 final_monotonic_hit | false |

**当前候选**：`results/strength_nontrend_sweep_hard_zero_m020`

**本轮配置变更**：
- 只改 `monotonic_loss_weight: 1.0 -> 2.0`
- `strength_lr_scale` 保持 `15.0`
- `freeze_backbone_for_strength` 保持 `true`
- `instruction_text_dropout_prob` 保持 `0.0`
- dedicated leaf 继续固定为 `discrete_strength_family/hard_zero`

**训练链路确认**：
- 目标仍是 `TEdit-main/datasets/discrete_strength_family/hard_zero`
- 本轮只做 config-only 候选 A，不触及 dataset / eval / model head

**候选 A 结果**：

| check | baseline `s012_d008` | candidate A `m020` |
|---|---:|---:|
| raw_monotonic_hit_rate | 0.6667 | 0.6667 |
| final_monotonic_hit_rate | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 4.9108e-04 | 9.7517e-04 |
| preservation_pass | true | true |
| sample_idx=1 strong_minus_weak_edit_gain | -3.2149e-02 | -2.5575e-02 |
| sample_idx=1 raw_monotonic_hit | false | false |
| sample_idx=1 final_monotonic_hit | false | false |

**sample_idx=1 quick probe（train split）**：
- raw/final 仍完全一致，`blend_gap_edit_region_mean_abs = 0.0`。
- 目标样本依旧是 **raw-stage inversion**，且方向未翻正：
  - weak `48.867878`
  - medium `48.872879`
  - strong `48.878307`
- `final_output_strength_mapping_by_scalar` 仍是单调递增：`0.920068 -> 0.960068 -> 1.000068`。
- 结论：Candidate A 没有暴露新的 final attenuation；失败点依旧在 raw-stage sample-level ordering 稳定性。

**quick eval（test, max-samples=3）**：
- aggregate 没退化，但也没修通 target sample。
- `sample_idx=1` 仍为唯一反向样本：
  - `pred_gain_by_strength = [0.981045, 0.968046, 0.955470]`
  - `pred_gap_seq = [-0.012999, -0.012576]`
  - `family_spearman_rho_strength_gain = -1.0`
- 其余样本仍保持单调，preservation 继续通过。

**本轮结论**：
- 单独把 `monotonic_loss_weight: 1.0 -> 2.0` 不能修复 `sample_idx=1` 的 raw inversion。
- aggregate `strong_minus_weak_edit_gain_mean` 有所增大，但 target sample 仍反向，因此 Candidate A 判定为 **未通过 stop condition**。
- 按既定顺序，下一候选转入 **B：只调整 `strength_lr_scale`**。

**执行状态**：
- 候选 A 已完成并记录。
- 候选 B 已完成并记录为失败；准备进入候选 C。

---

**当前候选**：`results/strength_nontrend_sweep_hard_zero_fbfalse`

**本轮配置变更**：
- 只改 `freeze_backbone_for_strength: true -> false`
- `monotonic_loss_weight` 保持 `2.0`
- `strength_lr_scale` 保持 `10.0`
- `instruction_text_dropout_prob` 保持 `0.0`
- dedicated leaf 继续固定为 `discrete_strength_family/hard_zero`

**候选 B 结果**：

| check | baseline `s012_d008` | candidate A `m020` | candidate B `lr010` |
|---|---:|---:|---:|
| raw_monotonic_hit_rate | 0.6667 | 0.6667 | 0.6667 |
| final_monotonic_hit_rate | 0.6667 | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 4.9108e-04 | 9.7517e-04 | 9.7483e-04 |
| preservation_pass | true | true | true |
| sample_idx=1 strong_minus_weak_edit_gain | -3.2149e-02 | -2.5575e-02 | -2.5576e-02 |
| sample_idx=1 raw_monotonic_hit | false | false | false |
| sample_idx=1 final_monotonic_hit | false | false | false |
| sample_idx=1 pred_gap_seq | N/A | `[-0.012999, -0.012576]` | `[-0.012999, -0.012576]` |

**quick eval（test, max-samples=3）**：
- `sample_idx=1` 仍为唯一反向样本：
  - `pred_gain_by_strength = [0.981052, 0.968052, 0.955476]`
  - `pred_gap_seq = [-0.012999, -0.012576]`
  - `family_spearman_rho_strength_gain = -1.0`
- aggregate `raw_monotonic_hit_rate` / `final_monotonic_hit_rate` 与 Candidate A 完全一致。
- `strong_minus_weak_edit_gain_mean` 与 Candidate A 仅有噪声级差异。
- `preservation_pass` 继续为 `true`。

**本轮结论**：
- 单独把 `strength_lr_scale: 15.0 -> 10.0` 没有带来可观变化。
- target sample 的 raw inversion 没有被翻正，因此 Candidate B 判定为 **未通过 stop condition**。
- 按既定顺序，下一候选转入 **C：只调整 `freeze_backbone_for_strength`**。

**执行状态**：
- 候选 C 配置已切换，等待训练。

## [2026-04-20] - hard_zero final flattening 已修通，但整体 monotonic 仍未过线

**本轮目标**：
- 只修 `discrete_strength_family/hard_zero`。
- 不改 dataset contract、不改 family routing、不改 eval 指标定义。
- 优先复用现有 output path：`final_output_strength_mapping` + `output_branch_carrier`。
- 严格按 `训练 -> quick probe -> quick eval -> 记录结果` 执行。

**本轮候选**：`results/strength_nontrend_sweep_hard_zero_s010_d010`

**本轮配置**：
- `ctrl_attrs: [[hard_zero]]`
- `freeze_backbone_for_strength: true`
- `strength_lr_scale: 10.0`
- `output_branch_carrier.skip_scale: 0.2`
- `output_branch_carrier.min_residual_to_skip_ratio: 0.1`
- `final_output_strength_mapping.gain_order_direction: increasing`
- `final_output_strength_mapping.scalar_prior_scale: 0.08`
- `final_output_strength_mapping.learned_max_delta: 0.08`
- `final_output_strength_mapping.min_gain/max_gain: 0.85 / 1.15`
- `final_output_strength_mapping.gain_order_margin: 0.02`
- `final_output_strength_mapping.gain_order_weight: 0.3`

**训练链路确认**：
- `resolved_runtime_config.json` 明确指向：
  - `TEdit-main/datasets/discrete_strength_family/hard_zero`
- runtime strength config 已正确吃到本轮 mapping / carrier 配置。
- `py_compile` 通过：
  - `TEdit-main/run_finetune.py`
  - `TEdit-main/train/finetuner.py`
  - `TEdit-main/models/conditional_generator.py`
  - `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`
  - `tool/tedit_wrapper.py`
  - `test_scripts/probe_tedit_strength_internal.py`
  - `test_scripts/evaluate_tedit_strength_effect.py`

**quick probe（train sample 0）**：
- `raw_edit_region_mean_abs_delta`：
  - weak `12.4479227`
  - medium `12.4481297`
  - strong `12.4483347`
- `final_edit_region_mean_abs_delta` 与 raw 完全一致。
- `blend_gap_edit_region_mean_abs = 0.0`
- `final_output_strength_mapping_by_scalar`：
  - weak `0.9200454`
  - medium `0.9600454`
  - strong `1.0000454`
- `final_edit_monotonic = true`
- 结论：**本轮已消除 raw->final flattening；final 不再把 strength 差异压没。**

**quick eval（test, max-samples=3, edit-steps=1）**：

| metric | value |
|---|---:|
| monotonic_hit_rate | 0.6667 |
| raw_monotonic_hit_rate | 0.6667 |
| final_monotonic_hit_rate | 0.6667 |
| attenuation_suspected_rate | 0.0000 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 |
| raw_strong_minus_weak_mean | 1.2779e-04 |
| final_strong_minus_weak_mean | 1.2779e-04 |
| family_spearman_rho_strength_gain_mean | 0.3333 |
| gain_calibration_mae_mean | 3.6463 |
| bg_mae_strong_minus_weak | 1.3669e-04 |
| preservation_pass | true |
| strength_visible_in_final | true |

**per-sample quick eval**：

| sample_idx | monotonic | strong_minus_weak_edit_gain | spearman |
|---|---:|---:|---:|
| 0 | true | +3.6001e-04 | 1.0 |
| 1 | false | -3.7718e-04 | -1.0 |
| 2 | true | +4.0054e-04 | 1.0 |

**本轮结论**：
- `hard_zero` 当前已基本排除 **raw->final flattening** 作为主瓶颈。
- 最新 probe 与 quick eval 表明，final output 已能保留 raw strength ordering，且不再出现额外衰减：
  - raw `weak < medium < strong`
  - final `weak < medium < strong`
  - `blend_gap = 0`
  - `attenuation_suspected_rate = 0.0`
  - 单样本上 final 与 raw 一致
- 相比旧基线（`~5.77e-05`），`strong_minus_weak_edit_gain_mean` 提升到 `1.28e-04`，约 2.2x，说明最终输出 gap 确实变大。
- 但整体 `final_monotonic_hit_rate` 仍停在 `0.6667`，没有越过旧基线；根因已转为 **sample-level 排序稳定性不足**，而不是 output path 末端压平。
- 当前剩余问题表现为：个别样本仍会发生反向，导致 family-level hit rate 卡在 `2/3`。

**下一步口径**：
- 不再优先修改 output path。
- 锁住本轮 output-path 设置，转入 `hard_zero` 单 family 的训练压力补强与难例诊断。
- 后续 sweep 继续保持一次只改一个因素：
  - 先保留 `freeze_backbone_for_strength: true`
  - 优先把 `strength_lr_scale: 10.0 -> 15.0`
  - 如仍不足，再上调 `gain_match_loss_weight`
  - 最后再考虑上调 `monotonic_loss_weight`
- 后续重点不再是“能否传到 final”，而是“为什么个别样本仍会翻序，尤其是当前反向样本的 source / region / target gain / baseline floor 是否存在 hard_zero 难例特征”。
- 不回退到 collection root，不扩展 family sweep，不改 eval 定义。

**附：候选 `results/strength_nontrend_sweep_hard_zero_s015_d010`（只改 `strength_lr_scale: 15.0`）**：
- quick probe 与 `s010_d010` 几乎一致：
  - raw / final 仍保持 `weak < medium < strong`
  - `blend_gap = 0`
  - `final_output_strength_mapping_by_scalar` 仅从 `0.920045 / 0.960045 / 1.000045` 微增到 `0.920068 / 0.960068 / 1.000068`
  - projector norm 略增，但未转化为更大的 sample-level final gap
- quick eval 与 `s010_d010` 实质相同：

| metric | s010_d010 | s015_d010 |
|---|---:|---:|
| final_monotonic_hit_rate | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 | 1.2779e-04 |
| gain_calibration_mae_mean | 3.6463 | 3.6463 |
| preservation_pass | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 |

- per-sample 也未发生实质变化：反向样本仍是 `sample_idx=1`，`strong_minus_weak_edit_gain = -3.7718e-04`。
- 结论：**单独把 `strength_lr_scale` 从 10 提到 15 没有带来可见收益，可判定为无效候选。**
- 因此下一候选应继续锁住 output-path 与 `strength_lr_scale=15.0`，转而轻调 `gain_match_loss_weight`。

**附：候选 `results/strength_nontrend_sweep_hard_zero_s015_g006`（固定 `strength_lr_scale: 15.0`，只改 `gain_match_loss_weight: 4.0 -> 6.0`）**：
- quick probe 与 `s015_d010` 实质一致：
  - raw / final 仍保持 `weak < medium < strong`
  - `blend_gap = 0`
  - `final_output_strength_mapping_by_scalar` 维持 `0.920068 / 0.960068 / 1.000068`
  - `output_branch_carrier` 诊断基本不变，说明 output-path 已锁定且未退化
- quick eval 仍无可见改善：

| metric | s010_d010 | s015_d010 | s015_g006 |
|---|---:|---:|---:|
| final_monotonic_hit_rate | 0.6667 | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 | 1.2779e-04 | 1.2779e-04 |
| gain_calibration_mae_mean | 3.6463 | 3.6463 | 3.6463 |
| preservation_pass | true | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 | 0.0000 |

- per-sample 仍未翻转：

**训练链路确认**：
- `resolved_runtime_config.json` 明确指向：
  - `TEdit-main/datasets/discrete_strength_family/hard_zero`
- runtime strength config 已正确吃到本轮 mapping / carrier 配置。
- `py_compile` 通过：
  - `TEdit-main/run_finetune.py`
  - `TEdit-main/train/finetuner.py`
  - `TEdit-main/models/conditional_generator.py`
  - `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`
  - `tool/tedit_wrapper.py`
  - `test_scripts/probe_tedit_strength_internal.py`
  - `test_scripts/evaluate_tedit_strength_effect.py`

**quick probe（train sample 0）**：
- `raw_edit_region_mean_abs_delta`：
  - weak `12.4479227`
  - medium `12.4481297`
  - strong `12.4483347`
- `final_edit_region_mean_abs_delta` 与 raw 完全一致。
- `blend_gap_edit_region_mean_abs = 0.0`
- `final_output_strength_mapping_by_scalar`：
  - weak `0.9200454`
  - medium `0.9600454`
  - strong `1.0000454`
- `final_edit_monotonic = true`
- 结论：**本轮已消除 raw->final flattening；final 不再把 strength 差异压没。**

**quick eval（test, max-samples=3, edit-steps=1）**：

| metric | value |
|---|---:|
| monotonic_hit_rate | 0.6667 |
| raw_monotonic_hit_rate | 0.6667 |
| final_monotonic_hit_rate | 0.6667 |
| attenuation_suspected_rate | 0.0000 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 |
| raw_strong_minus_weak_mean | 1.2779e-04 |
| final_strong_minus_weak_mean | 1.2779e-04 |
| family_spearman_rho_strength_gain_mean | 0.3333 |
| gain_calibration_mae_mean | 3.6463 |
| bg_mae_strong_minus_weak | 1.3669e-04 |
| preservation_pass | true |
| strength_visible_in_final | true |

**per-sample quick eval**：

| sample_idx | monotonic | strong_minus_weak_edit_gain | spearman |
|---|---:|---:|---:|
| 0 | true | +3.6001e-04 | 1.0 |
| 1 | false | -3.7718e-04 | -1.0 |
| 2 | true | +4.0054e-04 | 1.0 |

**本轮结论**：
- `hard_zero` 当前已基本排除 **raw->final flattening** 作为主瓶颈。
- 最新 probe 与 quick eval 表明，final output 已能保留 raw strength ordering，且不再出现额外衰减：
  - raw `weak < medium < strong`
  - final `weak < medium < strong`
  - `blend_gap = 0`
  - `attenuation_suspected_rate = 0.0`
  - 单样本上 final 与 raw 一致
- 相比旧基线（`~5.77e-05`），`strong_minus_weak_edit_gain_mean` 提升到 `1.28e-04`，约 2.2x，说明最终输出 gap 确实变大。
- 但整体 `final_monotonic_hit_rate` 仍停在 `0.6667`，没有越过旧基线；根因已转为 **sample-level 排序稳定性不足**，而不是 output path 末端压平。
- 当前剩余问题表现为：个别样本仍会发生反向，导致 family-level hit rate 卡在 `2/3`。

**下一步口径**：
- 不再优先修改 output path。
- 锁住本轮 output-path 设置，转入 `hard_zero` 单 family 的训练压力补强与难例诊断。
- 后续 sweep 继续保持一次只改一个因素：
  - 先保留 `freeze_backbone_for_strength: true`
  - 优先把 `strength_lr_scale: 10.0 -> 15.0`
  - 如仍不足，再上调 `gain_match_loss_weight`
  - 最后再考虑上调 `monotonic_loss_weight`
- 后续重点不再是“能否传到 final”，而是“为什么个别样本仍会翻序，尤其是当前反向样本的 source / region / target gain / baseline floor 是否存在 hard_zero 难例特征”。
- 不回退到 collection root，不扩展 family sweep，不改 eval 定义。

**附：候选 `results/strength_nontrend_sweep_hard_zero_s015_d010`（只改 `strength_lr_scale: 15.0`）**：
- quick probe 与 `s010_d010` 几乎一致：
  - raw / final 仍保持 `weak < medium < strong`
  - `blend_gap = 0`
  - `final_output_strength_mapping_by_scalar` 仅从 `0.920045 / 0.960045 / 1.000045` 微增到 `0.920068 / 0.960068 / 1.000068`
  - projector norm 略增，但未转化为更大的 sample-level final gap
- quick eval 与 `s010_d010` 实质相同：

| metric | s010_d010 | s015_d010 |
|---|---:|---:|
| final_monotonic_hit_rate | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 | 1.2779e-04 |
| gain_calibration_mae_mean | 3.6463 | 3.6463 |
| preservation_pass | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 |

- per-sample 也未发生实质变化：反向样本仍是 `sample_idx=1`，`strong_minus_weak_edit_gain = -3.7718e-04`。
- 结论：**单独把 `strength_lr_scale` 从 10 提到 15 没有带来可见收益，可判定为无效候选。**
- 因此下一候选应继续锁住 output-path 与 `strength_lr_scale=15.0`，转而轻调 `gain_match_loss_weight`。

**附：候选 `results/strength_nontrend_sweep_hard_zero_s015_g006`（固定 `strength_lr_scale: 15.0`，只改 `gain_match_loss_weight: 4.0 -> 6.0`）**：
- quick probe 与 `s015_d010` 实质一致：
  - raw / final 仍保持 `weak < medium < strong`
  - `blend_gap = 0`
  - `final_output_strength_mapping_by_scalar` 维持 `0.920068 / 0.960068 / 1.000068`
  - `output_branch_carrier` 诊断基本不变，说明 output-path 已锁定且未退化
- quick eval 仍无可见改善：

| metric | s010_d010 | s015_d010 | s015_g006 |
|---|---:|---:|---:|
| final_monotonic_hit_rate | 0.6667 | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 | 1.2779e-04 | 1.2779e-04 |
| gain_calibration_mae_mean | 3.6463 | 3.6463 | 3.6463 |
| preservation_pass | true | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 | 0.0000 |

- per-sample 仍未翻转：

| sample_idx | s015_d010 monotonic | s015_d010 strong_minus_weak | s015_g006 monotonic | s015_g006 strong_minus_weak |
|---|---:|---:|---:|---:|
| 0 | true | +3.6001e-04 | true | +3.6001e-04 |
| 1 | false | -3.7718e-04 | false | -3.7718e-04 |
| 2 | true | +4.0054e-04 | true | +4.0054e-04 |

- 结论：**把 `gain_match_loss_weight` 从 4 提到 6 也没有带来任何可见收益。**
- 当前 hard_zero 的主瓶颈更像是固定难例（`sample_idx=1`）上的排序稳定性/校准失配，而不是 output-path、`strength_lr_scale`、或轻量 `gain_match_loss_weight` 压力不足。
- 下一步若继续串行单候选，应保持 output-path、`strength_lr_scale=15.0`、`gain_match_loss_weight=6.0` 固定，转向 `monotonic_loss_weight` 的最小步长上调。

**附：候选 `results/strength_nontrend_sweep_hard_zero_s015_g006_m010`（固定 `strength_lr_scale: 15.0`、`gain_match_loss_weight: 6.0`，只改 `monotonic_loss_weight: 0.5 -> 1.0`）**：
- 训练成功，runtime 仍确认只消费 dedicated leaf：
  - `TEdit-main/datasets/discrete_strength_family/hard_zero`
- quick probe 已补跑成功（train `sample_idx=0`）：
  - `raw_edit_region_mean_abs_delta`: `12.6186609 -> 12.6240816 -> 12.6291418`
  - `final_edit_region_mean_abs_delta` 与 raw 完全一致
  - `blend_gap_edit_region_mean_abs = 0`
  - `final_output_strength_mapping_by_scalar = 0.9200682 / 0.9600681 / 1.0000682`
  - `final_output_gain_gate_mean_by_scalar = 1.0 / 1.0 / 1.0`
  - `final_edit_monotonic = true`
  - 说明 probe 侧继续确认：**output-path 没有额外衰减，strength 差异可稳定传到 final。**
- quick eval 与 `s015_g006` 仍完全一致：
  - `monotonic_hit_rate = raw_monotonic_hit_rate = final_monotonic_hit_rate = 0.6667`
  - `attenuation_suspected_rate = 0.0`
  - `strong_minus_weak_edit_gain_mean = raw_strong_minus_weak_mean = final_strong_minus_weak_mean = 1.2779e-04`
  - `preservation_pass = true`
  - 失败样本仍固定为 `sample_idx=1`
  - 因此这一步不是 probe 缺失导致的判断空洞，而是 **probe 成功后依然证实主瓶颈不在 final attenuation，而在固定难例的排序稳定性/校准失配**

| metric | s015_g006 | s015_g006_m010 |
|---|---:|---:|
| final_monotonic_hit_rate | 0.6667 | 0.6667 |
| strong_minus_weak_edit_gain_mean | 1.2779e-04 | 1.2779e-04 |
| gain_calibration_mae_mean | 3.6463 | 3.6463 |
| preservation_pass | true | true |
| attenuation_suspected_rate | 0.0000 | 0.0000 |

- per-sample 仍未翻转：

| sample_idx | s015_g006 monotonic | s015_g006 strong_minus_weak | s015_g006_m010 monotonic | s015_g006_m010 strong_minus_weak |
|---|---:|---:|---:|---:|
| 0 | true | +3.6001e-04 | true | +3.6001e-04 |
| 1 | false | -3.7718e-04 | false | -3.7718e-04 |
| 2 | true | +4.0054e-04 | true | +4.0054e-04 |

- eval 继续显示：
  - `raw_monotonic_hit_rate = final_monotonic_hit_rate = 0.6667`
  - `raw_to_final_monotonic_drop = 0.0`
  - `blend_gap = 0`
  - 说明 output-path 仍然健康，`monotonic_loss_weight: 1.0` 也没有改变 sample-level 排序
- 结论：**把 `monotonic_loss_weight` 从 0.5 提到 1.0 也没有带来任何可见收益。**
- 当前可确认：`strength_lr_scale`、`gain_match_loss_weight`、`monotonic_loss_weight` 这三步轻量压力 sweep 都未翻转固定难例 `sample_idx=1`；主瓶颈仍是该样本的排序稳定性/校准失配，而不是 final attenuation。

## [2026-04-20] - non-trend dedicated leaf 训练主线修通，当前瓶颈转为 final strength gap 过小

## [2026-04-20] - non-trend dedicated leaf 训练主线修通，当前瓶颈转为 final strength gap 过小

**本轮目标**：
- 继续 Claude 断点后的 pure-editing strength family 训练。
- `trend_injection` 先冻结不做。
- 重点把其他编辑类按 trend family 的成功经验修通：
  - dedicated family leaf dataset
  - one run only reads one family leaf
  - 不改监督公式
  - 不改 Trainer 主逻辑
  - 使用 `tedit` 环境和 `cuda:0`

**涉及 family**：
- `hard_zero`
- `step_change`
- `multiplier`
- `noise_injection`

**新增 / 使用的关键产物**：
- dedicated leaf datasets：
  - `TEdit-main/datasets/discrete_strength_family/hard_zero`
  - `TEdit-main/datasets/discrete_strength_family/step_change`
  - `TEdit-main/datasets/discrete_strength_family/multiplier`
  - `TEdit-main/datasets/discrete_strength_family/noise_injection`
- 单 family 临时配置：
  - `tmp/strength_family_configs/finetune_strength_<family>.yaml`
  - `tmp/strength_family_configs_nontrend_increasing/finetune_strength_<family>.yaml`
- smoke / smalltrain 结果：
  - `tmp/strength_dedicated_leaf_smoke/<family>`
  - `results/strength_dedicated_leaf_smalltrain/<family>`
  - `results/strength_dedicated_leaf_smalltrain_increasing/<family>`
  - `results/strength_dedicated_leaf_smalltrain_increasing_10ep/<family>`

**重要纠偏**：
- Claude 遗留训练进程曾写入 `results/strength_first_gpu_validation`，但 runtime 中 `resolved_dataset_folder` 仍指向 collection root：
  - `TEdit-main/datasets/discrete_strength_family`
- 这不符合当前主线要求；当前主线要求必须指向：
  - `TEdit-main/datasets/discrete_strength_family/<selector>`
- 已停止该偏离主线的旧进程，保留其输出作为历史产物，但不把它作为 dedicated leaf 训练结论。

**数据与 loader 验证**：
- 五个 leaf 均可被 `DiscreteStrengthFamilyDataset` 直接读取。
- 本轮关注的四个 non-trend leaf 均确认：
  - train: 8 families
  - valid: 3 families
  - test: 3 families
  - 首样本 `tool_name` 与 selector 一致
- runtime config 均确认 `resolved_dataset_folder` 指向 `<root>/<family>`，不再指向 collection root。

**第一阶段：完全复用 trend decreasing final mapping，链路通但效果方向不对 / 不可见**
- 配置来源：`tmp/strength_family_configs/finetune_strength_<family>.yaml`
- 保留 trend family 的 final output strength mapping：
  - `gain_order_direction=decreasing`
  - `scalar_prior_scale=-0.04`
- 输出目录：
  - `results/strength_dedicated_leaf_smalltrain/<family>`
- 训练设置：
  - `conda run -n tedit`
  - `cuda:0`
  - 4 epochs
  - `--strength-diagnostics 1`
  - `--strength-diagnostics-interval 1`
- 训练链路结果：
  - 四类均正常产出 `ckpts/model_best.pth`
  - 四类均正常产出 `resolved_runtime_config.json`
  - 四类均正常产出 `strength_diagnostics.jsonl`
  - diagnostics 每类 8 行
- `evaluate_tedit_strength_effect.py --max-samples 3 --edit-steps 1` 结果：

| family | strength_visible_in_final | monotonic_hit_rate | spearman | strong_minus_weak |
|---|---:|---:|---:|---:|
| hard_zero | false | 0.3333 | -0.3333 | -5.76e-05 |
| step_change | false | 0.0000 | -1.0000 | -1.46e-04 |
| multiplier | false | 0.0000 | -1.0000 | -1.51e-04 |
| noise_injection | false | 0.0000 | -1.0000 | -1.29e-04 |

**第一阶段结论**：
- 训练 / loader / checkpoint / eval 链路均正确。
- 但 non-trend family 不能直接照搬 trend 的 decreasing final mapping。
- 对 non-trend 来说，数据目标中的 target gain 随 strength 增大；decreasing mapping 会把最终输出方向压反或压平。

**第二阶段：只调整 non-trend final mapping 方向为 increasing，小样本正确性成立**
- 配置来源：`tmp/strength_family_configs_nontrend_increasing/finetune_strength_<family>.yaml`
- 只改 final output strength mapping 方向和先验符号：
  - `gain_order_direction=increasing`
  - `scalar_prior_scale=+0.04`
- 明确未改：
  - 监督公式
  - Trainer 主逻辑
  - 模型代码
  - dedicated leaf 数据 contract
- 输出目录：
  - `results/strength_dedicated_leaf_smalltrain_increasing/<family>`
- 训练设置：
  - 4 epochs
  - `tedit` 环境
  - `cuda:0`
- `evaluate_tedit_strength_effect.py --max-samples 3 --edit-steps 1` 结果：

| family | strength_visible_in_final | monotonic_hit_rate | spearman | strong_minus_weak | gain_mae |
|---|---:|---:|---:|---:|---:|
| hard_zero | true | 0.6667 | 0.3333 | +5.77e-05 | 3.6461 |
| step_change | true | 1.0000 | 1.0000 | +1.46e-04 | 1.2771 |
| multiplier | true | 1.0000 | 1.0000 | +1.52e-04 | 0.5632 |
| noise_injection | true | 1.0000 | 1.0000 | +1.28e-04 | 5.4351 |

**第二阶段结论**：
- non-trend 的 increasing final mapping 能让 final strength direction 变为可见且方向正确。
- `step_change` / `multiplier` / `noise_injection` 在 3-sample eval 上达到 monotonic 1.0。
- `hard_zero` 仍只有 0.6667，说明 hard_zero 还没有完全稳定。
- 虽然 `strength_visible_in_final=true`，但 `strong_minus_weak` 仍是 `1e-4` 量级，幅度很小。

**第三阶段：10 epoch 延长训练，方向保持但幅度未显著放大**
- 输出目录：
  - `results/strength_dedicated_leaf_smalltrain_increasing_10ep/<family>`
- 训练设置：
  - 10 epochs
  - `tedit` 环境
  - `cuda:0`
  - 从原始 pretrained 初始化，不从 4 epoch checkpoint 续训，避免混入续训变量。
- diagnostics：
  - 四类均 20 行，符合 train/valid diagnostics 输出预期。
- 训练 loss 走势：

| family | loss trend |
|---|---|
| hard_zero | 63.70 -> 53.50，仍明显下降 |
| step_change | 17.38 -> 16.84，缓慢下降 |
| multiplier | 17.10 -> 16.99，best loss 到 14.33 |
| noise_injection | 16.41 -> 16.19，best loss 到 14.93 |

- 10 epoch eval 结果：

| family | strength_visible_in_final | monotonic_hit_rate | spearman | strong_minus_weak | gain_mae |
|---|---:|---:|---:|---:|---:|
| hard_zero | true | 0.6667 | 0.3333 | +5.77e-05 | 3.6461 |
| step_change | true | 1.0000 | 1.0000 | +1.46e-04 | 1.2771 |
| multiplier | true | 1.0000 | 1.0000 | +1.48e-04 | 0.5631 |
| noise_injection | true | 1.0000 | 1.0000 | +1.30e-04 | 5.4351 |

**第三阶段结论**：
- 单纯把 4 epoch 拉到 10 epoch 没有显著扩大 final strength gap。
- `hard_zero` loss 仍在下降，但 final monotonic 没从 0.6667 提升。
- 其他三类排序稳定，但幅度仍非常小。

**当前核心问题**：
- 当前不再是路径、loader、selector、checkpoint 或 eval 脚本问题。
- 当前瓶颈是：
  - **non-trend final strength direction 已修正，但 final strength gap 过小。**
- 表现为：
  - `strength_visible_in_final=true`
  - `monotonic_hit_rate` 对三类成立
  - 但 `strong_minus_weak` 只有 `1e-4` 量级
  - `gain_calibration_mae` 对 `hard_zero`、`noise_injection` 仍很大
- `hard_zero` 额外问题：
  - 小样本 monotonic 只有 0.6667
  - 训练 loss 仍明显下降，可能还没学稳，也可能目标本身和 bounded final mapping 的表达能力不匹配。

**可能原因分析**：
1. **trend 的 bounded final mapping 机制可复用，但参数太保守**
   - 当前 non-trend 使用 `scalar_prior_scale=0.04`、`learned_max_delta=0.04`。
   - 这个配置足以翻正方向，但对 final time-series edit 的影响只有 `1e-4` 量级。
   - 对 non-trend 这种 target gain 范围更大的 family，当前 gain 上限可能太弱。

2. **global final-output gain 不是 family-specific 局部幅度控制**
   - final mapping 作用在最终 diffusion 输出层，属于全局 bounded gain。
   - 它能改变 final amplitude ordering，但不一定能匹配每类编辑的真实幅度目标。
   - 特别是 `hard_zero` 和 `noise_injection` 的语义不是简单 level/trend 幅度缩放，global gain 可能只能制造微弱排序，不能完成校准。

3. **模型内部已有 projector 信号，但输出端没有足够放大**
   - eval 中 `projector_signal_present=true`。
   - 说明 strength condition 没完全丢。
   - 但 final edit gain 差距极小，瓶颈更像在 `strength_cond -> final output mapping -> final series delta` 的后半段。

4. **10 epoch 不改变结论，说明不是简单训练步数不足**
   - 10 epoch 对 `multiplier` / `noise_injection` best loss 有改善，但 final gap 基本不变。
   - 这说明继续盲目加 epoch 收益有限。
   - `hard_zero` 可以例外继续观察，因为 loss 仍显著下降，但也不能只靠加 epoch。

5. **`hard_zero` 可能需要单独 family 处理**
   - hard zero 的目标是接近 flatline / shutdown，目标 gain 和 preservation 的冲突更强。
   - 当前同一套 bounded scalar gain 对它只得到 0.6667 monotonic。
   - 它可能需要更强 mapping、hard-zero-specific local target weighting，或更直接的 region-local output control；但这些都应作为下一步实验，不应直接改 Trainer 主线。

**下一步建议**：
- 保留 non-trend increasing 方向作为正确主线。
- 不建议继续盲目加 epoch。
- 建议做小型配置扫描，仍不改 Trainer / 监督公式：
  1. 扫 `scalar_prior_scale`：`0.04 -> 0.08 / 0.12`
  2. 扫 `learned_max_delta`：`0.04 -> 0.08 / 0.12`
  3. 保持 `min_gain/max_gain` 有界，先不放太大，避免背景漂移。
  4. 每次只用 `max-samples=3` 或 `max-samples=5` 做 quick eval。
  5. 优先看：
     - `strong_minus_weak`
     - `monotonic_hit_rate`
     - `preservation_pass`
     - `bg_mae_strong_minus_weak`
- 对 `hard_zero` 单独跟踪：
  - 如果更强 mapping 仍不能把 monotonic 提到 1.0，说明 hard_zero 可能需要单独的 local / region-aware 输出控制，而不是继续复用统一 bounded gain。

**已验证命令口径**：
- 训练均使用：
  - `conda run -n tedit python TEdit-main/run_finetune.py ... --device cuda:0`（device 来自 model/eval config）
  - `OMP_NUM_THREADS=1`
- eval 均使用：
  - `conda run -n tedit python test_scripts/evaluate_tedit_strength_effect.py --device cuda:0 --selector <family> --max-samples 3 --edit-steps 1`
- 编译检查通过：
  - `conda run -n tedit python -m py_compile TEdit-main/run_finetune.py TEdit-main/data/discrete_strength_family.py test_scripts/build_tedit_strength_trend_family_dataset.py test_scripts/evaluate_tedit_strength_effect.py test_scripts/probe_tedit_strength_internal.py`

**状态**：
- non-trend dedicated leaf 训练链路：已修通。
- non-trend direction：increasing 已验证为正确方向。
- non-trend 小样本排序：三类成立，hard_zero 未完全稳定。
- 当前未解决问题：final strength gap 太小，尚不能认为幅度控制质量达标。

## [2026-04-19] - multi-family semantic alignment / data-contract fix 冻结完成

**修改文件**：`TEdit-main/data/discrete_strength_family.py`、`test_scripts/build_tedit_strength_discrete_benchmark.py`、`test_scripts/build_tedit_strength_trend_family_dataset.py`、`tool/tedit_wrapper.py`、`TEdit-main/configs/synthetic/model_multi_weaver.yaml`、`TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`

**结论口径**：
- 已完成多 family 语义对齐支线的最小修复。
- 统一 strength scalar 维持不变。
- `trend_injection` 保留 proxy attrs。
- `step_change` 默认 `neutral`。
- `multiplier` / `hard_zero` / `noise_injection` 改为 neutral attrs。
- `task_id` 契约已补齐但主线默认关闭。
- 当前改动已通过 `tedit` 环境编译、`compileall`、多 family smoke 与 batch contract 验证，不触碰 held-out final-mapping 主线。

**验收结果**：
| 项目 | 结果 |
|---|---|
| 语义权威源 | family-level benchmark records 成立 |
| loader 行为 | 不再回退成错误 trend 语义猜测 |
| strength axis | 维持统一、连续、可比较的 `0.0 / 0.5 / 1.0` |
| trend family | `trend_injection -> proxy` |
| step family | `step_change -> neutral`（默认） |
| non-trend families | `multiplier / hard_zero / noise_injection -> neutral` |
| task side-channel | 契约补齐，默认 `use_task_id=false` |
| 主线隔离 | 不污染 held-out final-mapping 主线 |

**补充检查**：
| 项目 | 值 |
|---|---|
| smoke 中最大 `task_id` | `6` |
| `num_tasks` 配置 | `8` |
| 结论 | 满足 `num_tasks >= max(task_id)+1` |

**边界声明**：
- 该支线到此判定为 **完成并冻结**。
- 不继续在此支线上顺手扩展 trainer、loss 或 `task_id on` 主实验。
- 后续主注意力切回 held-out final-mapping 主线，继续围绕 `final_monotonic_hit_rate`、`final_strong_minus_weak_mean`、`preservation_pass` 与背景漂移控制推进。


## [2026-04-19] - final strength mapping 验证成立：反向 bounded gain 让最终幅度差可见

**修改文件**：`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`TEdit-main/models/conditional_generator.py`、`TEdit-main/configs/synthetic/model_multi_weaver.yaml`、`TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`、`TEdit-main/train/finetuner.py`、`test_scripts/probe_tedit_strength_internal.py`、`test_scripts/test_output_branch_carrier_contract.py`

**目标重申**：
- 原始目标是最终编辑效果在 weak / medium / strong 下出现可见、可评估的幅度差，而不是只让内部 carrier 或 stage diagnostic 变好。
- 上一轮固定 `final_output_strength_scale` 已证明 final head 是有效位点，但它只是手工 global gain；本轮目标是把它收敛成默认恒等、可训练、有界、可诊断的 final mapping。

**改动内容**：
- 新增 `final_output_strength_mapping`：
  - 默认关闭，旧 checkpoint 行为保持不变。
  - 作用点放在 `multipatch_mixer` 后、reshape 回最终 diffusion 输出前。
  - 形式为 bounded scalar gain：`gain = clamp(1 + scalar_prior_scale * (scalar - center) + learned_max_delta * tanh(head(strength_cond)))`。
  - `gain_head` 最后一层零初始化，避免启用 learned head 时破坏初始行为。
- 增加 `gain_order_direction`：
  - 支持 `increasing` 和 `decreasing`。
  - 本轮实验表明该位点是“预测噪声 gain 到最终编辑幅度”的反向映射，因此正式 finetune 配置使用 `decreasing`。
- 新增 final mapping 诊断：
  - `final_output_strength_mapping_mean/min/max`
  - `final_output_strength_mapping_by_scalar`
  - `final_strength_mapped_output`
- 在 `conditional_generator` 中接入最小 mapping-level order loss：
  - 只约束 final mapping gain 本身的 scalar order。
  - 权重来自 `final_output_strength_mapping.gain_order_weight`。
- 在 `finetuner` 的 strength 参数分组里纳入 `final_output_strength_mapping_head`，保证新增 head 使用 strength LR scale。

**关键实验**：
- 先跑正向 v1 mapping：`scalar_prior_scale=+0.1`。
  - gain 按 `0.9 / 1.0 / 1.1` 生效，`final_strength_mapped_output` 也被拉开。
  - 但最终 time-series edit gain 反向：both eval `final_strong_minus_weak_mean=-0.1294`，`final_monotonic_hit_rate=0.0`。
  - 结论：该位点不能按“预测噪声越大，最终编辑越强”的直觉处理。
- 再做反向探针：`scalar_prior_scale=-0.1`。
  - both eval `final_strong_minus_weak_mean=+0.1294`，`final_monotonic_hit_rate=1.0`。
  - preservation 失败，背景强弱差约 `0.1107`，说明方向对了但幅度过强。
- 最终正式配置使用保守反向 mapping：
  - `scalar_prior_scale=-0.04`
  - `learned_max_delta=0.04`
  - `min_gain=0.9`
  - `max_gain=1.1`
  - `gain_order_direction=decreasing`

**正式验证结果**：
- 训练命令使用 `tedit` 环境，输出到 `tmp/output_final_mapping_inverse004_3epoch`，3 epoch 正常完成。
- `tmp/final_mapping_inverse004_probe_both.json`：
  - `final_output_strength_mapping_by_scalar = {0.0000: 1.039996, 1.0000: 0.999996, 2.0000: 0.959996}`
  - 单样本 final edit region 从 weak `2.7303` 到 strong `2.7518`，方向转正。
- `tmp/final_mapping_inverse004_eval_both.json`：
  - final edit gain weak `5.8417`, medium `5.8681`, strong `5.8933`
  - `final_strong_minus_weak_mean = 0.05155`
  - `final_monotonic_hit_rate = 1.0`
  - `strength_visible_in_final = true`
  - `preservation_pass = true`
  - `bg_mae_strong_minus_weak = 0.04401`
- `tmp/final_mapping_inverse004_eval_label_only.json`：
  - final edit gain weak `5.8417`, medium `5.8681`, strong `5.8933`
  - `final_strong_minus_weak_mean = 0.05155`
  - `final_monotonic_hit_rate = 1.0`
  - `strength_visible_in_final = true`
  - `preservation_pass = true`

**验证命令**：
- `conda run -n tedit python -m py_compile ...` 通过。
- `conda run -n tedit python -m unittest test_scripts.test_output_branch_carrier_contract TEdit-main.tests.test_strength_control_phase1 TEdit-main.tests.test_finetuner_strength_diagnostics` 通过，23 tests OK。
- final mapping both / label_only probe 与 eval 均已跑通。

**关键结论**：
- 现在可以把主线表述为：
  - **strength information is present, and a bounded inverse final strength mapping can translate it into visible final-output amplitude differences.**
- 之前“final mapping 缺位”的判断被实验支持；但更精确地说，这个 final mapping 不是简单正向放大，而是对 diffusion predicted-noise head 使用保守反向 gain。
- 本轮首次同时满足：
  - final strong > weak
  - family-level monotonic hit rate 达到 1.0
  - both 和 label_only 都成立
  - preservation 仍通过

**剩余风险 / 下一步**：
- 当前验证仍是 3-sample smoke，不是完整 held-out 主实验。
- 反向 mapping 的合理性来自实际 diffusion 输出链路验证；后续需要在更大 test split 上确认不是小样本偶然。
- 背景漂移已经接近阈值，应避免继续粗暴加大 gain；下一步若要提升 gap，应优先做局部/掩码感知 mapping，而不是扩大 global gain。

**状态**：final strength mapping 主断点已打通，小样本最终可见幅度控制成立。

## [2026-04-19] - residual carrier 部分恢复后，主瓶颈后移到 final strength mapping

**修改文件**：`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`TEdit-main/models/conditional_generator.py`、`TEdit-main/configs/synthetic/model_multi_weaver.yaml`、`TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`、`TEdit-main/run_finetune.py`、`TEdit-main/train/finetuner.py`、`test_scripts/probe_tedit_strength_internal.py`、`test_scripts/test_output_branch_carrier_contract.py`、`tmp/output_branch_carrier_*`、`tmp/output_branch_scalar_order_*`

**目标重申**：
- 原始目标不是让内部某个 diagnostic 变好，而是让最终编辑输出在 weak / medium / strong 下呈现可见、可评估的幅度差异，同时尽量保持非编辑区不漂移。
- 当前主线应继续围绕 TEdit output head / final mapping，避免重新发散到 sampler、dataset、projector 或文本解析。

**改动内容**：
- 在 weaver output head 中补充 `output_branch_carrier`：
  - 默认关闭，旧 checkpoint 行为保持不变。
  - 可选将小比例 skip carrier 注入 residual branch，防止 `residual_content_branch` 完全无 carrier。
  - 新增 residual-vs-skip anti-collapse regularizer，并接入 finetune loss。
- 补齐分层 diagnostics：
  - `residual_content_branch`
  - `skip_branch`
  - `residual_carrier_source_branch`
  - `residual_carrier_restored_branch`
  - `residual_amplitude_branch`
  - `residual_merge`
  - `skip_aggregate`
  - `final_head_projection`
  - `final_head_relu`
  - `patch_decoder_concat`
  - `final_multipatch_output`
- 增加 branch-level scalar-aware supervision：
  - `output_branch_carrier.scalar_order_margin`
  - `output_branch_carrier.scalar_order_weight`
  - 直接约束 `residual_amplitude_branch` 的 strength order。
- 修复 runtime strength-control nested config shallow merge 问题，确保新增嵌套配置能从 finetune config 进入最终模型配置。

**验证结果**：
- `conda run -n tedit python -m py_compile ...` 通过。
- `conda run -n tedit python -m unittest test_scripts.test_output_branch_carrier_contract` 通过。
- `conda run -n tedit python -m unittest TEdit-main.tests.test_strength_control_phase1 TEdit-main.tests.test_finetuner_strength_diagnostics` 通过。
- `git diff --check` 通过。
- branch carrier 3 epoch 小实验显示 carrier energy 被恢复：
  - `residual_carrier_restored_abs_mean` 从 baseline 约 `0.5274` 提升到约 `0.5554-0.5677`。
  - `residual_restored_to_skip_abs_ratio` 有可观测提升。
  - preservation 仍通过。
- scalar-order 3 epoch 与 aggressive scalar-order 3 epoch 均显示：
  - branch scalar-order loss 已接入且为非零；
  - 即使用 aggressive 设置（`scalar_order_margin=0.01`, `scalar_order_weight=5.0`），`residual_amplitude_branch` strong-weak gap 仍只有 `1e-6` 量级；
  - `final_strong_minus_weak_mean` 仍在 `1e-7` 量级附近，最终强度差异没有真正成立。

**关键结论**：
- `residual carrier collapse` 是真实断点，但不是唯一断点。
- carrier energy 已能部分恢复，说明这不是纯假修复；但 restored/amplitude branch 并没有形成足够 strength separation。
- 当前主瓶颈已经从“carrier 完全死掉”推进为：
  - **carrier partially restored, but strength-conditioned final output mapping still fails.**
- 继续只加大 branch anti-collapse 或 branch scalar-order loss 的收益很低；从 aggressive ablation 看，短期内它不能把 final edit gain 拉开。

**下一步断点**：
- 以最终输出幅度差异为第一目标，不再把内部指标改善当成完成标准。
- 优先检查和调整从 `strength_cond -> strength_amplitude_head/final head -> final_multipatch_output` 的直接可见映射。
- 可考虑一个 identity-safe、默认关闭的 final-output strength gain path，但必须用 probe/eval 证明它确实改变最终编辑幅度，而不是只让内部数值好看。
- 后续实验仍使用 `tedit` 环境，产物优先写入 `tmp/`。

**后续补充验证**：
- 已实现默认关闭的 `final_output_strength_scale`，作用点在 weaver final head 的 `multipatch_mixer` 之后、reshape 回最终 diffusion 输出之前。
- identity-safe 默认配置已加入 `model_multi_weaver.yaml`：
  - `enabled=false`
  - `scale_per_unit=0.0`
  - `min_gain=max_gain=1.0`
- 临时配置 `tmp/final_output_strength_scale_model_configs.yaml` 使用 `scale_per_unit=0.1`，同一 checkpoint 不训练评估：
  - final edit gain 从几乎 flat 变为 weak `2.7230`, medium `2.7271`, strong `2.7320`
  - `final_strong_minus_weak_mean = 0.00892`
  - preservation 仍通过
- 临时配置 `tmp/final_output_strength_scale_02_model_configs.yaml` 使用 `scale_per_unit=0.2`：
  - probe sample final edit 从 weak `1.5781`, medium `1.7885`, strong `2.0237` 呈现明显幅度差
  - 3-sample eval final edit gain 为 weak `2.7192`, medium `2.7271`, strong `2.7351`
  - `final_strong_minus_weak_mean = 0.01587`
  - `final_monotonic_hit_rate = 0.3333`
  - preservation 仍通过，但背景强弱差也随 scale 增到约 `0.02218`
- 这说明 final-output gain path 可以直接产生最终幅度差，但还只是 coarse global gain，不能单独保证每个 family 都稳定单调。

**状态**：carrier 断点已部分修复；最终可见幅度控制仍未解决，下一步转向 final strength mapping。

## [2026-04-18] - P2 最小 rerun 恢复下游 stage diagnostics，可观测性恢复但控制质量未解

**修改文件**：`TEdit-main/models/conditional_generator.py`、`tool/tedit_wrapper.py`、`TEdit-main/models/conditioning/numeric_projector.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`TEdit-main/configs/synthetic/model_multi_weaver.yaml`、`test_scripts/probe_tedit_strength_internal.py`、`test_scripts/evaluate_tedit_strength_effect.py`、`results/p2_amplitude_head_probe.json`、`results/p2_amplitude_head_eval_both.json`、`results/p2_amplitude_head_eval_label_only.json`、`results/p2_amplitude_head_eval_text_only.json`

**改动内容**：
- 在 `conditional_generator` 的 DDIM forward 路径中补齐 `strength_label / strength_scalar / task_id / text_context` 传递，修复最小 rerun 中 forward 条件丢失
- 在 `numeric_projector` 中把离散 strength embedding 与显式 scalar 投影并联拼接，恢复 P2 `amplitude_decomposition` 所需的联合条件输入
- 在两套 diffusion residual block 中加入 amplitude head，并把输出阶段 diagnostics 拆成 `residual_content_branch / residual_amplitude_branch / amplitude_gamma / amplitude_beta`
- 统一 beta 路径方向口径并开放 `beta_upweight` 诊断分支；同时让 wrapper 在缺省配置下补齐 `device`
- 用最小 P2 rerun 重新跑 internal probe 与 downstream eval，检查下游 stage diagnostics 是否重新落盘

**关键结论**：
- DDIM forward conditioning 修复后，P2 rerun 已重新产出 downstream stage diagnostics，`stage_by_scalar`、stage transition summary 与输出阶段 amplitude diagnostics 都能回到 probe 结果中
- 这次修复解决的是“看不见下游响应”的 observability 断点，而不是“强度控制已完全成立”
- 当前可以确认：P2 线的 downstream diagnostics restoration 已验证，后续可以继续围绕 amplitude/output decomposition 判断问题发生在何层
- 但控制质量本身仍未完全解决；恢复可观测性不等于已经恢复 `weak < medium < strong` 的稳定输出行为

**关键决策**：
- P2 后续继续保留 amplitude/output decomposition 作为唯一主线，不回退为再次发散的多小实验并行
- 之后的判断应优先依赖恢复后的 downstream stage diagnostics，而不是只看最终 edit gain 单点现象
- 对外归档时要把“可观测性恢复”与“控制质量未解”明确拆开表述，避免把本次最小 rerun 夸大成完整修复

**状态**：下游诊断已恢复，控制质量仍待继续

## [2026-04-17] - beta 路径方向修复确认 gain 已回到正向响应

**修改文件**：`TEdit-main/data/discrete_strength_family.py`、`TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`、`TEdit-main/configs/synthetic/finetune_strength_trend_family_run1.yaml`、`TEdit-main/configs/synthetic/finetune_strength_trend_family_run2.yaml`、`TEdit-main/models/diffusion/diff_csdi_multipatch.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`test_scripts/probe_tedit_strength_internal.py`、`test_scripts/evaluate_tedit_strength_effect.py`、`results/beta_only_repair_short/*`、`results/beta_only_repair_longer/*`、`results/beta_direction_pass2/*`、`results/beta_direction_pass2_w1/*`

**改动内容**：
- 把 family strength scalar 口径从 `0.0 / 0.5 / 1.0` 改到 `0.0 / 1.0 / 2.0`，避免中档与强档间隔过窄
- 调整 beta 路径最终注入方向，并补充 `beta_upweight` 诊断模式，用于确认 beta 支路符号/方向是否与目标 gain 一致
- 将评测与 internal probe 从固定 `weak / medium / strong` 键名改成按实际 scalar 排序聚合，避免方向诊断仍被旧键名假设绑死
- 结合 `beta_only_repair` 与 `beta_direction_pass2` rerun，对 beta 路径修复前后做短程与延长版复核

**关键结论**：
- beta 路径的符号/方向断点已修复，edit gain 不再沿反方向响应；修复后 gain 方向重新跟随预期的正向 strength 增长，而不是出现“strong 比 weak 改得更少”的倒置
- 昨天这轮验证的核心结论不是“控制已经完全解决”，而是更窄且更重要的：beta path repaired to positive gain direction
- 这意味着此前 `beta_flip` 只能在推理期临时翻正的现象，已经被收敛为源代码路径上的正式修复，不再是纯诊断技巧
- 仍需继续检查方向修复之后，幅度 gap 是否足够大、是否能稳定传到更下游输出层

**关键决策**：
- 后续实验默认采用已修复的 beta 正向口径，不再把 beta-only 临时翻转当成主叙事
- 后续主问题切回“正向后为什么 gap 仍不够稳”，而不是重新争论方向是否反了
- 归档里明确保留这条 breakpoint：beta 方向问题已经被实证修复

**状态**：方向修复已确认，幅度控制仍待继续

## [2026-04-08] - 新线正式 GPU 训练完成，但 held-out monotonic 主实验仍失败

**修改文件**：`tmp/trend_strength_family_train_v1/*`、`tmp/strength_family_train_v2/0/trend_injection/ckpts/model_best.pth`、`tmp/strength_family_train_v2/0/trend_injection/strength_diagnostics.jsonl`、`tmp/strength_family_train_v2/test_benchmark_health.json`、`tmp/strength_family_train_v2/test_benchmark_health.md`、`tmp/strength_family_train_v2/test_trend_monotonic_eval.json`、`tmp/strength_family_train_v2/test_trend_monotonic_eval.md`、`TEdit-main/run_finetune.py`

**改动内容**：
- 构建正式 trend-only family 数据集：
  - `train=96 families`
  - `valid=24 families`
  - `test=24 families`
- 使用 phase1b strength checkpoint 作为初始化，从 `cuda:0` 上跑完 10 epoch 新线训练
- 新线训练使用：
  - `strength_label + instruction_text`
  - family-based `trend_injection` targets
  - `edit_region_loss + background_loss + monotonic_loss`
- 修正 `run_finetune.py` 的旧残留：为新线训练增加 `skip_final_evaluate`，避免训练结束后再掉回旧的 `cond_gen` 最终评估
- 在 held-out `test.json` 上运行 24-family monotonic 主实验

**关键结论**：
- 新线训练本身是跑通的：
  - epoch0 `best_valid_loss = 5.2286`
  - epoch1 `best_valid_loss = 4.8888`
  - epoch2 `best_valid_loss = 4.6433`
  - epoch3 `best_valid_loss = 4.6046`
  - 后续 epoch4-9 进入平台期
- strength diagnostics 显示 `strength_projector` 和三层 `strength_modulation` 都在持续更新，不是死链或零梯度训练
- held-out 24-family test benchmark 是健康的：
  - `complete_strength_rate = 1.0`
  - `target_monotonic_rate = 1.0`
  - `zero_background_leak_rate = 1.0`
- 但正式主实验结果仍然失败：
  - `edit_gain_mean`: weak `0.585786`, medium `0.585814`, strong `0.585769`
  - `monotonic_hit_rate = 0.0`
  - `strong_minus_weak_edit_gain_mean = -1.69e-05`
  - `target_mae_edit_region_mean`: weak `0.8442`, medium `1.3927`, strong `2.0039`
  - `preservation_pass_rate = 1.0`
- 这说明：
  - 训练口径已经从旧属性监督切到新线
  - 模型确实在学 strength 相关参数
  - 但输出层依然没有把 `weak / medium / strong` 转化为更大的编辑幅度
  - 当前主要失败已不再是“旧监督错位”，而是“即便在新监督下，strength condition 仍未足够影响最终 edit gain”

**关键决策**：
- 旧 `trend_types` 属性差异监督不再作为强度主线证据
- 新线的下一个排查重点不应是 benchmark 或文本解析，而应直接审查：
  - monotonic loss 的量级是否太弱
  - edit-region target loss 是否把三档都拉回同一个保守解
  - `edit_steps=10` 的评测设置是否掩盖 strength 差异

**状态**：训练链已收口，行为效果仍未成立

## [2026-04-16] - 强度诊断推进到“方向反了 / 输出层映射有误”阶段

**修改文件**：`TEdit-main/models/conditioning/numeric_projector.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`TEdit-main/models/conditional_generator.py`、`TEdit-main/train/finetuner.py`、`TEdit-main/run_finetune.py`、`tool/tedit_wrapper.py`、`test_scripts/evaluate_tedit_strength_effect.py`、`test_scripts/probe_tedit_strength_internal.py`、`test_scripts/probe_reversed_strength_scalar.py`、`test_scripts/run_tedit_strength_phase1_ablation.py`、`test_scripts/run_tedit_trend_monotonic_eval.py`、`results/strength_gainmult4_cuda0_validation/*`、`results/strength_second_gpu_validation_semantic_split/*`、`results/beta_only_repair_short/*`、`results/beta_only_repair_longer/*`、`results/beta_direction_pass2/*`、`results/beta_direction_pass2_w1/*`、`results/provenance_smoke*/*`

**改动内容**：
- 给 strength projector、modulation residual、generator loss、训练器增加分层诊断与审计输出，能够分别观察：
  - projector 输出是否随强度变化
  - modulation 的 `delta_gamma / delta_beta` 是否随强度变化
  - 中间 stage 响应是否保留 strength 差异
  - 最终输出是否仍保留 `weak < medium < strong`
- 补充三类专用脚本：
  - `evaluate_tedit_strength_effect.py`：对最终编辑结果做三档强度评估
  - `probe_tedit_strength_internal.py`：直接探测内部调制与 stage 响应
  - `probe_reversed_strength_scalar.py`：把强度标量反向喂入，验证方向是否被学反
- 新增一轮 `gainmult4`、`semantic split rerun`、`beta flip`、`beta_only_repair`、`beta_direction_loss`、`provenance smoke` 实验
- 给 `run_finetune.py` 增加 `resolved_runtime_config.json` 产物，固定记录运行时真实生效的 CLI 覆盖、合并后配置、checkpoint/data/output 路径与 git commit

**关键结论**：
- 当前问题已经不是“strength 通道不存在”：
  - recent probe 显示内部 modulation 对不同强度存在可观测差异
  - `delta_gamma / delta_beta` 会随 `0.0 -> 0.5 -> 1.0` 增大
- 当前主要故障也不再像是“完全零响应”，而更像是：
  - 最终输出层把强度差异压平
  - 或者把方向学反，出现 `strong` 比 `weak` 改得更少
- `semantic_split` rerun 的关键现象：
  - 正常喂入 scalar 时，label-only probe 常出现负相关或负的 `strong_minus_weak_edit_gain_mean`
  - 把 scalar 反向喂入后，整体趋势会显著翻正
- `beta_flip` probe 的关键现象：
  - 仅在推理时翻转 beta 路径符号，就能把局部 probe 从反向改成正向
  - 这表明当前最可疑的问题集中在 beta 路径及其到最终输出的映射
- `Pass 2A` 结论仍然保守：
  - `beta_direction_loss_weight=1.0` 已经接入训练入口并实际运行
  - 但 Gate 1 / Gate 2 仍未修复正式 monotonic 行为
  - 目前还不能宣称“已经不需要 beta-flip”

**关键决策**：
- 后续排查重点从“有没有 strength signal”切到“为什么 signal 在输出阶段变平或反向”
- provenance 记录必须保留，因为近期实验已细到“同一 checkpoint 不同运行口径会影响解释”
- 当前不应把问题重新表述成 benchmark 不健康或文本解析失效；主故障已经收敛到输出方向与最终映射

**状态**：已定位到输出方向层面的核心窄口，尚未完成修复

## [2026-04-08] - 新 strength-injection 主线切断旧属性监督并落地 family 训练路径

**修改文件**：`TEdit-main/data/discrete_strength_family.py`、`TEdit-main/data/__init__.py`、`TEdit-main/models/conditional_generator.py`、`TEdit-main/run_finetune.py`、`TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml`、`TEdit-main/configs/synthetic/model_multi_weaver.yaml`、`test_scripts/build_tedit_strength_trend_family_dataset.py`、`docs/pure_editing_how_much_protocol.md`、`docs/experiment_preparation.md`、`PIPELINE.md`

**改动内容**：
- 新增 `trend_injection` family-based 离散强度训练数据构造器，直接生成 `train/valid/test` 三个 split 的 `src_x / tgt_x / mask_gt / strength_label / instruction_text` 数据
- 新增 `discrete_strength_family` 数据集类型，用 family 作为最小训练单元，并在 loader 中保留 `family_sizes`
- 在 `ConditionalGenerator.fintune()` 上增加输出层监督：
  - `edit_region_loss_weight`
  - `background_loss_weight`
  - `monotonic_loss_weight`
  - `monotonic_margin`
- 调整 `run_finetune.py`，使新数据类型不再沿用旧的 `ctrl_attrs -> 子目录` 数据拼接逻辑
- 新增新线专用 finetune 配置，并把文档口径更新为“family 训练集 + 输出层强度监督”主线

**关键结论**：
- 新主线已经不再依赖 `trend_types` 这类属性差异桶化监督来“顺带学习强度”
- family dataset smoke 已通过：
  - `train=6 families / 18 samples`
  - `valid=3 families / 9 samples`
  - `test=3 families / 9 samples`
- 新 dataset loader smoke 已通过：
  - `batch_size=2 families` 会整理成 `6` 个 flat samples
  - 同时保留 `family_sizes=(2,)`
- 新 loss 路径已在 `cuda:0` 上通过单 batch 前向 smoke：
  - `num_steps=2`
  - `edit_steps=2`
  - `layers=1`
  - `multipatch_num=1`
  - `loss=2.328873872756958`

**关键决策**：
- 第一阶段训练主线正式切换到 family-based `trend_injection` 数据，不再把旧 synthetic 属性切换监督当作强度主训练集
- 输出层 monotonic 现在进入训练目标本身，不再只依赖 internal probe 或后验评估去发现问题
- 自由文本解析仍保留，但第一阶段模型本体训练与主评测优先使用同分布模板文本

**状态**：已实现，待正式 GPU 训练验证

## [2026-04-08] - 离散 benchmark 正式体检与 trend monotonic 主实验

**修改文件**：`test_scripts/check_tedit_strength_discrete_benchmark.py`、`test_scripts/run_tedit_trend_monotonic_eval.py`、`tmp/discrete_strength_health_30/benchmark_health.json`、`tmp/discrete_strength_health_30/benchmark_health.md`、`tmp/discrete_strength_trend18/benchmark_health.json`、`tmp/discrete_strength_trend18/benchmark_health.md`、`tmp/discrete_strength_trend18/trend_monotonic_eval.json`、`tmp/discrete_strength_trend18/trend_monotonic_summary.md`

**改动内容**：
- 对 30-family 的 mixed discrete benchmark 做正式体检，检查三档完整性、target 构造单调性、tool/duration 分布和背景泄漏
- 对 18-family 的 `trend_injection` benchmark 单独再做一轮 health check，确认主实验输入本身健康
- 新增直接消费 discrete benchmark family 的 trend monotonic 主实验 runner，固定 `GT family + GT region + strength_label + instruction_text`，不把 planner、tool routing 或 localization 混进第一轮结论
- 使用 `tmp/strength_phase1/finetune_run_bs8/0/trend_types/ckpts/model_best.pth` 跑完 18-family、54 次执行的中等规模 monotonic 主实验

**关键结论**：
- mixed 30-family benchmark 并非完全健康：
  - `complete_strength_rate = 1.0`
  - `target_monotonic_rate = 29 / 30 = 0.9667`
  - `zero_background_leak_rate = 1.0`
  - tool 分布为 5 个主工具族各 6 family，duration 分布为 short / medium / long 各 10 family
- 唯一失败 family 是 `family_025 (noise_injection, short)`，其 target edit gain 为：
  - weak `0.18864`
  - medium `0.17907`
  - strong `0.18714`
  说明当前 builder 的 `noise_injection` 构造在个别样本上会破坏三档单调性
- 18-family 的 trend-only benchmark 是健康的：
  - `complete_strength_rate = 1.0`
  - `target_monotonic_rate = 1.0`
  - `zero_background_leak_rate = 1.0`
  - `health_pass = true`
- 在 trend-only benchmark 上，当前 Phase 1b checkpoint 的输出层强度控制仍然失败：
  - `probe_gate.diff_0_2_linf = 9.85e-05`, `pass = true`
  - `edit_gain_mean`: weak `0.574873`, medium `0.574876`, strong `0.574864`
  - `monotonic_hit_rate = 0.0`
  - `strong_minus_weak_edit_gain_mean = -9.40e-06`
  - `bg_mae_strong_minus_weak = -3.11e-08`
  - `preservation_pass_rate = 1.0`
- 这说明当前 checkpoint 已满足“内部非零响应”的前置门槛，但输出层仍未形成 `weak < medium < strong`；失败原因不再是 benchmark 输入不健康，而是模型尚未把 strength condition 转化为可见的编辑幅度差

**关键决策**：
- mixed discrete benchmark 暂时不能直接作为全工具族主 benchmark 使用，必须先修 `noise_injection` family 的强度构造
- 当前 trend-only benchmark 可以继续作为 Phase 1 monotonic 主评测集
- 下一步不应回到 `task_id` 或连续强度，而应先围绕现有 `strength + text` 路线排查“为什么 target gap 在输出层没有转化为 edit gain gap”

**状态**：benchmark 部分健康，模型输出层未通过

## [2026-04-08] - Phase 1 强度控制验证收口到 `strength + text`

**修改文件**：`tmp/strength_phase1/model_configs.synthetic.pretrain.strength_enabled.yaml`、`tmp/strength_phase1/finetune.synthetic.phase1b.yaml`、`tmp/strength_phase1/evaluate.synthetic.phase1b.yaml`、`tmp/strength_phase1/probe/pretrained_probe.json`、`tmp/strength_phase1/probe/post_finetune_probe.json`、`tmp/strength_phase1/probe/sweep_summary.json`

**改动内容**：
- 将第一阶段主线从 `strength_label + task_id + pooled text_context` 收紧为 `strength_label + pooled text_context`
- 用临时实验配置明确打开 `strength_control.enabled=true`、保持 `use_text_context=true`、固定 `use_task_id=false`
- 在 `synthetic/trend_types` 上完成两步验证：
  - Phase 1a：直接对预训练权重做 internal probe
  - Phase 1b：基于预训练权重做 1 epoch、`freeze_backbone_for_strength=true` 的轻量 finetune，再重复同一 probe

**关键结论**：
- 预训练权重在 Phase 1a 下对 `weak / medium / strong` 完全不可分，5-sample probe 全部是 `diff_0_2_linf = 0`
- 轻量 finetune 后，5-sample probe 稳定出现非零差异，`mean(diff_0_2_linf) = 9.47e-05`
- 这说明 `strength + text -> projector / modulation` 的链路开始可分，但量级仍很小，不能把它当作“输出层强度控制已真实有效”
- 新增输出层简化评估脚本 `test_scripts/evaluate_tedit_strength_effect.py` 后，5-sample `trend_types` 小样本结果显示：
  - `edit_gain_mean`: weak `0.84457`, medium `0.84459`, strong `0.84453`
  - `monotonic_hit_rate = 0.0`
  - `strong_minus_weak_edit_gain_mean = -4.33e-05`
- 这表明当前 Phase 1b 权重虽然已经不是“完全零响应”，但输出层仍没有形成有效的强度顺序
- `run_finetune.py` 训练后自带的 `cond_gen` 评估在 `diff_csdi_multipatch_weaver.py` 触发维度错误，当前会阻断自动评估收尾

**关键决策**：
- `task_id` 保留代码实现，但从第一阶段 active mainline 中移出
- Phase 1 成功标准仍是“先证明可分性成立”，不是一次性拿到稳定单调性或论文级强度控制结果
- 输出层简化评估已补上，下一步应优先扩大样本数并排查为什么 `strong` 仍未超过 `weak`
- 在输出层趋势未成立前，不继续提前引入 `task_id` 或连续强度

**状态**：部分成立

## [2026-04-08] - 离散 strength mainline benchmark 与 parser 落地

**修改文件**：`modules/strength_parser.py`、`modules/llm.py`、`tool/ts_editors.py`、`test_scripts/build_tedit_strength_discrete_benchmark.py`、`test_scripts/evaluate_tedit_strength_effect.py`、`docs/pure_editing_how_much_protocol.md`、`docs/experiment_preparation.md`、`PIPELINE.md`

**改动内容**：
- 新增独立的离散强度文本解析器，输出 `strength_text / strength_label / constraint_type / parse_reason / confidence`
- 将强度解析接入 `normalize_llm_plan` 与 `_apply_explicit_prompt_hints`，使自由文本可直接补全 `intent.strength` 与 `parameters.strength_label`
- 新增离散主线 benchmark builder，按 family 方式固定 `source/tool/region/template`，仅改变 `weak / medium / strong`
- 在输出层评估脚本中补充 `bg_mae_strong_minus_weak` 与 preservation pass 判定
- 文档和 pipeline 入口改为离散 strength mainline，不再把 teacher stress benchmark 写成当前主 benchmark

**关键结论**：
- 文本强度解析已不再停留在“模板化传递”，而是具备了独立可检查的解析输出
- 新 benchmark 已能生成真正的 3-way family，而不是单样本后处理分桶
- 当前实现已经具备继续做离散 Phase 1 闭环实验的基础设施

**状态**：已实现，待更大样本实验验证

## [2026-04-08] - 锁定 pure-editing strength injection mainline

**修改文件**：`docs/pure_editing_how_much_protocol.md`、`agent/nodes.py`、`tool/tedit_wrapper.py`、`tool/ts_editors.py`、`tool/tool_description/tedit_tools.py`、`TEdit-main/models/conditioning/numeric_projector.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch.py`、`TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`、`TEdit-main/models/conditional_generator.py`、`TEdit-main/data/synthetic_finetune.py`、`test_scripts/build_tedit_strength_dataset.py`、`test_scripts/probe_tedit_strength_internal.py`

**改动内容**：
- 将 pure-editing `how much` 主线切换到 TEdit 内部强度注入
- 锁定条件路径为 `strength_label + task_id + pooled text_context`
- 在 agent -> tool description -> wrapper -> generator -> diffusion 内部 modulation 这条链路上打通强度参数
- 新增 `StrengthProjector / Numeric Projector` 与 `modulation_residual` 相关实现
- 增加 strength dataset augment 与 internal probe 脚本

**关键决策**：
- 主线不再以 teacher search 或 student distillation 作为方法叙事中心
- 当前只回答三个问题：内部是否真的注入、`weak < medium < strong` 是否单调、非编辑区是否稳定
- `run_pipeline.py` 不应重新引入 retired pure-editing student runtime override

**状态**：进行中

## [2026-03-24] - 进入 pure-editing tool-conditioned student 阶段

**修改文件**：`modules/pure_editing_student.py`、`test_scripts/train_pure_editing_student.py` 及相关 pure-editing 运行链

**改动内容**：
- 引入 pure-editing tool-conditioned student 路线，尝试用学生模型学习 `how much`
- 开始围绕 runtime-safe、softguard、capacity ablation 等方向迭代

**关键决策**：
- 当时默认将 student 视为主线候选
- 后续事实证明该路线不再适合作为当前 mainline 叙事

**状态**：已退休

## [2026-03-24] - teacher search 原型完成并形成首批 checkpoint

**修改文件**：`modules/pure_editing_how_much.py`、`tool/ts_editors.py`、`tool/tedit_wrapper.py`

**改动内容**：
- 建立 pure-editing teacher parameter search 原型
- 形成 20-sample 与 50-sample 的 teacher protocol checkpoint

**关键决策**：
- 先固定 `GT tool + GT region`，只搜索 `how much` 参数
- 用 teacher vs heuristic 对照确认主工具族是否存在正信号

**状态**：历史里程碑

## [2026-03-24] - volatility operator audit 与 canonical split 路线形成

**修改文件**：`modules/pure_editing_volatility.py`、`modules/pure_editing_how_much.py`、`docs/pure_editing_how_much_protocol.md`

**改动内容**：
- 对 `volatility_increase` 做 operator audit
- 比较 `global_subwindow`、`burst_local`、`envelope_noise`、`piecewise_envelope_noise`
- 推进 canonical split 与 routed closure 验证

**关键决策**：
- 确认 volatility 弱点主要来自 operator/objective 不对齐，而不只是搜索空间不够大
- 结论从“继续 student distillation”改为“先修 volatility 家族，再谈下一步”

**状态**：历史里程碑

## 当前主线状态（滚动摘要）

**当前主线**：
- `TEdit`
- `Pure Editing`
- 离散强度：`weak / medium / strong`
- 第一阶段条件路径：`strength_label`、`pooled text_context`
- 机制：`Numeric Projector / Strength Adapter + residual modulation injection`
- `task_id`：保留实现，移到后续阶段再评估是否重新引入

**当前证据基线**：
- `tmp/how_much/pure_editing/teacher_protocol_20.json`
- `tmp/how_much/pure_editing/teacher_protocol_stress50.json`
- `tmp/how_much/pure_editing/volatility24_audit.json`
- `tmp/how_much/pure_editing/volclosure24_seed29_split_validation_v3.json`
- `tmp/strength_phase1/probe/pretrained_probe.json`
- `tmp/strength_phase1/probe/post_finetune_probe.json`
- `tmp/strength_phase1/probe/sweep_summary.json`

**当前下一步**：
- 先补输出层验证，确认 edit-region 是否随 `weak / medium / strong` 出现可解释变化
- 排查 `run_finetune.py` 训练后 `cond_gen` 评估的维度错误，恢复自动评估收尾
- 在可分性确认后，再进入 monotonicity / preservation / ablation
