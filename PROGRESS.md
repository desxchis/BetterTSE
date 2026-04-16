# Research Progress Log - 科研代码进度记录

按时间倒序记录 pure-editing strength mainline 的关键推进，不混入 forecast-revision 主线。

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
