# BetterTSE 当前卡点、进展与判断复盘

## 1. 这轮工作现在真正卡在哪里

当前项目**不是卡在“任务是否成立”**，也**不是卡在“数据是否不够”**，更**不是卡在“框架还没搭起来”**。

当前真正卡住的问题已经收敛为一个更窄、但也更核心的点：

> **`how much to edit` 还没有被干净、稳定、可验证地解决。**

更具体地说，当前卡点分成四层。

### 1.1 研究主张和当前能力之间还有错位

现阶段 BetterTSE 在叙述上已经把问题推进到：

- `revision_needed`
- `where`
- `what`
- `how much`

但从现有设计与实验状态看，真正被扎实验证的是前 3 项，第四项仍然只是**部分显式化**，还没有形成一个可以强主张的闭环。

目前 `how much` 虽然已经从模糊 planner 字段，推进成了：

- 明确的 `edit_spec`
- 单独的 calibration metrics
- 单独的 calibration benchmark runner
- config-driven framework assembly

但它现在更像是：

- **可运行的评测脚手架**
- **可分析的分问题接口**
- **可继续训练/校准的工程入口**

而不是：

- 已经被证明有效的核心方法能力。

所以当前最核心的卡点不是“没有东西”，而是：

| 层面 | 现状 | 问题 |
|---|---|---|
| 任务定义 | 已经清楚 | 不再是主要问题 |
| 主干实验链 | 已经成立 | 不再是主要问题 |
| `how much`接口化 | 已完成第一版 | 但还没变成稳定方法结论 |
| 论文主张 | 想强调可控修正/幅度控制 | 证据还不够硬 |

### 1.2 真实数据上的“校准质量”仍然不够干净

从已有结论看，forecast revision 主线已经成立：

- controlled 上成立
- real structured event 上成立
- non-app gate 上成立
- native text 上也成立

但这条线成立，更多说明的是：

- 任务有效
- localized revision 比 global revision 更好
- 框架可迁移
- gate 可 abstain

它**还不能自动推出**：

- 模型已经学会了精确控制修正幅度
- 输入强弱和输出响应之间已经稳定可校准
- 真正解决了 `how much`

尤其从 case study 来看，真实样本上仍然存在：

- 强度判断偏强/偏弱
- region 偏大
- shape 对不上
- magnitude error 仍然偏高

典型表现：

| 数据集 | 好消息 | 暴露的问题 |
|---|---|---|
| Weather | magnitude 可做到很准 | 主要是理想受控环境 |
| XTraffic | localized 明显优于 global | 强度、区域、语义仍有粗糙性 |
| MTBench | native text 能跑通且收益为正 | 长文本语义到幅度/形状的映射仍不稳定 |

所以第二个卡点不是“方法完全失效”，而是：

> **真实数据上，revision 是有效的，但 calibration 还不够干净。**

### 1.3 当前监督来源对 `how much` 仍然不够“硬”

项目已经很明确地识别出一个理论风险：

- 如果教师来自模型内部搜索或规则筛选
- 学生学到的可能不是“真实可控规律”
- 而是当前系统偏差的再编码

这也是此前 repeatedly 提到的关键判断：

- 当前 teacher-student 路线定位不清
- 本质上偏向 self-distillation
- 会放大 bias
- 容易把启发式搜索包装成“学会控制”

因此当前第三个卡点其实是监督定义问题：

> **我们已经知道“不能再把 teacher search 当成核心答案”，但新的、足够独立的外部约束还没有完全补齐。**

现在 repo 里已经向前推进了一步：

- 把 `edit_spec_gt`、calibration metrics、oracle/semi-oracle runner 建起来
- 让 `how much` 至少变成可测量对象

但还没有完全走到：

- 外部 critic / reward / learned reliability 足够稳定
- 真实证明某种 learned calibration 比规则法更稳
- 可以自信宣称“已解决 how much”

### 1.4 当前更像是“框架稳定后的小核心难题”，不是“大方向未定”

这点很重要。

现在的卡，不是早期那种：

- 任务要不要做 forecast revision
- XTraffic 要不要继续
- MTBench 能不能纳入
- schema 要不要改
- benchmark 要不要重做

这些大方向问题，其实都已经被当前文档反复冻结了。

真正卡住的是：

> **主框架已经稳定后，剩下的最关键 headroom，几乎都集中在 calibration / `how much`。**

也就是说，这是一个**“收敛后的关键窄口”**，不是“发散期的不确定”。

---

## 2. 目前已经达成了哪些进展

## 2.1 已经把项目主线从混乱状态整理成“两条主线”

这是一个很重要但容易低估的进展。

当前仓库已经明确区分：

### Mainline A：Pure Time-Series Editing

- 目标：验证 BetterTSE 本身的编辑能力
- 重点：
  - localization
  - intent alignment
  - editability
  - outside-region preservation

### Mainline B：Forecast Revision as Downstream Application

- 目标：验证 BetterTSE 编辑能力在 forecasting 场景里的迁移价值
- 重点：
  - gain over base forecast
  - quality against revision target / future_gt
  - robustness across datasets/backbones

这一步的意义是：

- forecast revision 不再替代 editing 主线
- pure editing 仍然是方法核心
- forecast revision 被重新放回“下游应用验证”的位置

这直接修复了前一阶段一个很危险的问题：

> 项目一度有从“时序编辑方法”滑向“带文本的 forecasting trick” 的风险。

现在这个风险已经被文档和实验结构显式压住了。

## 2.2 已经把 forecast revision 主线的证据链跑通了

现阶段最扎实的成果之一，就是 forecast revision 的 evidence chain 已经建立起来，并且不是单点结果，而是一条层次清楚的链。

### 当前稳定证据链

| 顺序 | 数据集 | 角色 | 已建立的结论 |
|---|---|---|---|
| 1 | Weather v4 | controlled proof point | 任务有效，localized 优于 global |
| 2 | XTraffic v2 narrowed | main real benchmark | localized revision 可迁移到真实事件数据 |
| 3 | XTraffic v2 nonapp | real gate benchmark | 可以在真实 no-op 样本上 abstain |
| 4 | MTBench finance v2 | native-text complement | 框架可迁移到 native text + time series |
| 5 | MTBench finance v2 (100) | stability check | native-text 结果不是 smoke 偶然 |

对应已经形成的总体判断也相当一致：

- task is valid
- localized > global
- transfers to real data
- gate behavior is verified
- transfers to native text when schema matches domain

这说明项目**已经越过“这个方向靠不靠谱”的阶段**。

## 2.3 已经完成 benchmark role freeze，叙事开始稳定

当前文档已经反复强调并冻结了数据角色：

| 数据集 | 当前角色 |
|---|---|
| XTraffic | 主 benchmark / 主 empirical battlefield |
| MTBench | realism / native-text complement |
| CiK | protocol template，不是主战场 |
| Time-MMD / Time-IMM | 后续可选扩展，不是当前主线 |
| CGTSF 等 | 轻量补充 / appendix |

这一步很关键，因为之前真正的大风险之一其实不是性能，而是：

> benchmark-role sprawl 导致 paper narrative 被摊薄。

现在这个判断已经很明确：

- 当前风险不是数据覆盖不够
- 当前风险是 scope spread
- 当前不该再平行新开很多 benchmark 线

这意味着项目管理层面的“卡住”已经被明显缓解了。

## 2.4 已经验证了可复现性，结果不是一次性偶然产物

当前 checkpoint 还做了小规模 reproducibility rerun，并且结果是：

- all_exact_match = True
- max_abs_diff = 0.0

覆盖了：

- Weather v4 controlled
- XTraffic v2 narrowed
- XTraffic v2 nonapp
- MTBench finance v2 100

这件事的重要性在于：

- 当前证据链不是“只跑成功一次”
- 保存结果和 rerun 结果一致
- 可以把现阶段状态视为一个可冻结的断点

对后续继续做 calibration 非常重要，因为它给了一个稳定起点。

## 2.5 已经把 `how much` 从模糊概念变成了独立框架对象

这是当前最直接相关的进展。

过去 `how much` 更多像：

- prompt 里的强弱词
- planner 的隐式字段
- rule-based numeric scaling
- 执行期 heuristic

现在至少已经被显式化成：

### 代码/框架层面的进展

- 显式 `edit_spec` 预测
- `edit_spec -> executor params` 投影
- benchmark 支持 `edit_spec_gt`
- calibration metrics 已加入主 runner
- 单独 calibration benchmark runner 已加入
- config-driven framework planner 已加入

### 当前已经有的 calibration 指标

- `normalized_parameter_error`
- `peak_delta_error`
- `signed_area_error`
- `duration_error`
- `recovery_slope_error`
- 以及文档中持续强调的 `magnitude calibration error`

### 已完成的工程判断

- `how much` 不该继续只是 planner strength field
- 它必须变成单独可评测的子问题
- calibration 实验应先在 `oracle_region + oracle_intent` 下跑
- 再逐步接回 full pipeline

这说明：

> 现在不是“还没开始解决 `how much`”，而是“已经把问题从抽象口号推进到可执行实验框架，但还没收敛出最终答案”。

## 2.6 已经形成了对 teacher-student/self-distillation 风险的清晰判断

另一个非常重要的进展，不是代码，而是判断本身被厘清了。

当前文档已经明确指出：

- 现有 teacher 并不是真正独立教师
- 更像参数搜索 + 候选筛选 + 伪标签
- 学生优化的是 teacher_params，不是可控编辑行为本身
- 这会带来自蒸馏问题
- 会放大已有 bias 与 variance

这个判断其实很关键，因为它决定了后续路线不会再误入：

- 把 heuristic search 包装成 teacher
- 再把 teacher-student 当成论文主方法

也就是说，当前虽然还没完全拿出新的终局方案，但至少已经完成了一个重要纠偏：

> **我们已经知道什么方向不该继续作为核心答案。**

这本身就是显著进展。

---

## 3. 目前已经形成了哪些关键判断

## 3.1 已经成立的判断

### 判断 A：forecast revision 任务本身是成立的

这一点基本已经不再是争议点。

受控、真实事件、真实 non-app、native text 四条线都给出了正向支持。

### 判断 B：localized revision 明显优于 global revision

这已经是当前主线最稳定的经验结论之一，也是 forecast revision 成立的核心支柱。

### 判断 C：真实 no-op gate 是能做出来的

`revision_needed_match = 1.0` 和 `over_edit_rate = 0.0` 说明这条线不是只能“逢样本必改”。

### 判断 D：native-text transfer 不是伪命题，但前提是 schema 要做 domain match

MTBench 的结果说明：

- 不是所有 text 都能直接拿来用
- 但只要 revision schema 跟 domain 对齐，native text 是能形成正向 revision 信号的

### 判断 E：当前主风险不是 benchmark 不够多，而是 scope 和 narrative 被摊薄

因此当前最不该做的是：

- 再大幅改主框架
- 同时开多个新 benchmark
- 让 patchtst/new backbone 变成 blocker
- 重新定义问题边界

## 3.2 尚未成立、不能过早主张的判断

### 判断 F：还不能说项目已经解决了 `how much to edit`

原因不是完全没进展，而是：

- 当前更多是 scaffold 成立
- 不是强证据表明 learned calibration 已经稳定有效
- 真实数据的幅度对齐仍有明显误差

### 判断 G：还不能说扩散模型内部已经学会“原生幅度控制”

从 proposal 草案和相关设计判断看，当前更多还处于：

- 外部控制层
- 参数投影层
- calibration/评测层

而不是已经证明：

- 幅度条件稳定进入 diffusion 内部
- 且表现出单调、平滑、可校准的响应曲线

### 判断 H：teacher-student 不能再被直接当作核心创新主线

它可以保留为：

- oracle 上界
- semi-oracle 辅助路线
- 某些 calibration baseline

但不适合继续被包装成“核心方法已经解决 how much”。

---

## 4. 当前卡点为什么会让人感觉“项目卡住”

项目现在的停滞感，本质上来自一种典型的中后期状态：

- 大框架已经搭起来了
- 大方向已经收敛了
- 主要 benchmark 也已经跑通了
- 但是剩下的关键创新点更窄、更难、更不容易靠堆工程解决

因此当前的卡住感，不是因为没有成果，而是因为：

> **容易出结果的结构性问题大多已经解决，剩下的是最核心、最难替代的那部分。**

这会带来几个表象：

| 表象 | 实际原因 |
|---|---|
| 感觉一直在补文档和框架 | 因为主框架已接近冻结，正在为核心窄口让路 |
| 感觉 code 加了很多但结论没同步变强 | 因为 calibration scaffold ≠ calibration problem solved |
| 感觉 teacher/student、rule、oracle 都有，但主方法还不够锐利 | 因为真正缺的是稳定独立的 `how much` 解决方案 |
| 感觉真实数据能涨分，但论文主张还差一口气 | 因为“有效 revision”与“可控幅度建模”是两层命题 |

所以这个“卡住”不是坏事，它更像是：

- 项目已经从 exploratory 阶段进入 narrowing 阶段
- 但 narrowing 后留下的是最需要研究贡献的硬骨头

---

## 5. 现在最合理的整体复盘结论

## 5.1 一句话结论

当前 BetterTSE **已经完成了主框架与证据链的稳定化**，并证明了 forecast revision 这条应用线是成立的；**现在真正卡住的核心问题，已经收敛为 `how much to edit` 的校准与可控性还没有被充分、干净、独立地解决。**

## 5.2 分层结论

### 已经解决/基本稳定的部分

- 项目应以“两条主线”组织，而不是只剩 forecast revision
- forecast revision 任务定义成立
- localized revision 优于 global revision
- XTraffic 可作为主真实 benchmark
- MTBench 可作为 native-text complement
- non-app gate 可成立
- 数据角色与叙事边界已经冻结
- 当前结果具备小规模可复现性
- calibration scaffold 已经落地，可继续推进

### 当前真正卡住的部分

- `how much` 还没有形成论文级强结论
- 真实数据上的 calibration 仍不够干净
- 监督信号仍缺少足够独立、稳定的外部约束
- teacher-student 路线已被识别出理论问题，但替代性主方法尚未完全收敛

### 当前不该误判为“卡点”的部分

- 不是 benchmark 不够多
- 不是主框架还要继续大改
- 不是任务本身不成立
- 不是 native text 完全不 work
- 不是缺少 runnable code scaffold

---

## 6. 对当前状态的建议性判断

如果只按当前 repo 和已有文档来复盘，最稳妥的判断应该是：

1. **把当前阶段视作一个已经稳定的断点。**
   - 证据链已足够支撑当前主线成立。

2. **不要再把“框架重构”当成主要工作。**
   - 主框架现在已经不是瓶颈。

3. **把后续所有新增工作尽量约束到 calibration / `how much` 这个窄口。**
   - 这是最主要 headroom。

4. **对外表述要诚实分层。**
   - 现在可以强说“localized forecast revision framework is valid and transferable”。
   - 还不能同等强度地说“we have solved controllable magnitude modeling”。

5. **论文叙事上应避免把 scaffold 当成结果。**
   - 当前 `edit_spec + metrics + planner + framework` 是非常好的基础设施，
   - 但它们本身不是最终研究贡献闭环，除非后续实验把 controllability 真正坐实。

---

## 7. 一个最简短的最终判断

### 当前最核心的卡点

> 不是“能不能做 revision”，而是“能不能把 revision 的幅度控制做成可验证、可校准、能在真实数据上稳定成立的能力”。

### 当前最扎实的已完成进展

> forecast revision 主线已经成立，localized > global、real transfer、real gate、native-text transfer 都已经有稳定证据链，项目的大框架和 benchmark 角色也已经基本冻结。

### 当前最重要的认识

> 项目不是还在发散，而是已经收敛到一个真正决定研究贡献上限的窄口：`how much to edit`。
