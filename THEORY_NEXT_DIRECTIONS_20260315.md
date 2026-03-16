# 理论备忘：从意图到数值的瓶颈与后续破局点

时间: `2026-03-15`

## 当前核心判断

现阶段 BetterTSE 的主瓶颈已经不再是“LLM 是否能理解编辑意图”，而是：

- 如何把自然语言中的编辑意图稳定映射为可执行的连续数值参数
- 如何让这些参数跨数据集、跨量纲、跨特征仍然可比且可控

更直接地说，当前问题不是 `what to edit`，而是 `how much to edit`。

## 为什么这是主瓶颈

当前 pipeline 已经把下面几层做得相对清楚：

- `instruction -> intent`
- `instruction -> region`
- `intent -> canonical tool`

但最后一步仍然是弱的：

- `canonical tool -> exact numeric parameters`

这里困难的根本原因有 4 个。

### 1. 自然语言中的强度通常不是绝对量

像“明显升高”“维持高位”“更受限”“短时冲高”这类表达，本质上是相对概念，不是绝对数值。它们依赖：

- 当前序列基线
- 局部波动尺度
- 数据集单位
- 业务语境

同一句“显著升高”，在 Traffic、ETTh1、油价、病人血压里不是同一个量级。

### 2. 当前执行器参数空间和语言空间没有中间桥梁

现在执行层最终需要的是：

- `math_shift`
- `level_shift`
- `amplitude`
- `width`
- `trend strength`

但 LLM 输出的是：

- `up/down`
- `hump/step/flatline`
- `weak/medium/strong`

中间缺一个稳定的“无量纲编辑表示”。

### 3. TEdit 原生控制空间偏属性，不偏量纲

TEdit 的出发点是“改指定属性，同时保留其它属性”，它很适合：

- trend type
- trend direction
- season cycle

但它本身不是为“把这段抬高 1.7 个局部标准差并维持 18 步”这种精确可执行规格设计的。

### 4. 当前 benchmark 主要评估到了语义与定位，还没单独评估数值校准

你们已经在看：

- `t-IoU`
- `intent match`
- `editability`
- `preservability`

但还缺一个专门回答“数值调得准不准”的层。

## 从相关论文得到的最有价值启发

### 1. TEdit / TSE 证明了“属性编辑”这条路线成立，但它不是数值校准方案

`Towards Editing Time Series` 把任务定义为：修改指定属性、保留其他属性不变，核心是属性控制，不是自然语言到连续参数标定。  
来源:

- https://seqml.github.io/tse/
- https://openreview.net/pdf?id=qu5NTwZtxA

对我们最重要的启发：

- TEdit 很适合做编辑器后端
- 但它前面需要一个更强的参数化层，不然自然语言只能粗粒度落到属性标签

### 2. InstructTime 已经明确点出“编辑强度”必须单独建模

`Instruction-based Time Series Editing` 的核心贡献之一，就是把自然语言编辑引入时序编辑，并强调：

- instruction-based editing
- controllable edit strength
- local and global edits together

论文里通过共享多模态表示和插值去控制编辑强度。  
来源:

- https://arxiv.org/abs/2508.01504

对我们最有价值的点：

- “编辑强度”不应继续藏在 prompt 里靠 LLM 猜
- 它应该成为一个显式、连续、可监督的变量

### 3. VerbalTS 说明文本编辑可以精细到 token-level 对齐

`VerbalTS` 是“从文本生成时序”，但它在项目页里明确展示了：

- 修改文本中的局部 token
- 生成结果会跟随这些 token 做更细粒度编辑

来源:

- https://seqml.github.io/VerbalTS/

这对我们很重要，因为它支持一个判断：

- 文本不是只能做 coarse intent
- 文本里和“强度、持续时间、恢复方式”相关的局部短语，是可以被单独建模并映射到连续参数的

### 4. ChatTS / Time-MQA 更偏“理解与推理”，不是执行与校准

`ChatTS` 强调通过合成数据，把时间序列和文本对齐，从而增强理解与推理。  
`Time-MQA` 强调多任务问答和 reasoning dataset。  
来源:

- https://arxiv.org/abs/2412.03104
- https://arxiv.org/abs/2503.01875

它们给我们的启发不是“怎么做编辑器”，而是：

- synthetic instruction data 是必要的
- reasoning 和 alignment 可以通过构造数据显著提升

但它们并没有直接解决：

- 数值参数如何校准
- 编辑强度如何跨量纲泛化

### 5. Context is Key / Time Weaver / Time-IMM 共同说明：上下文与异构条件决定了量级

`Context is Key` 的 benchmark 里已经有：

- explicit dates
- bounded constraints
- covariate context

说明文本上下文不仅决定方向，还决定约束与数值边界。  
来源:

- https://servicenow.github.io/context-is-key-forecasting/v0/

`Time Weaver` 强调异构 metadata 对时序生成的 specificity 至关重要。  
来源:

- https://arxiv.org/abs/2403.02682

`Time-IMM` 则说明现实时序常常是 irregular、multimodal、messy，必须显式建模异构条件。  
来源:

- https://arxiv.org/abs/2506.10412

对我们最关键的结论是：

- 数值校准不能只看 instruction 文本
- 还必须看局部时序上下文、特征统计量和外部条件

### 6. InstructPix2Pix 证明了“instruction editing”成功的关键是合成监督对

`InstructPix2Pix` 的关键不是 prompt engineering，而是生成大规模 `(input, instruction, output)` 编辑对，再训练模型直接学会执行。  
来源:

- https://openaccess.thecvf.com/content/CVPR2023/html/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.html

这对 BetterTSE 的启发非常直接：

- 不能一直靠纯规则把语言硬映射到数值
- 后面必须构造“有精确参数真值”的 instruction-edit pairs，训练一个参数校准器

## 我认为最合理的下一阶段架构

不要再让 LLM 直接输出最终数值。后面应该拆成 4 层。

### Layer 1. Semantic Planner

输出：

- `region`
- `effect_family`
- `shape`
- `direction`
- `duration class`
- `strength class`

这一层你们已经做得基本对了。

### Layer 2. Dimensionless Edit Spec

新增一个无量纲编辑表示层，不直接输出原始单位值。

建议字段至少包括：

- `delta_level_z`
  - 相对局部标准差的水平变化
- `slope_ratio`
  - 相对局部趋势强度的变化倍数
- `amp_ratio`
  - 相对局部振幅的放大/缩小倍数
- `vol_ratio`
  - 相对局部波动强度倍数
- `duration_ratio`
  - 占目标窗口长度的比例
- `floor_ratio`
  - 对 `flatline/shutdown` 类任务，目标下压到局部分位点的程度

这一层是后面最关键的新中间层。

### Layer 3. Numeric Calibrator

输入：

- 原始局部窗口统计量
- planner 输出的 intent
- dimensionless edit spec
- 可选的外部上下文文本

输出：

- `math_shift`
- `level_shift`
- `amplitude`
- `width`
- `target attrs`

这个 calibrator 才应该真正负责“量纲落地”。

它可以有三种形态：

- 规则版
  - 先用局部均值、标准差、分位点手工映射
- 轻量学习版
  - 一个小 MLP/Transformer head
- 联合学习版
  - 和编辑器一起训练

### Layer 4. Executor

执行器继续允许两条路线并存：

- `TEdit-backed`
- `math-backed`

这不是妥协，而是必要设计：

- TEdit 适合属性编辑
- 数学工具适合可精确控制的局部量级变化

## 最值得优先做的破局点

### 方向一：先定义“无量纲编辑 DSL”

这是我认为最关键的一步。

如果不先把编辑规格从“原始单位”改写到“局部归一化单位”，后面所有学习都会混量纲。

一句话说：

- 先学 `relative edit`
- 再做 `unitful de-normalization`

### 方向二：单独训练一个 Parameter Calibrator

不要让 LLM 继续输出精确数值。

更合理的是：

1. LLM 输出 `intent + region + strength class`
2. calibrator 根据局部统计量输出具体参数
3. executor 执行

训练数据正好可以来自你们已有的合成 benchmark：

- 注入器天然知道 GT 变化幅度
- region 也有 GT
- tool type 也可以有 GT

这正好就是监督信号。

### 方向三：把评测拆出“数值校准层”

建议新增：

- `normalized_parameter_error`
  - 在无量纲空间比较 GT 和预测参数
- `tool_parameter_error`
  - 每个工具自己的参数误差
- `forecast_impact_error`
  - 如果任务面向 future correction，则比较编辑前后预测修正是否接近期望

否则你们后面很难知道：

- 是 region 错了
- 是 tool 选错了
- 还是数值调轻/调重了

## 如果只做一个新创新点，我会选什么

如果论文后面只想突出一个真正像样的创新点，我建议选：

**Instruction-to-Parameter Calibration for Time Series Editing**

核心思想：

- 不直接从 instruction 到编辑器
- 中间显式引入 `dimensionless edit spec`
- 再由 calibrator 做跨量纲落地

这个点比单纯再改 prompt 或再加几个工具更扎实，也更像真正的研究贡献。

## 更实际的落地顺序

### 阶段 1

- 冻结当前 `intent + region` pipeline
- 定义无量纲参数 schema
- 让生成器输出 GT numeric specs

### 阶段 2

- 先做 rule-based calibrator
- 跑 benchmark，确认是否比 LLM 直接给数值更稳

### 阶段 3

- 训练一个 small calibrator
- 对比：
  - `LLM direct numeric`
  - `rule calibrator`
  - `learned calibrator`

### 阶段 4

- 再考虑 end-to-end instruction-based editor
- 或把 TEdit 替换/增强为更适合 instruction editing 的后端

## 结论

当前阶段最值得继续投入的，不是再提高 `intent` 理解，也不是马上扩一大堆工具。

真正的破局点是：

**把“语言意图”与“连续数值执行”之间补上一层可监督、可泛化、跨量纲稳定的参数校准模块。**

如果这层打通了，你们后面的故事会非常完整：

- LLM 负责理解复杂文本
- localizer 负责时间边界
- calibrator 负责数值落地
- TEdit / 数学工具负责执行

这条链路比现在直接让 LLM 决定一切更稳，也更有论文价值。
