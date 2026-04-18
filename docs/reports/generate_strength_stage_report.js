const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,
  Header,
  Footer,
  AlignmentType,
  HeadingLevel,
  WidthType,
  BorderStyle,
  ShadingType,
  PageNumber,
  LevelFormat,
  PageBreak,
} = require("docx");

const outputPath = "/root/autodl-tmp/BetterTSE-main/tmp/strength_stage_report_20260416.docx";
const imagePath = "/root/autodl-tmp/BetterTSE-main/tmp/strength_diffusion_framework_20260416.png";

const border = { style: BorderStyle.SINGLE, size: 1, color: "C7CDD3" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 300 },
    ...opts,
    children: [new TextRun({ text, font: "Arial", size: 23, ...(opts.bold ? { bold: true } : {}) })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 300 },
    children: [new TextRun({ text, font: "Arial", size: 23 })],
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80, line: 300 },
    children: [new TextRun({ text, font: "Arial", size: 23 })],
  });
}

function cell(text, width, fill = "FFFFFF", bold = false) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      spacing: { after: 60, line: 280 },
      children: [new TextRun({ text, font: "Arial", size: 22, bold })],
    })],
  });
}

function simpleTable(rows, columnWidths) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths,
    rows: rows.map((row, rowIdx) => new TableRow({
      children: row.map((text, idx) => cell(text, columnWidths[idx], rowIdx === 0 ? "D9E7F2" : "FFFFFF", rowIdx === 0)),
    })),
  });
}

const doc = new Document({
  creator: "OpenAI Codex",
  title: "Strength Control Stage Report",
  description: "BetterTSE strength-control stage experiment report",
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 260, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 25, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [new TextRun({ text: "BetterTSE Strength Stage Report", font: "Arial", size: 18, color: "666666" })],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Page ", font: "Arial", size: 18, color: "666666" }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "666666" }),
              ],
            }),
          ],
        }),
      },
      children: [
        new Paragraph({
          heading: HeadingLevel.TITLE,
          alignment: AlignmentType.CENTER,
          spacing: { after: 220 },
          children: [new TextRun({ text: "BetterTSE 强度控制阶段实验汇报", bold: true, font: "Arial", size: 36 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
          children: [new TextRun({ text: "基于 2026-04-15 至 2026-04-16 本地代码与实验记录", font: "Arial", size: 24, color: "666666" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 260 },
          children: [new TextRun({ text: "用途：阶段性向老师汇报 strength 主线方法、诊断结果与下一步重点", font: "Arial", size: 24, color: "666666" })],
        }),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("一、这条实验线在解决什么问题")] }),
        p("目前这条 strength 线，不是在解决“模型会不会编辑”这个总问题，而是在解决更窄、更硬的一件事：在编辑类型、编辑区域、编辑方向都固定的前提下，只改变“改得轻一点、中等一点、明显一点”，模型能不能稳定地产生不同强弱的编辑结果。"),
        p("我们要求的不只是能改，还要求满足顺序关系：weak 改得最少，medium 居中，strong 改得最多，同时非编辑区尽量不要被带坏。最近两天的工作重点已经不是继续搭通道，而是定位：为什么强度信号已经进了模型内部，但最终输出还经常表现成平或者反。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("二、在整个 pipeline 开头，文本里的强度信息是怎样被读出来并变成模型可用参数的")] }),
        p("这一步很适合单独讲清楚，因为导师如果不看代码，很容易误以为我们只是把“轻微、明显”几个词原样喂给模型。实际上当前 pipeline 开头已经把文本里的强度信息拆成了两类：一类是人能直接看懂的文字强度表达，另一类是模型内部真正使用的强度标签和强度数值。"),
        numbered("首先，benchmark builder 会先写出一条自然语言指令。例如同一类局部趋势编辑，会生成“轻微抬升”“中等抬升”“明显抬升”这类文本。这里的“轻微 / 中等 / 明显”是给人看的强度描述。"),
        numbered("然后 builder 不只保存文字，还会同时保存两个结构化字段。第一个叫 `strength_label`，它表示离散档位，也就是 weak、medium、strong。第二个叫 `strength_scalar`，它表示连续强度值，当前主线固定成 weak=0.0、medium=0.5、strong=1.0。"),
        numbered("因此，文本里的“轻微”并不会在训练时以一种模糊方式存在，而是会被明确地对应到：`strength_label=weak`，`strength_scalar=0.0`。同理，“中等”对应 medium 和 0.5，“明显”对应 strong 和 1.0。"),
        numbered("进入训练或评估时，`tool/tedit_wrapper.py` 和下游模型调用不会只拿一段纯文本，而是会把 `instruction_text`、`strength_label`、`strength_scalar` 一起打包进 batch。这样模型既能看到自然语言表述，也能直接拿到明确的强度控制参数。"),
        numbered("到模型内部后，`StrengthProjector` 会分别处理这些信息：离散标签走 embedding，连续数值走小型感知机，文本走轻量词嵌入再做平均池化。最后三者被合成一个统一的强度条件向量，再继续注入扩散模型。"),
        p("所以如果用一句最容易懂的话说，我们现在在 pipeline 开头做的事情，不是“让模型自己猜轻微和明显差多少”，而是“先把文本里的强度信息翻译成结构化强度参数，再把文字和参数一起交给模型”。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("三、当前方法总览")] }),
        p("当前方法可以概括为四步。第一步，用一个已经具备基础合成编辑能力的 TEdit 预训练模型做底座，而不是从零训练整个扩散模型。第二步，单独构造一个 family 数据集，让同一组样本里只有强度变化，其他条件都固定。第三步，把强度作为显式控制信号注入扩散模型内部，而不是只把“轻微、明显”放在文字里。第四步，在训练目标里把强度监督抬成主约束，直接约束输出幅度、顺序和档位差距。"),
        p("也就是说，这条线不是 prompt 工程，也不是简单地在旧模型上轻微微调，而是一条“预训练底座 + family 强度数据 + 显式条件注入 + 强度主约束损失”的完整训练线。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 起始权重与微调基座")] }),
        p("当前 strength 主线最早的起点，是 TEdit 的合成预训练权重。标准底座是 `TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth`。这个权重原本学到的是基础时间序列编辑能力，不是专门为强度控制训练的。"),
        p("之后我们在这个底座上做 strength family 微调。再往后的 `semantic split`、`beta_only_repair`、`beta_direction_pass2` 等实验，不一定都从最原始的合成预训练权重起步，而是常常从上一轮 strength family checkpoint 继续做更细的诊断与修补。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.2 当前训练方法到底是什么")] }),
        p("严格地说，当前不是重新训练一套新模型，而是在预训练 TEdit 底座上，使用 `discrete_strength_family` 数据集做有监督微调。训练入口是 `TEdit-main/run_finetune.py`，训练器在 `TEdit-main/train/finetuner.py`。默认超参数是 epochs=10、batch_size=4、lr=1e-5。"),
        p("和旧式“只看扩散重建误差”的训练不同，当前训练把强度控制相关的损失放到了主位置，因此整个训练方向已经从“保留基本编辑能力”转成“逼模型学会按要求控制编辑幅度”。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("四、训练数据与 benchmark 设计")] }),
        p("当前 strength 主线用的是专门的 family 数据集，而不是普通混合编辑数据。配置里指定的数据集类型是 `discrete_strength_family`，数据目录默认为 `./datasets/discrete_strength_trend_family`。"),
        p("family 的核心思想是：在同一个 family 内，原始序列、编辑区域、编辑方向、持续时长、指令模板和操作类型全部固定，只允许强度变化。这样模型如果出现差异，就更有把握说明它真的在学“强度”，而不是在混学别的变量。"),
        p("这里的几个名字需要单独解释一下。`source_ts` 是原始时间序列，也就是还没被编辑的输入；`target_ts` 是目标时间序列，也就是该强度下我们希望模型改出来的结果；`mask_gt` 是可编辑区域的真值掩码，告诉我们“哪里允许改、哪里不该改”；`instruction_text` 是给模型看的自然语言指令；`strength_label` 和 `strength_scalar` 则是从同一条指令中抽出来的结构化强度信息。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1 当前主线用的是哪一种操作")] }),
        p("当前这条线的主训练数据只聚焦一种最干净的操作：trend_injection，也就是局部趋势抬升或压低。这样做的目的，是先把“改多大”这个问题单独拿出来，在最可控的设置里判断模型是否学会强度轴，而不是一开始把多种编辑形态混在一起。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2 强度档位和数值锚点")] }),
        p("目前 family 数据集固定了三档强度锚点：weak=0.0、medium=0.5、strong=1.0。这里既有文字档位，也有连续数值。文字档位方便按自然语言表达指令，数值锚点方便模型学习一条可插值的强度轴。"),
        p("对 trend_injection 来说，三档强度对应的目标幅度参数是：weak 的 amplitude_ratio=0.12，medium=0.24，strong=0.36。也就是说，目标序列本身就是按更大的编辑幅度构造出来的，不是只在标签上说“更强”。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3 数据规模与划分")] }),
        p("当前 family builder 的默认设置是：seq_len=192，train_families=96，valid_families=24，test_families=24。每个 family 会生成 weak、medium、strong 三个样本，因此数据的最核心结构不是单样本，而是三档成组的 family。"),

        simpleTable(
          [
            ["项目", "当前主线设置"],
            ["样本组织方式", "family：同源、同区域、同模板，只变强度"],
            ["主操作族", "trend_injection"],
            ["强度锚点", "weak=0.0, medium=0.5, strong=1.0"],
            ["trend 幅度参数", "0.12 / 0.24 / 0.36"],
            ["默认序列长度", "192"],
            ["默认 family 划分", "train 96 / valid 24 / test 24"],
          ],
          [3000, 6360]
        ),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.4 给导师看的一个具体样例")] }),
        p("为了更直观，可以把一个 family 想成下面这样一组三个样本。三者用的是同一条原始序列、同一个局部编辑区域、同一种趋势抬升任务，只是强度不同。"),
        simpleTable(
          [
            ["字段", "weak 样本", "medium 样本", "strong 样本"],
            ["instruction_text", "请在这段区域做轻微抬升", "请在这段区域做中等抬升", "请在这段区域做明显抬升"],
            ["strength_label", "weak", "medium", "strong"],
            ["strength_scalar", "0.0", "0.5", "1.0"],
            ["source_ts", "同一条原始序列", "同一条原始序列", "同一条原始序列"],
            ["mask_gt", "同一段可编辑区域", "同一段可编辑区域", "同一段可编辑区域"],
            ["target_ts", "轻微抬升后的目标序列", "中等抬升后的目标序列", "明显抬升后的目标序列"],
            ["trend amplitude_ratio", "0.12", "0.24", "0.36"],
          ],
          [1800, 2520, 2520, 2520]
        ),
        p("这个样例的关键点是：除了强度，其他条件都不变。导师只要抓住这一点，就能理解为什么我们后面会用“单调性”和“强度差距”去监督模型。因为这三个样本天然就应该满足：weak 改得最少，strong 改得最多。"),
        p("换句话说，family 设计本身就在告诉模型：这里真正要学的是“同一种编辑任务里的强弱关系”，而不是“换一种区域、换一种模板、换一种任务”。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("五、强度信号如何进入模型")] }),
        p("当前方法不是只把强度写进文字指令，而是有一条单独的强度条件通路。输入给模型的强度相关信息有三种：离散档位 strength_label、连续数值 strength_scalar，以及文本 instruction_text。"),
        p("这三类信息会先进入 `StrengthProjector`。它内部大致做的是：离散标签走 embedding，数值强度走一个小型多层感知机，文本走轻量词嵌入加平均池化，然后把这些信息拼起来，再映射成一个统一的强度条件向量。当前配置里，num_strength_bins=3，emb_dim=32，hidden_dim=64，out_dim=64，use_text_context=true，use_task_id=false。"),
        p("这里有一个很重要的边界：当前 strength 主线已经不再依赖 task_id 作为核心条件，而是明确强调 strength_label、strength_scalar 和文本本身。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 注入到 diffusion 模型的哪个位置")] }),
        p("强度向量不会只在输入端用一下，而是继续进入扩散模型内部的调制路径。当前模式叫 `modulation_residual`。可以把它理解成：模型原本就有一套基础调制参数，而强度条件会额外产生一组“增量调制量”，再去修改原来的调制方式。"),
        p("代码里最重要的两个量是 delta_gamma 和 delta_beta。它们是强度条件产生的调制增量，会加到基础调制上，从而影响扩散网络内部的特征流。当前默认还会乘一个 `strength_gain_multiplier=4.0`，目的是防止 strength 分支太弱，被原有主干完全淹没。"),
        p("如果传入了 strength_scalar，这两个增量还会按数值强度进一步缩放。也就是说，当前模型里“强度”既是离散桶，也是连续控制轴。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.2 当前 diffusion 框架示意图")] }),
        p("下面这张图不是逐层张量图，而是为了汇报当前主线而画的结构示意图，重点标出：family 数据、强度编码、扩散骨干、输出，以及训练时真正起作用的损失约束。"),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 160 },
          children: [
            new ImageRun({
              type: "png",
              data: fs.readFileSync(imagePath),
              transformation: { width: 620, height: 379 },
              altText: {
                title: "Strength diffusion framework",
                description: "Current schematic of BetterTSE strength-control diffusion framework",
                name: "strength_diffusion_framework",
              },
            }),
          ],
        }),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("六、我们训练的是模型的哪一部分")] }),
        p("默认主线不是“只训练 projector”。当前配置里 `freeze_backbone_for_strength=false`，也就是说默认并不冻结整个扩散主干。更准确地说，主线默认训练整套相关路径，但会特别提高强度相关参数的学习率。"),
        p("训练器会把参数分成普通参数组和强度参数组。强度参数主要包括 `strength_projector`、`strength_modulation`、`strength_input_projection` 等模块。它们使用 `strength_lr_scale=5.0`，也就是学习率放大五倍。"),
        p("为什么这么做？因为我们不是从零训练，而是从已经会编辑的模型起步。新加的 strength 分支如果仍用和主干完全一样的小学习率，常常来不及学到可见的控制能力。"),
        p("最近两天也做了很多局部诊断变体，包括：冻结主干只让强度路径更自由、只做 beta 路径修复、加大 strength gain multiplier、做 label-only 或 both 的条件拆分。这些实验不是主线训练配置，但对定位故障很重要。"),

        simpleTable(
          [
            ["组件或策略", "当前主线", "最近两天的诊断变体"],
            ["预训练底座", "TEdit synthetic pretrain", "同底座或强度 checkpoint 继续跑"],
            ["扩散主干", "默认参与训练", "有冻结主干变体"],
            ["StrengthProjector", "参与训练", "参与训练"],
            ["strength_modulation / input_projection", "参与训练", "参与训练"],
            ["学习率", "主干 lr=1e-5，strength 组放大 5 倍", "有 gainmult4、beta_only_repair 等变体"],
          ],
          [2600, 2900, 3860]
        ),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("七、当前损失目标构成：强度监督已经是主约束")] }),
        p("这条线最关键的变化在训练目标。现在不再是把强度当成一个很弱的辅助信号，而是明确把强度监督提到了主位置。传统扩散损失还保留着，但权重已经明显让位给幅度和顺序相关约束。"),

        simpleTable(
          [
            ["损失项", "当前权重", "直白解释"],
            ["diffusion_loss", "0.2", "保住模型原本的基本编辑能力，不让整体生成质量彻底坏掉"],
            ["edit_region_loss", "0.5", "编辑区要贴近目标，该改的地方真改过去"],
            ["background_loss", "0.2", "不该改的背景尽量不要动"],
            ["gain_match_loss", "4.0", "实际改动幅度要接近目标要求的幅度"],
            ["monotonic_loss", "0.5", "更高强度不能比更低强度改得更少"],
            ["family_relative_gain_loss", "3.0", "不仅顺序要对，档位之间的间隔也不能塌掉"],
            ["constant_gain_penalty", "2.0", "惩罚 weak/medium/strong 三档几乎一样"],
            ["numeric_only_loss", "4.0", "就算只给数值强度，也得学会控制幅度"],
            ["beta_direction_loss", "默认 0.0，近期实验开到 1.0", "针对最近发现的“方向学反了”问题新增的修补项"],
          ],
          [2550, 1750, 5060]
        ),
        p("从这些权重可以看得很清楚：现在主导训练方向的，不是传统扩散重建，而是“编辑区是否改对、背景是否稳、输出幅度是否对、顺序是否对、三档是否真有差别”。你如果把当前训练概括成“强度监督已经成为主约束”，这是忠于代码和配置的。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("7.1 几个容易混淆的符号解释")] }),
        simpleTable(
          [
            ["符号或字段", "实际含义"],
            ["source_ts", "原始时间序列，模型要在它的基础上做编辑"],
            ["target_ts", "该强度档位下希望得到的目标序列"],
            ["mask_gt", "可编辑区域的真值掩码，只表示哪里该改"],
            ["strength_label", "离散强度档位，例如 weak / medium / strong"],
            ["strength_scalar", "连续强度值，目前主线锚点是 0.0 / 0.5 / 1.0"],
            ["edit gain", "输出与原序列在编辑区上的平均绝对差，可理解为“实际改了多大”"],
            ["target edit gain", "目标序列相对原序列在编辑区上的平均绝对差，可理解为“应该改多大”"],
            ["delta_gamma / delta_beta", "强度条件注入扩散网络后形成的两组调制增量"],
          ],
          [2600, 6760]
        ),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("八、一次训练 batch 是怎么走完的")] }),
        p("下面按照一次实际训练 batch 的逻辑，把这条线从数据读入讲到参数更新。这里不省略中间环节，因为它正好能把当前方法的关键点全部串起来。"),
        numbered("数据加载器从 `discrete_strength_family` 中取出一个 batch。每个样本都包含原始序列、目标序列、编辑区域、强度标签、强度数值和指令文本。因为数据按 family 组织，所以同一组样本之间的强度是可比较的。"),
        numbered("训练器把这些字段整理成模型可读的 batch。除了常规的源序列和目标序列，还会把 `strength_label`、`strength_scalar`、`instruction_text` 一并送下去。"),
        numbered("扩散模型内部先走原来的基础编辑路径，同时调用 `StrengthProjector` 把强度条件编码成一个统一向量。"),
        numbered("这个向量进入内部调制模块，产生 delta_gamma 和 delta_beta，再加到基础调制上，从而改变扩散网络在不同强度下的内部响应。"),
        numbered("模型输出当前 batch 的编辑结果。训练时不仅看输出像不像目标，还会额外计算编辑区的实际改动幅度、背景扰动，以及同一 family 内的顺序关系。"),
        numbered("`ConditionalGenerator` 里把多项强度监督损失加总：编辑区损失、背景损失、幅度匹配、单调性、family 相对差距、常量惩罚、numeric-only 约束，以及在近期实验里可选的 beta 方向损失。"),
        numbered("训练器按参数组回传梯度。强度相关参数组会用更高的学习率更新，因此 projector 和调制分支会比普通主干更积极地适应新任务。"),
        numbered("如果开启了诊断，训练器还会定期记录 strength diagnostics，例如 projector 是否有区分、调制增量是否有区分、generator 层面实际 edit gain 是否随着强度变化。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("九、评估与诊断：我们现在到底看哪些指标")] }),
        p("当前强度线不是只看一个平均误差，而是把“内部是否有信号”和“最终输出是否对”分开看。最近两天 Claude 的记录里，诊断已经分成至少三层：projector 层、调制层、最终输出层。"),
        p("对外层面最关键的是强度效果评估。核心字段包括：`edit_gain_mean`、`target_mae_edit_region_mean`、`bg_mae_mean`、`monotonic_hit_rate`、`strong_minus_weak_edit_gain_mean` 和 `family_spearman_rho_strength_gain_mean`。"),
        simpleTable(
          [
            ["字段", "怎么看"],
            ["edit_gain_mean", "模型实际改了多大，值越大表示编辑幅度越大"],
            ["target_mae_edit_region_mean", "编辑区离目标还有多远，越小越好"],
            ["bg_mae_mean", "背景区被带坏了多少，越小越好"],
            ["monotonic_hit_rate", "有多少 family 真正满足 weak < medium < strong"],
            ["strong_minus_weak_edit_gain_mean", "strong 平均比 weak 多改了多少；负数就说明方向反了"],
            ["family_spearman_rho_strength_gain_mean", "强度和输出幅度整体相关性；正数说明方向对，负数说明方向反"],
            ["projector_pairwise_l2 / modulation pairwise l2", "不同强度在内部表示上是否真的有差别"],
          ],
          [3200, 6160]
        ),
        p("最近还增加了一组很重要的分层诊断：`raw_edit_region_mean_abs_delta`、`final_edit_region_mean_abs_delta`、`raw_final_gap_edit_region_mean`、`preservation_attenuation_ratio_mean`。这组字段用来区分：问题到底出在一开始就没拉开，还是中间有差异，最后又被某条路径压平了。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("十、最近两天 strength 诊断时间线与结果解释")] }),
        p("下面的时间线只聚焦 strength 这条线，不混 forecast revision。核心原则是：每一轮实验都回答一个更窄的问题。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.1 4 月 15 日白天：Phase-1 ablation，先问“是不是文本或主干压住了强度”")] }),
        p("这轮主要看 `strength_gainmult4_cuda0_validation`。试了 baseline、冻结主干、文本 dropout、冻结主干再文本 dropout。结果没有支持“只是文本盖住了数值轴”这种简单解释。"),
        bullet("四个版本的 both_rho 和 label_rho 都是 -0.8333。"),
        bullet("baseline 的 monotonic_adj 只有 0.0833，冻结主干版本甚至到 0。"),
        bullet("也就是说，改动这些表面因素，并没有把强度顺序修正过来。"),
        p("这轮实验的意义是先排除几种最直观的解释：不是简单的文本干扰，也不是简单的主干过强。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.2 4 月 15 日晚上：semantic split，开始怀疑“方向可能学反了”")] }),
        p("`strength_second_gpu_validation_semantic_split` 这一轮把条件拆成了 both、label_only，并做正向和反向 scalar probe。这里出现了两种非常关键的现象。"),
        bullet("早期的某些结果是完全平的。也就是 weak、medium、strong 三档 `edit_gain_mean` 基本一样，模型看起来像根本不认强度。"),
        bullet("rerun 后，结果不再完全平，而是变成明显反向。`probe_forward_label_only_rerun` 中，0.0000 / 0.5000 / 1.0000 的 edit_gain_mean 分别约为 12.2296 / 12.2187 / 12.2084。"),
        bullet("对应地，strong_minus_weak_edit_gain_mean = -0.02124，family_spearman_rho_strength_gain_mean = -0.5，monotonic_hit_rate = 0.25。"),
        p("这组数字的直觉解释很简单：模型不是完全没有响应，而是“越强反而改得越少”。这比“完全没学到”更具体，也更危险，因为它说明模型有可能把强度轴的方向学反了。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.3 反向 scalar probe：目前最强的一条证据")] }),
        p("在 `probe_reversed_label_only_rerun` 里，我们故意把数值强度反过来喂：原本 weak 对应 0.0、strong 对应 1.0，现在把它交换。结果整体趋势会一起翻转。"),
        bullet("反向之后，0.0000 / 0.5000 / 1.0000 的 edit_gain_mean 变成约 12.2087 / 12.2187 / 12.2296。"),
        bullet("对应 strong_minus_weak_edit_gain_mean 变成 +0.02097。"),
        bullet("family_spearman_rho_strength_gain_mean 从 -0.5 翻到 +0.5。"),
        bullet("monotonic_hit_rate 从 0.25 提到 0.75。"),
        p("这一轮的解释非常关键：模型不是不用这条强度轴，而是更像把“数值更大”理解成“改得更弱”。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.4 beta flip probe：当前最可疑的路径集中在 beta 这一侧")] }),
        p("接下来又做了一轮更窄的 probe：不改训练权重，只在推理时把 beta 路径的符号翻掉，看输出会不会跟着变。结果是会变。"),
        bullet("不翻 beta 时，raw_edit_region_mean_abs_delta 随强度下降，约为 32.8710 / 32.8499 / 32.8297。"),
        bullet("只翻 beta 符号后，raw_edit_region_mean_abs_delta 变成上升，约为 32.8710 / 32.8774 / 32.8834。"),
        bullet("与此同时，调制层内部并不是没信号。以 baseline probe 为例，modulation_delta_gamma_pairwise_l2 的 0.0000_1.0000 约为 0.1721，modulation_delta_beta_pairwise_l2 的 0.0000_1.0000 约为 0.1155。"),
        p("这说明问题已经不是“强度没进模型”。内部调制差异是存在的，真正不对的是它到最终输出的方向映射，而 beta 路径目前最像是主要嫌疑点。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.5 beta_only_repair 与 Pass 2A：修复已经开始，但还没通过主评测")] }),
        p("在 `beta_only_repair_short` 和 `beta_only_repair_longer` 里，我们继续盯 beta 路径做更窄的修补。结果和 beta flip probe 一致：不翻时依然反向，翻了以后局部 probe 会转正。这说明现象不是偶然 seed。"),
        p("之后又把 `beta_direction_loss` 接入训练入口，做了 `beta_direction_pass2_w1`。但结果仍然保守。Gate 2 的强度效果里，edit_gain_mean 仍是 12.2296 / 12.2241 / 12.2188，strong_minus_weak_edit_gain_mean 仍是 -0.01084，family_spearman_rho_strength_gain_mean 仍是 -0.5，monotonic_hit_rate 仍只有 0.25。"),
        p("因此最近两天最准确的结论是：方向性问题已经被定位得很窄，但正式修复还没有完成。`pass2a_conclusion.md` 也明确写了，当前还不能声称 weak→strong 的单调性已经修好，更不能声称后续已经不需要 beta flip。"),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("10.6 provenance repair：这不是性能修复，而是结果可信度修复")] }),
        p("最近还做了 `provenance_smoke`。它的作用不是提性能，而是把每次运行真正生效的配置固定下来，写入 `resolved_runtime_config.json`。原因是 strength 诊断已经细到：如果运行时配置有漂移，就会直接影响我们对结果的解释。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("十一、当前阶段最稳妥的结论")] }),
        bullet("第一，family 数据集和 benchmark 是健康的，问题不在数据定义混乱。"),
        bullet("第二，强度通道已经真实进入模型内部，不再处于“只是设计草图”的阶段。"),
        bullet("第三，最近最强的证据不是“完全没反应”，而是“方向可能学反了”。"),
        bullet("第四，内部调制信号可见，说明主要故障更像发生在 beta 路径及其到最终输出的映射。"),
        bullet("第五，正式主评测还没有显示 monotonicity 已经稳定修复，因此当前不能宣称 strength 控制已经成立。"),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("十二、下一阶段最该继续做什么")] }),
        p("如果只从最近两天的代码和结果出发，不发散讲别的方向，那么下一阶段最值得继续推进的不是再扩数据，也不是再增加新的自然语言模板，而是继续沿“方向性修复”这条链往下做。"),
        bullet("优先确认 beta 路径究竟是在什么位置把方向带反，是调制本身的符号问题，还是输出头对该调制的解释方向不对。"),
        bullet("把当前已经存在的 probe 与正式 Gate 2 评测更紧地对齐，避免出现“局部 probe 看起来修好了，但主评测没跟上”的情况。"),
        bullet("继续保留 provenance 固化，所有后续结论都应该锚定到 `resolved_runtime_config.json`，而不是旧的 YAML 快照。"),
        bullet("在修方向之前，不建议过早扩展到更复杂的编辑族，否则会把核心问题重新埋回噪声里。"),

        new Paragraph({ children: [new PageBreak()] }),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("附录 A：当前主线关键配置一页表")] }),
        simpleTable(
          [
            ["类别", "当前主线值"],
            ["base checkpoint", "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"],
            ["dataset type", "discrete_strength_family"],
            ["main operation", "trend_injection"],
            ["strength anchors", "weak=0.0, medium=0.5, strong=1.0"],
            ["optimizer base lr", "1e-5"],
            ["strength lr scale", "5.0"],
            ["freeze backbone", "false"],
            ["epochs / batch size", "10 / 4"],
            ["strength control mode", "modulation_residual"],
            ["projector dims", "emb 32 / hidden 64 / out 64"],
            ["gain multiplier", "4.0"],
          ],
          [3300, 6060]
        ),

        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("附录 B：汇报时可以直接说的一段总括")] }),
        p("目前 BetterTSE 的 strength 主线已经从“尝试把强度塞进模型”推进到了“强度信号已经进入扩散模型内部，但最终输出还没有形成稳定、单调、可验证的幅度控制”。在方法上，我们是以 TEdit 的合成预训练权重为底座，使用只改变强度的 family 数据集做微调，把离散档位、连续数值和文本条件一起编码成强度向量，再通过内部调制残差注入扩散骨干；训练时也不再以传统扩散误差为主，而是用幅度匹配、单调性、family 间隔保持和 numeric-only 一致性等强度监督作为主要约束。最近两天的诊断最重要的发现是：内部调制层已经能看到强度差异，但最终输出经常平掉或方向反转，因此当前最可疑的部位已经缩小到 beta 路径及其输出映射上。"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
