# CEPT Transport：可证伪实验计划

## 研究假设

不再验证“response history 会传播”。要验证的是：在控制每个 target、relation、layer、head 的 attention 质量后，**真实 source-event incidence 是否能预测 evidence effect 经哪些历史 token 传递**。如果只保留关系质量或重连端点也同样好，图结构没有贡献，图建模主张失败。

## 无标签 intervention teacher

对 evidence 中表面长度不变的数值片段构造 factual/counterfactual 输入，并记录目标 token log-prob。Teacher 产生：

- total evidence effect；
- 把改变过的 evidence K/V 补回 counterfactual receiver 的 `direct_seed_rescue`；该量允许影响经更早 response 表征继续传播，因此是“由 seed 启动的总递归支持”，不是自然直接效应；
- 在同一个 counterfactual receiver 上补回 factual response-history K/V 得到的 `alternate_history_effect = Y01-Y00`；
- 同时补回 changed-evidence 与 response-history K/V 的 `joint_seed_history_rescue`；
- `representation_residual = total_effect - joint_seed_history_rescue` 和 `seed_history_interaction = joint - seed - history`，用于识别图变量覆盖不足及不可加交互；
- 分块 response-history K/V rescue，用于同一事件内的 block 排序；
- self-patch error，作为干预数值误差的经验 null。

目标保留 effect 符号。训练 source 上的 self-patch error q95 给出 null floor，正 total effect q90 给出尺度。只有 `|TE| <= floor` 是 null；`TE < -floor` 的方向冲突事件单独计数并从当前单方向目标排除，不能伪装成零支持。可靠正例与 null 分别求均值后等权组合，避免大量零 target 淹没正效应。正 TE 若无法由 seed/history 图变量表示也排除并报告，而不是强造错误监督。Block effect 独立标定并做同事件 pairwise ranking，不假定各块效应可加或和为 1。

## Attention-only transport student

Student 只读取 canonical prediction-event graph：`EY/QY/RY`、完整 layer/head attention trace、真实 source/target incidence、discarded attention 的 unknown mass。禁止 token ID embedding、hidden state、logits、entropy、绝对位置、回答长度、GMM 和通用 GNN/MLP。

模型学习 `3 × layer × head` 个非负 surrogate gate，以及每层一个 `[0,1]` operational residual-persistence 系数。它不是 residual stream 的物理分解；`no_residual` 必须作为对照。状态在 `token × transformer layer` 上展开：预测事件 `t` 使用 predictor row `t-1`，response 位置 `p` 作为 K/V source 时读取其上一层状态，diagonal attention 因而是合法的跨层 self-state 边，而不是伪造的前一 token 边。

```text
direct_t  = gated changed-evidence -> Y_t
history_t = sum_{s<t} gated R_s -> Y_t * support_s
support_t = direct_t + history_t
```

Teacher 的抽样母体必须先由 verified train attention-cache 的精确文件集合生成，绑定 dataset hash、cache-manifest hash、generator/task selector、response ID 清单及其 SHA-256；禁止从全体多-generator RAGTruth 抽样后在 graph join 阶段静默丢弃。训练与部署使用同一个 seed operator：第一个合法的 equal-token numeric counterfactual 实际改变的位置。无候选样本明确记为不可评分，不再把训练时的稀疏 numeric seed 外推成“所有 evidence 都是 seed”。因此当前结论只覆盖该 cache sampling frame 与 numeric-intervention-eligible/scorable 子集的交集，而不是全体 RAGTruth。

旧 cache 没有第一个 response token 所需的 prompt-final predictor row。该 event 保留 `row_available=false`，但从主要 token/response 指标中排除，并显式报告 coverage。

## 主分数

主 token 风险不是泛化的 `1 - support`，而是模型条件下 retained、gated、layer-averaged response-history exposure 的 operational interval 下端：

```text
unsupported_history_lower_t
  = observed_RY_fraction_t - evidence_supported_RY_upper_t
```

discarded attention 无法确定 relation，因此按 evidence-favourable 方式进入 operational interval；旧 cache 缺失的 prompt rows 则在第一层后对因果可达非 seed prompt state 使用 `[0,1]`。这些不是完整 Transformer K/V 的数学严格界，因为 MLP、LayerNorm 和 residual amplitude 未被观测。`overall_evidence_deficit_lower = 1 - support_upper` 仅作辅助分析。Response 分数预注册为可评分 token 风险最高 10% 的均值。

## Label-free Gate 2

按 `source_id` 建立 train、checkpoint-selection validation、untouched mechanism holdout 三个互不重叠分区，对五个预注册变体分别训练、冻结和评分：

- `true`：真实 incidence 和完整递归；
- `rewired`：保留 target/relation/channel/value，并在 target × log-lag bucket 内置换 RY source assignment；
- `mass_only`：保留 relation/channel mass，删除真实 source incidence；
- `one_hop`：保留一次从 direct evidence 支持出发的 RR hop，移除多步递归。
- `no_residual`：保留真实 incidence，但把 operational residual persistence 固定为 0。

checkpoint 只由 selection validation 选择。在至少 20 个 source 的 untouched mechanism holdout 上按 source 计算 paired loss，并用 source-group bootstrap 得到 `control loss - true loss` 的 95% CI。真实 incidence 必须同时优于 rewired、mass-only、one-hop、no-residual、未训练 gate 和 constant-zero。Gate 还要求可靠正 target 的 source/event 数、support/history target 方差、support/history 的 source-cluster bootstrap Spearman 下界大于零、同一预测事件内 history-block pair ranking accuracy 的 bootstrap 下界大于 0.5、rewired 实际改变比例，以及部署分数的 coverage/非退化/区间宽度/低 nuisance correlation。所有 informativeness 判决都在 train-side mechanism holdout 完成；official test 诊断不参与开标签决策。

## 冻结后评估

Gate 2 通过后，才读取官方 `y_token` 并评估所有预注册变体。报告 numeric-eligible/scorable 子集上的 token/response AUROC、AUPRC、coverage，并对 `true-control` 的 AUROC/AUPRC 差异做 source-cluster paired bootstrap。检测期另设 `detection_claim`：token 与 response 上都必须满足最小独立 source、正负 source 和正负样本数；`true-control` 的 AUROC/AUPRC 差值 CI 必须同时优于五个结构对照。两个 endpoint 都必须优于 response length、evidence/query token count、deployment seed count 和 response-level unknown attention mass；token 端还必须优于 normalized position 与 token-level unknown mass。每个 nuisance 同时预注册正负方向，不能看标签后翻转方向。并且 true 自身的 `AUROC-0.5` 与 `AUPRC-prevalence` bootstrap CI 下界都必须大于零。不能因为 true 比一个更差的对照好，或只打印 primary AUROC，就声称“图有效”。Onset/continuation 和 span 指标只能在实现并预注册后使用。

## 当前适用范围和下一步

旧的 49-sample Gate 0/1 运行使用 `HISTORY_BLOCK_SIZE=0`，也没有 `direct_seed_rescue` 和结构化 teacher，不能复用。新 runner 默认 300 个 teacher 样本、block size 4、最多 8 个 block。

在扩大训练前依次完成：

1. 用现有数值 counterfactual 检查 Gate 2 是否通过；
2. 若失败，先检查 teacher 信噪比、图 row coverage 和 source-event alignment，不看 hallucination 标签调参；
3. 扩展 entity/date/field counterfactual；
4. 重新提取 prompt-final predictor row，并保存按 relation 的 discarded mass；
5. 在 extractor manifest 中记录精确 model source signature；旧 cache 缺少该 provenance 时只允许无标签 pilot，不允许打开标签；
6. 多随机种子重复训练，最后才做标签评估和 onset/continuation 分层。
