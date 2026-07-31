# 研究方案：无标签的证据归因图学习用于 RAG 幻觉检测

## Problem Anchor

- **Bottom-line problem**：在不使用幻觉标注训练模型的前提下，把仓库中基于固定 attention 阈值的 token 属性图，改造成能从无标签数据中学习结构并定位 RAG 幻觉的图模型。
- **Must-solve bottleneck**：当前 `tau=0.05` 将不同层、不同头、不同序列长度的 attention 用同一阈值硬切成边；token 粒度过细，attention 边也不等价于事实支持关系。检测阶段又直接使用 token 级 `y_token` 和 BCE，因而不是无监督检测。
- **Non-goals**：不重新训练生成 LLM；不依赖额外人工标注或外部知识库；不堆叠多个独立检测器；不把“无监督”偷换成在 RAGTruth 标签上调参。
- **Constraints**：优先复用仓库现有 attention 提取与 PyG 管线；RAGTruth 标签仅用于最终评测；默认可访问开源生成模型内部状态；暂按单机 1–2 张 24GB GPU 设计，实际资源需再确认。
- **Success condition**：训练、早停、模型选择及异常阈值均不读取幻觉标签；在 token/span AUROC、AUPRC 上优于固定阈值图和非图无监督基线，并在跨任务、跨生成模型测试中保持稳定。

## Technical Gap

当前流程是：

```text
prompt + response
  -> 全层全头 attention
  -> self-attention 对角线作为节点属性
  -> 任一 head/layer attention > 0.05 即连边
  -> CHARM 消息传递
  -> token 标签 BCE 分类
```

失效点不只是“缺少一种无监督 loss”，而是图的语义本身不对齐任务：

1. 节点代表 subword token，幻觉通常是 claim/span 级现象；
2. 边代表大于固定阈值的内部 attention，不等同于“证据支持该 claim”；
3. 不同 head/layer 的数值分布直接拼接，固定阈值缺少可比性；
4. 监督分类器可能只学习模型、任务或位置偏差，而不是证据依赖异常。

### 两条候选路线

- **Route A：最小改造**。保留 token 节点，以每个目标 token 的 top-k/分位数候选边替代固定阈值，再用 GraphMAE 式 masked reconstruction 产生异常分数。优点是实现快；缺点是仍把 token attention 当成事实支持关系，研究上限较低。
- **Route B：语义证据图（推荐）**。把图改为 evidence unit 与 response claim 的异构有向图，用无标签的 masked reconstruction、证据置换对比和多视图一致性联合学习软边。优点是图结构与“是否被上下文支持”直接对齐；代价是需要 span 切分和一次新的图构建器。

选择 Route B。最主要的论文贡献不是“用了 GraphMAE”，而是：**用反事实证据一致性，在无幻觉标签条件下学习 claim–evidence 软拓扑，并以拓扑不一致作为幻觉异常信号。**

## Method Thesis

- **One-sentence thesis**：一个被上下文支持的 claim，应当在语义保持的证据视图中稳定连接到相同证据、在错误证据置换后失去该连接；利用这一无标签规律可以同时学习图结构与幻觉异常分数。
- **Why this is the smallest adequate intervention**：只新增一个软结构学习器和一个共享图编码器/解码器；生成 LLM 与文本编码器冻结，不增加监督分类头。
- **Why this route is timely**：它结合 foundation model 的冻结语义表示与 graph masked modeling，但核心训练信号仍来自数据本身而非 LLM-as-a-judge。

## Contribution Focus

- **Dominant contribution**：反事实证据一致性驱动的无监督 claim–evidence 图结构学习。
- **Supporting contribution**：由重构误差、证据连接缺失和跨视图不稳定组成的无标签 span 异常分数。
- **Explicit non-contributions**：不声称首次使用语义图；不声称 attention 是因果解释；不提出新的通用 GNN backbone。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**：生成 LLM、tokenizer、sentence/claim splitter、可选的冻结 sentence encoder。
- **New trainable components**：
  1. relation-aware soft edge scorer；
  2. 共享的 heterogeneous GNN encoder + lightweight decoder。
- **Tempting additions intentionally not used**：LLM judge、强化学习、外部知识图谱、独立 NLI 分类器、监督 hallucination head。

### System Overview

```text
source/prompt ---------------------- response
    |                                  |
evidence units e1...em             claims c1...cn
    |                                  |
frozen hidden/attention/semantic attributes
                    |
       candidate heterogeneous DAG
       e -> c, previous-c -> c
                    |
       soft edge scorer q_phi(e_ij)
                    |
        masked heterogeneous GNN
                    |
 reconstruction + counterfactual consistency
                    |
      per-claim anomaly score S(ci)
                    |
       map claim score back to tokens
```

### Representation Design

- **Evidence nodes**：prompt/source 按句子或长度受控的 semantic chunks 切分。
- **Claim nodes**：response 按句子内的谓词—论元/内容短语切分；最小实现可先使用 clause spans。
- **Node attributes**：
  - span 内冻结 hidden states 的 mean/max pooling；
  - token log-probability、entropy；
  - span 的位置、长度、node type；
  - 指向 source 与 previous response 的 attention mass、entropy、跨 head/layer 方差及高频能量。
- **Candidate edges**：
  - `evidence -> claim`；
  - `previous claim -> claim`，保持自回归有向无环约束；
  - 每个 claim 仅保留按 attention rank 或 semantic similarity 得到的 top-M 候选，避免全连接。
- **Edge attributes**：多层多头 attention 统计、span embedding cosine、相对位置、relation type。
- **Soft topology**：

  \[
  q_{ij}=\operatorname{sigmoid}(g_\phi[x_i,x_j,a_{ij},r_{ij}])
  \]

  训练时用连续边权消息传递；推理时无需把边硬阈值化。

### Self-supervised Training

对每个无标签样本产生三类视图：

1. **原始视图** \(G\)；
2. **语义保持视图** \(G^+\)：对无关 source chunks 做 dropout、对 head/layer 做 dropout、轻微移动 chunk 边界；
3. **反事实证据视图** \(G^-\)：从 batch 内置换 source chunks，或删除当前 claim 最可能依赖的证据。

总目标：

\[
\mathcal L =
\mathcal L_{\text{mask}}
+\lambda_c\mathcal L_{\text{cons}}
+\lambda_n\mathcal L_{\text{cf}}
+\lambda_s\mathcal L_{\text{sparse}}.
\]

- **Masked latent reconstruction** \(\mathcal L_{\text{mask}}\)：遮蔽 claim/evidence 的一部分 node attributes，让 GNN 从邻域恢复冻结 teacher embedding；避免直接重构超高维原始 attention。
- **Positive-view consistency** \(\mathcal L_{\text{cons}}\)：同一 claim 在 \(G\) 与 \(G^+\) 中的 embedding 和 evidence-edge 分布保持一致。
- **Counterfactual evidence loss** \(\mathcal L_{\text{cf}}\)：真实上下文下 claim–evidence agreement 应高于 batch 内置换上下文，采用 margin/InfoNCE；这是自动构造的训练信号，不使用幻觉标签。
- **Sparse causal prior** \(\mathcal L_{\text{sparse}}\)：约束每个 claim 仅保留少量主要证据，并禁止未来 claim 指向过去 claim。

为避免模型把所有节点都重构得很好，只在 latent space 预测冻结 teacher 表示，并对 edge scorer 加熵/稀疏约束；反事实 source swap 防止 identity shortcut。

### Unsupervised Anomaly Score

对 claim \(c_i\)：

\[
S_i =
\alpha\,E^{\text{mask}}_i
+\beta\,(1-\max_{e_j}q_{ji})
+\gamma\,JS(q_i(G),q_i(G^+))
+\delta\,\max(0,m-A_i(G)+A_i(G^-)).
\]

分别表示：

1. masked latent reconstruction error；
2. 无可靠 evidence parent；
3. 语义保持扰动下拓扑不稳定；
4. 正确证据没有比置换证据产生更强 agreement。

将 claim score 广播回其 token span，即可与 RAGTruth 的 token labels 对齐评测。主指标用不依赖阈值的 AUROC/AUPRC。若部署必须输出二值结果，在无标签训练分数上用 median + MAD 或 two-component mixture 的交点定阈值，并单独报告阈值敏感性。

### Inference Path

1. 冻结生成模型抽取一次 hidden states、attention 与 token probabilities；
2. 切分 evidence/claim spans 并池化属性；
3. edge scorer 产生软图；
4. 对 3–5 个轻量 positive views 重复图编码；
5. 计算 \(S_i\)，映射到 token/span。

推理不需要 RAGTruth 标签，也不调用外部 judge。若生成模型不可访问，可退化为 frozen proxy encoder 的语义属性，但这属于另一实验设置。

### Failure Modes and Diagnostics

- **多数训练数据本身包含幻觉**：正常模式会被污染。诊断 score 分布与任务/模型的相关性；用 robust loss、trimmed mean 或低分样本自步学习缓解。
- **source swap 产生“仍可支持”的假负例**：用跨 source_id、低 embedding similarity 的 chunk 置换，并过滤高相似负例。
- **claim splitter 误切**：先做 sentence/clause 两种粒度一致性；不让 LLM judge 参与主方法。
- **模型仅学习 response 位置或 task type**：做 source-shuffle、去 attention、去 hidden、位置随机化消融。
- **attention 不是忠实归因**：将 attention 仅作为候选和属性，而不是硬真值；与 hidden-only、LRP/gradient attribution 版本比较。

### Novelty and Elegance Argument

最近的 semantic-level internal reasoning graph 已把上下文与回答切成语义片段，并通过 LRP 构图，但其边仍由 top-k/离散梯度规则选择，最终对 AlignScore 进行下游交叉熵训练。本方案不把“语义图”本身当作新意，而把以下机制作为区别：

1. 图拓扑由无标签的反事实证据一致性学习，而非固定阈值/top-k；
2. 检测是 graph anomaly scoring，而非带标签二分类；
3. 正确 source、语义保持视图和置换 source 构成可验证的自监督结构信号。

## Claim-Driven Validation Sketch

### Claim 1：自监督学习的 claim–evidence 软拓扑比固定 attention 图更能表示 grounding

- **Minimal experiment**：在 RAGTruth 上仅用无标签 train split 学习；比较固定 `tau=0.05`、per-node top-k、semantic similarity graph、SIRG-style adaptive heuristic 与 proposed soft graph。
- **Decisive metric**：token/span AUROC、AUPRC；另用人工标签仅做评测，不用于选择超参。
- **Necessary ablations**：去掉 \(\mathcal L_{\text{cf}}\)、固定结构只训练 encoder、token nodes 替代 claim nodes。
- **Expected evidence**：软图在跨 task 与跨 generator 测试中提升，尤其 source-grounding 类型幻觉。

### Claim 2：异常分数的有效性来自证据反事实，而非一般语言不确定性

- **Minimal experiment**：比较完整分数、仅 log-prob/entropy、仅 reconstruction、仅 evidence-disconnect。
- **Decisive metric**：跨模型 AUROC 及 source-shuffle 后 score 增量；grounded claim 的增量应显著小于 hallucinated claim。
- **Simplification check**：若 \(\mathcal L_{\text{cons}}\) 或 previous-claim edges 无稳定收益，删除而不继续叠模块。

## Experiment Handoff Inputs

- **Must-prove claims**：标签自由；学习结构优于启发式结构；证据反事实提供独立信号。
- **Must-run ablations**：token vs claim、hard vs soft edge、无 counterfactual loss、无 attention、无 hidden。
- **Critical datasets / metrics**：RAGTruth QA/Data-to-Text/Summary；AUROC、AUPRC、跨 task/model 泛化。
- **Highest-risk assumptions**：大多数训练 claim 有可恢复的正常 grounding；source swap 能构造足够可靠的负视图。

## Compute & Timeline Estimate

- **Estimated GPU-hours**：attention/hidden 抽取约 20–80 GPU 小时，取决于模型和序列长度；图 SSL 原型约 10–30 GPU 小时。该估计需在 100 个样本 pilot 后修正。
- **Data / annotation cost**：训练 0 人工标签；RAGTruth 标签只用于最终离线评测。
- **Timeline**：1 周完成数据与语义图构建，1 周完成 SSL 与异常评分，1 周做最小对照与跨模型验证。
