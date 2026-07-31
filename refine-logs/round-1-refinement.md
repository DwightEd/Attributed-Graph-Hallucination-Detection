# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：在不使用幻觉标注训练模型的前提下，把仓库中基于固定 attention 阈值的 token 属性图，改造成能从无标签数据中学习结构并定位 RAG 幻觉的图模型。
- **Must-solve bottleneck**：当前 `tau=0.05` 将不同层、不同头、不同序列长度的 attention 用同一阈值硬切成边；token 粒度过细，attention 边也不等价于事实支持关系。检测阶段又直接使用 token 级 `y_token` 和 BCE，因而不是无监督检测。
- **Non-goals**：不重新训练生成 LLM；不依赖额外人工标注或外部知识库；不堆叠多个独立检测器；不把“无监督”偷换成在 RAGTruth 标签上调参。
- **Constraints**：优先复用仓库现有 attention 提取与 PyG 管线；RAGTruth 标签仅用于最终评测；默认可访问开源生成模型内部状态；暂按单机 1–2 张 24GB GPU 设计，实际资源需再确认。
- **Success condition**：训练、早停、模型选择及异常阈值均不读取幻觉标签；在 token/span AUROC、AUPRC 上优于固定阈值图和非图无监督基线，并在跨任务、跨生成模型测试中保持稳定。

## Anchor Check

- **Original bottleneck**：固定阈值 attention token 图不能稳定表示 claim 的证据依赖，且当前 detector 依赖幻觉标签。
- **Why the revised method still addresses it**：修订后直接从无标签 counterfactual source deletion 学习 evidence-to-claim 边，并用 graph anomaly score 代替 BCE。
- **Reviewer suggestions rejected as drift**：没有引入外部事实核验、LLM judge 或生成模型微调，因为这些会把问题变成另一类验证系统。

## Simplicity Check

- **Dominant contribution after revision**：counterfactual evidence influence distillation——用 evidence deletion 对生成分布的影响作为自动边目标。
- **Components removed or merged**：删除 positive-view consistency、batch source-swap InfoNCE 和独立 anomaly head；把 soft edge scorer 纳入 graph encoder。
- **Reviewer suggestions rejected as unnecessary complexity**：未增加 NLI、RL 或多 agent。
- **Why the remaining mechanism is the smallest adequate route**：一个自动边信号解决“如何无标签构图”，一个 masked graph objective 解决“如何无标签检测异常”。

## Changes Made

### 1. 把启发式负视图替换为可测量的 counterfactual edge target

- **Reviewer said**：source swap 有假负例，且贡献不够锐利。
- **Action**：对每个 response claim 的 top-M evidence candidates 做 leave-one-evidence-out teacher-forced forward，测量实际 token distribution / hidden representation 的变化。
- **Reasoning**：边不再由统一 attention 阈值定义，而由“这段证据是否真的影响该 claim 的生成状态”定义；整个信号无需幻觉标签。
- **Impact on core method**：形成明确的结构学习目标，并与仅靠 attention 或 semantic similarity 的图区分开。

### 2. 精简训练与异常分数

- **Reviewer said**：四项 loss 与四项 score 形成贡献扩散。
- **Action**：训练仅保留 edge distillation、masked latent reconstruction 和轻量 sparsity regularizer；异常分数仅保留 evidence support deficit、reconstruction residual、self-reliance。
- **Reasoning**：所有量都可解释为图结构或属性异常。
- **Impact on core method**：更容易实现、消融和陈述。

## Revised Proposal

# 研究方案：Counterfactual Evidence Graph（CEG）

## Technical Gap

仓库当前把每个 token 当节点，以 `attention > 0.05` 产生 `j -> i` 边，再用人工 token labels 训练 CHARM。这里存在两个不一致：

1. **拓扑不一致**：attention 数值只是内部路由信号，固定阈值不能跨 layer/head/length 比较，也不能说明 source span 是否支持 response claim。
2. **学习目标不一致**：BCE 学到的是标签边界，而不是“正常的证据依赖结构”；去掉标签后现有 pipeline 没有训练信号。

CEG 将图建模问题改写为：**从无标签的 counterfactual generation behavior 中恢复 evidence-to-claim 软依赖图，并把偏离正常证据依赖的 claim 视为属性图异常。**

## Method Thesis

- **One-sentence thesis**：若 source chunk \(e_j\) 是 claim \(c_i\) 的真实生成证据，删除 \(e_j\) 应显著改变 \(c_i\) 的 teacher-forced token distribution 或 hidden trajectory；这一 counterfactual influence 可作为无标签边监督。
- **Smallest adequate intervention**：冻结生成 LLM，只训练一个 relation-aware graph encoder/decoder；不训练 hallucination classifier。
- **Modern primitive usage**：生成 LLM 充当 counterfactual signal teacher，而非文本 judge；graph masked modeling 学习正常结构。

## Contribution Focus

- **Dominant contribution**：无幻觉标签的 counterfactual evidence-edge learning。
- **Supporting contribution**：基于证据图结构/属性残差的 span anomaly score。
- **Explicit non-contributions**：语义片段图、GNN backbone、masked autoencoding 本身不作为新贡献。

## Proposed Method

### 1. Semantic heterogeneous DAG

节点集合：

- \(V_E=\{e_j\}\)：source/prompt evidence chunks；
- \(V_C=\{c_i\}\)：response clauses/claims。

边类型：

- \(e_j \rightarrow c_i\)：evidence support；
- \(c_k \rightarrow c_i, k<i\)：response self-reliance / discourse dependency。

只给每个 claim 生成 top-M candidates。候选召回使用 per-claim attention rank 与 frozen semantic similarity 的并集；此步只控制计算量，不决定最终边。

节点属性：

- span pooled hidden states；
- log-probability、entropy、position、length、node type；
- source-attention mass、entropy、head/layer variance、高频能量。

边属性：

- attention statistics；
- span cosine similarity；
- relative position；
- relation type。

### 2. Counterfactual edge targets

完整输入下，以 teacher forcing 计算 claim \(c_i\) 的实际 token distribution \(P_i\) 与 pooled hidden state \(h_i\)。对候选 evidence \(e_j\) 做删除或 attention-mask，再计算 \(P_i^{-j}, h_i^{-j}\)：

\[
d_{ji} =
\eta\,\frac{1}{|c_i|}\sum_{t\in c_i} JS(P_t,P_t^{-j})
+(1-\eta)\,[1-\cos(h_i,h_i^{-j})].
\]

对同一 claim 内的 \(d_{ji}\) 做 robust rank normalization，得到软边目标 \(\tilde q_{ji}\in[0,1]\)。response-to-response 边可用删除 previous claim 的同样方式产生目标。

软 edge scorer：

\[
q_{ji}=\sigma(g_\phi(x_j,x_i,a_{ji},r_{ji})).
\]

用 Huber/BCE-soft loss 蒸馏：

\[
\mathcal L_{\text{edge}}=\sum_{(j,i)}\operatorname{Huber}(q_{ji},\tilde q_{ji}).
\]

训练后，推理只需完整输入的一次 forward 和 edge scorer，无需再做所有 deletion passes。若离线计算昂贵，可只对 top-3 evidence candidates、随机 30% 训练 claims 计算 deletion target。

### 3. Robust masked graph modeling

在学得的软图上遮蔽部分 claim node attributes，让 heterogeneous GNN 从 evidence parents 和 previous claims 恢复冻结 teacher latent：

\[
\mathcal L_{\text{mask}} =
\sum_i \rho\left(\hat z_i-\operatorname{sg}(z_i^{teacher})\right).
\]

使用 warm-up 后的 trimmed loss：每个 batch 仅对 reconstruction residual 较低的 \(1-\epsilon\) claims 回传 \(\mathcal L_{\text{mask}}\)，降低无标签训练集中已有幻觉对“正常模式”的污染。这里 \(\epsilon\) 固定为一个小范围（如 0.1），不能根据测试标签搜索。

总目标：

\[
\mathcal L=\mathcal L_{\text{edge}}
+\lambda_m\mathcal L_{\text{mask}}
+\lambda_s\sum_i\|q_{\cdot i}\|_1.
\]

新 trainable component 仍只有一个 graph encoder/decoder，edge scorer 是其中的输入门控。

### 4. Unsupervised anomaly score

对 claim \(c_i\)：

\[
S_i =
\alpha\underbrace{(1-\max_{e_j\in V_E}q_{ji})}_{\text{support deficit}}
+\beta\underbrace{\|\hat z_i-z_i^{teacher}\|}_{\text{attribute/structure residual}}
+\gamma\underbrace{
\frac{\sum_{k<i}q_{ki}}
{\sum_jq_{ji}+\sum_{k<i}q_{ki}+\varepsilon}
}_{\text{self-reliance}}.
\]

三个分量分别回答：

1. 是否存在强 source evidence parent；
2. claim 是否能被正常 evidence neighborhood 解释；
3. claim 是否主要依赖先前回答而非 source。

训练与模型选择不读取 `y_token`。权重 \(\alpha,\beta,\gamma\) 可在无标签 validation 上通过稳定性准则设定，最小实现先设等权并做无标签尺度标准化。主评测报告 AUROC/AUPRC；二值 threshold 使用 train score 的 median + MAD 或 two-component mixture。

### 5. Integration into This Repository

#### `get_attention.py`

- 将 `output_hidden_states=False` 改为 `True`；
- 保存 1–3 个预选层的 hidden states、token logits/log-probs；
- 保留 labels 字段只为后续评测，数据提取与训练函数不得读取；
- 把 hard-coded model/filter/path 改为 CLI/config。

#### `processed_graphs_attribute.py`

- 删除全局 `tau=0.05` 作为正式构图规则；
- 新增 span-to-token mapping；
- 生成 evidence/claim nodes、candidate relations 与 pooled attributes；
- 训练集离线计算 top-M deletion influence targets；
- 输出 `node_type`, `span`, `candidate_edge_index`, `edge_attr`, `edge_target`。

#### `train_charm_grid.py`

- 保留 PyG batching 框架；
- 将 `CHARM + BCEWithLogitsLoss(y_token)` 替换为 relation-aware soft-edge encoder + masked latent decoder；
- `evaluate()` 才可加载 `y_token`；
- early stopping 使用无标签 validation reconstruction/edge loss，不能使用 val AUPR；
- 删除 `compute_pos_weight()` 与监督阈值选择。

#### `GNN_train.py`

该文件读取 `he_incidence_index/he_attr/he_mark`，与当前 `processed_graphs_attribute.py` 输出不一致，且 hypergraph 分支引用未传入的局部变量。第一版 CEG 不应同时维护 graph 与 hypergraph 两条路线；先以 `train_charm_grid.py` 为唯一入口。

### 6. Failure Modes and Diagnostics

- **Parametric-memory masking effect弱，但 claim 仍被 source 支持**：candidate 边融合 semantic similarity；异常分数不只依赖 deletion influence。单独报告 seen-fact QA 与 source-specific D2T。
- **删除 chunk 破坏 prompt 格式**：用 attention mask 或等长 neutral mask，确保 token position 对齐；两者做 pilot 对比。
- **训练污染**：检查不同 trim ratio 下排名稳定性；若极敏感，说明“多数正常模式”假设不成立。
- **只学任务/位置偏差**：做位置打乱、移除 attention、跨 task/model 测试。
- **切分误差**：sentence 与 clause 两种固定切分器比较；不在 label 上选择 splitter。

## Novelty and Elegance Argument

与 semantic internal reasoning graph 的关键差异不是节点名字，而是边的来源和训练范式：

- 现有语义图通常由 LRP/attention 后再用 top-k 或离散梯度选边，并训练监督 discriminator；
- CEG 的边目标来自 source deletion 对生成分布的可测量影响；
- edge learner 将昂贵 counterfactual influence 蒸馏为单次前向软图；
- detector 通过 graph anomaly residual 工作，不需要幻觉标签。

因此论文的单一主张是：**counterfactual generation behavior 可以替代人工标签，学习适合 hallucination anomaly detection 的 evidence graph topology。**

## Claim-Driven Validation Sketch

### Claim 1：counterfactual soft edges 比 attention 阈值/top-k 更能恢复 grounding

- **Minimal experiment**：固定同一 node attributes 与 GNN，比较 `tau=0.05`、attention top-k、semantic kNN、SIRG-style adaptive edges、CEG edges。
- **Metric**：token/span AUROC 与 AUPRC；跨 task/model transfer。
- **Must-run ablation**：去掉 deletion target，改为 attention regression；token nodes 替代 claim nodes。
- **Expected evidence**：CEG 特别改善 D2T/QA 中 source-specific unsupported claims。

### Claim 2：收益来自 graph anomaly modeling，而非简单 deletion score

- **Minimal experiment**：比较 raw deletion influence、非图 MLP/Isolation Forest、仅 support deficit、完整 CEG score。
- **Metric**：AUROC/AUPRC；train-score threshold 下的 Macro-F1 作为次要指标。
- **Deletion check**：若 masked graph residual 不优于 raw edge score，则删除 GNN，承认最小非图方法更合适。

## Experiment Handoff Inputs

- **Must-prove**：训练/early stop/model selection 零标签；counterfactual edges 优于 heuristic edges；图残差带来独立收益。
- **Highest-risk assumption**：source deletion influence 能区分 evidence grounding 与 parametric recall。
- **Pilot first**：每个任务各 100 个样本，top-3 candidates；先画 grounded/hallucinated claim 的 \(d_{ji}\) 分布，再决定是否扩大训练。

## Compute & Timeline Estimate

- **Pilot**：约 300 样本 × 每 claim 3 次 masked forward，预计 5–15 GPU 小时。
- **Full preprocessing**：依序列长度与 claim 数约 40–150 GPU 小时；可缓存 KV/hidden 或仅抽样 claims。
- **Graph training**：单张 24GB GPU，约 10–30 GPU 小时。
- **Timeline**：3–5 天 pilot，1 周实现 CEG 数据格式与训练，1 周完成两个核心实验块。
