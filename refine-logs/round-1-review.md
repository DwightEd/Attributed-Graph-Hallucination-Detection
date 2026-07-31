# Round 1 方法审查

> **审查来源说明**：`research-refine` 要求使用 GPT-5.5 Codex reviewer backend，但当前环境未暴露 `mcp__codex__codex` / `codex-reply`，工具发现也没有返回等价后端。以下是本地方法审计，不冒充外部审查；因此本轮不能给出 `READY` 结论。

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 9 | 保持了“零幻觉标签训练 + 改变图结构”的核心问题。 |
| Method Specificity | 8 | 节点、边、属性、loss 和 score 已可实现，但三个视图与四项 loss 的交互仍偏复杂。 |
| Contribution Quality | 7 | claim–evidence 图、masked reconstruction、view consistency、source swap 都有合理性，但主贡献不够单一。 |
| Frontier Leverage | 8 | 合理使用冻结 LLM 表示与 graph masked modeling，没有强行加入 LLM judge。 |
| Feasibility | 7 | 语义切分和多视图可行；但 source swap 可能产生假负例，且多次图编码增加调参与解释成本。 |
| Validation Focus | 8 | 两个 claim-driven 实验基本充分。 |
| Venue Readiness | 7 | 与 2026 年 SIRG 的语义图接近，需要一个更明确的机制级差异。 |
| **Overall** | **7.8** | `REVISE` |

## Critical / Important Issues

### 1. Dominant contribution is diffuse

- **Weakness**：当前同时强调语义图、软结构学习、masked reconstruction、positive-view consistency、source swap 与复合异常分数。
- **Fix**：把主机制收敛为“用 source deletion 的 counterfactual influence 生成无标签 edge targets，再把该信号蒸馏成单次前向的软证据图”。masked graph autoencoding 仅作为异常建模，不再并列宣称多视图学习。
- **Priority**：CRITICAL。

### 2. Source swap is a noisy proxy for factual support

- **Weakness**：batch 内错误 source 可能碰巧支持 claim；反之，正确 source 对模型 logits 的影响也可能很小。
- **Fix**：对每个 claim 的 top-M candidate evidence 做 leave-one-evidence-out teacher-forced forward，以实际 token log-probability / hidden-state change 定义边强度；semantic similarity 只用于候选生成与补偿 parametric-memory 情形。
- **Priority**：CRITICAL。

### 3. Fully unsupervised thresholding needs separation from ranking evaluation

- **Weakness**：MAD/GMM threshold 可以使用，但其稳定性不应与 AUROC 混在一起。
- **Fix**：主结果用 AUROC/AUPRC；二值 F1 单列“deployment threshold”设置，并保证 threshold 只拟合无标签 train scores。
- **Priority**：IMPORTANT。

### 4. Contaminated training distribution

- **Weakness**：RAGTruth 无标签训练集合并非全是正常样本，普通 autoencoder 可能学习复现异常。
- **Fix**：使用 per-claim trimmed reconstruction loss 或 warm-up 后保留低分 normal core，不新增网络模块。
- **Priority**：IMPORTANT。

## Simplification Opportunities

1. 删除独立的 positive-view consistency loss。
2. 删除 batch source-swap InfoNCE，改为可测量的 source-deletion influence。
3. 将四项异常分数缩减为三项：结构支持缺失、masked reconstruction error、self-reliance ratio。

## Modernization Opportunities

1. 使用冻结 LLM 的 teacher-forced counterfactual influence 作为自动 edge supervision，而不是把 LLM 当文本 judge。
2. 使用 latent target reconstruction，避免直接重构高维、噪声大的全层 attention。

## Drift Warning

NONE。

## Verdict

`REVISE`。方向正确，但需要把“多视图无监督图模型”收敛为一个更锐利的 counterfactual edge-learning 机制。
