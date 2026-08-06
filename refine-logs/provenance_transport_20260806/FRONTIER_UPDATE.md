# CEPT 前沿调研更新：从历史依赖转向证据来源

## 结论

“生成 token 会依赖 response history”是自回归模型的结构事实，不值得再作为研究假设。真正需要检验的是：**历史 token 的 attention routing，是否与由 evidence intervention 标定的 ancestry surrogate 一致。**

同一条 `response -> response` 路径可能是两种完全不同的机制：

```text
evidence -> r1 -> r2       grounded relay
             r1 -> r2       self-reinforcing relay（缺少 evidence ancestry）
```

因此 CEPT 不把 RR attention 高直接判成幻觉，而是估计 retained RR exposure 中，有多少能由 evidence counterfactual 的 K/V 救援效应解释。当前 teacher-forced 图没有证明自由生成时的错误会被放大，所以输出必须称为 `unsupported response-history routing exposure`，不能称为已经发现的 self-reinforcement 或物理信息流。整体 evidence deficit 只作为辅助分数。

## 与最接近工作的边界

- [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) 使用 context/generated-token attention 比例训练有标签线性分类器；它不追踪 response history 是否继承 evidence。
- [SIRG](https://aclanthology.org/2026.findings-acl.1385/) 构造语义片段级 LRP 推理图，并用有标签判别器检测幻觉；它与只比较拓扑模式的 [TOHA](https://aclanthology.org/2026.acl-long.704/) 不应混为同一类方法。
- [TPA](https://aclanthology.org/2026.acl-long.1159/) 分解 Query、RAG Context、Past Token、Self 等来源，再训练有标签 XGBoost；它没有识别 grounded history relay。
- [Two Pathways to Truthfulness](https://aclanthology.org/2026.acl-long.1173/) 用 attention knockout 和 token patching 区分问题锚定与答案锚定路径，但目标不是外部证据经 response history 的逐 token 中介。
- [CausalGaze](https://aclanthology.org/2026.findings-acl.1943/) 的因果量来自有标签检测器损失对 attention edge 的梯度，并非外部 evidence intervention。
- [ContextCite](https://arxiv.org/abs/2409.00729) 用上下文片段纳入/排除干预学习 attribution surrogate，但不分解 direct evidence 与 response-history-mediated evidence effect。
- [CoDA](https://aclanthology.org/2026.findings-acl.576/) 发现中后层 context routing 弱化并进行生成干预；[Attribution Blind Spot](https://arxiv.org/abs/2605.26778) 进一步提醒 attribution 与真实使用之间可能失配。这正是 CEPT 必须设置 intervention teacher 和结构对照的原因。
- [First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/abs/2507.20836) 已提出 onset 与 conditional hallucination 的差异，因此 CEPT 可把二者作为预注册分层分析，但不能声称该区分本身新颖。

截至 2026-08-06，在本次检索覆盖的同行评审论文和相关预印本中，尚未发现单一方法同时结合：evidence intervention、response-history path-specific mediation、observational–causal transport gap，以及无幻觉标签的 attention-routing student distillation。这是**组合层面的检索差异**，不是“首个方法”的证明。

## 方法和主张边界

“attention-only”只指部署时的 student 输入。Teacher 明确使用模型 K/V intervention 和目标 token log-prob，因此整个研究不能被描述为只使用 attention。

当前 counterfactual adapter 只实现 `numeric_digit_surface_preserving_v1`。它适合验证 transport 机制，但不足以支撑一般 RAGTruth 证据来源主张；正式实验还需要 entity、date 和结构化字段等表面保持的 counterfactual。

即使扩展干预，方法最多检测**相对于给定 source 的 faithfulness/routing risk**，不能判断开放世界事实真伪：错误 source 可能支持错误答案，正确参数知识也可能绕过 source。

每个 `relation × layer × head` gate 是预测 K/V 干预效果的低容量 surrogate 权重。因为 teacher 并未逐 head 干预，不能把这些 gate 宣称为已识别的“因果 head”。

旧 cache 缺少 prompt query rows、完整 residual/MLP 路径以及首个 response predictor row，因此当前上下界只是 attention-subgraph 条件下的 operational interval。RAGTruth test 也已在本项目历史实验中多次查看，后续结果属于 exploratory evidence，而不是 pristine confirmatory evidence。

## 数据依据

主要数据集为 [RAGTruth](https://arxiv.org/abs/2401.00396/)。训练阶段不读取其 hallucination 标签；标签只能在无标签 Gate 2 通过、checkpoint、公式和 prediction hash 冻结后用于事后评估。
