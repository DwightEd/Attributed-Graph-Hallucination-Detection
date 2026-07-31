# Review Summary

**Problem**：无幻觉标签地改变当前 attention token 图建模，并完成 RAG 幻觉定位。  
**Initial Approach**：固定 `tau=0.05` 的 token attention 图 + CHARM + token BCE。  
**Date**：2026-07-31  
**Rounds**：1 个本地审计/修订轮次  
**Final Local Assessment**：8.5 / 10  
**Final Verdict**：REVISE（外部 reviewer backend 不可用，不能判定 READY）

## Problem Anchor

在不使用幻觉标注训练、早停、选模和定阈值的前提下，将固定 attention 阈值 token 图改为从无标签 counterfactual behavior 学习的 evidence–claim 软图，并以图异常定位 hallucinated spans。

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | 多视图、source swap、四项 loss 使贡献扩散；与已有 semantic graph 的差异不足 | 删除 source-swap 与独立 consistency loss；改用 source-deletion influence 自动产生 edge targets；缩减为 edge distillation + masked graph modeling | 基本解决 | deletion influence 可能混淆 contextual grounding 与 parametric recall |

## Overall Evolution

- 从“在现有 token 图上使用无监督 loss”转为“重新定义节点、边和训练信号”。
- 主贡献收敛为 counterfactual evidence-edge learning。
- 保留一个 graph encoder/decoder，不增加监督 classifier、NLI、judge 或 RL。
- 把 RAGTruth labels 严格隔离到最终 evaluation。

## Final Status

- **Anchor status**：preserved。
- **Focus status**：tight。
- **Modernity status**：appropriately frontier-aware。
- **Strongest part**：使用生成模型自身的 source-deletion sensitivity 作为无标签边监督，再蒸馏为低成本软图。
- **Remaining weakness**：需要先做 300 样本 pilot 验证 deletion influence 与 hallucination label 的分布分离；当前仓库也缺 README/requirements 且存在运行错误，尚未实现方案。
