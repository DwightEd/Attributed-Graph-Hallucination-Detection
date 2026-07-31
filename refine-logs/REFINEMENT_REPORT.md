# Refinement Report

**Problem**：无监督改变属性图建模以检测 RAG hallucination。  
**Initial Approach**：固定 attention 阈值 token graph + supervised BCE。  
**Date**：2026-07-31  
**Rounds**：1  
**Final Local Assessment**：8.5 / 10  
**Final Verdict**：REVISE（未获得技能要求的外部 GPT-5.5 复审）

## Problem Anchor

训练全流程不读取 hallucination labels；学习的图必须显式表示 evidence 对 response claims 的支持关系；最终以无监督异常分数定位 span。

## Output Files

- Review summary：`refine-logs/REVIEW_SUMMARY.md`
- Final proposal：`refine-logs/FINAL_PROPOSAL.md`
- Initial proposal：`refine-logs/round-0-initial-proposal.md`
- Local audit：`refine-logs/round-1-review.md`
- Full refinement：`refine-logs/round-1-refinement.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 initial/local audit | 9 | 8 | 7 | 8 | 7 | 8 | 7 | 7.8 | REVISE |
| 1 refined/local assessment | 9 | 9 | 8.5 | 8 | 7.5 | 8.5 | 8 | 8.5 | REVISE* |

\* 方向已明显收敛，但未进行技能指定的外部复审，不能标记 READY。

## Method Evolution Highlights

1. 节点从 subword token 改为 evidence chunks 与 response claims。
2. 边从统一 attention 阈值改为 source-deletion counterfactual influence 的无标签蒸馏。
3. 检测从 BCE classifier 改为 support deficit、masked reconstruction residual 与 self-reliance 组成的 graph anomaly score。

## Pushback / Drift Log

| Round | Potential Suggestion | Author Response | Outcome |
|---|---|---|---|
| 1 | 加入 NLI/LLM judge 增强事实核验 | 会把内部无监督图学习漂移成外部 verifier，因此不采用 | rejected |
| 1 | 保留 source-swap contrastive view | 假负例较多，以可测量 deletion influence 取代 | replaced |

## Remaining Weaknesses

- Counterfactual deletion 需要额外 forward passes，预处理成本高。
- 生成模型的 parametric memory 可能导致“有文本支持但删除后影响小”。
- claim splitter 与 source chunking 可能影响结构稳定性。
- 仓库当前没有可复现配置或依赖文件，且训练脚本含未定义变量。

## Raw Reviewer Responses

没有可用的外部 reviewer response。`round-1-review.md` 是明确标注的本地方法审计，不是 GPT-5.5 输出。

## Next Steps

先做每个 RAGTruth task 100 个样本、每个 claim top-3 evidence candidates 的 pilot。只有当 deletion influence 对 grounded/hallucinated spans 呈现可复现分离时，再进入完整实现与实验计划。
